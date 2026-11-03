# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for non-blocking polling of multiproc executor futures."""

import time
from collections import deque

import pytest
import zmq

from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from vllm.utils.network_utils import get_open_zmq_ipc_path
from vllm.v1.executor.multiproc_executor import FutureWrapper


class Response:
    """A response that can be collected once it has arrived."""

    def __init__(self, value: object, calls: list[object]):
        self.value = value
        self.arrived = False
        self.calls = calls

    def ready(self) -> bool:
        return self.arrived

    def get(self) -> object:
        assert self.arrived, "get_response() would block"
        self.calls.append(self.value)
        return self.value


def test_poll_collects_arrived_responses_in_order():
    queue: deque[FutureWrapper] = deque()
    calls: list[object] = []
    first, second = Response("a", calls), Response("b", calls)
    f1 = FutureWrapper(queue, first.get, response_ready=first.ready)
    f2 = FutureWrapper(queue, second.get, response_ready=second.ready)

    assert not f2.poll()
    assert calls == []

    # A response for the later future can't be collected before the earlier one.
    second.arrived = True
    assert not f1.poll()
    assert calls == []

    first.arrived = True
    assert f2.poll()
    assert calls == ["a", "b"]
    assert f1.result() == "a"
    assert f2.result() == "b"
    assert not queue


def test_poll_stops_at_first_response_not_yet_arrived():
    queue: deque[FutureWrapper] = deque()
    calls: list[object] = []
    first, second = Response("a", calls), Response("b", calls)
    f1 = FutureWrapper(queue, first.get, response_ready=first.ready)
    f2 = FutureWrapper(queue, second.get, response_ready=second.ready)

    first.arrived = True
    assert not f2.poll()
    assert f1.done() and f1.result() == "a"
    assert list(queue) == [f2]

    second.arrived = True
    assert f2.poll()
    assert f2.result() == "b"


def test_poll_without_readiness_check_collects_response():
    queue: deque[FutureWrapper] = deque()
    future = FutureWrapper(queue, lambda: "value")
    assert future.poll()
    assert future.result() == "value"


def test_poll_sets_exception_from_response():
    def fail():
        raise ValueError("worker failed")

    queue: deque[FutureWrapper] = deque()
    future = FutureWrapper(queue, fail, response_ready=lambda: True)
    assert future.poll()
    with pytest.raises(ValueError, match="worker failed"):
        future.result()


def test_dispatch_starts_response_condition_busy_window():
    """After a period with no responses, dispatching a new call makes waits
    for its response spin rather than sleep, so it is seen promptly."""
    context = zmq.Context()
    condition = SpinCondition(
        is_reader=True,
        context=context,
        notify_address=get_open_zmq_ipc_path(),
        busy_loop_s=60,
    )
    try:
        condition.last_read = time.monotonic() - 120  # Idle for a while.
        start = time.monotonic()
        condition.wait(timeout_ms=200)
        assert time.monotonic() - start >= 0.15  # Slept.

        response = Response("a", [])
        FutureWrapper(
            deque(),
            response.get,
            response_ready=response.ready,
            response_condition=condition,
        )
        start = time.monotonic()
        condition.wait(timeout_ms=5000)
        assert time.monotonic() - start < 0.1  # Spun.
    finally:
        for sock in (
            condition.local_notify_socket,
            condition.read_cancel_socket,
            condition.write_cancel_socket,
        ):
            sock.close(linger=0)
        context.term()
