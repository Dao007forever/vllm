# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeing prefix-cache entries that can no longer hit (DeadEntryTracker)."""

import random

import pytest
import torch

from tests.v1.core.utils import (
    full_layer,
    mamba_layer,
    new_kv_cache_group,
    sliding_window_layer,
)
from vllm.distributed.kv_events import BlockRemoved
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.dead_entry_tracker import DeadEntryTracker
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4
ATTN, STATE = 0, 1


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def make_pool(num_blocks: int = 64, state_groups=(STATE,)):
    pool = BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )
    tracker = DeadEntryTracker(pool, ATTN, tuple(state_groups))
    pool.dead_tracker = tracker
    return pool, tracker


def make_request(
    request_id: str, tokens: list[int], block_size: int = BLOCK_SIZE
) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, 100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, sha256),
    )


def cache_state(pool, request, idx, group=STATE):
    state = pool.get_new_blocks(1)[0]
    table = [pool.null_block] * idx + [state]
    pool.cache_full_blocks(request, table, idx, idx + 1, BLOCK_SIZE, group)
    return state


def run(pool, tracker, request, num_blocks, state_at, hit=None):
    """Cache ``num_blocks`` attention blocks of ``request`` (reusing the
    blocks of ``hit`` for its first blocks) and states at ``state_at``; then
    free everything as a finished request would."""
    hit = hit or []
    pool.touch(hit)
    attn = list(hit) + pool.get_new_blocks(num_blocks - len(hit))
    pool.cache_full_blocks(request, attn, len(hit), num_blocks, BLOCK_SIZE, ATTN)
    states = [cache_state(pool, request, idx) for idx in state_at]
    pool.free_blocks(reversed(attn))
    pool.free_blocks(states)
    tracker.drain()
    return attn, states


def cached(pool, request, idx, group):
    key = make_block_hash_with_group_id(request.block_hashes[idx], group)
    return bool(pool.cached_blocks_for(key))


def free_queue_head(pool, n):
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()[:n]]


def test_attention_past_the_last_state_is_freed_at_release():
    pool, tracker = make_pool()
    request = make_request("a", list(range(6 * BLOCK_SIZE)))
    attn, _ = run(pool, tracker, request, 6, state_at=[3])
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 4 + [
        False
    ] * 2
    assert cached(pool, request, 3, STATE)
    assert tracker.num_freed == [2, 0]
    # The freed blocks are reused first.
    assert set(free_queue_head(pool, 2)) == {attn[4].block_id, attn[5].block_id}
    # The freed blocks are junction hints.
    assert request.block_hashes[4] in tracker.ghost


def test_evicting_the_state_frees_its_attention_prefix():
    pool, tracker = make_pool()
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    attn, (state,) = run(pool, tracker, request, 4, state_at=[3])
    free_before = pool.get_num_free_blocks()
    pool.evict_blocks({state.block_id})
    tracker.drain()
    assert not any(cached(pool, request, i, ATTN) for i in range(4))
    # Freeing only reorders the free queue.
    assert pool.get_num_free_blocks() == free_before
    assert set(free_queue_head(pool, 4)) == {b.block_id for b in attn}
    assert not tracker.parent


def test_evicting_an_attention_block_kills_everything_below():
    pool, tracker = make_pool()
    request = make_request("a", list(range(6 * BLOCK_SIZE)))
    attn, _ = run(pool, tracker, request, 6, state_at=[2, 5])
    pool.evict_blocks({attn[3].block_id})
    tracker.drain()
    # The state at 5 lost block 3; blocks 4 and 5 lost their state. The state
    # at 2 and its prefix still hit.
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 3 + [
        False
    ] * 3
    assert cached(pool, request, 2, STATE)
    assert not cached(pool, request, 5, STATE)


def test_lru_eviction_of_a_state_frees_its_attention_prefix():
    pool, tracker = make_pool(num_blocks=8)
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    run(pool, tracker, request, 4, state_at=[3])
    # 7 usable blocks: 2 uncached, then the 4 attention blocks and the state.
    # Taking 3 evicts the attention tail first (LRU), which kills the state.
    pool.get_new_blocks(3)
    tracker.drain()
    assert not any(cached(pool, request, i, ATTN) for i in range(4))
    assert not cached(pool, request, 3, STATE)
    assert not tracker.parent


def test_shared_prefix_survives_while_a_branch_has_a_state():
    pool, tracker = make_pool()
    shared = list(range(2 * BLOCK_SIZE))
    x = make_request("x", shared + list(range(100, 100 + 2 * BLOCK_SIZE)))
    y = make_request("y", shared + list(range(200, 200 + 2 * BLOCK_SIZE)))
    attn_x, (state_x,) = run(pool, tracker, x, 4, state_at=[3])
    run(pool, tracker, y, 4, state_at=[3], hit=attn_x[:2])
    pool.evict_blocks({state_x.block_id})
    tracker.drain()
    assert [cached(pool, x, i, ATTN) for i in range(4)] == [True, True, False, False]
    assert all(cached(pool, y, i, ATTN) for i in range(4))
    assert cached(pool, y, 3, STATE)


def test_blocks_in_use_are_never_freed():
    pool, tracker = make_pool()
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    attn = pool.get_new_blocks(4)
    pool.cache_full_blocks(request, attn, 0, 4, BLOCK_SIZE, ATTN)
    tracker.drain()
    # Running request: no state yet, but its blocks are held.
    assert all(cached(pool, request, i, ATTN) for i in range(4))
    pool.free_blocks(reversed(attn))
    tracker.drain()
    assert not any(cached(pool, request, i, ATTN) for i in range(4))


def test_move_block_hashes_is_not_an_eviction():
    pool, tracker = make_pool()
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    _, (state,) = run(pool, tracker, request, 4, state_at=[3])
    pool.touch([state])
    copy = pool.get_new_blocks(1)[0]
    pool.move_block_hashes(state, copy)
    pool.free_blocks([state, copy])
    tracker.drain()
    assert all(cached(pool, request, i, ATTN) for i in range(4))
    assert cached(pool, request, 3, STATE)


def test_state_cached_before_its_attention_blocks():
    """The coordinator caches groups in group order, so a state (group 0 in
    real models) can be cached before the attention blocks of its chain."""
    pool, tracker = make_pool()
    request = make_request("a", list(range(6 * BLOCK_SIZE)))
    state = cache_state(pool, request, 3)
    attn = pool.get_new_blocks(6)
    pool.cache_full_blocks(request, attn, 0, 6, BLOCK_SIZE, ATTN)
    pool.free_blocks(reversed(attn))
    pool.free_blocks([state])
    tracker.drain()
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 4 + [
        False
    ] * 2
    assert cached(pool, request, 3, STATE)


def test_mid_chain_eviction_drops_the_freed_subtree():
    """A block taken mid-chain frees everything below it; the freed nodes
    must leave the tree at once, or each one's queued event walks the
    remaining subtree again (quadratic: seconds per eviction on 600k-token
    chains)."""
    pool, tracker = make_pool(num_blocks=256)
    request = make_request("a", list(range(200 * BLOCK_SIZE)))
    attn, _ = run(pool, tracker, request, 200, state_at=[199])
    visits = 0
    subtree = tracker._subtree

    def counting_subtree(root):
        nonlocal visits
        for node in subtree(root):
            visits += 1
            yield node

    tracker._subtree = counting_subtree  # type: ignore[method-assign]
    pool.evict_blocks({attn[10].block_id})
    tracker.drain()
    assert not any(cached(pool, request, i, ATTN) for i in range(10, 200))
    assert all(request.block_hashes[i] not in tracker.parent for i in range(10, 200))
    assert visits < 3 * 200


def test_several_state_groups():
    """The KV cache grouping splits state layers into several groups. A
    position hits only with a state in every group; losing one frees the rest
    and the attention prefix that only it supported."""
    s1, s2 = 1, 2
    pool, tracker = make_pool(state_groups=(s1, s2))
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    attn = pool.get_new_blocks(4)
    pool.cache_full_blocks(request, attn, 0, 4, BLOCK_SIZE, ATTN)
    states = {g: cache_state(pool, request, 3, g) for g in (s1, s2)}
    pool.free_blocks(reversed(attn))
    pool.free_blocks(states.values())
    tracker.drain()
    assert all(cached(pool, request, i, ATTN) for i in range(4))
    assert cached(pool, request, 3, s1) and cached(pool, request, 3, s2)
    pool.evict_blocks({states[s1].block_id})
    tracker.drain()
    assert not cached(pool, request, 3, s2)
    assert not any(cached(pool, request, i, ATTN) for i in range(4))
    assert not tracker.parent


def test_reset_prefix_cache_resets_the_tracker():
    pool, tracker = make_pool()
    request = make_request("a", list(range(6 * BLOCK_SIZE)))
    run(pool, tracker, request, 6, state_at=[3])
    assert tracker.parent and tracker.ghost
    assert pool.reset_prefix_cache()
    assert pool.dead_tracker is tracker
    assert not tracker.parent and not tracker.ghost


# ---- KVCacheManager / coordinator level -----------------------------------

MANAGER_BLOCK_SIZE = 16
NUM_STATE_GROUPS = 3


def hybrid_config(
    num_blocks: int, block_size: int = MANAGER_BLOCK_SIZE, sliding_window: bool = False
):
    """Three Mamba groups (align mode) and one full-attention group, the
    layout vLLM gives Qwen3-Next; optionally a sliding-window group too (a
    layout the tracker does not support)."""
    attn = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    mamba = MambaSpec(block_size=block_size, shapes=((1, 1),), dtypes=(torch.float32,))
    groups = [
        new_kv_cache_group(
            [f"mamba{i}"],
            mamba_layer(mamba, mamba_cache_mode="align", num_speculative_blocks=0),
        )
        for i in range(NUM_STATE_GROUPS)
    ]
    groups.append(new_kv_cache_group(["attn"], full_layer(attn)))
    if sliding_window:
        groups.append(
            new_kv_cache_group(["swa"], sliding_window_layer(attn, 2 * block_size))
        )
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


ATTN_GROUP = NUM_STATE_GROUPS


def make_manager(
    num_blocks: int, free_dead: bool, sliding_window: bool = False, **kwargs
) -> KVCacheManager:
    return KVCacheManager(
        hybrid_config(num_blocks, sliding_window=sliding_window),
        max_model_len=8192,
        scheduler_block_size=MANAGER_BLOCK_SIZE,
        hash_block_size=MANAGER_BLOCK_SIZE,
        enable_caching=True,
        free_dead_prefix_entries=free_dead,
        **kwargs,
    )


def serve(manager: KVCacheManager, request: Request) -> int:
    """Prefill ``request`` in one step and finish it; returns its hit."""
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    assert (
        manager.allocate_slots(request, request.num_tokens - num_hit, num_hit, blocks)
        is not None
    )
    request.num_computed_tokens = request.num_tokens
    manager.free(request)
    return num_hit


def tokens(seed: int, num_blocks: int) -> list[int]:
    return [seed * 1000 + i for i in range(num_blocks * MANAGER_BLOCK_SIZE)]


def attn_cached(manager: KVCacheManager, request: Request, idx: int) -> bool:
    return (
        manager.block_pool.get_cached_block(request.block_hashes[idx], [ATTN_GROUP])
        is not None
    )


def state_cached(manager: KVCacheManager, request: Request, idx: int, group: int):
    return (
        manager.block_pool.get_cached_block(request.block_hashes[idx], [group])
        is not None
    )


def test_flag_attaches_tracker_only_for_supported_layouts():
    on = make_manager(64, free_dead=True)
    assert on.dead_tracker is not None
    assert on.block_pool.dead_tracker is on.dead_tracker
    assert on.dead_tracker.attn == ATTN_GROUP
    assert on.dead_tracker.states == tuple(range(NUM_STATE_GROUPS))

    off = make_manager(64, free_dead=False)
    assert off.dead_tracker is None and off.block_pool.dead_tracker is None
    assert off.coordinator.dead_tracker is None

    # Pure attention: the flag is ignored.
    attn = AttentionSpec(
        block_size=MANAGER_BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[new_kv_cache_group(["attn"], full_layer(attn))],
    )
    manager = KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=MANAGER_BLOCK_SIZE,
        hash_block_size=MANAGER_BLOCK_SIZE,
        free_dead_prefix_entries=True,
    )
    assert manager.dead_tracker is None and manager.block_pool.dead_tracker is None

    # Two full-attention groups: unsupported, runs without the tracker.
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["a0"], full_layer(attn)),
            new_kv_cache_group(["a1"], full_layer(attn)),
            *hybrid_config(64).kv_cache_groups[:1],
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=MANAGER_BLOCK_SIZE,
        hash_block_size=MANAGER_BLOCK_SIZE,
        free_dead_prefix_entries=True,
    )
    assert manager.dead_tracker is None and manager.block_pool.dead_tracker is None


def _trace(manager: KVCacheManager, seed: int) -> list:
    """A random multi-turn workload; returns everything observable."""
    rng = random.Random(seed)
    out: list = []
    histories: dict[int, list[int]] = {}
    for step in range(60):
        session = rng.randrange(6)
        history = histories.get(session, [session * 100_000])
        history = history + [rng.randrange(50_000) for _ in range(rng.randrange(8, 70))]
        histories[session] = history
        request = make_request(str(step), history, MANAGER_BLOCK_SIZE)
        blocks, num_hit, boundary = manager.get_computed_blocks(request)
        new_blocks = manager.allocate_slots(
            request, request.num_tokens - num_hit, num_hit, blocks
        )
        out.append((num_hit, boundary, new_blocks is not None))
        if new_blocks is None:
            continue
        out.append(manager.get_block_ids(request.request_id))
        request.num_computed_tokens = request.num_tokens
        manager.free(request)
        out.append(
            [
                b.block_id
                for b in manager.block_pool.free_block_queue.get_all_free_blocks()
            ]
        )
        out.append(manager.take_events())
    return out


def test_flag_on_changes_supported_runs_only():
    """The tracker changes a run only where it is attached: with the flag on
    for a layout it does not support, the run is identical to flag off."""
    off = make_manager(48, free_dead=False, enable_kv_cache_events=True)
    assert off.block_pool.dead_tracker is None
    on = make_manager(48, free_dead=True, enable_kv_cache_events=True)
    # Sanity: this workload creates dead entries for the tracker to free.
    assert _trace(on, seed=0) != _trace(off, seed=0)
    assert on.dead_tracker is not None and sum(on.dead_tracker.num_freed) > 0

    off = make_manager(
        64, free_dead=False, sliding_window=True, enable_kv_cache_events=True
    )
    on = make_manager(
        64, free_dead=True, sliding_window=True, enable_kv_cache_events=True
    )
    assert on.dead_tracker is None and on.block_pool.dead_tracker is None
    assert _trace(on, seed=0) == _trace(off, seed=0)


@pytest.mark.parametrize("free_dead", [False, True])
def test_dead_attention_chain_is_freed_after_its_state_is_evicted(free_dead):
    collector = KVCacheMetricsCollector(sample_rate=1.0)
    manager = make_manager(
        64,
        free_dead=free_dead,
        enable_kv_cache_events=True,
        metrics_collector=collector,
    )
    pool = manager.block_pool
    # ``live`` is cached first, so LRU evicts it before ``dead``.
    live = make_request("live", tokens(1, 2), MANAGER_BLOCK_SIZE)
    dead = make_request("dead", tokens(2, 4), MANAGER_BLOCK_SIZE)
    assert serve(manager, live) == 0
    assert serve(manager, dead) == 0
    assert all(attn_cached(manager, dead, i) for i in range(4))
    assert all(state_cached(manager, dead, 3, g) for g in range(NUM_STATE_GROUPS))
    manager.take_events()
    collector.drain_events()
    free_list = pool.free_block_queue.get_all_free_blocks()
    # Never-cached blocks come first in the free queue, then ``live``.
    num_uncached = next(i for i, b in enumerate(free_list) if b.block_hash)

    # A state of ``dead`` is evicted (e.g. a KV connector reported it).
    state = pool.get_cached_block(dead.block_hashes[3], [0])[0]
    manager.evict_blocks({state.block_id})
    events = manager.take_events()

    # Without that state nothing of ``dead`` can hit.
    probe = make_request("probe", tokens(2, 4) + [7] * 5, MANAGER_BLOCK_SIZE)
    assert manager.get_computed_blocks(probe)[1] == 0

    if not free_dead:
        # The useless entries stay until LRU reaches them.
        assert all(attn_cached(manager, dead, i) for i in range(4))
        assert all(state_cached(manager, dead, 3, g) for g in (1, 2))
        assert len(events) == 1
    else:
        assert not any(attn_cached(manager, dead, i) for i in range(4))
        assert not any(state_cached(manager, dead, 3, g) for g in (1, 2))
        assert manager.dead_tracker.num_freed == [4, 2]
        # One BlockRemoved per dropped entry, with the right group.
        removed = [(e.group_idx, e.block_hashes[0]) for e in events]
        assert all(isinstance(e, BlockRemoved) for e in events)
        assert sorted(g for g, _ in removed) == [0, 1, 2] + [ATTN_GROUP] * 4
        # Residency metrics were closed for every evicted block.
        assert len(collector.drain_events()) == 7
        assert not any(
            b.block_hash is not None
            for b in pool.free_block_queue.get_all_free_blocks()[:6]
        )

    # Taking as many blocks as were freed: with the flag the freed blocks
    # are reused and ``live`` keeps its hit; without it LRU evicts ``live``.
    held = pool.get_new_blocks(num_uncached + 6)
    resend = make_request("resend", tokens(1, 2) + [9] * 3, MANAGER_BLOCK_SIZE)
    hit = manager.get_computed_blocks(resend)[1]
    assert hit == (2 * MANAGER_BLOCK_SIZE if free_dead else 0)
    pool.free_blocks(held)


def test_lru_eviction_through_allocate_slots_frees_the_chain():
    """The drain after allocate_slots frees what that allocation's LRU
    evictions killed."""
    manager = make_manager(40, free_dead=True)
    pool = manager.block_pool
    dead = make_request("dead", tokens(2, 4), MANAGER_BLOCK_SIZE)
    serve(manager, dead)
    assert all(attn_cached(manager, dead, i) for i in range(4))
    # Hold every never-cached block, so the next allocation must evict.
    num_uncached = sum(
        b.block_hash is None for b in pool.free_block_queue.get_all_free_blocks()
    )
    held = pool.get_new_blocks(num_uncached)
    request = make_request("new", tokens(3, 1), MANAGER_BLOCK_SIZE)
    blocks, num_hit, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(request, request.num_tokens, 0, blocks) is not None
    # Some of ``dead`` was evicted; the drain freed the rest of it.
    assert not any(attn_cached(manager, dead, i) for i in range(4))
    assert not any(state_cached(manager, dead, 3, g) for g in range(NUM_STATE_GROUPS))
    assert sum(manager.dead_tracker.num_freed) > 0
    manager.free(request)
    pool.free_blocks(held)
