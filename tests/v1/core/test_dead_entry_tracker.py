# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeing prefix-cache entries that can no longer hit (DeadEntryTracker)."""

import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import (
    chunked_local_layer,
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
from vllm.v1.core.sched.scheduler import Scheduler
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
    assert tracker.num_freed == [2, 0, 0]
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
    num_blocks: int,
    block_size: int = MANAGER_BLOCK_SIZE,
    sliding_window: int | None = None,
    num_attn_groups: int = 1,
    chunked_local: bool = False,
    eagle: str | None = None,
    attn_block_size: int | None = None,
    swa_block_size: int | None = None,
):
    """Three Mamba groups (align mode) and one full-attention group, the
    layout vLLM gives Qwen3-Next; optionally more full-attention groups, a
    sliding-window group of ``sliding_window`` tokens, or a chunked-local
    group (a layout the tracker does not support). ``eagle`` marks the
    groups holding EAGLE draft layers: "attn" (the target's attention group,
    as with an MTP layer), "draft" (an extra full-attention group), "swa"
    (the sliding-window group) or "mamba" (unsupported). Full-attention and
    sliding-window groups use ``block_size`` unless given their own."""
    attn = AttentionSpec(
        block_size=attn_block_size or block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    swa = AttentionSpec(
        block_size=swa_block_size or block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
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
    for i in range(1, num_attn_groups):
        groups.append(new_kv_cache_group([f"attn{i}"], full_layer(attn)))
    if sliding_window is not None:
        groups.append(
            new_kv_cache_group(["swa"], sliding_window_layer(swa, sliding_window))
        )
    if chunked_local:
        groups.append(
            new_kv_cache_group(["local"], chunked_local_layer(attn, 4 * block_size))
        )
    if eagle == "draft":
        groups.append(new_kv_cache_group(["draft"], full_layer(attn)))
    if eagle is not None:
        index = {
            "attn": NUM_STATE_GROUPS,
            "draft": len(groups) - 1,
            "swa": len(groups) - 1,
            "mamba": 0,
        }[eagle]
        groups[index] = replace(groups[index], is_eagle_group=True)
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )


ATTN_GROUP = NUM_STATE_GROUPS


def make_manager(
    num_blocks: int,
    free_dead: bool,
    sliding_window: int | None = None,
    num_attn_groups: int = 1,
    chunked_local: bool = False,
    eagle: str | None = None,
    block_size: int = MANAGER_BLOCK_SIZE,
    attn_block_size: int | None = None,
    **kwargs,
) -> KVCacheManager:
    """Hashes at ``MANAGER_BLOCK_SIZE``: a larger Mamba ``block_size`` turns on
    fine-grained (partial) hash hits; the sliding-window group then uses the
    hash block size (the only one at which partial hits stay on)."""
    if eagle is not None:
        kwargs["use_eagle"] = True
    return KVCacheManager(
        hybrid_config(
            num_blocks,
            block_size=block_size,
            sliding_window=sliding_window,
            num_attn_groups=num_attn_groups,
            chunked_local=chunked_local,
            eagle=eagle,
            attn_block_size=attn_block_size,
            swa_block_size=MANAGER_BLOCK_SIZE,
        ),
        max_model_len=8192,
        scheduler_block_size=block_size,
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
    assert on.dead_tracker.attn == (ATTN_GROUP,)
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

    # Several full-attention groups and a sliding-window group (e.g. a draft
    # model's layers): supported.
    manager = make_manager(
        64, free_dead=True, sliding_window=2 * MANAGER_BLOCK_SIZE, num_attn_groups=2
    )
    tracker = manager.dead_tracker
    assert tracker is not None
    assert tracker.attn == (ATTN_GROUP, ATTN_GROUP + 1)
    # cdiv(2 * 16 - 1, 16) blocks before a hit.
    assert tracker.swa_window_blocks == {ATTN_GROUP + 2: 2}

    # EAGLE drops on attention groups: supported.
    manager = make_manager(
        64, free_dead=True, sliding_window=2 * MANAGER_BLOCK_SIZE, eagle="swa"
    )
    assert manager.dead_tracker is not None
    assert manager.dead_tracker.eagle == {ATTN_GROUP + 1}
    # On a Mamba group (draft models have none): unsupported.
    manager = make_manager(64, free_dead=True, eagle="mamba")
    assert manager.dead_tracker is None and manager.block_pool.dead_tracker is None

    # A chunked-local group: unsupported, runs without the tracker.
    manager = make_manager(64, free_dead=True, chunked_local=True)
    assert manager.dead_tracker is None and manager.block_pool.dead_tracker is None

    # Fine-grained (partial) hash hits: Mamba and attention blocks of four
    # hash blocks, a sliding-window group at the hash block size.
    manager = make_manager(
        64, free_dead=True, block_size=4 * MANAGER_BLOCK_SIZE, sliding_window=40
    )
    assert manager.coordinator.enable_partial_hash_hits
    tracker = manager.dead_tracker
    assert tracker is not None and tracker.fine_grained
    assert tracker.attn_block_units == {ATTN_GROUP: 4}
    assert tracker.swa_window_blocks == {ATTN_GROUP + 1: 3}
    assert tracker.ghost_lookup == (ATTN_GROUP,)
    # A sliding-window block larger than the hash block turns partial hits
    # off, and blocks are then not hashed at their size: unsupported.
    manager = KVCacheManager(
        hybrid_config(64, block_size=4 * MANAGER_BLOCK_SIZE, sliding_window=40),
        max_model_len=8192,
        scheduler_block_size=4 * MANAGER_BLOCK_SIZE,
        hash_block_size=MANAGER_BLOCK_SIZE,
        enable_caching=True,
        free_dead_prefix_entries=True,
    )
    assert not manager.coordinator.enable_partial_hash_hits
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
        64, free_dead=False, chunked_local=True, enable_kv_cache_events=True
    )
    on = make_manager(
        64, free_dead=True, chunked_local=True, enable_kv_cache_events=True
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
        assert manager.dead_tracker.num_freed == [4, 2, 0]
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


# ---- several attention groups, sliding-window groups ----------------------

SWA = 2


def make_swa_pool(window_blocks: int, attn_groups=(ATTN,), num_blocks: int = 96):
    pool = BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BLOCK_SIZE
    )
    tracker = DeadEntryTracker(pool, attn_groups, (STATE,), {SWA: window_blocks})
    pool.dead_tracker = tracker
    return pool, tracker


def cache_chain(pool, request, num_blocks, group):
    blocks = pool.get_new_blocks(num_blocks)
    pool.cache_full_blocks(request, blocks, 0, num_blocks, BLOCK_SIZE, group)
    return blocks


def test_several_attention_groups():
    """Attention layers can span several groups: a position has its
    attention only while every group has it."""
    other = 3
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BLOCK_SIZE)
    tracker = DeadEntryTracker(pool, (ATTN, other), (STATE,))
    pool.dead_tracker = tracker
    request = make_request("a", list(range(4 * BLOCK_SIZE)))
    attn = {g: cache_chain(pool, request, 4, g) for g in (ATTN, other)}
    state = cache_state(pool, request, 3)
    for blocks in attn.values():
        pool.free_blocks(reversed(blocks))
    pool.free_blocks([state])
    tracker.drain()
    assert all(cached(pool, request, i, g) for i in range(4) for g in (ATTN, other))
    pool.evict_blocks({attn[other][2].block_id})
    tracker.drain()
    assert not any(cached(pool, request, i, g) for i in range(4) for g in (ATTN, other))
    assert not cached(pool, request, 3, STATE)
    assert not tracker.parent


def _swa_run(window_blocks, num_blocks, state_at, state_first=False):
    pool, tracker = make_swa_pool(window_blocks)
    request = make_request("a", list(range(num_blocks * BLOCK_SIZE)))
    states = [cache_state(pool, request, i) for i in state_at] if state_first else []
    attn = cache_chain(pool, request, num_blocks, ATTN)
    swa = cache_chain(pool, request, num_blocks, SWA)
    if not state_first:
        states = [cache_state(pool, request, i) for i in state_at]
    pool.free_blocks(reversed(attn))
    pool.free_blocks(reversed(swa))
    pool.free_blocks(states)
    tracker.drain()
    return pool, tracker, request, swa


@pytest.mark.parametrize("state_first", [False, True])
def test_window_blocks_outside_every_window_are_freed(state_first):
    """A hit at a state needs the window ending there; W blocks outside every
    usable state's window are freed when released. A state cached before its
    window blocks (group order) becomes usable once they arrive."""
    pool, tracker, request, _ = _swa_run(2, 6, [3], state_first)
    assert [cached(pool, request, i, SWA) for i in range(6)] == [
        False, False, True, True, False, False,
    ]  # fmt: skip
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 4 + [
        False
    ] * 2
    assert cached(pool, request, 3, STATE)
    assert tracker.num_freed == [2, 0, 4]


def test_losing_a_window_block_kills_the_state():
    pool, tracker, request, swa = _swa_run(2, 6, [3])
    pool.evict_blocks({swa[2].block_id})
    tracker.drain()
    assert not cached(pool, request, 3, STATE)
    assert not any(cached(pool, request, i, g) for i in range(6) for g in (ATTN, SWA))
    assert not tracker.parent


def test_window_shared_by_two_states_outlives_the_newer_one():
    """W blocks in the windows of two states stay while either is usable."""
    pool, tracker, request, _ = _swa_run(3, 6, [3, 4])
    assert [cached(pool, request, i, SWA) for i in range(6)] == [False] + [True] * 4 + [
        False
    ]
    key = make_block_hash_with_group_id(request.block_hashes[4], STATE)
    pool.evict_blocks({pool.cached_blocks_for(key)[0].block_id})
    tracker.drain()
    # Block 4 now only served the evicted state; 1-3 are the older one's window.
    assert [cached(pool, request, i, SWA) for i in range(6)] == [False] + [True] * 3 + [
        False
    ] * 2
    assert cached(pool, request, 3, STATE)
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 4 + [
        False
    ] * 2


def test_window_reaching_the_start_of_the_sequence():
    """Near the start, a hit needs every block from the start, not a full
    window."""
    pool, tracker, request, _ = _swa_run(3, 4, [1])
    assert cached(pool, request, 1, STATE)
    assert [cached(pool, request, i, SWA) for i in range(4)] == [True, True] + [
        False
    ] * 2
    assert [cached(pool, request, i, ATTN) for i in range(4)] == [True, True] + [
        False
    ] * 2


def _chunked_trace(manager: KVCacheManager, seed: int, chunk_blocks: int = 3) -> list:
    """A random multi-turn workload with branches and chunked prefill (so
    sliding-window blocks are released while requests run); returns the hit
    and shared-prefix junction of every request."""
    rng = random.Random(seed)
    out: list = []
    histories: dict[int, list[int]] = {}
    for step in range(80):
        session = rng.randrange(5)
        history = histories.get(session, [session * 100_000])
        if rng.random() < 0.25 and len(history) > 2 * MANAGER_BLOCK_SIZE:
            history = history[: rng.randrange(MANAGER_BLOCK_SIZE, len(history))]
        history = history + [
            rng.randrange(50_000) for _ in range(rng.randrange(8, 120))
        ]
        histories[session] = history
        request = make_request(str(step), history, MANAGER_BLOCK_SIZE)
        blocks, num_hit, boundary = manager.get_computed_blocks(request)
        out.append((num_hit, boundary))
        computed = num_hit
        first = True
        while computed < request.num_tokens:
            n = min(chunk_blocks * MANAGER_BLOCK_SIZE, request.num_tokens - computed)
            if first:
                new = manager.allocate_slots(request, n, num_hit, blocks)
                first = False
            else:
                new = manager.allocate_slots(request, n)
            assert new is not None
            computed += n
            request.num_computed_tokens = computed
            manager.new_step_starts()
        manager.free(request)
    return out


@pytest.mark.parametrize(
    "sliding_window,num_attn_groups,eagle",
    [
        (None, 2, None),
        (2 * MANAGER_BLOCK_SIZE, 1, None),
        (3 * MANAGER_BLOCK_SIZE + 5, 2, None),
        # EAGLE: an MTP layer in the target's attention group, a separate
        # full-attention draft, a sliding-window draft.
        (None, 1, "attn"),
        (None, 1, "draft"),
        (2 * MANAGER_BLOCK_SIZE, 1, "swa"),
    ],
)
def test_never_frees_an_entry_that_could_hit(sliding_window, num_attn_groups, eagle):
    """Without memory pressure LRU never evicts, so every entry the tracker
    frees is one LRU would have kept: if any of them could serve a hit, a
    later request would hit less (or find another junction) than with the
    flag off."""
    kwargs = dict(
        sliding_window=sliding_window,
        num_attn_groups=num_attn_groups,
        eagle=eagle,
        enable_kv_cache_events=True,
    )
    off = make_manager(20_000, free_dead=False, **kwargs)
    on = make_manager(20_000, free_dead=True, **kwargs)
    trace_off = _chunked_trace(off, seed=1)
    assert not any(isinstance(e, BlockRemoved) for e in off.take_events())
    trace_on = _chunked_trace(on, seed=1)
    assert trace_on == trace_off
    # Sanity: the workload hits and branches, and the tracker had dead
    # entries to free (attention past the last state; window blocks).
    assert sum(hit for hit, _ in trace_off) > 0
    assert any(boundary for _, boundary in trace_off)
    assert on.dead_tracker is not None
    freed = on.dead_tracker.num_freed
    assert freed[0] > 0
    assert (freed[2] > 0) == (sliding_window is not None)


def test_eagle_keeps_the_next_block_of_a_usable_state():
    """With EAGLE, a hit at a state also needs its next block in the eagle
    groups; that block is kept with no state below it, until the state goes."""
    draft = 3
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BLOCK_SIZE)
    tracker = DeadEntryTracker(pool, (ATTN, draft), (STATE,), eagle_group_ids=(draft,))
    pool.dead_tracker = tracker
    request = make_request("a", list(range(6 * BLOCK_SIZE)))
    attn = {g: cache_chain(pool, request, 6, g) for g in (ATTN, draft)}
    state = cache_state(pool, request, 3)
    for blocks in attn.values():
        pool.free_blocks(reversed(blocks))
    pool.free_blocks([state])
    tracker.drain()
    assert [cached(pool, request, i, ATTN) for i in range(6)] == [True] * 4 + [
        False
    ] * 2
    assert [cached(pool, request, i, draft) for i in range(6)] == [True] * 5 + [False]
    pool.evict_blocks({state.block_id})
    tracker.drain()
    assert not any(cached(pool, request, i, g) for i in range(6) for g in (ATTN, draft))
    assert not tracker.parent


# ---- fine-grained (partial) hash hits --------------------------------------
# Hashes every BLOCK_SIZE tokens (a unit); attention and Mamba blocks span
# UNITS of them, so a hit can end inside a block.

UNITS = 4
FINE_BLOCK = UNITS * BLOCK_SIZE


def make_fine_pool(eagle: bool = False):
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BLOCK_SIZE)
    tracker = DeadEntryTracker(
        pool,
        ATTN,
        (STATE,),
        eagle_group_ids=(ATTN,) if eagle else (),
        attn_block_units={ATTN: UNITS},
        ghost_lookup_group_ids=(ATTN,),
    )
    pool.dead_tracker = tracker
    return pool, tracker


def fine_attn(pool, request, num_blocks):
    """Full attention blocks ``0 .. num_blocks - 1``."""
    blocks = pool.get_new_blocks(num_blocks)
    pool.cache_full_blocks(request, blocks, 0, num_blocks, FINE_BLOCK, ATTN)
    return blocks


def fine_entry(pool, request, unit, group=ATTN, block=None):
    """An entry ending at ``unit``: a partial attention block or a state."""
    block = block or pool.get_new_blocks(1)[0]
    pool.cache_partial_block(request, block, (unit + 1) * BLOCK_SIZE, group, FINE_BLOCK)
    return block


def release(pool, tracker, blocks):
    pool.free_blocks(reversed(blocks))
    tracker.drain()


def fine_cached(pool, request, units, group=ATTN):
    return [cached(pool, request, u, group) for u in units]


def test_fine_full_and_partial_entries_a_state_needs():
    """A state inside block 2 needs full blocks 0-2 (or a partial entry at or
    past it in block 2); a partial entry in block 3 serves no state."""
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(5 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 3)
    blocks.append(fine_entry(pool, request, 13))
    blocks.append(fine_entry(pool, request, 9, STATE))
    release(pool, tracker, blocks)
    assert fine_cached(pool, request, (3, 7, 11, 13)) == [True, True, True, False]
    assert cached(pool, request, 9, STATE)
    assert tracker.num_freed == [1, 0, 0]
    # The freed entry still extends the junction: full blocks 0-2, then the
    # partial entry at unit 13.
    assert tracker.ghost_attention_hit(request.block_hashes, 5 * FINE_BLOCK) == (
        14 * BLOCK_SIZE
    )
    assert tracker.ghost_attention_hit(request.block_hashes, 13 * BLOCK_SIZE) == (
        12 * BLOCK_SIZE
    )


@pytest.mark.parametrize("state_unit,kept", [(9, True), (10, True), (11, False)])
def test_fine_partial_entry_serves_states_up_to_it(state_unit, kept):
    """A partial entry at unit 10 serves hits up to it inside its block; a
    state past it needs the full block (or a later partial entry), which
    usually arrives later, so the state stays."""
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 2)
    blocks.append(fine_entry(pool, request, 10))
    state = fine_entry(pool, request, state_unit, STATE)
    release(pool, tracker, blocks + [state])
    assert fine_cached(pool, request, (3, 7, 10)) == [True, True, kept]
    assert cached(pool, request, state_unit, STATE)
    # Without the state nothing of the chain can hit.
    pool.evict_blocks({state.block_id})
    tracker.drain()
    assert fine_cached(pool, request, (3, 7, 10)) == [False] * 3
    assert not tracker.parent


def test_fine_losing_a_partial_entry_kills_nothing():
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 2)
    partial = fine_entry(pool, request, 10)
    state = fine_entry(pool, request, 9, STATE)
    release(pool, tracker, blocks + [partial, state])
    pool.evict_blocks({partial.block_id})
    tracker.drain()
    assert fine_cached(pool, request, (3, 7)) == [True, True]
    assert cached(pool, request, 9, STATE)
    assert tracker.num_freed == [0, 0, 0]


def test_fine_losing_a_full_block_kills_below_it():
    """Losing full block 1 kills the state in block 2 and block 2; the state
    inside block 1 stays (a partial entry or the full block may return)."""
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 3)
    states = [fine_entry(pool, request, u, STATE) for u in (5, 9)]
    release(pool, tracker, blocks + states)
    assert fine_cached(pool, request, (3, 7, 11)) == [True] * 3
    pool.evict_blocks({blocks[1].block_id})
    tracker.drain()
    assert fine_cached(pool, request, (3, 7, 11)) == [True, False, False]
    assert cached(pool, request, 5, STATE)
    assert not cached(pool, request, 9, STATE)


def test_fine_promotion_is_not_an_eviction():
    """A running request's partial tail block is re-keyed as a full block
    once it fills: the states it served stay, nothing is freed."""
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 2) + pool.get_new_blocks(1)
    fine_entry(pool, request, 9, block=blocks[2])
    state = fine_entry(pool, request, 9, STATE)
    tracker.drain()
    # Block 2 fills: partial (unit 9) -> full (unit 11).
    pool.cache_full_blocks(request, blocks, 2, 3, FINE_BLOCK, ATTN)
    release(pool, tracker, blocks + [state])
    assert fine_cached(pool, request, (3, 7, 9, 11)) == [True, True, False, True]
    assert cached(pool, request, 9, STATE)
    assert tracker.num_freed == [0, 0, 0]


def test_fine_block_keyed_twice_is_freed_with_its_last_entry():
    """A full block also keyed at a partial boundary inside it (a request
    whose prompt ends there hit it): freeing the dead partial entry would
    evict the needed full one, so the block stays until both are dead."""
    pool, tracker = make_fine_pool()
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 3)
    fine_entry(pool, request, 9, block=blocks[2])
    state = fine_entry(pool, request, 11, STATE)
    release(pool, tracker, blocks + [state])
    assert fine_cached(pool, request, (9, 11)) == [True, True]
    pool.evict_blocks({state.block_id})
    tracker.drain()
    assert fine_cached(pool, request, (3, 7, 9, 11)) == [False] * 4
    assert not tracker.parent


def test_fine_eagle_needs_the_next_unit():
    """An EAGLE group matches one unit past the hit: a hit at unit 9 needs
    unit 10 covered, so the partial entry at 10 is kept and the one at 9 is
    not; a state at the end of block 1 keeps entries in block 2."""
    pool, tracker = make_fine_pool(eagle=True)
    request = make_request("a", list(range(4 * FINE_BLOCK)))
    blocks = fine_attn(pool, request, 2)
    partials = [fine_entry(pool, request, u) for u in (9, 10)]
    state = fine_entry(pool, request, 9, STATE)
    release(pool, tracker, blocks + partials + [state])
    assert fine_cached(pool, request, (3, 7, 9, 10)) == [True, True, False, True]
    assert tracker.ghost_attention_hit(request.block_hashes, 4 * FINE_BLOCK) == (
        10 * BLOCK_SIZE
    )

    pool, tracker = make_fine_pool(eagle=True)
    blocks = fine_attn(pool, request, 2)
    partials = [fine_entry(pool, request, u) for u in (8, 10)]
    state = fine_entry(pool, request, 7, STATE)
    release(pool, tracker, blocks + partials + [state])
    assert fine_cached(pool, request, (3, 7, 8, 10)) == [True] * 4


PARTIAL_BLOCK_SIZE = 4 * MANAGER_BLOCK_SIZE


def _split_stub(manager: KVCacheManager, eagle: bool, chunk_tokens: int):
    """``self`` for the real ``Scheduler._mamba_block_aligned_split``."""
    partial = manager.coordinator.enable_partial_hash_hits
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=PARTIAL_BLOCK_SIZE),
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        max_num_scheduled_tokens=chunk_tokens,
        use_eagle=eagle,
        use_eagle_block_drop=eagle,
        num_hash_lookahead_tokens=0,
        hash_block_size=MANAGER_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_partial_cache_hit=partial,
        mamba_shared_prefix_checkpoint=(
            partial and manager.mamba_shared_prefix_checkpoint
        ),
    )


def _partial_trace(
    manager: KVCacheManager, seed: int, chunk_tokens: int, split: bool, eagle: bool
) -> list:
    """``_chunked_trace`` for fine-grained hits: prefill in chunks of
    ``chunk_tokens``, ended where the scheduler's Mamba split ends them
    (``split``) or exactly there (inside a block); the junction is handed to
    the request as the scheduler does, and half of the turns decode before
    the next one (filling and re-keying partial blocks)."""
    stub = _split_stub(manager, eagle, chunk_tokens)
    rng = random.Random(seed)
    out: list = []
    histories: dict[int, list[int]] = {}

    def step_done():
        _, retained = manager.take_kv_cache_block_copies()
        manager.block_pool.free_blocks(retained)
        manager.new_step_starts()

    for step in range(80):
        session = rng.randrange(5)
        history = histories.get(session, [session * 100_000])
        if rng.random() < 0.25 and len(history) > 2 * PARTIAL_BLOCK_SIZE:
            history = history[: rng.randrange(MANAGER_BLOCK_SIZE, len(history))]
        history = history + [
            rng.randrange(50_000) for _ in range(rng.randrange(8, 150))
        ]
        request = make_request(str(step), history, MANAGER_BLOCK_SIZE)
        blocks, num_hit, boundary = manager.get_computed_blocks(request)
        request.shared_prefix_boundary = boundary
        out.append((num_hit, boundary))
        first = True
        while request.num_computed_tokens < request.num_tokens:
            hit = num_hit if first else 0
            start = request.num_computed_tokens + hit
            if start >= request.num_tokens:
                break
            n = min(chunk_tokens, request.num_tokens - start)
            if split:
                n = Scheduler._mamba_block_aligned_split(stub, request, n, hit)
            assert n > 0
            assert (
                manager.allocate_slots(
                    request,
                    n,
                    num_new_computed_tokens=hit,
                    new_computed_blocks=blocks if first else None,
                )
                is not None
            )
            request.num_computed_tokens = start + n
            step_done()
            first = False
        if rng.random() < 0.5:
            for _ in range(rng.randrange(1, 100)):
                request.append_output_token_ids(rng.randrange(50_000))
                assert manager.allocate_slots(request, 1) is not None
                request.num_computed_tokens += 1
                step_done()
            history = list(request.all_token_ids)
        histories[session] = history
        manager.free(request)
    return out


def _forbid_lru_eviction(manager: KVCacheManager) -> None:
    queue = manager.block_pool.free_block_queue
    popleft_n = queue.popleft_n

    def checked(n):
        blocks = popleft_n(n)
        assert all(b.block_hash is None for b in blocks), "LRU evicted"
        return blocks

    queue.popleft_n = checked  # type: ignore[method-assign]


PARTIAL_LAYOUTS = {
    "mamba+attn": dict(),
    "+swa": dict(sliding_window=40),
    "eagle-attn": dict(eagle="attn"),
    "eagle-draft": dict(eagle="draft"),
    "eagle-swa": dict(sliding_window=40, eagle="swa"),
    "attn-at-hash-size": dict(attn_block_size=MANAGER_BLOCK_SIZE),
}


@pytest.mark.parametrize("layout", PARTIAL_LAYOUTS)
@pytest.mark.parametrize(
    "chunk_tokens,split",
    [
        # Chunks ending inside blocks (states and partial entries there).
        (3 * MANAGER_BLOCK_SIZE, False),
        (7 * MANAGER_BLOCK_SIZE, False),
        # The scheduler's chunk ends: block boundaries, the prompt's last hash
        # boundary and the junction.
        (40, True),
    ],
)
def test_never_frees_an_entry_that_could_hit_partial(layout, chunk_tokens, split):
    """``test_never_frees_an_entry_that_could_hit`` with fine-grained hits:
    hash blocks of 16 tokens, Mamba (and attention) blocks of 64."""
    kwargs = dict(
        PARTIAL_LAYOUTS[layout],
        block_size=PARTIAL_BLOCK_SIZE,
        enable_kv_cache_events=True,
    )
    eagle = "eagle" in kwargs
    off = make_manager(20_000, free_dead=False, **kwargs)
    on = make_manager(20_000, free_dead=True, **kwargs)
    assert on.coordinator.enable_partial_hash_hits
    assert on.dead_tracker is not None and on.dead_tracker.fine_grained
    # (Partial entries are re-keyed, so BlockRemoved events do not show LRU
    # evictions here.)
    _forbid_lru_eviction(off)
    trace_off = _partial_trace(off, 1, chunk_tokens, split, eagle)
    trace_on = _partial_trace(on, 1, chunk_tokens, split, eagle)
    assert trace_on == trace_off
    # Sanity: hits end inside blocks, and with EAGLE the junction is used.
    assert any(hit % PARTIAL_BLOCK_SIZE for hit, _ in trace_off)
    if eagle:
        assert any(boundary for _, boundary in trace_off)
    freed = on.dead_tracker.num_freed
    if not split:
        # Attention past the last state (chunk ends inside blocks cache no
        # state there), and window blocks.
        assert freed[0] > 0
    assert (freed[2] > 0) == ("sliding_window" in kwargs)
