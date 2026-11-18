# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Free prefix-cache entries that can no longer serve a hit.

A hybrid model with full-attention groups A, state groups S (Mamba, align
mode) and optionally sliding-window groups W, all at one block size, can only
hit at a position p that

* has every A block up to p cached (in every A group),
* has a cached state at p in every S group, and
* has, in every W group, the blocks of the window ending at p cached: the
  last ``cdiv(sliding_window - 1, block_size)`` blocks up to p, or every
  block from the start if p is closer to it.

vLLM's KV cache grouping gives every group the same number of layers, so a
model's A, S and W layers can each span several groups. A state at p is
*usable* if it is cached in every S group and its windows are complete.
Three kinds of cached entries can therefore never hit:

* a state that is not usable, or whose A prefix has lost a block,
* an A block with no usable state at or below it on its chain, and
* a W block outside the window of every usable state below it.

With EAGLE, the groups holding draft layers (eagle groups, A or W) match one
block past the hit and drop it, so a hit at p also needs p's next block in
every eagle group. That block is usually cached later than the state: the
state at the end of a turn ends inside a partial block, and the next turn of
the conversation caches the block after it. A missing next block therefore
does not make a state dead. Instead, the eagle groups' blocks at the children
of a usable state are kept, even with no usable state below them.

LRU frees them only when it reaches them, and until then they take the place
of entries that could still hit. This tracker keeps the prefix tree of cached
blocks and the states hanging off it, and frees dead entries as soon as they
appear. Frees cascade: losing an A block kills everything below it; losing a
W block makes the states whose window it was in unusable; losing the last
usable state below an A block kills that block, and W blocks no usable state
needs are freed. Freed blocks go to the front of the free queue, so they are
reused before any live entry is evicted.

A entries freed this way stay in a hash-only list (per group) so that
junction detection (an A prefix deeper than the hybrid hit) still sees them:
the junction is the coordinator's first full-attention lookup replayed over
cached and freed entries. The list holds as many entries as the pool has
blocks, so a freed entry is forgotten about when LRU would have reused its
block.

An A block can lose its entry while a running request holds it and blocks
below it (a KV connector reporting it invalid). Nothing below it can hit any
more, but the held blocks cannot be freed yet: the subtree is marked broken,
its states stop counting, and its blocks are freed when released -- unless
the lost block is cached again first.

Fine-grained (partial) hash hits
--------------------------------
With ``hash_block_size`` smaller than the Mamba block size, a hit can end at
any hash-block boundary (a *unit*), also inside a cache block. Every entry is
keyed by the chain hash at its last unit: a full A block of ``s`` units by
the hash of its last unit, a partial A entry (a block's prefix, registered at
a prompt's last hash boundary) and a state by the hash of the unit they end
at; W groups then hash at their block size, one unit. The tree then has a
node per unit, with every ancestor present. The full-attention lookup matches
full A blocks from the start, then the highest partial entry inside the next
block, so a hit at unit ``u`` in A block ``k`` needs the full A blocks before
``k`` and, inside ``k``, the full block or a partial entry at or past ``u``;
an eagle A group matches one unit past the hit and drops it, so it needs that
for ``u + 1``. States and W blocks hit as above with units for blocks.

A state is usable as before (states and windows only): the A entry inside its
block usually arrives later (the next chunk or turn fills the block), so its
absence does not make the state dead, as with the eagle next block. Hence an
A entry ending at unit ``x`` of block ``k`` is kept while a usable state needs
it: one at a unit in ``[start(k), x]`` on its chain (one unit earlier for
eagle groups), or, for a full block, anywhere below it. Losing a full A block
kills everything below it; losing a partial entry kills nothing.

Hooks are called by BlockPool while it mutates its queues, so they only record
work (and update counts); ``drain`` frees the dead entries at a safe point.
"""

import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)

# Seconds between two log lines with the freed counts.
_LOG_INTERVAL_S = 60.0

# Indices into ``DeadEntryTracker.num_freed``.
FREED_ATTN, FREED_STATE, FREED_SWA = 0, 1, 2


def _as_tuple(groups: int | Iterable[int]) -> tuple[int, ...]:
    return (groups,) if isinstance(groups, int) else tuple(groups)


class DeadEntryTracker:
    def __init__(
        self,
        block_pool: "BlockPool",
        attn_group_ids: int | Iterable[int],
        state_group_ids: int | Iterable[int],
        swa_window_blocks: dict[int, int] | None = None,
        eagle_group_ids: Iterable[int] = (),
        attn_block_units: dict[int, int] | None = None,
        ghost_lookup_group_ids: Iterable[int] | None = None,
        ghost_capacity: int | None = None,
    ):
        """Track the given groups of ``block_pool``.

        Args:
            block_pool: The pool all tracked groups allocate from.
            attn_group_ids: Full-attention groups.
            state_group_ids: Mamba groups (align mode).
            swa_window_blocks: Sliding-window groups, each with the number of
                contiguous blocks a hit needs (``cdiv(window - 1,
                block_size)``).
            eagle_group_ids: Full-attention or sliding-window groups that
                drop one matched block (one unit if fine-grained) for EAGLE.
            attn_block_units: Fine-grained (partial) hash hits: the block size
                of every full-attention group in hash blocks (units). Sliding-
                window groups then hash at their block size. None: every
                group hashes at its block size and hits are block-aligned.
            ghost_lookup_group_ids: The full-attention groups the
                coordinator looks up first, whose match sets the shared-prefix
                junction (``ghost_attention_hit``); defaults to every
                full-attention group.
            ghost_capacity: Freed A entries kept for junction detection;
                defaults to the pool's number of blocks.

        """
        self.block_pool = block_pool
        self.attn = _as_tuple(attn_group_ids)
        self.states = _as_tuple(state_group_ids)
        self.swa_window_blocks = dict(swa_window_blocks or {})
        assert self.attn and self.states
        assert all(n >= 1 for n in self.swa_window_blocks.values())
        self.eagle = frozenset(eagle_group_ids)
        assert self.eagle <= set(self.attn) | set(self.swa_window_blocks)
        self.fine_grained = attn_block_units is not None
        self.attn_block_units = dict(attn_block_units or {})
        self.ghost_lookup = tuple(
            self.attn if ghost_lookup_group_ids is None else ghost_lookup_group_ids
        )
        assert self.ghost_lookup and set(self.ghost_lookup) <= set(self.attn)
        if self.fine_grained:
            assert set(self.attn_block_units) == set(self.attn)
            assert all(n >= 1 for n in self.attn_block_units.values())
        # Units below a state whose A entries it may need (its block, or with
        # eagle the block of the next unit): bounds the recheck when it goes.
        self._max_block_units = max(self.attn_block_units.values(), default=1)
        # The longest window, in blocks.
        self._max_window = max(self.swa_window_blocks.values(), default=1)
        # Prefix tree over the block hashes of every tracked group.
        self.parent: dict[BlockHash, BlockHash | None] = {}
        # Fine-grained only: the unit index of every node.
        self.unit: dict[BlockHash, int] = {}
        self.children: defaultdict[BlockHash, set[BlockHash]] = defaultdict(set)
        # Cached blocks per hash in each group (duplicates are possible).
        self.attn_count: dict[int, defaultdict[BlockHash, int]] = {
            g: defaultdict(int) for g in self.attn
        }
        self.state_count: dict[int, defaultdict[BlockHash, int]] = {
            g: defaultdict(int) for g in self.states
        }
        self.swa_count: dict[int, defaultdict[BlockHash, int]] = {
            g: defaultdict(int) for g in self.swa_window_blocks
        }
        # Hashes with a usable state, counted in ``live_below`` of their
        # ancestors.
        self.counted: set[BlockHash] = set()
        # Usable states in the subtree of a node, the node included.
        self.live_below: defaultdict[BlockHash, int] = defaultdict(int)
        # Work recorded by the hooks. ``_killed``: (node, group) of full A
        # blocks lost.
        self._killed: deque[tuple[BlockHash, int]] = deque()
        self._unsupported: deque[BlockHash] = deque()
        self._released: list[BlockHash] = []
        # W blocks to check against the windows of the usable states.
        self._swa_check: deque[BlockHash] = deque()
        # Fine-grained only: nodes whose A entries to check against the
        # usable states.
        self._attn_check: deque[BlockHash] = deque()
        self._draining = False
        # Nodes that lost an A block (in the given groups) while blocks below
        # them were held: nothing at or below them can hit until it is cached
        # again.
        self.broken: dict[BlockHash, set[int]] = {}
        # A entries (hash with group id) freed as dead, for junction detection.
        self.ghost: OrderedDict[BlockHash, None] = OrderedDict()
        self.ghost_capacity = (
            block_pool.num_gpu_blocks if ghost_capacity is None else ghost_capacity
        )
        self.num_freed = [0, 0, 0]  # [attention blocks, states, W blocks]
        self._num_logged = [0, 0, 0]
        self._last_log = time.monotonic()

    # ---- hooks (record only) -------------------------------------------
    def on_insert(self, group_id: int, block_hashes: Sequence[BlockHash], unit: int):
        """``block_hashes[unit]`` (a request's hashes, one per hash block)
        was cached in ``group_id``."""
        if (
            group_id not in self.attn_count
            and group_id not in self.state_count
            and group_id not in self.swa_count
        ):
            return
        block_hash = block_hashes[unit]
        if self.fine_grained:
            self._insert_fine(group_id, block_hashes, unit)
            return
        parent = block_hashes[unit - 1] if unit else None
        if block_hash not in self.parent:
            self.parent[block_hash] = parent
            if parent is not None:
                self.children[parent].add(block_hash)
            # Groups cache in group order, so a state can arrive before the
            # A blocks of its chain: adopt states already counted below.
            below = sum(
                self.live_below.get(c, 0) for c in self.children.get(block_hash, ())
            )
            if below:
                self._add_live(block_hash, below)
        if group_id in self.attn_count:
            self.attn_count[group_id][block_hash] += 1
            self._maybe_repair(block_hash, group_id)
        elif group_id in self.state_count:
            self.state_count[group_id][block_hash] += 1
            self._recount(block_hash)
        else:
            self.swa_count[group_id][block_hash] += 1
            # A window may have just become complete.
            for node in self._window_owners(
                block_hash, self.swa_window_blocks[group_id]
            ):
                self._recount(node)

    def on_remove(self, group_id: int, block_hash: BlockHash, rekey: bool = False):
        """A cache entry went away; ``rekey`` if its block was re-keyed at a
        longer prefix of the same contents (a partial entry promoted)."""
        if block_hash not in self.parent:
            return
        if self.fine_grained:
            self._remove_fine(group_id, block_hash, rekey)
            return
        if group_id in self.attn_count:
            counts = self.attn_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                # The hash no longer marks a shared prefix. (A freed copy is
                # added back by ``_free``.)
                self.ghost.pop(
                    make_block_hash_with_group_id(block_hash, group_id), None
                )
                self._killed.append((block_hash, group_id))
        elif group_id in self.state_count:
            counts = self.state_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                self._recount(block_hash)
                self._unsupported.append(block_hash)
        elif group_id in self.swa_count:
            counts = self.swa_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                # The states whose window it was in are no longer usable.
                for node in self._window_owners(
                    block_hash, self.swa_window_blocks[group_id]
                ):
                    self._recount(node)
                self._swa_check.append(block_hash)  # prune it if now empty

    def on_release(self, group_id: int, block_hash: BlockHash):
        """A block's last reference went away (e.g. a request finished, or a
        W block slid out of a running request's window)."""
        if block_hash not in self.parent:
            return
        if group_id in self.attn_count:
            self._released.append(block_hash)
            if self.fine_grained:
                self._attn_check.append(block_hash)
        elif group_id in self.state_count:
            self._released.append(block_hash)
        elif group_id in self.swa_count:
            self._swa_check.append(block_hash)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            self.block_pool,
            self.attn,
            self.states,
            self.swa_window_blocks,
            self.eagle,
            self.attn_block_units if self.fine_grained else None,
            self.ghost_lookup,
            self.ghost_capacity,
        )

    # ---- fine-grained hooks ------------------------------------------------
    def _insert_fine(
        self, group_id: int, block_hashes: Sequence[BlockHash], unit: int
    ) -> None:
        # Add the missing nodes of the chain: a node's ancestors are always
        # in the tree (only nodes without children leave it).
        first = unit
        while first >= 0 and block_hashes[first] not in self.parent:
            first -= 1
        for i in range(first + 1, unit + 1):
            node = block_hashes[i]
            parent = block_hashes[i - 1] if i else None
            self.parent[node] = parent
            self.unit[node] = i
            if parent is not None:
                self.children[parent].add(node)
        block_hash = block_hashes[unit]
        if group_id in self.attn_count:
            self.attn_count[group_id][block_hash] += 1
            self._maybe_repair(block_hash, group_id)
        elif group_id in self.state_count:
            self.state_count[group_id][block_hash] += 1
            self._recount(block_hash)
        else:
            self.swa_count[group_id][block_hash] += 1
            for node in self._window_owners(
                block_hash, self.swa_window_blocks[group_id]
            ):
                self._recount(node)

    def _remove_fine(self, group_id: int, block_hash: BlockHash, rekey: bool) -> None:
        if group_id in self.attn_count:
            counts = self.attn_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] > 0:
                return
            del counts[block_hash]
            if not rekey:
                # Evicted: the hash no longer marks a shared prefix. (A freed
                # copy is added back by ``_free``.)
                self.ghost.pop(
                    make_block_hash_with_group_id(block_hash, group_id), None
                )
            if self._is_full(block_hash, group_id):
                # Nothing below a lost full A block can hit.
                self._killed.append((block_hash, group_id))
            else:
                # A partial entry: the hits it served can still come from the
                # full block or another partial entry; nothing else depends on
                # it.
                self._swa_check.append(block_hash)  # prune it if now empty
        elif group_id in self.state_count:
            counts = self.state_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                self._recount(block_hash)
                self._unsupported.append(block_hash)
        elif group_id in self.swa_count:
            counts = self.swa_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                for node in self._window_owners(
                    block_hash, self.swa_window_blocks[group_id]
                ):
                    self._recount(node)
                self._swa_check.append(block_hash)

    # ---- tree helpers ----------------------------------------------------
    def _has_attn(self, node: BlockHash) -> bool:
        return all(c.get(node) for c in self.attn_count.values())

    def _any_attn(self, node: BlockHash) -> bool:
        return any(c.get(node) for c in self.attn_count.values())

    def _has_state(self, node: BlockHash) -> bool:
        """A state in every state group."""
        return all(c.get(node) for c in self.state_count.values())

    def _any_state(self, node: BlockHash) -> bool:
        return any(c.get(node) for c in self.state_count.values())

    def _any_swa(self, node: BlockHash) -> bool:
        return any(c.get(node) for c in self.swa_count.values())

    def _holds_anything(self, node: BlockHash) -> bool:
        return self._any_attn(node) or self._any_state(node) or self._any_swa(node)

    def _is_eagle_child(self, node: BlockHash) -> bool:
        """``node`` is the next block a hit at its parent's usable state
        needs: it is cached in every eagle group."""
        if not self.eagle:
            return False
        parent = self.parent.get(node)
        if parent is None or parent not in self.counted:
            return False
        return all(
            (self.attn_count[g] if g in self.attn_count else self.swa_count[g]).get(
                node
            )
            for g in self.eagle
        )

    def _window_complete(self, node: BlockHash) -> bool:
        """Every W group has the blocks of the window ending at ``node``:
        ``node`` and its ancestors, ``window`` blocks in all, or up to the
        start of the sequence."""
        for group_id, window in self.swa_window_blocks.items():
            counts = self.swa_count[group_id]
            cur: BlockHash | None = node
            for k in range(window):
                if cur is None:
                    break  # reached the start of the sequence
                if not counts.get(cur):
                    return False
                if k == window - 1:
                    break
                parent = self.parent.get(cur)
                if parent is not None and parent not in self.parent:
                    return False  # the chain above is not cached
                cur = parent
        return True

    def _usable(self, node: BlockHash) -> bool:
        """A hit can end at ``node`` (given its A prefix)."""
        return (
            self._has_state(node)
            and self._window_complete(node)
            and not (self.broken and self._under_broken(node))
        )

    def _under_broken(self, node: BlockHash | None) -> bool:
        while node is not None and node in self.parent:
            if node in self.broken:
                return True
            node = self.parent[node]
        return False

    def _break(self, root: BlockHash, group_id: int, nodes: list[BlockHash]) -> None:
        """``root`` lost its A block in ``group_id``: the states at and below
        it stop counting, so their blocks are freed once released."""
        self.counted.difference_update(nodes)
        num_live = self.live_below.get(root, 0)
        if num_live:
            for node in nodes:
                self.live_below.pop(node, None)
            self._add_live(self.parent[root], -num_live)
        self.broken.setdefault(root, set()).add(group_id)

    def _maybe_repair(self, node: BlockHash, group_id: int) -> None:
        """An A block of ``group_id`` was cached at ``node``: if that is the
        last one it lost, its subtree can hit again."""
        lost = self.broken.get(node)
        if lost is None:
            return
        lost.discard(group_id)
        if not lost:
            del self.broken[node]
            for child in self._subtree(node):
                self._recount(child)

    def _recount(self, node: BlockHash) -> None:
        usable = node in self.parent and self._usable(node)
        if usable and node not in self.counted:
            self.counted.add(node)
            self._add_live(node, 1)
        elif not usable and node in self.counted:
            self.counted.discard(node)
            self._add_live(node, -1)
            # Its states are now dead, and so may be its A prefix, the W
            # blocks of its window and its children's eagle blocks.
            self._unsupported.append(node)
            if self.swa_window_blocks:
                self._swa_check.extend(self._ancestors(node, self._max_window))
            if self.eagle:
                children = self.children.get(node, ())
                self._unsupported.extend(children)
                self._swa_check.extend(children)
            if self.fine_grained:
                # A entries inside its block (eagle: the next unit's block)
                # that only it needed.
                self._attn_check.extend(
                    self._window_owners(node, self._max_block_units + 1)
                )

    def _is_full(self, node: BlockHash, group_id: int) -> bool:
        """``node`` ends an A block of ``group_id`` (fine-grained)."""
        return (self.unit[node] + 1) % self.attn_block_units[group_id] == 0

    def _attn_needed(self, node: BlockHash, group_id: int) -> bool:
        """Fine-grained: a usable state needs the A entry of ``group_id`` at
        ``node``, the end of a full block or a partial entry inside block
        ``k``. A hit at unit ``u`` (``u + 1`` for eagle groups) inside ``k``
        is served by the full block or a partial entry at or past it, and a
        hit past ``k`` needs the full block."""
        unit = self.unit[node]
        units = self.attn_block_units[group_id]
        shift = 1 if group_id in self.eagle else 0
        if (unit + 1) % units == 0 and self.live_below.get(node, 0):
            return True
        # A usable state on its chain at a unit in [start(k), unit], shifted.
        low, high = unit - unit % units - shift, unit - shift
        cur: BlockHash | None = node
        while cur is not None and unit >= low:
            if unit <= high and cur in self.counted:
                return True
            cur = self.parent.get(cur)
            unit -= 1
        return False

    def _ancestors(self, node: BlockHash, n: int) -> list[BlockHash]:
        """``node`` and up to ``n - 1`` of its ancestors."""
        out: list[BlockHash] = []
        cur: BlockHash | None = node
        while cur is not None and cur in self.parent and len(out) < n:
            out.append(cur)
            cur = self.parent[cur]
        return out

    def _window_owners(self, node: BlockHash, window: int) -> list[BlockHash]:
        """Nodes whose window of ``window`` blocks contains ``node``: ``node``
        and its descendants less than ``window`` levels below it."""
        out = [node]
        frontier = [node]
        for _ in range(window - 1):
            frontier = [c for n in frontier for c in self.children.get(n, ())]
            if not frontier:
                break
            out.extend(frontier)
        return out

    def _swa_needed(self, node: BlockHash, group_id: int) -> bool:
        """A usable state has ``node`` in its window of ``group_id``, or
        needs it as its next block (eagle groups)."""
        if group_id in self.eagle and (
            self.parent.get(node) in self.counted
            if self.fine_grained
            else self._is_eagle_child(node)
        ):
            return True
        return any(
            owner in self.counted
            for owner in self._window_owners(node, self.swa_window_blocks[group_id])
        )

    def _free_states(self, node: BlockHash) -> None:
        for g, counts in self.state_count.items():
            if counts.get(node):
                self._free(g, node)

    def _free_attn(self, node: BlockHash, keep_eagle: bool = False) -> None:
        for g, counts in self.attn_count.items():
            if counts.get(node) and not (keep_eagle and g in self.eagle):
                self._free(g, node)

    def _free_swa(self, node: BlockHash, keep_eagle: bool = False) -> None:
        for g, counts in self.swa_count.items():
            if counts.get(node) and not (keep_eagle and g in self.eagle):
                self._free(g, node)

    def _add_live(self, node: BlockHash | None, delta: int):
        while node is not None and node in self.parent:
            self.live_below[node] += delta
            if self.live_below[node] <= 0:
                del self.live_below[node]
            node = self.parent[node]

    def _subtree(self, root: BlockHash) -> Iterable[BlockHash]:
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(self.children.get(node, ()))

    def _top_unsupported(self, node: BlockHash) -> BlockHash | None:
        """Highest ancestor-or-self of ``node`` with no usable state below it."""
        if self.live_below.get(node, 0):
            return None
        while True:
            parent = self.parent.get(node)
            if parent is None or parent not in self.parent:
                return node
            if self.live_below.get(parent, 0):
                return node
            node = parent

    def _detach(self, node: BlockHash) -> BlockHash | None:
        parent = self.parent.pop(node)
        self.unit.pop(node, None)
        self.children.pop(node, None)
        self.live_below.pop(node, None)
        self.counted.discard(node)
        self.broken.pop(node, None)
        if parent is not None and parent in self.children:
            self.children[parent].discard(node)
            if not self.children[parent]:
                del self.children[parent]
        return parent

    def _drop_node_if_empty(self, node: BlockHash):
        while (
            node in self.parent
            and not self._holds_anything(node)
            and not self.children.get(node)
        ):
            parent = self._detach(node)
            if parent is None or parent not in self.parent:
                return
            node = parent

    def _prune(self, nodes: list[BlockHash]) -> None:
        """Drop the nodes of a just-freed subtree that hold nothing any more.

        ``nodes`` is in preorder, so walking it backwards visits children
        first. Without this, every freed descendant's queued event would walk
        the remaining subtree again: quadratic in the chain length. Nodes
        still cached (e.g. held by a running request) stay.
        """
        for node in reversed(nodes):
            if (
                node not in self.parent
                or self._holds_anything(node)
                or self.children.get(node)
            ):
                continue
            self._detach(node)

    # ---- freeing -----------------------------------------------------------
    def _free(self, group_id: int, block_hash: BlockHash) -> None:
        if self.fine_grained:
            self._free_fine(group_id, block_hash)
            return
        key = make_block_hash_with_group_id(block_hash, group_id)
        for block in self.block_pool.cached_blocks_for(key):
            if self.block_pool.drop_cached_block(block):
                if group_id in self.attn_count:
                    self.num_freed[FREED_ATTN] += 1
                    self._add_ghosts([key])
                elif group_id in self.state_count:
                    self.num_freed[FREED_STATE] += 1
                else:
                    self.num_freed[FREED_SWA] += 1

    def _free_fine(self, group_id: int, block_hash: BlockHash) -> None:
        """Fine-grained ``_free``. A block can hold several entries (a full A
        block also keyed at a partial boundary inside it, by a request whose
        prompt ends there); it is freed only once all of them are dead."""
        key = make_block_hash_with_group_id(block_hash, group_id)
        for block in self.block_pool.cached_blocks_for(key):
            keys = self.block_pool.cached_block_keys(block)
            if any(
                other != key
                and self._needed(get_group_id(other), get_block_hash(other))
                for other in keys
            ):
                continue
            if not self.block_pool.drop_cached_block(block):
                continue
            if group_id in self.attn_count:
                self.num_freed[FREED_ATTN] += 1
                self._add_ghosts(keys)
            elif group_id in self.state_count:
                self.num_freed[FREED_STATE] += 1
            else:
                self.num_freed[FREED_SWA] += 1

    def _add_ghosts(self, keys: Iterable[BlockHash]) -> None:
        for key in keys:
            self.ghost[key] = None
            self.ghost.move_to_end(key)
        while len(self.ghost) > self.ghost_capacity:
            self.ghost.popitem(last=False)

    def _needed(self, group_id: int, node: BlockHash) -> bool:
        """Fine-grained: a usable state needs the entry."""
        if node not in self.parent:
            return False
        if group_id in self.attn_count:
            return self._attn_needed(node, group_id)
        if group_id in self.state_count:
            return node in self.counted
        if group_id in self.swa_count:
            return self._swa_needed(node, group_id)
        return True

    def _free_subtree(self, top: BlockHash, nodes: list[BlockHash]) -> None:
        """Free every block of a subtree with no usable state, then recheck
        the W blocks above it whose windows reached into it."""
        if self.fine_grained:
            self._free_subtree_fine(top, nodes)
            return
        for node in nodes:
            self._free_states(node)
            # A usable parent's hit still needs this block in the eagle groups.
            keep_eagle = self._is_eagle_child(node)
            self._free_attn(node, keep_eagle)
            self._free_swa(node, keep_eagle)
        parent = self.parent.get(top)
        self._prune(nodes)
        if parent is not None and parent in self.parent:
            # The subtree may have held the last usable state below
            # ``parent``; the freed states' own events were dropped with
            # their pruned nodes.
            self._unsupported.append(parent)
            if self.swa_window_blocks:
                self._swa_check.extend(self._ancestors(parent, self._max_window))

    def _free_subtree_fine(self, top: BlockHash, nodes: list[BlockHash]) -> None:
        """Fine-grained ``_free_subtree``: the subtree's states, then the A
        and W entries no usable state (above it) needs."""
        for node in nodes:
            self._free_states(node)
        for node in nodes:
            self._free_unneeded(node)
        parent = self.parent.get(top)
        self._prune(nodes)
        if parent is not None and parent in self.parent:
            self._unsupported.append(parent)
            if self.swa_window_blocks:
                self._swa_check.extend(self._ancestors(parent, self._max_window))

    def _free_unneeded(self, node: BlockHash) -> None:
        for group_id, counts in self.attn_count.items():
            if counts.get(node) and not self._attn_needed(node, group_id):
                self._free(group_id, node)
        for group_id, counts in self.swa_count.items():
            if counts.get(node) and not self._swa_needed(node, group_id):
                self._free(group_id, node)

    def drain(self) -> None:
        """Free every entry the recorded events made dead (cascading)."""
        if self._draining:
            return
        self._draining = True
        try:
            seen_released: set[BlockHash] = set()
            while (
                self._killed
                or self._unsupported
                or self._released
                or self._attn_check
                or self._swa_check
            ):
                if self._killed:
                    # A full A block is gone (and was not cached again since):
                    # nothing below it can hit.
                    root, group_id = self._killed.popleft()
                    if root in self.parent and not self.attn_count[group_id].get(root):
                        nodes = list(self._subtree(root))
                        self._break(root, group_id, nodes)
                        self._free_subtree(root, nodes)
                    continue
                if self._unsupported or self._released:
                    if self._unsupported:
                        node = self._unsupported.popleft()
                    else:
                        node = self._released.pop()
                        if node in seen_released:
                            continue
                        seen_released.add(node)
                    if node not in self.parent:
                        continue
                    if node not in self.counted and self._any_state(node):
                        # Not usable: a state lost in one group, or a window
                        # block gone.
                        self._free_states(node)
                    top = self._top_unsupported(node)
                    if top is not None:
                        # No usable state at or below ``top``: none of its
                        # blocks can ever hit.
                        self._free_subtree(top, list(self._subtree(top)))
                    continue
                if self._attn_check:
                    node = self._attn_check.popleft()
                    if node in self.parent:
                        self._free_unneeded(node)
                        self._drop_node_if_empty(node)
                    continue
                node = self._swa_check.popleft()
                if node not in self.parent:
                    continue
                for group_id, counts in self.swa_count.items():
                    if counts.get(node) and not self._swa_needed(node, group_id):
                        self._free(group_id, node)
                self._drop_node_if_empty(node)
        finally:
            self._draining = False
        self._maybe_log()

    def _maybe_log(self) -> None:
        if self.num_freed == self._num_logged:
            return
        now = time.monotonic()
        if now - self._last_log < _LOG_INTERVAL_S:
            return
        self._last_log = now
        self._num_logged = list(self.num_freed)
        logger.info(
            "Freed dead prefix-cache entries so far: %d attention blocks, "
            "%d Mamba states, %d sliding-window blocks.",
            *self.num_freed,
        )

    def ghost_attention_hit(
        self, block_hashes: Sequence[BlockHash], max_length: int
    ) -> int:
        """Tokens the coordinator's first full-attention lookup
        (``FullAttentionManager.find_longest_cache_hit``) would match had dead
        A entries not been freed, counting cached and freed entries."""
        if not self.ghost:
            return 0
        hash_block_size = self.block_pool.hash_block_size
        units = self.attn_block_units.get(self.ghost_lookup[0], 1)
        eagle = self.ghost_lookup[0] in self.eagle
        num_units = len(block_hashes)
        if eagle:
            max_length = min(max_length + hash_block_size, num_units * hash_block_size)

        def matched(idx: int) -> bool:
            block_hash = block_hashes[idx]
            return all(
                self.attn_count[g].get(block_hash)
                or make_block_hash_with_group_id(block_hash, g) in self.ghost
                for g in self.ghost_lookup
            )

        # Full blocks from the start, then the highest partial entry inside
        # the next block.
        hit = 0
        for _ in range(max_length // (units * hash_block_size)):
            if hit + units > num_units or not matched(hit + units - 1):
                break
            hit += units
        if units > 1:
            last = min(hit + units - 1, max_length // hash_block_size, num_units)
            for idx in range(last - 1, hit - 1, -1):
                if matched(idx):
                    hit = idx + 1
                    break
        if eagle and hit:
            hit -= 1
        return hit * hash_block_size
