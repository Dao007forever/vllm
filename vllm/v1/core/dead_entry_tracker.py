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

A blocks freed this way stay in a bounded hash-only list so that junction
detection (an A prefix deeper than the hybrid hit) still sees them.

Hooks are called by BlockPool while it mutates its queues, so they only record
work (and update counts); ``drain`` frees the dead entries at a safe point.
"""

from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

_GHOST_CAPACITY = 1 << 20

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
                drop one matched block for EAGLE.

        """
        self.block_pool = block_pool
        self.attn = _as_tuple(attn_group_ids)
        self.states = _as_tuple(state_group_ids)
        self.swa_window_blocks = dict(swa_window_blocks or {})
        assert self.attn and self.states
        assert all(n >= 1 for n in self.swa_window_blocks.values())
        self.eagle = frozenset(eagle_group_ids)
        assert self.eagle <= set(self.attn) | set(self.swa_window_blocks)
        # The longest window, in blocks.
        self._max_window = max(self.swa_window_blocks.values(), default=1)
        # Prefix tree over the block hashes of every tracked group.
        self.parent: dict[BlockHash, BlockHash | None] = {}
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
        # Work recorded by the hooks.
        self._killed: deque[BlockHash] = deque()
        self._unsupported: deque[BlockHash] = deque()
        self._released: list[BlockHash] = []
        # W blocks to check against the windows of the usable states.
        self._swa_check: deque[BlockHash] = deque()
        self._draining = False
        # A hashes freed as dead, for junction detection.
        self.ghost: OrderedDict[BlockHash, None] = OrderedDict()
        self.num_freed = [0, 0, 0]  # [attention blocks, states, W blocks]

    # ---- hooks (record only) -------------------------------------------
    def on_insert(self, group_id: int, block_hash: BlockHash, parent: BlockHash | None):
        if (
            group_id not in self.attn_count
            and group_id not in self.state_count
            and group_id not in self.swa_count
        ):
            return
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
            if self._has_attn(block_hash):
                self.ghost.pop(block_hash, None)
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

    def on_remove(self, group_id: int, block_hash: BlockHash):
        if block_hash not in self.parent:
            return
        if group_id in self.attn_count:
            counts = self.attn_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                self._killed.append(block_hash)
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
        elif group_id in self.swa_count:
            self._swa_check.append(block_hash)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            self.block_pool,
            self.attn,
            self.states,
            self.swa_window_blocks,
            self.eagle,
        )

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
        return self._has_state(node) and self._window_complete(node)

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
        if group_id in self.eagle and self._is_eagle_child(node):
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
        self.children.pop(node, None)
        self.live_below.pop(node, None)
        self.counted.discard(node)
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
        if nodes:
            parent = self.parent.get(nodes[0])
            if parent is not None:
                self._drop_node_if_empty(parent)
        if nodes and nodes[0] in self.parent:
            self._drop_node_if_empty(nodes[0])

    # ---- freeing -----------------------------------------------------------
    def _free(self, group_id: int, block_hash: BlockHash) -> None:
        key = make_block_hash_with_group_id(block_hash, group_id)
        for block in self.block_pool.cached_blocks_for(key):
            if self.block_pool.drop_cached_block(block):
                if group_id in self.attn_count:
                    self.num_freed[FREED_ATTN] += 1
                    self.ghost[block_hash] = None
                    if len(self.ghost) > _GHOST_CAPACITY:
                        self.ghost.popitem(last=False)
                elif group_id in self.state_count:
                    self.num_freed[FREED_STATE] += 1
                else:
                    self.num_freed[FREED_SWA] += 1

    def _free_subtree(self, top: BlockHash, nodes: list[BlockHash]) -> None:
        """Free every block of a subtree with no usable state, then recheck
        the W blocks above it whose windows reached into it."""
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

    def drain(self) -> None:
        """Free every entry the recorded events made dead (cascading)."""
        if self._draining:
            return
        self._draining = True
        try:
            seen_released: set[BlockHash] = set()
            while (
                self._killed or self._unsupported or self._released or self._swa_check
            ):
                if self._killed:
                    # An A block is gone: nothing at or below it can hit.
                    root = self._killed.popleft()
                    if root not in self.parent:
                        continue
                    self._free_subtree(root, list(self._subtree(root)))
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
                node = self._swa_check.popleft()
                if node not in self.parent:
                    continue
                for group_id, counts in self.swa_count.items():
                    if counts.get(node) and not self._swa_needed(node, group_id):
                        self._free(group_id, node)
                self._drop_node_if_empty(node)
        finally:
            self._draining = False

    def extend_with_ghosts(
        self, block_hashes: list[BlockHash], start_block: int, max_blocks: int
    ) -> int:
        """Blocks past ``start_block`` an A prefix would have reached had dead
        A blocks not been freed (cached or recently freed as dead)."""
        if not self.ghost:
            return start_block
        idx = start_block
        while idx < max_blocks:
            block_hash = block_hashes[idx]
            if block_hash not in self.ghost and not self._has_attn(block_hash):
                break
            idx += 1
        return idx
