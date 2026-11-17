# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Free prefix-cache entries that can no longer serve a hit.

A hybrid model with one full-attention group A and one or more state groups S
(Mamba, align mode, all at A's block size) can only hit at a position whose A
blocks are all cached and that has a cached state in every S group. vLLM's
KV cache grouping splits a model's state layers into several groups of the
attention group's layer count, so S can have more than one group. Two kinds of
cached entries can therefore never hit:

* a state whose A prefix has lost a block, and
* an A block with no cached state at or below it on its chain.

LRU frees them only when it reaches them, and until then they take the place
of entries that could still hit. This tracker keeps the prefix tree of cached
A blocks and the states hanging off it, and frees dead entries as soon as they
appear. Frees cascade: freeing an A block kills the states and A blocks below
it; freeing the last state below an A block kills that block. Freed blocks go
to the front of the free queue, so they are reused before any live entry is
evicted.

A blocks freed this way stay in a bounded hash-only list so that junction
detection (an A prefix deeper than the hybrid hit) still sees them.

Hooks are called by BlockPool while it mutates its queues, so they only record
work; ``drain`` frees the dead entries at a safe point.
"""

from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

_GHOST_CAPACITY = 1 << 20


class DeadEntryTracker:
    def __init__(
        self,
        block_pool: "BlockPool",
        attn_group_id: int,
        state_group_ids: int | tuple[int, ...],
    ):
        self.block_pool = block_pool
        self.attn = attn_group_id
        if isinstance(state_group_ids, int):
            state_group_ids = (state_group_ids,)
        self.states = tuple(state_group_ids)
        # Prefix tree over block hashes of either group.
        self.parent: dict[BlockHash, BlockHash | None] = {}
        self.children: defaultdict[BlockHash, set[BlockHash]] = defaultdict(set)
        # Cached blocks per hash in each group (duplicates are possible).
        self.attn_count: defaultdict[BlockHash, int] = defaultdict(int)
        self.state_count: dict[int, defaultdict[BlockHash, int]] = {
            g: defaultdict(int) for g in self.states
        }
        # Hashes with a state in every state group, counted in ``live_below``
        # of their ancestors.
        self.counted: set[BlockHash] = set()
        # Cached states in the subtree of a node, the node included.
        self.live_below: defaultdict[BlockHash, int] = defaultdict(int)
        # Work recorded by the hooks.
        self._killed: deque[BlockHash] = deque()
        self._unsupported: deque[BlockHash] = deque()
        self._released: list[BlockHash] = []
        self._draining = False
        # A hashes freed as dead, for junction detection.
        self.ghost: OrderedDict[BlockHash, None] = OrderedDict()
        self.num_freed = [0, 0]  # [attention blocks, states]

    # ---- hooks (record only) -------------------------------------------
    def on_insert(self, group_id: int, block_hash: BlockHash, parent: BlockHash | None):
        if group_id != self.attn and group_id not in self.state_count:
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
        if group_id == self.attn:
            self.attn_count[block_hash] += 1
            self.ghost.pop(block_hash, None)
        else:
            self.state_count[group_id][block_hash] += 1
            if block_hash not in self.counted and self._has_state(block_hash):
                self.counted.add(block_hash)
                self._add_live(block_hash, 1)

    def on_remove(self, group_id: int, block_hash: BlockHash):
        if block_hash not in self.parent:
            return
        if group_id == self.attn:
            self.attn_count[block_hash] -= 1
            if self.attn_count[block_hash] <= 0:
                del self.attn_count[block_hash]
                self._killed.append(block_hash)
        elif group_id in self.state_count:
            counts = self.state_count[group_id]
            counts[block_hash] -= 1
            if counts[block_hash] <= 0:
                del counts[block_hash]
                if block_hash in self.counted:
                    self.counted.discard(block_hash)
                    self._add_live(block_hash, -1)
                self._unsupported.append(block_hash)

    def on_release(self, group_id: int, block_hash: BlockHash):
        """An A block's last reference went away (e.g. a request finished)."""
        if group_id == self.attn and block_hash in self.parent:
            self._released.append(block_hash)

    def reset(self):
        self.__init__(self.block_pool, self.attn, self.states)  # type: ignore[misc]

    # ---- tree helpers ----------------------------------------------------
    def _has_state(self, node: BlockHash) -> bool:
        """A state in every state group: a hit can end here."""
        return all(c.get(node) for c in self.state_count.values())

    def _any_state(self, node: BlockHash) -> bool:
        return any(c.get(node) for c in self.state_count.values())

    def _free_states(self, node: BlockHash) -> None:
        for g, counts in self.state_count.items():
            if counts.get(node):
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
        """Highest ancestor-or-self of ``node`` with no state below it."""
        if self.live_below.get(node, 0):
            return None
        while True:
            parent = self.parent.get(node)
            if parent is None or parent not in self.parent:
                return node
            if self.live_below.get(parent, 0):
                return node
            node = parent

    def _drop_node_if_empty(self, node: BlockHash):
        while (
            node in self.parent
            and not self.attn_count.get(node)
            and not self._any_state(node)
            and not self.children.get(node)
        ):
            parent = self.parent.pop(node)
            self.children.pop(node, None)
            self.live_below.pop(node, None)
            if parent is None or parent not in self.children:
                return
            self.children[parent].discard(node)
            if not self.children[parent]:
                del self.children[parent]
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
                or self.attn_count.get(node)
                or self._any_state(node)
                or self.children.get(node)
            ):
                continue
            parent = self.parent.pop(node)
            self.children.pop(node, None)
            self.live_below.pop(node, None)
            if parent is not None and parent in self.children:
                self.children[parent].discard(node)
                if not self.children[parent]:
                    del self.children[parent]
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
                self.num_freed[group_id != self.attn] += 1
                if group_id == self.attn:
                    self.ghost[block_hash] = None
                    if len(self.ghost) > _GHOST_CAPACITY:
                        self.ghost.popitem(last=False)

    def drain(self) -> None:
        """Free every entry the recorded events made dead (cascading)."""
        if self._draining:
            return
        self._draining = True
        try:
            seen_released: set[BlockHash] = set()
            while self._killed or self._unsupported or self._released:
                if self._killed:
                    # An A block is gone: nothing at or below it can hit.
                    root = self._killed.popleft()
                    if root not in self.parent:
                        continue
                    parent = self.parent[root]
                    nodes = list(self._subtree(root))
                    for node in nodes:
                        self._free_states(node)
                        if node != root and self.attn_count.get(node):
                            self._free(self.attn, node)
                    self._prune(nodes)
                    if parent is not None and parent in self.parent:
                        # The subtree may have held the last state below
                        # ``parent``; the freed states' own events were
                        # dropped with their pruned nodes.
                        self._unsupported.append(parent)
                    continue
                if self._unsupported:
                    node = self._unsupported.popleft()
                else:
                    node = self._released.pop()
                    if node in seen_released:
                        continue
                    seen_released.add(node)
                if node not in self.parent:
                    continue
                if not self._has_state(node):
                    # A state lost in one group leaves the others unusable.
                    self._free_states(node)
                top = self._top_unsupported(node)
                if top is None:
                    continue
                # No state at or below ``top``: its A blocks can never hit.
                nodes = list(self._subtree(top))
                for sub in nodes:
                    if self.attn_count.get(sub):
                        self._free(self.attn, sub)
                self._prune(nodes)
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
            if block_hash not in self.ghost and not self.attn_count.get(block_hash):
                break
            idx += 1
        return idx
