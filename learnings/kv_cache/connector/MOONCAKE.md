# MooncakeStoreConnector — `ReqMeta`

This note focuses on one data class: `ReqMeta` in
[vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py](../../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py).
It is the per-request "work order" the scheduler emits on every engine
step. The worker reads it and decides what to put into Mooncake (save)
and what to get from Mooncake (load).

If you haven't yet, skim [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) — `ReqMeta`
is just this connector's concrete subclass of the generic per-request
slice you put inside a `KVConnectorMetadata`.

## Where `ReqMeta` Sits

```text
scheduler (MooncakeStoreScheduler)
   │  per scheduler tick:
   │    build_connector_meta(scheduler_output)
   │
   ▼
MooncakeStoreConnectorMetadata     ◄── one per engine step
   ├── unfinished_request_ids
   ├── preempted_req_ids
   └── requests: list[ReqMeta]     ◄── one per request scheduled this step
                                       (or with a pending load)

worker (MooncakeConnectorWorker / send + recv threads)
   for req in meta.requests:
     if req.can_save:           send_thread.add_request(req)
     if req.load_spec.can_load: recv_thread.add_request(req)
```

`ReqMeta` is constructed by `ReqMeta.from_request_tracker(...)` inside
[scheduler.py](../../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py)
and consumed by the send/recv threads' `_handle_request(req_meta)` in
[worker.py](../../../vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py).

## The Companion Class: `RequestTracker`

`ReqMeta` is rebuilt every step. `RequestTracker` persists across steps —
the scheduler keeps one per in-flight request in `self._request_trackers`.
Its job is to remember "how far have we saved this request so far?" so the
next step knows where to resume.

| Field                  | Meaning                                                                                       |
|------------------------|-----------------------------------------------------------------------------------------------|
| `req_id`               | vLLM request ID.                                                                              |
| `token_len`            | Token length covered by the *current* prefill chunk (grows as chunked prefill progresses).    |
| `allocated_block_ids`  | Per-group block IDs for this request in the GPU paged cache. Updated when new blocks alloc'd. |
| `num_saved_tokens`     | High-watermark of tokens already saved to Mooncake. Drives the "anything new to save?" check. |
| `token_ids`            | Cached token-id list, used to populate KV cache events.                                       |
| `prefill_end_tokens`   | Snapshot of the prefill range. For a fresh req = `len(prompt)`. For a preempt-resume req it includes previously-generated tokens that have to be re-prefilled. Used to compute `is_last_chunk`. |

Keep this distinction in mind when reading `ReqMeta`: most fields are
*derived from* the tracker plus this-step state, not carried across steps.

## The `ReqMeta` Fields

```python
@dataclass
class ReqMeta:
    req_id: str
    token_len_chunk: int
    block_ids: tuple[list[int], ...]
    block_hashes: list[BlockHash]

    can_save: bool | None = None
    load_spec: LoadSpec | None = None
    is_last_chunk: bool | None = None
    current_event: torch.cuda.Event | None = None

    token_ids: list[int] | None = None
    original_block_size: int | None = None
```

The fields fall into four groups: **identity**, **save plan**, **load
plan**, and **side-channel info for events / synchronization**.

### Identity & Addressing

#### `req_id: str`

The vLLM request ID. Worker threads use it as the dedup key for in-flight
saves (`stored_requests`), the cancel key under store pressure, and the
completion key reported back in `get_finished()` as `done_sending` /
`done_recving`.

#### `block_ids: tuple[list[int], ...]`

Per-KV-cache-group lists of GPU paged-cache block IDs holding this
request's KV. `block_ids[g]` is "the blocks for group `g`."

It's a **tuple of lists**, not a flat list, because of the Hybrid Memory
Allocator. A hybrid model (e.g. full-attn + sliding-window, or attn +
Mamba) has multiple `KVCacheGroup`s and each group has its own block pool.
The scheduler constructs this as:

```python
# scheduler.py — build_connector_meta
if isinstance(request.block_ids, tuple):
    unfolded_block_ids = tuple(b.copy() for b in request.block_ids)
else:
    unfolded_block_ids = (request.block_ids.copy(),)   # 1-tuple for legacy
```

The send thread iterates groups and calls
`db.prepare_value(start, end, block_ids_per_group[g_idx])` to translate
"these tokens of group `g`" → "these GPU memory addresses in group `g`'s
KV tensor."

#### `block_hashes: list[BlockHash]`

Content hashes for this request's blocks, computed at the engine's
configured `hash_block_size`. These are vLLM's prefix-cache block hashes,
*reused* as the global keys into the Mooncake distributed store.

A single hash → a single Mooncake key:

```python
# data.py — ChunkedTokenDatabase
PoolKey(self.metadata, chunk_hash=block_hash.hex())
# rendered as:
"model_name@tp_rank:i@pcp...@dcp...@pp_rank:j@group:g@<hex hash>"
```

If the connector's effective `block_size` is larger than `hash_block_size`
(LCM stretching for HMA), `ChunkedTokenDatabase.process_tokens` merges
multiple consecutive `BlockHash`es into one chunk hash via
`BlockHashListWithBlockSize`. That's why you see two block-size knobs in
the scheduler: `_block_size` (effective) and `_hash_block_size`
(per-block-hash granularity).

Because the key is content-addressed, **a store hit from one engine run
can be served to a different engine run as long as model/tp/pp identity
matches**. That's the whole point of Mooncake.

### Save Plan

#### `token_len_chunk: int`

How many tokens of this request should be considered for save in this
step. Block-aligned when `discard_partial_chunks=True` (the default):

```python
# data.py — from_request_tracker
num_tokens_to_save = (
    (input_token_len // block_size * block_size)
    if discard_partial_chunks
    else input_token_len
)
```

`input_token_len` is `tracker.token_len`, i.e. all tokens whose KV will
exist after this step's forward — which is "old prefix + this step's
scheduled tokens." Aligning down to the block size means we never save a
partial last block (a block that the forward will keep writing into next
step). That last block becomes saveable next step, once it's full.

The send thread further floors this to the **HMA LCM block size** before
using it:

```python
# worker.py — KVCacheStoreSendingThread._handle_request
token_len = req_meta.token_len_chunk // lcm_block_size * lcm_block_size
```

That's needed because for hybrid models the smallest unit at which *all*
groups agree on a block boundary is the LCM of each group's `block_size`.

#### `can_save: bool | None`

`True` if anything in this `ReqMeta` should actually be saved this step,
`False` otherwise. The decision lives in `from_request_tracker`:

```python
chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size
skip_save = skip_save or num_tokens_to_save < chunk_boundary
```

`chunk_boundary` is "the next block boundary strictly past what we've
already saved." If the new chunk doesn't even reach that boundary, skip —
there's nothing new whole-block to put. `from_request_tracker` also
returns `None` (no `ReqMeta` added to the metadata) if `skip_save and
load_spec is None` — i.e. nothing to save *and* nothing to load.

`force_skip_save` is set when `kv_role == "kv_consumer"`: a pure consumer
side of P/D disagg never writes back to the store.

When `can_save` is `True`, the send thread is the one that updates
`req_meta.current_event` and queues the request into Mooncake's
`batch_put_from_multi_buffers`.

#### `is_last_chunk: bool | None`

`True` if this chunk reaches the end of the prefill range:

```python
# scheduler.py
last_chunk_tokens_num = (prefill_end // block_size * block_size)
is_last_chunk = (request_tracker.token_len >= last_chunk_tokens_num)
```

The send thread uses this to decide when the request's store is fully
done — once the last chunk's store completes, the request's id can be
returned from `get_finished()` as `finished_sending`, and the scheduler
will release the blocks (recall `request_finished` returned `True` to
defer free while saving was in flight). Without `is_last_chunk` the
worker couldn't tell mid-prefill chunks apart from the final one.

### Load Plan

#### `load_spec: LoadSpec | None`

If non-`None`, this request wants to load KV from Mooncake into its GPU
blocks. Definition:

```python
@dataclass
class LoadSpec:
    vllm_cached_tokens: int     # tokens already covered by local prefix cache
    kvpool_cached_tokens: int   # tokens we found in Mooncake
    can_load: bool              # blocks allocated and ready to receive
    token_len: int = 0          # actual range to load (filled in by worker)
```

How each field becomes set:

- `vllm_cached_tokens` / `kvpool_cached_tokens`: filled in
  `get_num_new_matched_tokens` after `client.lookup()` returns how many
  cached tokens Mooncake has. (Subtracts one if 100% hit so vLLM still
  generates from at least one real token.)
- `can_load`: set to `True` in `update_state_after_alloc` once the
  scheduler has actually allocated the blocks the load needs.
- `token_len`: set by the worker in `get_finished()`, right before it
  hands the request to the recv thread. The recv thread uses it (not
  `token_len_chunk`) as the load range.

The recv thread also uses `vllm_cached_tokens` as a *mask offset* — it
won't re-fetch tokens that the local prefix cache already covered:

```python
# worker.py — KVCacheStoreRecvingThread._handle_request
mask_num = req_meta.load_spec.vllm_cached_tokens // block_size * block_size
... db.process_tokens(token_len, req_meta.block_hashes, mask_num)
```

If `load_spec.can_load` is `False`, the worker skips the load. The
scheduler sets it `False` when `num_external_tokens == 0` (no blocks were
actually allocated for external load, e.g. because the local prefix cache
already covered everything).

### Side-channel: Events & Synchronization

#### `current_event: torch.cuda.Event | None`

This field is **not set by the scheduler**. It's filled in on the worker
side in `get_finished()` just before the request is handed to the send
thread:

```python
# worker.py — get_finished
current_event = None
for request in meta.requests:
    if request.can_save:
        current_event = torch.cuda.Event()
        current_event.record()   # records on the current (compute) stream
        break

for request in meta.requests:
    if not request.can_save:
        continue
    request.current_event = current_event
    ...
    self.kv_send_thread.add_request(request)
```

One event is recorded on the compute stream after the forward, shared
across all this-step save requests. The send thread synchronizes on it
before issuing the Mooncake put:

```python
if current_event is not None:
    current_event.synchronize()
```

Without this barrier, the Mooncake put could race the forward pass that
writes KV into those same paged blocks — you'd ship half-written KV. It's
the worker-side equivalent of `wait_for_save` for *async* saves on a
background thread.

This is why the field exists on `ReqMeta` even though it's worker-only:
the send thread receives only `ReqMeta`, so the event has to ride along.

### Side-channel: KV-event Reporting

These two fields exist purely so the connector can emit accurate
`BlockStored` events on the KV event bus (consumers outside vLLM that
mirror the global prefix cache):

#### `token_ids: list[int] | None`

The tokens corresponding to this chunk, so the emitted event can include
them:

```python
# worker.py — KVCacheStoreSendingThread._handle_request
token_ids = req_meta.token_ids[s:e] if req_meta.token_ids is not None else None
stored_event = BlockStored(
    block_hashes=[new_block_hashes[idx]],
    token_ids=token_ids,
    block_size=req_meta.original_block_size,
    ...,
)
```

It's populated from `tracker.token_ids` (which the scheduler initialized
from `prefill_tokens[:num_tokens_to_compute]`). `None` is fine if KV
events are disabled.

#### `original_block_size: int | None`

The engine's user-visible block size — i.e. `cache_config.block_size`,
not the LCM-stretched effective `_block_size` the connector uses
internally. External KV-event consumers index by the user-visible size,
so the event has to carry the original.

## Construction Logic In One Diagram

```text
RequestTracker
  ├── token_len             ◄── grows each step (chunked prefill)
  └── num_saved_tokens      ◄── high-watermark; updated when can_save=True

ReqMeta.from_request_tracker:

  num_tokens_to_save = floor(token_len / block_size) * block_size     (if discard_partial)
  chunk_boundary     = ceil((num_saved_tokens + 1) / block_size) * block_size
  skip_save          = (caller forced skip) OR num_tokens_to_save < chunk_boundary

  if skip_save and no load_spec: return None
  if not skip_save: tracker.num_saved_tokens = num_tokens_to_save

  ReqMeta(
    req_id              = tracker.req_id
    token_len_chunk     = num_tokens_to_save           # save range this step
    block_ids           = tracker.allocated_block_ids  # per-group blocks
    block_hashes        = caller-supplied              # prefix-cache hashes
    can_save            = not skip_save
    load_spec           = caller-supplied (or None)    # only set when external hit
    is_last_chunk       = caller-supplied              # uses tracker.prefill_end_tokens
    token_ids           = tracker.token_ids            # for KV events
    original_block_size = engine's cache_config.block_size
    current_event       = None                         # worker fills in later
  )
```

## Cheat Sheet

| Field                  | Set where?                                 | Used by                  | Why                                                                        |
|------------------------|--------------------------------------------|--------------------------|----------------------------------------------------------------------------|
| `req_id`               | tracker                                    | both                     | dedup, cancel, completion reporting                                        |
| `token_len_chunk`      | tracker.token_len → floor to block         | send (re-floor to LCM)   | how many tokens to save this step                                          |
| `block_ids`            | tracker.allocated_block_ids                | both                     | translate tokens → GPU memory addresses, per HMA group                     |
| `block_hashes`         | from request.block_hashes                  | both                     | derive Mooncake `PoolKey`s (content-addressed)                             |
| `can_save`             | derived from tracker.num_saved_tokens      | send                     | gate: save this step?                                                      |
| `load_spec`            | scheduler load_specs dict                  | recv, and worker.get_finished | load amounts; `.can_load`/`.token_len` gate the recv thread          |
| `is_last_chunk`        | tracker.token_len ≥ prefill end            | send                     | know when to report `finished_sending` for this req                        |
| `current_event`        | worker.get_finished()                      | send                     | block save behind compute-stream forward; avoids torn KV                   |
| `token_ids`            | tracker.token_ids                          | send                     | populate `BlockStored.token_ids`                                           |
| `original_block_size`  | scheduler (`cache_config.block_size`)      | send                     | populate `BlockStored.block_size` at user-visible granularity              |

## Reading Tips

- **Save path** — read `from_request_tracker` and
  `KVCacheStoreSendingThread._handle_request` side by side. The scheduler
  decides what's saveable; the send thread translates ranges to addresses,
  dedups against `batch_is_exist`, and `batch_put_from_multi_buffers`.
- **Load path** — read `get_num_new_matched_tokens` →
  `update_state_after_alloc` → `worker.get_finished` (where it stamps
  `load_spec.token_len`) → `KVCacheStoreRecvingThread._handle_request`.
- The two fields most likely to confuse on first read are `token_len_chunk`
  (save) vs `load_spec.token_len` (load) — they are independent ranges
  living on the same `ReqMeta`.
