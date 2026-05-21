# A Beginner's Guide to vLLM's Scheduler

This is the on-ramp. The scheduler sits at the heart of every vLLM step,
deciding *which* requests get how many tokens of compute on the next
forward pass. The code in [`vllm/v1/core/sched/scheduler.py`](../../vllm/v1/core/sched/scheduler.py)
is dense — it folds prefill, decode, chunked prefill, speculative
decoding, encoder inputs, KV transfer, and preemption into one loop. This
guide builds the intuitions from scratch with small examples, so the
production code reads as confirmations rather than revelations.

Everything below is a re-explanation of what the source already encodes.
Pointers into [`scheduler.py`](../../vllm/v1/core/sched/scheduler.py),
[`output.py`](../../vllm/v1/core/sched/output.py), and
[`request.py`](../../vllm/v1/request.py) appear throughout for when you
want the engineering depth.

---

## 1. The problem: many requests, one GPU, one forward pass

A GPU running an LLM does one thing per "step": a single forward pass of
the model. That pass takes some batch of tokens as input and produces one
output position per input position. The catch is that LLM inference has
two very differently-shaped operations happening simultaneously:

- **Prefill**: a request just arrived with, say, 3,000 prompt tokens. To
  produce its first output token, all 3,000 must run through the model
  once. That's one request × 3,000 tokens.
- **Decode**: a request that's already produced 47 output tokens wants
  to produce token 48. It only needs *1* token of forward-pass work — the
  KV cache (covered in [`learnings/kv_cache/BEGINNER_GUIDE.md`](../kv_cache/BEGINNER_GUIDE.md))
  holds everything else. That's one request × 1 token.

If you ran these as separate kernels you'd waste the GPU constantly:
prefill is compute-bound, decode is memory-bound, and neither fills the
machine alone. The trick that makes vLLM fast is **continuous batching**:
on every step, pack as many tokens from as many requests as the GPU can
chew through in one shot, mixing prefill chunks and decode tokens
freely.

That packing decision is the scheduler's job. The output it produces is a
plan for *one* forward pass: which requests participate, how many tokens
each one contributes, where in the KV cache their tokens write, and a
handful of related bookkeeping fields. Then the worker runs the forward
pass, returns its results, and the scheduler runs again — typically tens
of times per second per engine.

→ The interface lives in [`scheduler.py:308`](../../vllm/v1/core/sched/scheduler.py#L308)
(`Scheduler.schedule`) and the abstract contract is in
[`interface.py:52`](../../vllm/v1/core/sched/interface.py#L52).

## 2. There's no "prefill mode" or "decode mode"

Read the comment at the top of `schedule()` ([`scheduler.py:309-318`](../../vllm/v1/core/sched/scheduler.py#L309-L318)):

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each
> request just has the `num_computed_tokens` and `num_tokens_with_spec`.
> At each step, the scheduler tries to assign tokens to the requests so
> that each request's `num_computed_tokens` can catch up its
> `num_tokens_with_spec`.

This is the central design idea. Every request has two scalar counters:

- **`num_tokens`**: the total token count the request "wants" — prompt
  tokens plus any output tokens already generated, plus speculative
  draft tokens. (Strictly, `num_tokens_with_spec`; see §11.)
- **`num_computed_tokens`**: how many of those have already been pushed
  through the model.

A request is *done* with its current input when these two are equal. On
each step the scheduler picks some "delta" — how many tokens to chew
through this step — and that delta is exactly what `num_scheduled_tokens`
records for the request.

A few concrete cases all fit the same shape:

| Situation | `num_tokens` | `num_computed_tokens` | Delta this step |
|-----------|-------------:|----------------------:|----------------:|
| Fresh request, 100-token prompt | 100 | 0 | up to 100 |
| Same request, fully prefilled, 1st decode | 101 | 100 | 1 |
| Same request, 48th decode | 148 | 147 | 1 |
| Long prompt, chunked prefill, mid-prefill | 3000 | 512 | up to 512 more |
| Spec decode draft of 4 tokens | 148 + 4 | 147 | 5 (1 verify + 4 draft) |

Every code path in the scheduler is computing that delta and respecting
some constraint on it. The constraints are: token budget, KV-cache space,
encoder budget, model max length, max running requests, and a few
correctness invariants. The rest is bookkeeping.

→ See [`request.py:240-245`](../../vllm/v1/request.py#L240-L245) for
`num_tokens` and `num_tokens_with_spec`.

## 3. `num_scheduled_tokens`: the central piece of output metadata

The scheduler's output is a [`SchedulerOutput`](../../vllm/v1/core/sched/output.py#L181)
object. Of all the fields on it, **`num_scheduled_tokens`** is the one
you should learn first — almost everything else either feeds it or is
derived from it.

```python
# req_id -> num_scheduled_tokens
# Number of tokens scheduled for each request.
num_scheduled_tokens: dict[str, int]
# Total number of tokens scheduled for all requests.
# Equal to sum(num_scheduled_tokens.values())
total_num_scheduled_tokens: int
```

It is a `dict[str, int]` — request ID to the number of tokens that
request contributes to the upcoming forward pass. If a request isn't in
this dict, it isn't scheduled this step.

A simple worked example. Suppose at the end of `schedule()` we have:

```
num_scheduled_tokens = {
    "req-A": 1,        # decoding, 1 new token
    "req-B": 1,        # decoding, 1 new token
    "req-C": 1,        # decoding, 1 new token
    "req-D": 512,      # chunked prefill, 512 of 3000 prompt tokens
}
total_num_scheduled_tokens = 515
```

The model runner will assemble a batch with 515 token positions:
positions 0..2 are decode tokens (one each for A, B, C), and positions
3..514 are the next 512 prompt tokens for D's prefill. The forward pass
runs once, returns 515 hidden-state vectors, and the sampler picks one
new token for the *last* position of each request — so A, B, C get a new
decoded token, and D advances its `num_computed_tokens` by 512 (still
mid-prefill, no sampling yet).

This packing is exactly continuous batching. There's no separate
"prefill batch" and "decode batch" — they're stitched together token-by-
token.

A quick note on the post-forward bookkeeping: at the end of
`schedule()`, [`_update_after_schedule`](../../vllm/v1/core/sched/scheduler.py#L930)
adds the scheduled count to each request's `num_computed_tokens`
immediately, *before* the forward has even run. This may sound wrong —
we don't know yet whether the GPU finished without rejection. The
reason it's fine: in plain (non-spec) decoding the count is exactly
right, and for speculative decoding any rejected tokens get subtracted
back in `update_from_output` (§13). Pre-advancing means the next call
to `schedule()` can immediately consider the request as having more
context, which is what makes chunked prefill flow smoothly step over
step.

→ See [`output.py:191-196`](../../vllm/v1/core/sched/output.py#L191-L196)
for the field definitions and
[`scheduler.py:866-882`](../../vllm/v1/core/sched/scheduler.py#L866-L882)
for where `SchedulerOutput` is assembled.

## 4. The token budget: `max_num_batched_tokens`

How does the scheduler know when to stop adding tokens to the next
forward pass? Two main knobs:

- **`max_num_batched_tokens`** (token budget): the maximum number of
  *tokens* across all scheduled requests in one step. This is what
  caps `total_num_scheduled_tokens`.
- **`max_num_seqs`** (running cap): the maximum number of distinct
  requests that can be in `RUNNING` state at once. This caps the size
  of the batch dimension.

Internally, [`schedule()`](../../vllm/v1/core/sched/scheduler.py#L308) initializes
`token_budget = self.max_num_scheduled_tokens` ([`scheduler.py:327`](../../vllm/v1/core/sched/scheduler.py#L327))
and then *every* scheduling decision in the main loop subtracts from it:

```python
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
```

When `token_budget` hits zero, the loop stops. That's the entire
budgeting mechanism.

### Why a token budget and not a request budget?

Tokens are what fills the GPU. A batch of 8 requests doing decode (8
tokens total) and a batch of 1 request doing prefill (3,000 tokens
total) have wildly different forward-pass costs even though one has 8×
more requests. Capping by request count would over-pack on prefill
and under-pack on decode. Capping by token count gives the GPU a
predictable amount of work each step — kernels can be tuned for that
budget, CUDA graphs can capture it, attention metadata can be sized to
it.

The natural follow-up: "how should I choose `max_num_batched_tokens`?"
Higher means bigger batches, better throughput, more memory pressure
per step, and potentially worse tail latency (a decode token sitting
behind a 8k-token prefill chunk waits 8k tokens worth of compute).
Lower means more uniform latency but less throughput. The default
tries to land in a reasonable zone; production deployments tune it.

### `max_num_seqs`: why also cap requests?

Most attention backends keep per-request metadata (block tables, slot
mappings, etc.) sized to `max_num_seqs`. The scheduler enforces it
explicitly in [`scheduler.py:528`](../../vllm/v1/core/sched/scheduler.py#L528):

```python
if len(self.running) == self.max_num_running_reqs:
    break
```

The cap mostly bites in long-context decode scenarios where each
request contributes only 1 token (so the token budget is barely
touched) but you can pile up many of them in parallel.

→ [`scheduler.py:101-106`](../../vllm/v1/core/sched/scheduler.py#L101-L106)
sets both caps.

## 5. Two queues: `running` and `waiting`

The scheduler tracks every live request in one of two collections:

- **`self.running`** ([`scheduler.py:165`](../../vllm/v1/core/sched/scheduler.py#L165)):
  a plain Python `list[Request]` of requests currently in active
  decoding/prefilling. They have KV-cache blocks allocated.
- **`self.waiting`** ([`scheduler.py:162`](../../vllm/v1/core/sched/scheduler.py#L162)):
  a [`RequestQueue`](../../vllm/v1/core/sched/request_queue.py) — FCFS
  by default, optionally a min-heap on priority — of requests not yet
  scheduled (just arrived, just preempted, or waiting on something
  asynchronous).

The lifecycle of one request through these queues:

```
[client adds] → waiting
             ↓ (first time schedule picks it)
              running
             ↓ (decodes for some steps, then finishes)
              [drop, free blocks]

              OR

             ↓ (preempted because someone else needs blocks)
              waiting   (rejoins the front; will retry)
```

A handful of intermediate states model async behavior:
`WAITING_FOR_REMOTE_KVS` (KV transfer from a peer or storage),
`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`, `WAITING_FOR_STREAMING_REQ`.
These all live in a parallel `self.skipped_waiting` queue — a holding
area for requests that are still "waiting" in spirit but can't be
serviced this step. They re-enter the main `waiting` queue once their
dependency clears. The full status enum is in
[`request.py:316-332`](../../vllm/v1/request.py#L316-L332).

## 6. The schedule loop, in two passes

`schedule()` has a clear two-pass structure (a third pass for the
KV-connector metadata sits at the end, but it doesn't change the token
plan):

1. **Pass A — RUNNING requests first.** Walk `self.running` in order,
   give each one as many new tokens as fit in the token budget (plus
   KV blocks, encoder budget, model max len, etc.). This pass is
   [`scheduler.py:344-512`](../../vllm/v1/core/sched/scheduler.py#L344-L512).
2. **Pass B — then WAITING requests.** If the token budget isn't
   exhausted and we still have room in `max_num_seqs`, pop requests
   from `self.waiting`, try to allocate their blocks, and admit them
   to `self.running` for this step. This pass is
   [`scheduler.py:523-802`](../../vllm/v1/core/sched/scheduler.py#L523-L802).

The order matters. **Running requests always get fed first.** This is
deliberate: a request that's been generating for thousands of steps
has invested thousands of forward passes of compute; you don't want
to starve it by admitting a new one and pushing it out of the cache.
Decode-heavy workloads naturally fill the budget with cheap 1-token
decodes from running requests, then if there's any budget left,
admit a few new ones for prefill.

If the running pass *preempts* anyone (had to evict a running
request because there were no free blocks), the waiting pass is
**skipped entirely** that step — see
[`scheduler.py:524`](../../vllm/v1/core/sched/scheduler.py#L524):

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    # ... pass B ...
```

This is a stability measure: if we're already evicting to make room
for the currently-running set, admitting a fresh request would just
trigger more eviction.

→ The whole flow lives in [`schedule()`](../../vllm/v1/core/sched/scheduler.py#L308),
roughly 500 lines but very linear once you know the two-pass shape.

## 7. Chunked prefill: prefill in bites, not gulps

Suppose a request arrives with an 8,000-token prompt and your
`max_num_batched_tokens` is 1,024. A naive scheduler would refuse to
schedule it (won't fit) or hog 8 entire steps to do nothing else.

vLLM does neither. The scheduler simply caps `num_new_tokens` for that
request at the remaining budget:

```python
num_new_tokens = min(num_new_tokens, token_budget)
```

So step 1 schedules 1,024 of the 8,000 prompt tokens for that request.
The next step picks up where it left off — `num_computed_tokens` is
now 1,024, so `num_new_tokens = 8000 - 1024 = 6976`, capped to budget
again, gives another ~1,024-token chunk. After ~8 steps, the prefill
is done and the request transitions into normal decode.

Crucially, those 8 steps are not idle for everyone else. The chunk
takes some of the budget; *other* requests' decode tokens take the
rest. So 1,024-token chunks of prefill ride alongside (budget − 1024)
worth of decode work from other requests. Decode latency for those
other requests goes up only slightly per step, instead of cratering
for 8 full steps.

There's also a longer cap independent of the budget,
`scheduler_config.long_prefill_token_threshold` ([`scheduler.py:369-370`](../../vllm/v1/core/sched/scheduler.py#L369-L370),
[L634-636](../../vllm/v1/core/sched/scheduler.py#L634-L636)) — a
per-step ceiling on how many prefill tokens a *single* request can
contribute. If set, it forces even more aggressive chunking when a
single long prompt would otherwise dominate one step.

### What about prefix caching?

If the request shares a prefix with an already-cached request, the KV
cache manager returns the matching cached blocks before allocation —
[`scheduler.py:570-602`](../../vllm/v1/core/sched/scheduler.py#L570-L602).
Those cached tokens count toward `num_computed_tokens` immediately
(set via [`scheduler.py:783`](../../vllm/v1/core/sched/scheduler.py#L783)),
which means the scheduler "skips ahead" through the cached part of the
prompt and only schedules the uncached suffix. A request with a 3,000-
token prompt that's 90% cached schedules just ~300 tokens for prefill,
not 3,000.

## 8. Allocating blocks, and when allocation fails

Tokens have to land *somewhere* in the KV cache. After computing
`num_new_tokens`, the scheduler asks the KV cache manager:

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

This returns either a `KVCacheBlocks` object (success) or `None` (out
of memory). The block-allocation mechanics — block pool, ref counts,
column-packed tensors — are exactly what the KV cache guide covers
([`learnings/kv_cache/BEGINNER_GUIDE.md`](../kv_cache/BEGINNER_GUIDE.md)).
For the scheduler, the only question is: did we get blocks or not?

If allocation *fails* for a running request, the scheduler **preempts
the youngest running request** (FCFS) or the lowest-priority one
(PRIORITY), frees its blocks, and retries. This is the loop at
[`scheduler.py:422-466`](../../vllm/v1/core/sched/scheduler.py#L422-L466):

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(...)
    if new_blocks is not None:
        break

    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        preempted_req = self.running.pop()  # youngest, FCFS

    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break  # can't preempt ourselves — give up
```

The preempted request goes back to the front of the `waiting` queue.
It'll retry next step, and because of prefix caching may even pick up
much of its already-computed state for free.

For a *waiting* request being admitted, the rule is simpler: if
`allocate_slots` returns `None`, the loop breaks and we stop admitting
new requests this step (see [`scheduler.py:712-719`](../../vllm/v1/core/sched/scheduler.py#L712-L719)).
We don't preempt running requests to make room for *new* ones; that
would be the opposite of the running-first priority.

## 9. The walk-through: one step from scratch

Let's trace a concrete step. Setup:

- `max_num_batched_tokens = 64`
- `max_num_seqs = 4`
- `running = [R1, R2]` where:
    - R1 is mid-decode, has 200 prompt tokens, 47 output tokens
    (`num_computed_tokens = 247`, `num_tokens_with_spec = 247`).
    - R2 just finished prefill, ready for 1st decode
    (`num_computed_tokens = 100`, `num_tokens = 100`, but
    `num_tokens_with_spec = 101` — see §11; for now assume 1 decode
    token).
- `waiting = [W1, W2]`:
    - W1 has a 300-token prompt, all uncached.
    - W2 has a 50-token prompt, fully cached from a previous identical
    request (yay prefix cache).

**Step 1: enter pass A (running). Token budget = 64.**

- **R1**: `num_new_tokens = 247 - 247 = 0`. Wait, that means there's
  nothing to schedule? Actually for a running request mid-decode,
  `num_tokens_with_spec` was incremented by the most recent sampler
  output (or stayed at 247 if R1 is paused waiting for output). Let's
  say last step it generated a new token, so `num_tokens = 248`. Then
  `num_new_tokens = 248 - 247 = 1`. Allocate one slot. Budget = 63.
  `num_scheduled_tokens["R1"] = 1`.
- **R2**: 1 decode token. Allocate one slot. Budget = 62.
  `num_scheduled_tokens["R2"] = 1`.

**Step 2: enter pass B (waiting). Budget = 62, running = 2 < 4.**

- **W1**: 300 prompt tokens, none cached.
  `num_new_tokens = 300 - 0 = 300`, capped to budget = 62.
  Allocate 62 tokens' worth of blocks. Budget = 0.
  `num_scheduled_tokens["W1"] = 62`. W1 is now RUNNING and admitted —
  but only chunk 1 of its prefill landed this step. The next ~5 steps
  will finish the prefill (assuming the budget stays free of larger
  loads).
- **W2**: budget = 0, so the loop exits before even peeking at it.
  (W2 will be admitted next step.)

**Step 3: assemble output.**

```python
SchedulerOutput(
    scheduled_new_reqs=[NewRequestData(req_id="W1", ...)],
    scheduled_cached_reqs=CachedRequestData(req_ids=["R1", "R2"], ...),
    num_scheduled_tokens={"R1": 1, "R2": 1, "W1": 62},
    total_num_scheduled_tokens=64,
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    num_common_prefix_blocks=[0, ...],
    finished_req_ids=set(),
    free_encoder_mm_hashes=[],
    ...
)
```

The worker takes this, builds a batch of 64 token positions (1 + 1 +
62), runs the forward pass, returns 64 hidden states. The sampler
samples a new token for R1 and R2 (their last positions); W1's last
position holds prefill chunk 1's last hidden state but is not yet
sampled because the prefill isn't done — instead its
`num_computed_tokens` advances by 62.

Then `update_from_output` runs, R1's and R2's new tokens are appended
to their `output_token_ids`, and the cycle repeats.

→ See [`schedule()`](../../vllm/v1/core/sched/scheduler.py#L308) for
the full top-to-bottom flow.

## 10. `SchedulerOutput`: what each field carries

Let's catalog the fields. Open [`output.py:181-241`](../../vllm/v1/core/sched/output.py#L181-L241)
alongside this section.

### The token plan

- **`num_scheduled_tokens: dict[str, int]`** — covered in §3. The
  central piece.
- **`total_num_scheduled_tokens: int`** — just
  `sum(num_scheduled_tokens.values())`. Cached because it gets used
  for budget assertions and shape inference downstream.

### Per-request payloads, in two flavors

The model-runner side caches request metadata across steps (so the
prompt, sampling params, LoRA, etc., don't get re-sent every step).
The scheduler exploits this by sending *full* data on first schedule
and only *deltas* on subsequent ones.

- **`scheduled_new_reqs: list[NewRequestData]`** — for requests being
  scheduled for the first time *or* resumed after preemption (when
  `use_v2_model_runner` is set; otherwise resumed requests are listed
  separately in older code paths). Carries the full prompt, sampling
  params, multimodal features, etc. See
  [`NewRequestData`](../../vllm/v1/core/sched/output.py#L31).
- **`scheduled_cached_reqs: CachedRequestData`** — for requests that
  were already known to the worker. Carries only what *changed* this
  step: newly appended block IDs, the latest computed-token count,
  output-token count, and (in PP) the new token IDs from the previous
  sampler. See [`CachedRequestData`](../../vllm/v1/core/sched/output.py#L112).

The build of `scheduled_cached_reqs` is at
[`_make_cached_request_data`](../../vllm/v1/core/sched/scheduler.py#L999).

### Spec decode

- **`scheduled_spec_decode_tokens: dict[str, list[int]]`** — for
  requests participating in speculative decoding this step, the draft
  tokens the kernel will *verify*. Absent from the dict if the request
  has no drafts. See §11.

### Multimodal / encoder

- **`scheduled_encoder_inputs: dict[str, list[int]]`** — for
  multimodal requests, which encoder inputs (images, audio clips, etc.)
  the vision/audio encoder should run *in this step*. Indices point
  into the request's `mm_features` list.
- **`free_encoder_mm_hashes: list[str]`** — encoder outputs the worker
  can drop from its encoder cache because their owning requests have
  finished or moved past them. See `_free_encoder_inputs` in the
  scheduler.

### Prefix-cache fast path: cascade attention

- **`num_common_prefix_blocks: list[int]`** — one int per KV-cache
  group, giving the number of leading blocks that **all** running
  requests share. If nonzero, the attention kernel can use "cascade"
  attention to compute that shared region once instead of per-request.
  Set from `kv_cache_manager.get_num_common_prefix_blocks` ([`scheduler.py:820-825`](../../vllm/v1/core/sched/scheduler.py#L820-L825)).

### Lifecycle / cleanup

- **`finished_req_ids: set[str]`** — requests finished between the
  previous step and this one. The worker uses this to drop cached
  per-request state.
- **`preempted_req_ids: set[str] | None`** — requests preempted in
  *this* schedule call. Only populated for the v2 model runner; used
  to clear any per-request state on the worker.

### Async-only fields

- **`has_structured_output_requests: bool`** and
  **`pending_structured_output_tokens: bool`** — used by async
  scheduling to coordinate grammar bitmask computation.
- **`num_invalid_spec_tokens: dict[str, int] | None`** — adjustments
  to acceptance-rate stats when grammar invalidates draft tokens.

### Connector metadata

- **`kv_connector_metadata` / `ec_connector_metadata`** — opaque
  payloads built by the KV transfer / encoder cache connectors
  (P/D disaggregation, prefix offloading, etc.). The scheduler doesn't
  inspect these; it just builds and forwards them.

### Memory hygiene

- **`new_block_ids_to_zero: list[int] | None`** — block IDs freshly
  allocated this step that the worker must memset to zero before use.
  Set only when the cache config flags `needs_kv_cache_zeroing`
  (typically for Mamba-style state caches where stale NaN/garbage
  would corrupt the recurrence).

## 11. Speculative decoding: extra tokens per step

In speculative decoding, a small "draft" model proposes the next *k*
tokens, and the main model verifies them in *one* forward pass instead
of *k* passes. From the scheduler's viewpoint, this looks like a
running request that wants to advance `num_computed_tokens` by `1 + k`
in one step instead of `1`:

- `num_tokens` is the canonical token count (prompt + output).
- `num_tokens_with_spec = num_tokens + len(spec_token_ids)`.
- The decode step processes `1 + len(spec_token_ids)` tokens: 1 for the
  position after the last accepted real token (always there) plus one
  for each draft token being verified.

That's why `scheduled_spec_decode_tokens` exists as a separate dict.
The model runner needs to know *which* draft tokens are being verified
so it can compare them against the actual sampler output and accept or
reject. Rejected drafts roll back `num_computed_tokens` after the
forward, in `update_from_output` ([`scheduler.py:1310-1334`](../../vllm/v1/core/sched/scheduler.py#L1310-L1334)):

```python
if scheduled_spec_token_ids and generated_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = len(generated_token_ids) - 1
    num_rejected = num_draft_tokens - num_accepted
    if request.num_computed_tokens > 0:
        request.num_computed_tokens -= num_rejected
```

So `num_scheduled_tokens` for a spec-decoding request is consistently
`1 + num_drafts`, and the *number of actual new positions in the
sequence* is `1 + num_accepted` after the forward.

A subtler interaction: the scheduler reserves blocks for *lookahead*
tokens (`num_lookahead_tokens`, [`scheduler.py:216-218`](../../vllm/v1/core/sched/scheduler.py#L216-L218))
when allocating, so that even if all drafts are accepted, there's
already storage waiting for them — no allocation in the hot path of
verification.

## 12. Async scheduling: stay one step ahead

Plain scheduling waits for the GPU forward to finish before scheduling
the next step:

```
step N:   schedule → forward → update_from_output → schedule (step N+1)
```

That gap between "forward done" and "schedule done" leaves the GPU
idle. Async scheduling ([`async_scheduler.py`](../../vllm/v1/core/sched/async_scheduler.py))
fixes this by scheduling step N+1 *before* step N's forward pass
returns:

```
step N:   schedule → forward (overlapped with next schedule)
step N+1: schedule (using placeholder outputs) → ...
```

The trick is that step N's outputs are unknown when step N+1's
scheduler runs. So async scheduling treats the as-yet-unsampled
tokens as **placeholders**: bump `num_output_placeholders` for each
request, optimistically allocate blocks for them, and write
placeholder spec-token IDs (`-1`) into `request.spec_token_ids`.

The override in [`async_scheduler.py:18-35`](../../vllm/v1/core/sched/async_scheduler.py#L18-L35)
shows the pattern:

```python
cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
request.num_output_placeholders += 1 + cur_num_spec_tokens
request.spec_token_ids = self._spec_token_placeholders
```

When the forward actually returns, `update_from_output` reconciles:
real tokens replace placeholders, and `num_output_placeholders` is
decremented accordingly.

This is why `Request.num_output_placeholders` shows up in so many
expressions in `schedule()`. The scheduler can't ignore them — they
count as "computed tokens" for the purpose of figuring out what's
next, even though the actual sampling hasn't happened yet.

## 13. The other half: `update_from_output`

`schedule()` produces a plan. After the worker runs the forward pass,
[`update_from_output`](../../vllm/v1/core/sched/scheduler.py#L1246) is
the scheduler's other big method. Its job is to:

1. Append the newly sampled tokens to each request's
   `output_token_ids`.
2. Subtract any rejected speculative tokens from
   `num_computed_tokens`.
3. Free encoder inputs whose decoder positions have now been
   processed.
4. Check stop conditions (EOS, max tokens, stop strings, repetition)
   via `check_stop` ([`utils.py:94`](../../vllm/v1/core/sched/utils.py#L94)).
5. Mark finished requests, route them through the KV-connector for
   any "request finished" hooks, and put their IDs into
   `self.finished_req_ids` for the next step's `SchedulerOutput`.
6. Emit `EngineCoreOutputs` for each client — a per-client batched
   summary of new tokens, logprobs, finish reasons, etc.

The two methods together — `schedule` then `update_from_output` — are
the entire heartbeat of the engine. Everything else is bookkeeping
that flows from these two.

## 14. Why this design holds together

A few invariants quietly make the whole thing work:

- **One scheduler thread per engine.** Just like the KV cache's free
  list and ref counts, the scheduler relies on single-threaded
  ownership: no two `schedule()` calls run concurrently against the
  same state, so block allocation, ref counts, and queue mutations
  don't need locks.
- **`num_scheduled_tokens` is the contract.** The worker is told
  exactly how many tokens per request to process. There's no implicit
  "the worker figures out where to stop." If `num_scheduled_tokens["R1"]
  = 5`, the worker writes K/V into 5 slots, no more, no fewer. Drifts
  here would break the cache silently.
- **Pre-advance `num_computed_tokens`.** Bumping it in
  `_update_after_schedule` (before the forward) lets the *next*
  scheduling step start work without waiting for the forward to
  finish. Async scheduling depends on this; plain scheduling tolerates
  the small risk because spec-decode rollback is the only case where
  the bump can be wrong, and `update_from_output` handles that.
- **Running requests get fed first.** Preserves invested compute and
  prevents new requests from starving old ones. Combined with FCFS, it
  also bounds tail latency for the requests that have been around
  longest.
- **Preempt the youngest, not the oldest.** Same argument as the KV
  cache guide's §11 — the youngest has the least computed work and so
  is cheapest to redo when it comes back.

## 15. The big takeaways

If you remember nothing else:

1. **There's no prefill or decode "mode."** Every request just has
   `num_computed_tokens` chasing `num_tokens_with_spec`. The scheduler
   picks how much of that delta to take this step.

2. **`num_scheduled_tokens` is the central piece of output.** A
   `dict[req_id, int]` telling the worker how many tokens each
   request contributes to the next forward pass. Everything else on
   `SchedulerOutput` either feeds it or is derived from it.

3. **Continuous batching is just packing tokens, not "merging modes."**
   Prefill chunks and decode tokens flow through the same kernel in the
   same batch. The token budget caps the total.

4. **Running first, waiting second.** Decode-heavy workloads naturally
   fill the budget from running requests; only leftover budget admits
   new ones. If preemption happens, no waiting requests are admitted
   that step.

5. **Chunked prefill is just `min(num_new_tokens, token_budget)`.** A
   8k-token prompt naturally gets split across multiple steps without
   any prefill-specific code paths.

6. **Allocation drives preemption.** When `allocate_slots` returns
   `None`, the scheduler frees the youngest running request and
   retries. Preempted requests rejoin the *front* of `waiting`.

7. **Async scheduling overlaps schedule with forward.** Output
   placeholders (`num_output_placeholders`, sentinel spec token IDs)
   let `schedule()` work on step N+1 while step N's forward is still
   running. Reconciliation happens in `update_from_output`.

8. **Speculative decoding looks like "delta of `1 + k`" to the
   scheduler.** The verifier kernel uses
   `scheduled_spec_decode_tokens` to know which drafts to check;
   rejected ones are subtracted from `num_computed_tokens` after the
   forward.

9. **The single-threaded scheduler is load-bearing.** Same as in the
   KV cache: lock-free ref counting, lock-free queue mutation,
   lock-free block-table updates all rely on it.

10. **Failures are usually silent.** A wrong `num_scheduled_tokens`
    value, a stale block ID, a forgotten preempted-request cleanup —
    none of these crash the GPU. They corrupt the KV cache and the
    model gets quietly worse. The code is dense with assertions for a
    reason ([`scheduler.py:805-815`](../../vllm/v1/core/sched/scheduler.py#L805-L815)
    is a good representative sample).

When you're ready for more depth:

- [`scheduler.py`](../../vllm/v1/core/sched/scheduler.py) itself — read
  `schedule()` top to bottom once you've internalized the two-pass
  structure.
- [`output.py`](../../vllm/v1/core/sched/output.py) — the data types
  flowing between scheduler and worker.
- [`async_scheduler.py`](../../vllm/v1/core/sched/async_scheduler.py) —
  short, but a great look at how placeholders thread the needle.
- [`request_queue.py`](../../vllm/v1/core/sched/request_queue.py) — the
  FCFS vs. PRIORITY queues.
- [`learnings/kv_cache/BEGINNER_GUIDE.md`](../kv_cache/BEGINNER_GUIDE.md)
  — the KV cache machinery the scheduler talks to via `allocate_slots`
  and `get_computed_blocks`.
