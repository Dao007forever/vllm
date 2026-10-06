# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm.v1.hisparse.binding as attn_utils_module
from tests.v1.attention.utils import dense_kv_cache_tensor, dense_kv_cache_views
from vllm.config.compilation import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MultipleOf
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.hisparse.binding import allocate_hisparse_kv_caches
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseResidentSpec,
    KernelBlockGeometry,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
    compute_layout_strides,
    get_kernel_block_geometry,
    group_kernel_blocks,
    require_compact_kernel_block_ids,
)
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.gpu import attn_utils
from vllm.v1.worker.gpu.attn_utils import (
    FastPrefillHelper,
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
    prepare_kernel_slots_per_block,
)


@pytest.mark.parametrize(
    ("enabled", "block_size", "main_sizes", "indexer_sizes", "expected"),
    [
        (True, 256, [64], [64], 64),
        (True, 64, [32, 64], [16, 32], 32),
        (True, 64, [MultipleOf(16)], [32], 32),
        (True, 64, [64], [32], None),
        (False, 256, [64], [64], 256),
    ],
)
def test_get_kv_cache_spec_resolves_hisparse_block_size(
    monkeypatch, enabled, block_size, main_sizes, indexer_sizes, expected
):
    """Resolve shared MLA geometry before planning; leave other specs alone."""
    specs = {
        "main": MLAAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
        ),
        "indexer": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_role=SparseCacheRole.INDEXER,
        ),
        "dense": FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
        ),
    }
    layers = {}
    for name, sizes in zip(specs, [main_sizes, indexer_sizes, [block_size]]):
        backend = SimpleNamespace(
            get_name=lambda name=name: name,
            customize_spec=AttentionBackend.customize_spec,
            get_supported_kernel_block_sizes=lambda sizes=sizes: sizes,
        )
        layers[name] = SimpleNamespace(
            get_kv_cache_spec=lambda _, spec=specs[name]: spec,
            get_attn_backend=lambda backend=backend: backend,
        )
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(
        attn_utils_module, "get_hisparse_kv_cache_groups", lambda *_: []
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=object() if enabled else None)
    )
    if expected is None:
        with pytest.raises(ValueError, match="supported by every sparse"):
            attn_utils.get_kv_cache_spec(config)
        return

    resolved = attn_utils.get_kv_cache_spec(config)
    assert resolved["main"].block_size == resolved["indexer"].block_size == expected
    assert resolved["dense"] is specs["dense"]
    assert all(spec.block_size == block_size for spec in specs.values())


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport, varlen_bound: int | None = None):
        self.support = support
        self.varlen_bound = varlen_bound

    def get_cudagraph_support(self, *_args):
        return self.support

    def get_varlen_cudagraph_max_query_len(self, *_args):
        return self.varlen_bound


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_varlen_cudagraph_unsupported_backend_checks_scoped_bounds():
    """ALWAYS passes without a bound, other builders need one at least as wide as
    the requested length, and NEVER fails whatever bound a builder reports."""
    config: Any = SimpleNamespace()
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )

    def group(
        backend: Any,
        layer_name: str,
        support: AttentionCGSupport,
        bound: int | None = None,
    ):
        builder: Any = _FakeMetadataBuilder(support, bound)
        attn_group = AttentionGroup(backend, [layer_name], spec, 0)
        attn_group.metadata_builders = [builder]
        return attn_group

    target = group(_TargetBackend, "target", AttentionCGSupport.ALWAYS)
    draft = group(_DraftBackend, "draft", AttentionCGSupport.UNIFORM_BATCH, 8)
    never = group(_DraftBackend, "never", AttentionCGSupport.NEVER, 8)

    unsupported = attn_utils.get_varlen_cudagraph_unsupported_backend
    assert unsupported([[target, draft]], config, 8) is None
    assert unsupported([[target, draft]], config, 9) == ("_DraftBackend", 8)
    assert (
        unsupported([[target, draft]], config, 9, checked_layer_names={"target"})
        is None
    )
    assert unsupported([[target, never]], config, 1) == ("_DraftBackend", None)


@pytest.mark.parametrize("num_speculative_tokens", [0, 7])
def test_flashinfer_sparse_full_graphs_exclude_prefill_keep_varlen_decode(
    num_speculative_tokens,
):
    """Prefill must fall back without disabling adaptive verification graphs."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
        FlashInferMLASparseTRTLLMBackend,
    )

    backend = FlashInferMLASparseTRTLLMBackend
    config: Any = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens, parallel_drafting=False
        ),
        use_v2_model_runner=True,
    )
    spec = MLAAttentionSpec(
        block_size=64, num_kv_heads=1, head_size=576, dtype=torch.bfloat16
    )
    group = AttentionGroup(backend, ["target"], spec, 0)
    group.metadata_builders = [object.__new__(backend.get_builder_cls())]
    groups = [[group]]
    support = get_attn_cg_support(groups, config)
    compilation = CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL)
    mode = compilation.resolve_cudagraph_mode_and_sizes(
        support.min_cg_support,
        support.min_cg_attn_backend,
        uniform_decode_query_len=1 + num_speculative_tokens,
        use_v2_model_runner=True,
    )
    assert mode.mixed_mode() != CUDAGraphMode.FULL
    assert mode.decode_mode() == CUDAGraphMode.FULL
    assert get_query_lens_mismatch_unsupported_backend(groups) is None
    unsupported = attn_utils.get_varlen_cudagraph_unsupported_backend
    assert unsupported(groups, config, 1 + num_speculative_tokens) is None
    assert unsupported(groups, config, 2 + num_speculative_tokens) == (
        backend.__name__,
        1 + num_speculative_tokens,
    )


def test_get_kv_sharing_fast_prefill_eligible_layers(monkeypatch: pytest.MonkeyPatch):
    """Fast prefill applies to the contiguous suffix of KV-sharing layers.

    Draft-model layers register after the target model's and may share KV, so
    they must not extend (or break) the target's eligible suffix.
    """

    def check(
        layer_names: list[str],
        shared: dict[str, str],
        draft_layer_names: set[str] | None = None,
    ) -> set[str]:
        monkeypatch.setattr(
            attn_utils,
            "get_layers_from_vllm_config",
            lambda *a, **k: {name: None for name in layer_names},
        )
        monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda *a: shared)
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_sharing_fast_prefill=True)
        )
        return attn_utils.get_kv_sharing_fast_prefill_eligible_layers(
            vllm_config, draft_layer_names
        )

    # No KV sharing: nothing is eligible.
    assert check(["t0", "t1"], {}) == set()

    # Trailing run of sharing layers (YOCO-style second half).
    assert check(["t0", "t1", "t2", "t3"], {"t2": "t1", "t3": "t1"}) == {"t2", "t3"}

    # A non-sharing layer after a sharing one breaks the suffix.
    assert check(["t0", "t1", "t2", "t3"], {"t1": "t0", "t3": "t0"}) == {"t3"}

    # KV-sharing draft layers at the end are collected without an exclusion...
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
    ) == {"t2", "t3", "d0", "d1"}

    # ...so the runner excludes them: skipped, not collected, and they do not
    # break the target's trailing run.
    assert check(
        ["t0", "t1", "t2", "t3", "d0", "d1"],
        {"t2": "t1", "t3": "t1", "d0": "t1", "d1": "t1"},
        draft_layer_names={"d0", "d1"},
    ) == {"t2", "t3"}

    # Feature flag off: nothing is eligible even with sharing layers.
    monkeypatch.setattr(
        attn_utils, "get_layers_from_vllm_config", lambda *a, **k: {"t0": None}
    )
    monkeypatch.setattr(
        attn_utils, "get_shared_kv_cache_layers", lambda *a: {"t0": "t0"}
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False)
    )
    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(vllm_config) == set()


@pytest.mark.parametrize(
    "num_tokens", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128]
)
@pytest.mark.parametrize("num_active_loras", [1, 2, 4])
def test_fast_prefill_dispatch_preserves_active_lora_count(
    num_tokens: int, num_active_loras: int
):
    """Fast-prefill padding must match the main dispatch's LoRA variant."""

    class FakeCudaGraphManager:
        device = "cpu"

        def __init__(self):
            self.dispatch_calls = []

        def dispatch(self, **kwargs):
            self.dispatch_calls.append(kwargs)
            # A captured no-LoRA graph is padded to 8 tokens, while the
            # active-LoRA path stays eager at the unpadded token count.
            if kwargs["num_active_loras"] == 0:
                num_tokens = 8
                mode = CUDAGraphMode.PIECEWISE
            else:
                num_tokens = kwargs["num_tokens"]
                mode = CUDAGraphMode.NONE
            return BatchExecutionDescriptor(
                cg_mode=mode,
                num_tokens=num_tokens,
                num_reqs=kwargs["num_reqs"],
                num_active_loras=kwargs["num_active_loras"],
                num_ubatches=1,
            )

    manager = FakeCudaGraphManager()
    helper = FastPrefillHelper(manager, max_num_tokens=max(32, num_tokens))
    metadata = helper.prepare(
        torch.arange(num_tokens, dtype=torch.int32),
        num_reqs=1,
        cu_num_logits_np=np.array([0, num_tokens], dtype=np.int32),
        has_prefill=True,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=1,
            num_active_loras=num_active_loras,
            num_ubatches=1,
        ),
        num_active_loras=num_active_loras,
    )

    assert metadata is not None
    assert metadata.num_logits_indices == num_tokens
    assert metadata.logits_indices_padded.shape[0] == num_tokens
    assert metadata.max_logits_per_req == num_tokens
    assert manager.dispatch_calls[-1]["num_active_loras"] == num_active_loras


class _FakeSharedHostRegion:
    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.base_tensor = torch.empty(1, dtype=torch.int8)

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def test_profiling_cleanup_releases_tp_shared_region_once(monkeypatch):
    """TP-shared profiling pools must use region-aware chunk cleanup."""
    region = _FakeSharedHostRegion()
    runtime = SimpleNamespace(
        _host_cache=object(),
        registered_host_pool=region.base_tensor,
        hot_backing=object(),
        shared_host_region=region,
    )
    forward_context = {
        "layer": SimpleNamespace(
            hisparse_cache=SimpleNamespace(runtime=runtime),
        )
    }
    released = []

    def release_pinned_state(runtimes, pinned_host_pools, shared_host_region):
        released.append((runtimes, pinned_host_pools, shared_host_region))

    monkeypatch.setattr(
        attn_utils_module,
        "release_pinned_state",
        release_pinned_state,
    )

    attn_utils_module.release_hisparse_profiling_cache(forward_context)

    assert released == [([runtime], [], region)]


@pytest.mark.parametrize("failure_phase", ["allocation", "binding", "buffers"])
def test_init_hisparse_rolls_back_shared_region(monkeypatch, failure_phase):
    """A failure after mmap allocation must not leak the shared registration."""
    region = _FakeSharedHostRegion()
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=1),
    )

    def allocate(*args):
        args[-1].shared_region = region
        if failure_phase == "allocation":
            raise RuntimeError("initialization failed")
        return {}

    def bind(**kwargs):
        if failure_phase == "binding":
            raise RuntimeError("initialization failed")
        return []

    def buffers(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(attn_utils_module, "allocate_hisparse_kv_caches", allocate)
    monkeypatch.setattr(attn_utils_module, "bind_hisparse_kv_caches", bind)
    monkeypatch.setattr(
        attn_utils_module, "initialize_hisparse_runtime_buffers", buffers
    )
    with pytest.raises(RuntimeError, match="initialization failed"):
        attn_utils_module.init_hisparse_kv_cache(
            SimpleNamespace(),
            torch.device("cpu"),
            [],
            vllm_config,
            {},
            SimpleNamespace(),
        )
    assert region.cleanup_calls == 1


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    (
        "kernel_block_sizes",
        "storage_block_size",
        "expected_num_blocks",
        "expected_num_states",
    ),
    [
        (None, None, 4, 64),
        ([256], None, 4, 64),
        ([64], None, 16, 16),
        ([64], 256, 4, 64),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    storage_block_size: int | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
        storage_block_size=storage_block_size,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        (KVCacheLayout.BLHNC, 1),
        # Block rows hold both layers, so kernel block ids skip the other
        # layer's page.
        (KVCacheLayout.BLHNC, 2),
        (KVCacheLayout.BLNHC, 2),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )
    blocks = [group_kernel_blocks(cache, num_blocks) for cache in caches]

    for layer_idx, layer_blocks in enumerate(blocks):
        assert layer_blocks.shape[:2] == (num_blocks, physical_per_logical)
        for block_idx in range(num_blocks):
            for physical_idx in range(physical_per_logical):
                layer_blocks[block_idx, physical_idx].fill_(
                    100 * layer_idx + 10 * block_idx + physical_idx
                )
    expected = [layer_blocks.clone() for layer_blocks in blocks]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, layer_blocks in enumerate(blocks):
        want = expected[layer_idx].clone()
        want[2] = expected[layer_idx][0]
        torch.testing.assert_close(layer_blocks, want)


@pytest.mark.parametrize("layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC])
def test_split_cross_layer_views_cover_each_kernel_page_once(layout: KVCacheLayout):
    """Kernel ids skip the other layer's page, so the layers' views tile the
    buffer exactly once and each block keeps its own row."""
    num_blocks, num_layers, ppl = 3, 2, 4
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw, spec, num_blocks, num_layers, layout, kernel_block_size=16 // ppl
    )

    slots_per_block = num_layers * ppl
    row_bytes = num_layers * spec.page_size_bytes
    for layer_idx, cache in enumerate(caches):
        assert get_kernel_block_geometry(cache, num_blocks) == KernelBlockGeometry(
            ppl, slots_per_block
        )
        assert cache.shape[0] == (num_blocks - 1) * slots_per_block + ppl
        layer_blocks = group_kernel_blocks(cache, num_blocks)
        for block_idx in range(num_blocks):
            assert (
                layer_blocks[block_idx].data_ptr() - raw.data_ptr()
                == block_idx * row_bytes + layer_idx * spec.page_size_bytes
            )
        layer_blocks += 1

    assert torch.equal(raw.view(torch.float32), torch.ones(raw.numel() // 4))


def test_split_padded_pages_skip_the_padding():
    num_blocks, ppl = 3, 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=96,  # 64-byte page + 32 bytes = one kernel page of padding
    )
    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8)
    (cache,) = dense_kv_cache_views(
        raw, spec, num_blocks, 1, KVCacheLayout.LBHNC, kernel_block_size=4 // ppl
    )

    assert get_kernel_block_geometry(cache, num_blocks) == KernelBlockGeometry(ppl, 3)
    group_kernel_blocks(cache, num_blocks).fill_(1)

    rows = raw.view(torch.float32).view(num_blocks, -1)
    real = spec.real_page_size_bytes // 4
    assert torch.equal(rows[:, :real], torch.ones(num_blocks, real))
    assert torch.equal(rows[:, real:], torch.zeros_like(rows[:, real:]))


@pytest.mark.parametrize(
    "layout,page_size_padded,num_kv_heads",
    [
        # 80 bytes is not a multiple of the 32-byte kernel block.
        (KVCacheLayout.LBHNC, 80, 1),
        # Heads sit outside the block, so a kernel block is not one byte run.
        (KVCacheLayout.LHBNC, None, 2),
    ],
)
def test_split_rejects_blocks_off_the_kernel_block_grid(
    layout: KVCacheLayout, page_size_padded: int | None, num_kv_heads: int
):
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=num_kv_heads,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=page_size_padded,
    )
    raw = torch.zeros(num_blocks * 2 * spec.page_size_bytes, dtype=torch.int8)
    with pytest.raises(ValueError, match="kernel block"):
        dense_kv_cache_views(raw, spec, num_blocks, 1, layout, kernel_block_size=2)


@pytest.mark.parametrize(
    "layout,rejected",
    [(KVCacheLayout.LBHNC, False), (KVCacheLayout.BLHNC, True)],
)
def test_compact_only_consumers_reject_spaced_kernel_ids(
    layout: KVCacheLayout, rejected: bool
):
    num_blocks, num_layers = 3, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=2, dtype=torch.float32
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw, spec, num_blocks, num_layers, layout, kernel_block_size=2
    )
    if rejected:
        with pytest.raises(NotImplementedError, match="TestConnector"):
            require_compact_kernel_block_ids(caches, "TestConnector")
    else:
        require_compact_kernel_block_ids(caches, "TestConnector")


def test_block_table_kernel_ids_address_their_pages():
    """Block table ids from the per-group spacing land on each layer's pages."""
    layout = KVCacheLayout.BLHNC
    num_blocks, ppl, kernel_block_size = 3, 4, 4
    layers = ["layer.0", "layer.1"]
    spec = FullAttentionSpec(
        block_size=ppl * kernel_block_size,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    size = num_blocks * len(layers) * spec.page_size_bytes
    tensor = dense_kv_cache_tensor(
        torch.empty(size, dtype=torch.int8),
        spec,
        num_blocks,
        len(layers),
        layout,
        layer_names=layers,
    )
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )

    (slots_per_block,) = prepare_kernel_slots_per_block(
        config, [kernel_block_size], layout
    )
    assert slots_per_block == len(layers) * ppl
    kernel_ids = BlockTable.map_to_kernel_blocks(
        np.array([2, 0]), ppl, np.arange(ppl).reshape(1, -1), slots_per_block
    )
    assert kernel_ids.tolist() == [16, 17, 18, 19, 0, 1, 2, 3]

    caches = allocate_kv_cache(config, torch.device("cpu"), layout, [kernel_block_size])
    base = caches[layers[0]].data_ptr()
    kernel_page_bytes = spec.page_size_bytes // ppl
    row_bytes = len(layers) * spec.page_size_bytes
    for layer_idx, layer in enumerate(layers):
        for i, kernel_id in enumerate(kernel_ids.tolist()):
            block_id, page_idx = [2, 0][i // ppl], i % ppl
            assert caches[layer][kernel_id].data_ptr() - base == (
                block_id * row_bytes
                + layer_idx * spec.page_size_bytes
                + page_idx * kernel_page_bytes
            )


def test_allocate_hisparse_kv_caches_host_pool_and_view_less_specs():
    """Host tensors get their own backing; view-less specs keep the raw one."""
    spec = FullAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=4, dtype=torch.float32
    )
    page = spec.page_size_bytes
    resident_spec = HiSparseResidentSpec(block_size=2, page_size=page)
    device_size = 4 * page
    config = KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=3,
        kv_cache_tensors=[
            KVCacheTensor(
                size=3 * page,
                layers=["source"],
                layer_stride=3 * page,
                block_stride=page,
                host_resident=True,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["indexer"],
                layer_stride=device_size,
                block_stride=page,
            ),
            KVCacheTensor(
                size=device_size,
                layers=["resident"],
                layer_stride=device_size,
                block_stride=page,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["source"], spec, host_resident=True),
            KVCacheGroupSpec(["indexer"], spec),
            KVCacheGroupSpec(["resident"], resident_spec),
        ],
    )
    host_buffers: list[torch.Tensor] = []

    def host_allocator(size: int) -> torch.Tensor:
        host_buffers.append(torch.zeros(size, dtype=torch.int8))
        return host_buffers[-1]

    caches = allocate_hisparse_kv_caches(
        config,
        torch.device("cpu"),
        KVCacheLayout.LBHNC,
        [2, 2, 2],
        SimpleNamespace(allocate=host_allocator),
    )
    assert len(config.kv_cache_tensors) == 3

    assert [buf.numel() for buf in host_buffers] == [3 * page]
    assert caches["source"].shape[0] == 3
    assert (
        caches["source"].untyped_storage().data_ptr()
        == host_buffers[0].untyped_storage().data_ptr()
    )
    assert caches["indexer"].shape[0] == 4
    backing = caches["resident"]
    assert backing.dtype == torch.int8 and backing.numel() >= device_size
    assert (
        backing.untyped_storage().data_ptr()
        == caches["indexer"].untyped_storage().data_ptr()
    )
