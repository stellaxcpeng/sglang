import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HEAD_DIM = 128


def _fake_dynamic_quant(rows, dst_type=torch.int8):
    assert rows.ndim == 2
    assert dst_type == torch.int8
    rows_fp32 = rows.float()
    scale = rows_fp32.abs().amax(dim=-1) / 127.0
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(rows_fp32 / safe_scale.unsqueeze(-1)).clamp(-127, 127)
    return quant.to(torch.int8), scale.to(torch.float32)


def _patch_dynamic_quant(monkeypatch):
    monkeypatch.setattr(
        torch.ops.npu,
        "npu_dynamic_quant",
        _fake_dynamic_quant,
        raising=False,
    )


def _make_indexer(index_topk=8):
    indexer = Indexer.__new__(Indexer)
    indexer.head_dim = HEAD_DIM
    indexer.index_topk = index_topk
    return indexer


def test_quantize_index_k_is_per_token(monkeypatch):
    _patch_dynamic_quant(monkeypatch)
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.index_head_dim = HEAD_DIM
    pool._logged_int8_index_quantization = False

    index_k = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0] * 32,
            [-4.0, -2.0, 0.0, 2.0] * 32,
            [0.0] * HEAD_DIM,
        ],
        dtype=torch.bfloat16,
    )
    quant, scale = pool._quantize_index_k(index_k)

    assert quant.shape == index_k.shape
    assert quant.dtype == torch.int8
    assert scale.shape == (index_k.shape[0],)
    assert scale.dtype == torch.float16
    assert scale[1] > scale[0]
    torch.testing.assert_close(
        quant.float() * scale.float().unsqueeze(-1),
        index_k.float(),
        atol=float(scale.max()),
        rtol=0,
    )


def test_set_index_k_buffer_writes_quant_and_scale_to_same_slots(monkeypatch):
    _patch_dynamic_quant(monkeypatch)

    def fake_scatter(destination, indices, updates):
        destination[indices.view(-1).long()] = updates
        return destination

    monkeypatch.setattr(
        memory_pool_npu,
        "torch_npu",
        SimpleNamespace(npu_scatter_nd_update_=fake_scatter),
        raising=False,
    )

    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.enable_index_k_int8 = True
    pool.index_head_dim = HEAD_DIM
    pool.start_layer = 0
    pool._logged_int8_index_quantization = False
    pool.index_k_buffer = torch.zeros(1, 2, 4, 1, HEAD_DIM, dtype=torch.int8)
    pool.index_k_scale_buffer = torch.zeros(1, 2, 4, 1, dtype=torch.float16)

    loc = torch.tensor([1, 6], dtype=torch.int64)
    source = torch.stack(
        [
            torch.linspace(-1, 1, HEAD_DIM),
            torch.linspace(-4, 4, HEAD_DIM),
        ]
    ).to(torch.bfloat16)
    expected_q, expected_scale = pool._quantize_index_k(source)

    pool.set_index_k_buffer(layer_id=0, loc=loc, index_k=source)

    flat_q = pool.index_k_buffer[0].view(-1, 1, HEAD_DIM)
    flat_scale = pool.index_k_scale_buffer[0].view(-1, 1, 1)
    torch.testing.assert_close(flat_q[loc].squeeze(1), expected_q)
    torch.testing.assert_close(
        flat_scale[loc].view(-1), expected_scale.view(-1)
    )
    assert torch.count_nonzero(flat_q[[0, 2, 3, 4, 5, 7]]) == 0


def test_contiguous_buffer_infos_include_index_scale_group():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.layer_num = 2
    pool.index_head_dim = HEAD_DIM
    pool.k_buffer = torch.zeros(2, 3, 4, 1, 16, dtype=torch.bfloat16)
    pool.v_buffer = torch.zeros(2, 3, 4, 1, 8, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.zeros(2, 3, 4, 1, HEAD_DIM, dtype=torch.int8)
    pool.index_k_scale_buffer = torch.zeros(2, 3, 4, 1, dtype=torch.float16)

    ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
    expected_buffers = [
        *pool.k_buffer,
        *pool.v_buffer,
        *pool.index_k_buffer,
        *pool.index_k_scale_buffer,
    ]

    assert ptrs == [buffer.data_ptr() for buffer in expected_buffers]
    assert lens == [buffer.nbytes for buffer in expected_buffers]
    assert item_lens == [buffer[0].nbytes for buffer in expected_buffers]


def test_get_index_k_buffer_preserves_int8_dtype():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.layer_transfer_counter = None
    pool.start_layer = 0
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.index_k_buffer = torch.zeros(1, 2, 16, 1, HEAD_DIM, dtype=torch.int8)

    index_k = pool.get_index_k_buffer(layer_id=0)

    assert index_k.data_ptr() == pool.index_k_buffer[0].data_ptr()
    assert index_k.dtype == torch.int8


def test_cpu_offload_round_trip_includes_index_scale(monkeypatch):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.layer_num = 1
    pool.index_head_dim = HEAD_DIM
    pool.kv_lora_rank = 16
    pool.qk_rope_head_dim = 8
    pool.cpu_offloading_chunk_size = 2
    pool.k_buffer = torch.randn(1, 2, 4, 1, 16, dtype=torch.bfloat16)
    pool.v_buffer = torch.randn(1, 2, 4, 1, 8, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.randint(
        -127, 128, (1, 2, 4, 1, HEAD_DIM), dtype=torch.int16
    ).to(torch.int8)
    pool.index_k_scale_buffer = torch.rand(1, 2, 4, 1, dtype=torch.float16)
    indices = torch.tensor([1, 6], dtype=torch.int64)

    expected = [
        pool.k_buffer[0].view(-1, 1, 16)[indices].clone(),
        pool.v_buffer[0].view(-1, 1, 8)[indices].clone(),
        pool.index_k_buffer[0].view(-1, 1, HEAD_DIM)[indices].clone(),
        pool.index_k_scale_buffer[0].view(-1, 1, 1)[indices].clone(),
    ]
    cpu_copy = pool.get_cpu_copy(indices)
    pool.k_buffer.zero_()
    pool.v_buffer.zero_()
    pool.index_k_buffer.zero_()
    pool.index_k_scale_buffer.zero_()

    pool.load_cpu_copy(cpu_copy, indices)

    actual = [
        pool.k_buffer[0].view(-1, 1, 16)[indices],
        pool.v_buffer[0].view(-1, 1, 8)[indices],
        pool.index_k_buffer[0].view(-1, 1, HEAD_DIM)[indices],
        pool.index_k_scale_buffer[0].view(-1, 1, 1)[indices],
    ]
    for restored, reference in zip(actual, expected):
        torch.testing.assert_close(restored, reference)


def test_quant_lightning_indexer_receives_expected_contract(monkeypatch):
    _patch_dynamic_quant(monkeypatch)
    captured = {}

    def fake_quant_lightning_indexer(
        query, key, weights, query_scale, key_scale, **kw
    ):
        captured.update(
            query=query,
            key=key,
            weights=weights,
            query_scale=query_scale,
            key_scale=key_scale,
            kwargs=kw,
        )
        output = torch.arange(query.shape[0] * 8, dtype=torch.int32).view(
            query.shape[0], 1, 8
        )
        return output, torch.empty(0)

    monkeypatch.setattr(
        dsa_indexer,
        "torch_npu",
        SimpleNamespace(npu_quant_lightning_indexer=fake_quant_lightning_indexer),
        raising=False,
    )
    monkeypatch.setattr(Indexer, "_logged_npu_quant_lightning_indexer", False)
    indexer = _make_indexer(index_topk=8)

    query = torch.randn(3, 32, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randint(
        -127, 128, (2, 16, 1, HEAD_DIM), dtype=torch.int16
    ).to(torch.int8)
    key_scale = torch.rand(2, 16, 1, dtype=torch.float16)
    weights = torch.randn(3, 32, dtype=torch.bfloat16)
    output = indexer._run_npu_quant_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        key_scale=key_scale,
        actual_seq_lengths_query=torch.tensor([3]),
        actual_seq_lengths_key=torch.tensor([3]),
        block_table=torch.tensor([[0]], dtype=torch.int64),
    )

    assert output.shape == (3, 1, 8)
    assert captured["query"].dtype == torch.int8
    assert captured["key"].dtype == torch.int8
    assert captured["weights"].dtype == torch.float16
    assert captured["query_scale"].shape == (3, 32)
    assert captured["query_scale"].dtype == torch.float16
    assert captured["key_scale"].shape == key.shape[:-1]
    assert captured["key_scale"].dtype == torch.float16
    assert captured["kwargs"]["layout_query"] == "TND"
    assert captured["kwargs"]["layout_key"] == "PA_BSND"
    assert captured["kwargs"]["query_quant_mode"] == 0
    assert captured["kwargs"]["key_quant_mode"] == 0
    assert captured["kwargs"]["sparse_mode"] == 3
    assert captured["kwargs"]["sparse_count"] == 8
    assert captured["kwargs"]["block_table"].dtype == torch.int32


@pytest.mark.parametrize("query_heads", [0, 65])
def test_quant_lightning_indexer_rejects_unsupported_query_heads(query_heads):
    indexer = _make_indexer()
    with pytest.raises(ValueError, match="Q_N"):
        indexer._run_npu_quant_lightning_indexer(
            query=torch.empty(2, query_heads, HEAD_DIM, dtype=torch.bfloat16),
            key=torch.empty(1, 16, 1, HEAD_DIM, dtype=torch.int8),
            weights=torch.empty(2, query_heads, dtype=torch.float16),
            key_scale=torch.empty(1, 16, 1, dtype=torch.float16),
            actual_seq_lengths_query=torch.tensor([2]),
            actual_seq_lengths_key=torch.tensor([2]),
            block_table=torch.tensor([[0]], dtype=torch.int32),
        )


def test_quant_lightning_indexer_rejects_metadata_batch_mismatch(monkeypatch):
    _patch_dynamic_quant(monkeypatch)
    indexer = _make_indexer()
    with pytest.raises(ValueError, match="metadata batch mismatch"):
        indexer._run_npu_quant_lightning_indexer(
            query=torch.empty(2, 32, HEAD_DIM, dtype=torch.bfloat16),
            key=torch.empty(1, 16, 1, HEAD_DIM, dtype=torch.int8),
            weights=torch.empty(2, 32, dtype=torch.float16),
            key_scale=torch.empty(1, 16, 1, dtype=torch.float16),
            actual_seq_lengths_query=torch.tensor([1, 2]),
            actual_seq_lengths_key=torch.tensor([2]),
            block_table=torch.tensor([[0]], dtype=torch.int32),
        )


def test_cp_prev_and_next_share_quant_indexer_helper():
    indexer = _make_indexer()
    prev_output = torch.zeros(3, 1, 8, dtype=torch.int32)
    next_output = torch.ones(2, 1, 8, dtype=torch.int32)
    indexer._run_npu_quant_lightning_indexer = MagicMock(
        side_effect=[prev_output, next_output]
    )

    key = torch.empty(1, 16, 1, HEAD_DIM, dtype=torch.int8)
    key_scale = torch.empty(1, 16, 1, dtype=torch.float16)
    outputs = indexer.do_npu_cp_balance_indexer(
        q=torch.empty(5, 32, HEAD_DIM, dtype=torch.bfloat16),
        past_key_states=key,
        past_key_scales=key_scale,
        indexer_weights=torch.empty(5, 32, dtype=torch.float16),
        actual_seq_lengths_q=(torch.tensor([3]), torch.tensor([2])),
        actual_seq_lengths_kv=(torch.tensor([5]), torch.tensor([5])),
        block_table=torch.tensor([[0]], dtype=torch.int32),
    )

    assert outputs[0] is prev_output
    assert outputs[1] is next_output
    assert indexer._run_npu_quant_lightning_indexer.call_count == 2
    first_call = indexer._run_npu_quant_lightning_indexer.call_args_list[0].kwargs
    second_call = indexer._run_npu_quant_lightning_indexer.call_args_list[1].kwargs
    assert first_call["query"].shape == (3, 32, HEAD_DIM)
    assert second_call["query"].shape == (2, 32, HEAD_DIM)
    assert first_call["key_scale"] is key_scale
    assert second_call["key_scale"] is key_scale


def test_model_runner_int8_keeps_main_mla_kv_in_model_dtype(monkeypatch):
    _patch_dynamic_quant(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_quant_lightning_indexer=lambda *args, **kwargs: None),
    )

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "npu"
    runner.dtype = torch.bfloat16
    runner.server_args = SimpleNamespace(kv_cache_dtype="int8")
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepseekV32ForCausalLM"],
            index_topk=2048,
            index_head_dim=HEAD_DIM,
        )
    )

    runner.configure_kv_cache_dtype()

    assert runner.kv_cache_dtype == torch.bfloat16


def test_model_runner_rejects_index_k_int8_on_non_npu():
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.server_args = SimpleNamespace(kv_cache_dtype="int8")

    with pytest.raises(ValueError, match="limited to DSA index_k"):
        runner.configure_kv_cache_dtype()
