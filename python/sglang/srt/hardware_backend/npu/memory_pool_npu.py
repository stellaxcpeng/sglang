import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.common import is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

if is_npu():
    import torch_npu

logger = logging.getLogger(__name__)


def _init_npu_conv_state(
    conv_state_in, conv_state_shape, speculative_num_draft_tokens: Optional[int] = None
):
    extra_conv_len = 0
    if speculative_num_draft_tokens is not None:
        extra_conv_len = speculative_num_draft_tokens - 1

    # conv_state shape (layers, pool_size, conv_wind + draft_step, dim) for conv1d ascendc ops require dim as last dim
    conv_state = [
        torch.zeros(
            size=(
                conv_state_in.shape[0],
                conv_state_in.shape[1],
                conv_shape[1] + extra_conv_len,
                conv_shape[0],
            ),
            dtype=conv_state_in.dtype,
            device=conv_state_in.device,
        )
        for conv_shape in conv_state_shape
    ]
    return conv_state


class NPUMHATokenToKVPool(MHATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.use_fia = get_bool_env_var("ASCEND_USE_FIA", "False")
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

    def _create_buffers(self):
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # [size, head_num, head_dim] for each layer
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            # Continuous memory improves the efficiency of Ascend`s transmission backend,
            # while other backends remain unchanged.
            self.kv_buffer = torch.zeros(
                (
                    2,
                    self.layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer = self.kv_buffer[0]
            self.v_buffer = self.kv_buffer[1]

            if self.use_fia:
                self.k_buffer = []
                self.v_buffer = []
                for i in range(self.layer_num):
                    k_buffer_layer = self.kv_buffer[0][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    v_buffer_layer = self.kv_buffer[1][i].view(
                        -1, 1, self.head_num, self.head_dim
                    )
                    self.k_buffer.append(k_buffer_layer)
                    self.v_buffer.append(v_buffer_layer)

    # for disagg
    def get_contiguous_buf_infos(self):
        # layer_num x [seq_len, head_num, head_dim]
        # layer_num x [page_num, page_size, head_num, head_dim]
        kv_data_ptrs = [
            self.get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self.get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self.get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        if self.use_fia:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes * self.page_size
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        else:
            kv_item_lens = [
                self.get_key_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ] + [
                self.get_value_buffer(i)[0].nbytes
                for i in range(self.start_layer, self.start_layer + self.layer_num)
            ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if self.use_fia:
            k_buffer_layer = self.k_buffer[layer_id - self.start_layer]
            v_buffer_layer = self.v_buffer[layer_id - self.start_layer]

            torch_npu.npu_scatter_nd_update_(
                k_buffer_layer,
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.head_num, self.head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                v_buffer_layer,
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.head_num, self.head_dim),
            )
        else:
            loc = loc.to(torch.int32)
            torch_npu._npu_reshape_and_cache(
                key=cache_k,
                value=cache_v,
                key_cache=self.k_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                value_cache=self.v_buffer[layer_id - self.start_layer].view(
                    -1, self.page_size, self.head_num, self.head_dim
                ),
                slot_indices=loc,
            )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    # Parent MHATokenToKVPool.get_cpu_copy / load_cpu_copy use
    # `self.k_buffer[layer_id][chunk_indices]` which indexes the first dim.
    # NPUMHATokenToKVPool stores buffers as
    #   (num_pages, page_size, head_num, head_dim)            # use_fia=False
    #   (num_pages*page_size, 1, head_num, head_dim)          # use_fia=True
    def get_cpu_copy(self, indices):
        torch.npu.synchronize()
        buf_of_layers = []
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            buf_of_layers.append([k_layer, v_layer])
        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            v_layer = self.v_buffer[local_layer_id].view(
                -1, self.head_num, self.head_dim
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[local_layer_id][i // chunk_size][0],
                    kv_cache_cpu[local_layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
        torch.npu.synchronize()


class NPUMLATokenToKVPool(MLATokenToKVPool):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: Optional[int],
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super(MLATokenToKVPool, self).__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.index_head_dim = index_head_dim
        self.kv_quant_tile_size = 128
        if self.store_dtype == torch.int8:
            assert self.kv_lora_rank % self.kv_quant_tile_size == 0, (
                "INT8 MLA KV cache requires kv_lora_rank to be divisible by "
                f"{self.kv_quant_tile_size}, but got {self.kv_lora_rank}."
            )
        self.kv_scale_tiles = max(
            1,
            (self.kv_lora_rank + self.kv_quant_tile_size - 1)
            // self.kv_quant_tile_size,
        )
        self._logged_int8_rope_quantization = False
        self._logged_int8_rope_dequantization = False
        self._logged_int8_index_quantization = False

        self.custom_mem_pool = None

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            # The padded slot 0 is used for writing dummy outputs from padded tokens.
            self.k_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.kv_lora_rank,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.k_buffer_scale = None
            if self.store_dtype == torch.int8:
                self.k_buffer_scale = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.kv_scale_tiles,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            self.v_buffer = torch.zeros(
                (
                    layer_num,
                    self.size // self.page_size + 1,
                    self.page_size,
                    1,
                    self.qk_rope_head_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer_scale = None
            if self.store_dtype == torch.int8:
                self.v_buffer_scale = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            self.index_k_buffer = None
            self.index_k_scale_buffer = None
            if self.index_head_dim is not None:
                index_k_dtype = self.store_dtype
                self.index_k_buffer = torch.zeros(
                    (
                        layer_num,
                        self.size // self.page_size + 1,
                        self.page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=index_k_dtype,
                    device=self.device,
                )
                if self.store_dtype == torch.int8:
                    self.index_k_scale_buffer = torch.zeros(
                        (
                            layer_num,
                            self.size // self.page_size + 1,
                            self.page_size,
                            1,
                        ),
                        dtype=torch.float16,
                        device=self.device,
                    )

        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        kv_size_bytes = 0
        for k_cache in self.k_buffer:
            kv_size_bytes += get_tensor_size_bytes(k_cache)
        for v_cache in self.v_buffer:
            kv_size_bytes += get_tensor_size_bytes(v_cache)
        if self.store_dtype == torch.int8:
            assert self.k_buffer_scale is not None
            for scale_cache in self.k_buffer_scale:
                kv_size_bytes += get_tensor_size_bytes(scale_cache)
            assert self.v_buffer_scale is not None
            for scale_cache in self.v_buffer_scale:
                kv_size_bytes += get_tensor_size_bytes(scale_cache)
        if self.index_head_dim is not None:
            assert hasattr(self, "index_k_buffer")
            for index_k_cache in self.index_k_buffer:
                kv_size_bytes += get_tensor_size_bytes(index_k_cache)
            if self.store_dtype == torch.int8:
                assert self.index_k_scale_buffer is not None
                for scale_cache in self.index_k_scale_buffer:
                    kv_size_bytes += get_tensor_size_bytes(scale_cache)
        return kv_size_bytes

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return (
            self.k_buffer[layer_id - self.start_layer],
            self.v_buffer[layer_id - self.start_layer],
        )

    def get_state_buf_infos(self):
        if self.index_head_dim is None:
            return [], [], []
        data_ptrs = [self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)]
        data_lens = [self.index_k_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)]
        return data_ptrs, data_lens, item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype == torch.int8:
            return self.k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.k_buffer[layer_id - self.start_layer]

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype == torch.int8:
            return self.v_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        assert self.k_buffer_scale is not None
        return self.k_buffer_scale[layer_id - self.start_layer]

    def get_value_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.v_buffer_scale[layer_id - self.start_layer]

    def get_index_k_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_scale_buffer[layer_id - self.start_layer]

    def get_dequantized_value_buffer(
        self, layer_id: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        v_buffer = self.v_buffer[layer_id - self.start_layer]
        if self.store_dtype != torch.int8:
            if self.store_dtype != self.dtype:
                return v_buffer.view(self.dtype)
            return v_buffer

        v_scale = self.v_buffer_scale[layer_id - self.start_layer]
        if not self._logged_int8_rope_dequantization:
            logger.info(
                "Using dequantized k_rope from INT8 v_buffer. "
                f"v_buffer_shape={tuple(v_buffer.shape)}, "
                f"v_scale_shape={tuple(v_scale.shape)}, "
                f"dequant_dtype={dtype}"
            )
            self._logged_int8_rope_dequantization = True
        return v_buffer.to(dtype) * v_scale.unsqueeze(-1).to(dtype)

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if self.store_dtype == torch.int8:
            return self.index_k_buffer[layer_id - self.start_layer]
        if self.store_dtype != self.dtype:
            return self.index_k_buffer[layer_id - self.start_layer].view(self.dtype)
        return self.index_k_buffer[layer_id - self.start_layer]

    # for disagg
    def get_contiguous_buf_infos(self):
        # Returns: c_kv (k_buffer), k_rope (v_buffer), scales, and optionally index_k.
        # INT8 mode transfers raw INT8 k_rope plus a per-token rope scale.
        kv_data_ptrs = [self.k_buffer[i].data_ptr() for i in range(self.layer_num)] + [
            self.v_buffer[i].data_ptr() for i in range(self.layer_num)
        ]
        kv_data_lens = [self.k_buffer[i].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i].nbytes for i in range(self.layer_num)
        ]
        kv_item_lens = [self.k_buffer[i][0].nbytes for i in range(self.layer_num)] + [
            self.v_buffer[i][0].nbytes for i in range(self.layer_num)
        ]
        if self.store_dtype == torch.int8:
            # c_kv_scale (FP32): one scale per token per 128-wide latent tile.
            kv_data_ptrs += [
                self.k_buffer_scale[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.k_buffer_scale[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.k_buffer_scale[i][0].nbytes for i in range(self.layer_num)
            ]
            kv_data_ptrs += [
                self.v_buffer_scale[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.v_buffer_scale[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.v_buffer_scale[i][0].nbytes for i in range(self.layer_num)
            ]
        if self.index_head_dim is not None:
            kv_data_ptrs += [
                self.index_k_buffer[i].data_ptr() for i in range(self.layer_num)
            ]
            kv_data_lens += [
                self.index_k_buffer[i].nbytes for i in range(self.layer_num)
            ]
            kv_item_lens += [
                self.index_k_buffer[i][0].nbytes for i in range(self.layer_num)
            ]
            if self.store_dtype == torch.int8:
                kv_data_ptrs += [
                    self.index_k_scale_buffer[i].data_ptr()
                    for i in range(self.layer_num)
                ]
                kv_data_lens += [
                    self.index_k_scale_buffer[i].nbytes
                    for i in range(self.layer_num)
                ]
                kv_item_lens += [
                    self.index_k_scale_buffer[i][0].nbytes
                    for i in range(self.layer_num)
                ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def _quant_scale_to_antiquant_scale(self, quant_scale: torch.Tensor) -> torch.Tensor:
        """Convert KV quant scale to antiquant scale for FIA.

        npu_dynamic_quant scale direction must be confirmed on the target CANN
        version. Existing NPU dynamic-quant call sites consume the returned scale
        as the dequant multiplier, so keep identity here.
        FIA's antiquant_scale parameter expects the multiplier, so this is identity.
        Kept as a helper so a microtest can adjust the direction if needed.
        """
        return quant_scale

    def _quantize_kv_lora_rank_per_tile(
        self, cache_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic-quantize MLA c_kv with one scale per 128-wide tile."""
        tile_rows = cache_k.contiguous().view(
            -1,
            self.kv_scale_tiles,
            self.kv_quant_tile_size,
        ).view(
            -1,
            self.kv_quant_tile_size,
        )
        quant_rows, tile_scale = torch.ops.npu.npu_dynamic_quant(
            tile_rows, dst_type=torch.int8
        )
        cache_k_i8 = quant_rows.view_as(cache_k)
        cache_k_scale = tile_scale.view(*cache_k.shape[:-1], self.kv_scale_tiles)
        return cache_k_i8, cache_k_scale

    def _quantize_rope_per_token(
        self, cache_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic-quantize MLA k_rope with one scale per token."""
        rows = cache_v.contiguous().view(-1, self.qk_rope_head_dim)
        cache_v_i8, cache_v_scale = torch.ops.npu.npu_dynamic_quant(
            rows, dst_type=torch.int8
        )
        if not self._logged_int8_rope_quantization:
            logger.info(
                "Quantizing k_rope into INT8 v_buffer. "
                f"source_shape={tuple(cache_v.shape)}, "
                f"quant_shape={tuple(cache_v_i8.view_as(cache_v).shape)}, "
                f"scale_shape={tuple(cache_v_scale.view(*cache_v.shape[:-1]).shape)}, "
                f"source_dtype={cache_v.dtype}"
            )
            self._logged_int8_rope_quantization = True
        return cache_v_i8.view_as(cache_v), cache_v_scale.view(*cache_v.shape[:-1])

    def _quantize_index_k(
        self, index_k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic-quantize DSA index_k with one scale per token."""
        rows = index_k.contiguous().view(-1, self.index_head_dim)
        index_k_i8, index_k_scale = torch.ops.npu.npu_dynamic_quant(
            rows, dst_type=torch.int8
        )
        index_k_i8 = index_k_i8.view_as(index_k)
        index_k_scale = index_k_scale.view(*index_k.shape[:-1]).to(torch.float16)
        if not self._logged_int8_index_quantization:
            logger.info(
                "Quantizing index_k into INT8 index_k_buffer. "
                f"source_shape={tuple(index_k.shape)}, "
                f"quant_shape={tuple(index_k_i8.shape)}, "
                f"scale_shape={tuple(index_k_scale.shape)}, "
                f"source_dtype={index_k.dtype}"
            )
            self._logged_int8_index_quantization = True
        return index_k_i8, index_k_scale

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id

        if cache_v is None:
            cache_k, cache_v = cache_k.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        if self.store_dtype == torch.int8:
            assert cache_k.shape[-2] == 1, (
                "INT8 MLA KV cache expects one KV head for kv-quant sparse attention, "
                f"but got cache_k shape {tuple(cache_k.shape)}."
            )
            # INT8 mode: dynamic-quantize c_kv per 128-wide tile and
            # k_rope per token. Attention kernels consume dequantized k_rope.
            cache_k_i8, cache_k_scale = self._quantize_kv_lora_rank_per_tile(cache_k)
            antiquant_scale = self._quant_scale_to_antiquant_scale(cache_k_scale)
            cache_v_i8, cache_v_scale = self._quantize_rope_per_token(cache_v)
            rope_antiquant_scale = self._quant_scale_to_antiquant_scale(cache_v_scale)

            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.kv_lora_rank
                ),
                loc.view(-1, 1),
                cache_k_i8.view(-1, 1, self.kv_lora_rank),
            )
            torch_npu.npu_scatter_nd_update_(
                self.k_buffer_scale[layer_id - self.start_layer].view(
                    -1, 1, self.kv_scale_tiles
                ),
                loc.view(-1, 1),
                antiquant_scale.view(-1, 1, self.kv_scale_tiles),
            )

            torch_npu.npu_scatter_nd_update_(
                self.v_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.qk_rope_head_dim
                ),
                loc.view(-1, 1),
                cache_v_i8.view(-1, 1, self.qk_rope_head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                self.v_buffer_scale[layer_id - self.start_layer].view(-1, 1, 1),
                loc.view(-1, 1),
                rope_antiquant_scale.view(-1, 1, 1),
            )
        else:
            # BF16 / FP8 path: no quantization, just dtype conversion and store.
            if cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)

            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)

            torch_npu.npu_scatter_nd_update_(
                self.k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.kv_lora_rank
                ),
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.kv_lora_rank),
            )
            torch_npu.npu_scatter_nd_update_(
                self.v_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.qk_rope_head_dim
                ),
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.qk_rope_head_dim),
            )

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        if self.store_dtype == torch.int8:
            index_k_i8, index_k_scale = self._quantize_index_k(index_k)
            torch_npu.npu_scatter_nd_update_(
                self.index_k_buffer[layer_id - self.start_layer].view(
                    -1, 1, self.index_head_dim
                ),
                loc.view(-1, 1),
                index_k_i8.view(-1, 1, self.index_head_dim),
            )
            torch_npu.npu_scatter_nd_update_(
                self.index_k_scale_buffer[layer_id - self.start_layer].view(-1, 1, 1),
                loc.view(-1, 1),
                index_k_scale.view(-1, 1, 1),
            )
            return

        index_dtype = torch.bfloat16 if self.store_dtype == torch.int8 else self.dtype
        if index_k.dtype != index_dtype:
            index_k = index_k.to(index_dtype)
        if self.store_dtype != torch.int8 and self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)

        torch_npu.npu_scatter_nd_update_(
            self.index_k_buffer[layer_id - self.start_layer].view(
                -1, 1, self.index_head_dim
            ),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )

    def _chunk_copy_npu_to_cpu(self, buf_of_layers, indices):
        chunk_size = self.cpu_offloading_chunk_size
        out = []
        for tensors_per_layer in buf_of_layers:  # [k_buf, v_buf, ik_buf/None]
            layer_chunks = []
            for i in range(0, len(indices), chunk_size):
                ci = indices[i : i + chunk_size]
                layer_chunks.append(
                    [
                        t[ci].to("cpu", non_blocking=True)
                        for t in tensors_per_layer
                        if t is not None
                    ]
                )
            out.append(layer_chunks)
        return out

    def get_cpu_copy(self, indices):
        torch.npu.synchronize()
        buf_of_layers = []
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            scale_layer = (
                self.k_buffer_scale[local_layer_id].view(
                    -1, 1, self.kv_scale_tiles
                )
                if self.store_dtype == torch.int8
                else None
            )
            v_scale_layer = (
                self.v_buffer_scale[local_layer_id].view(-1, 1, 1)
                if self.store_dtype == torch.int8
                else None
            )
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            ik_scale_layer = (
                self.index_k_scale_buffer[local_layer_id].view(-1, 1, 1)
                if has_ik and self.store_dtype == torch.int8
                else None
            )
            buf_of_layers.append(
                [k_layer, v_layer, scale_layer, v_scale_layer, ik_layer, ik_scale_layer]
            )

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(buf_of_layers, indices)
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        has_ik = self.index_head_dim is not None
        for local_layer_id in range(self.layer_num):
            k_layer = self.k_buffer[local_layer_id].view(-1, 1, self.kv_lora_rank)
            v_layer = self.v_buffer[local_layer_id].view(-1, 1, self.qk_rope_head_dim)
            scale_layer = (
                self.k_buffer_scale[local_layer_id].view(
                    -1, 1, self.kv_scale_tiles
                )
                if self.store_dtype == torch.int8
                else None
            )
            v_scale_layer = (
                self.v_buffer_scale[local_layer_id].view(-1, 1, 1)
                if self.store_dtype == torch.int8
                else None
            )
            ik_layer = (
                self.index_k_buffer[local_layer_id].view(-1, 1, self.index_head_dim)
                if has_ik
                else None
            )
            ik_scale_layer = (
                self.index_k_scale_buffer[local_layer_id].view(-1, 1, 1)
                if has_ik and self.store_dtype == torch.int8
                else None
            )
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                k_cpu, v_cpu = chunk[0], chunk[1]
                next_idx = 2
                assert k_cpu.shape[0] == len(chunk_indices)
                k_layer[chunk_indices] = k_cpu.to(k_layer.device, non_blocking=True)
                v_layer[chunk_indices] = v_cpu.to(v_layer.device, non_blocking=True)
                if self.store_dtype == torch.int8:
                    scale_cpu = chunk[next_idx]
                    next_idx += 1
                    scale_layer[chunk_indices] = scale_cpu.to(
                        scale_layer.device, non_blocking=True
                    )
                    v_scale_cpu = chunk[next_idx]
                    next_idx += 1
                    v_scale_layer[chunk_indices] = v_scale_cpu.to(
                        v_scale_layer.device, non_blocking=True
                    )
                if has_ik:
                    ik_cpu = chunk[next_idx]
                    next_idx += 1
                    ik_layer[chunk_indices] = ik_cpu.to(
                        ik_layer.device, non_blocking=True
                    )
                    if self.store_dtype == torch.int8:
                        ik_scale_cpu = chunk[next_idx]
                        ik_scale_layer[chunk_indices] = ik_scale_cpu.to(
                            ik_scale_layer.device, non_blocking=True
                        )
        torch.npu.synchronize()
