from __future__ import annotations

import torch


def expand_dsa_sparse_indices(
    topk_indices: torch.Tensor, num_query_tokens: int
) -> torch.Tensor:
    """Pad dummy rows and expand [T, K] for NPU sparse attention."""
    assert topk_indices.dim() in (2, 3), (
        f"Expected DSA top-k indices with 2 or 3 dims, got {topk_indices.dim()}"
    )
    num_topk_rows = topk_indices.shape[0]
    assert num_topk_rows <= num_query_tokens, (
        f"DSA top-k rows ({num_topk_rows}) exceed query rows ({num_query_tokens})"
    )
    if num_topk_rows < num_query_tokens:
        topk_indices = torch.cat(
            (
                topk_indices,
                topk_indices.new_full(
                    (num_query_tokens - num_topk_rows, *topk_indices.shape[1:]),
                    -1,
                ),
            ),
            dim=0,
        )
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def align_lightning_indexer_graph_metadata(
    query_rows: int,
    actual_seq_lengths_kv: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad NPU lightning-indexer graph metadata to the captured query batch.

    Attention-TP gathering can expand a decode query to the graph's static token
    width while request-level KV metadata still contains only the real rows.  A
    zero KV length plus a zero block-table row represents an inert graph-padding
    query and keeps the three TND batch dimensions equal.
    """
    kv_rows = actual_seq_lengths_kv.shape[0]
    block_rows = block_table.shape[0]

    assert kv_rows <= query_rows, (
        "NPU DSA graph has more key-length rows than query rows: "
        f"{kv_rows} > {query_rows}"
    )
    assert block_rows <= query_rows, (
        "NPU DSA graph has more block-table rows than query rows: "
        f"{block_rows} > {query_rows}"
    )

    if kv_rows < query_rows:
        actual_seq_lengths_kv = torch.cat(
            (
                actual_seq_lengths_kv,
                actual_seq_lengths_kv.new_zeros(query_rows - kv_rows),
            ),
            dim=0,
        )

    if block_rows < query_rows:
        block_table = torch.cat(
            (
                block_table,
                block_table.new_zeros(
                    (query_rows - block_rows, *block_table.shape[1:])
                ),
            ),
            dim=0,
        )

    return actual_seq_lengths_kv, block_table
