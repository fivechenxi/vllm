# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )


@pytest.mark.parametrize(
    "num_reqs,num_tokens,expected_prefills",
    [
        (3, 12, [False, False, False]),
        (3, 14, [False, True, True]),
        (1, 4096, [True]),
    ],
)
def test_make_dummy_classifies_synced_prefill_shapes(
    num_reqs: int, num_tokens: int, expected_prefills: list[bool]
):
    """DP dummy batches must retain prefill semantics after shape sync.

    With speculative decode query length 4, a dummy rank synchronized to a
    prefill-heavy peer can receive longer queries. Treating those as decodes
    violates hybrid recurrent-state activation limits (for example K3
    RecoverSSM) even though the batch contains no real local requests.
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(
        num_reqs,
        num_tokens,
        buffers,
        decode_query_len=4,
    )

    assert batch.is_prefilling_np.tolist() == expected_prefills
    assert batch.has_prefill == any(expected_prefills)
