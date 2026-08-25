# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

pytestmark = pytest.mark.cpu_test


class _StagedSharedExperts:
    def __init__(self) -> None:
        self.staged = False
        self.discard_calls = 0

    def __call__(self, shared_experts_input, order) -> None:
        self.staged = True

    def discard_output(self) -> None:
        self.staged = False
        self.discard_calls += 1


class _FailingRoutedExperts:
    quant_method = SimpleNamespace(is_monolithic=True)

    def forward_monolithic(self, **kwargs):
        raise MemoryError("injected routed-MoE workspace OOM")


def test_routed_expert_failure_discards_staged_shared_output():
    """A routed-expert failure must not poison the next invocation."""
    runner = object.__new__(MoERunner)
    runner._shared_experts = _StagedSharedExperts()
    runner.routed_experts = _FailingRoutedExperts()

    with pytest.raises(MemoryError, match="injected routed-MoE workspace OOM"):
        runner._apply_quant_method(
            hidden_states=torch.empty(1),
            router_logits=torch.empty(1),
            shared_experts_input=torch.empty(1),
        )
