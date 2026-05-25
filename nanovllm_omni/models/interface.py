from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from nanovllm_omni.outputs import OmniOutput
from nanovllm_omni.worker.utils import RunnerState


@runtime_checkable
class SupportsStepExecution(Protocol):
    """Minimal state-driven step execution contract for nano diffusion pipelines."""

    supports_step_execution: bool

    def prepare_encode(self, state: RunnerState, **kwargs: Any) -> RunnerState:
        """Prepare request-level inputs and return initialized state."""

    def denoise_step(self, state: RunnerState, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward."""

    def step_scheduler(self, state: RunnerState, noise_pred: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Advance one scheduler step."""

    def post_decode(self, state: RunnerState, **kwargs: Any) -> OmniOutput:
        """Decode final latents into output media."""


def supports_step_execution(pipeline: object) -> bool:
    return isinstance(pipeline, SupportsStepExecution)
