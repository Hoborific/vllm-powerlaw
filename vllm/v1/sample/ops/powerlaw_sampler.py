# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import BatchUpdate

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Transformation constants (match reference)
DISTRIBUTION_WIDTH = 0.3
PEAK_LOGIT_VALUE = 5.0
INV_WIDTH = 1.0 / DISTRIBUTION_WIDTH


class PowerLawSampler:
    """Stateful power-law sampler.

    Note: This is intentionally *not* a LogitsProcessor; it performs actual token
    selection and updates its adaptive state based on the chosen token.
    """

    def __init__(self, vllm_config: "VllmConfig", device: torch.device):
        self.device = device

        # Defaults (can be overridden per request via SamplingParams.extra_args)
        self.default_target = float(os.environ.get("VLLM_POWERLAW_TARGET", -1.0))
        self.default_decay = float(os.environ.get("VLLM_POWERLAW_DECAY", 0.90))

        # req_index -> (target, decay, weighted_sum, total_weight)
        self.request_states: dict[int, tuple[float, float, float, float]] = {}

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args
        if extra_args is None:
            return

        if "power_law_target" in extra_args:
            target = extra_args["power_law_target"]
            if not isinstance(target, (int, float)):
                raise ValueError(
                    f"power_law_target must be a number, got {type(target)}"
                )
            if target >= 0.0 and not (0.0 <= float(target) <= 1.0):
                raise ValueError(
                    "power_law_target must be within [0.0, 1.0] when enabled; "
                    "negative values disable"
                )

        if "power_law_decay" in extra_args:
            decay = extra_args["power_law_decay"]
            if not isinstance(decay, (int, float)):
                raise ValueError(f"power_law_decay must be a number, got {type(decay)}")
            if not (0.0 <= float(decay) <= 1.0):
                raise ValueError("power_law_decay must be in range [0.0, 1.0]")

    def _get_request_params(self, sampling_params: SamplingParams) -> tuple[float, float]:
        extra_args = sampling_params.extra_args

        if extra_args and "power_law_target" in extra_args:
            target = float(extra_args["power_law_target"])
        else:
            target = self.default_target

        if extra_args and "power_law_decay" in extra_args:
            decay = float(extra_args["power_law_decay"])
        else:
            decay = self.default_decay

        return target, decay

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        # Remove first
        for index in batch_update.removed:
            self.request_states.pop(index, None)

        # Add / replace
        for index, params, _prompt_tok_ids, _output_tok_ids in batch_update.added:
            target, decay = self._get_request_params(params)

            if target < 0.0:
                # Disabled for this request
                self.request_states.pop(index, None)
                continue

            target = max(0.0, min(1.0, target))
            decay = max(0.0, min(0.99, decay))
            self.request_states[index] = (target, decay, 0.0, 0.0)

        # Move / swap
        for adx, bdx, _ in batch_update.moved:
            if adx in self.request_states:
                self.request_states[bdx] = self.request_states.pop(adx)

    def enabled(self) -> bool:
        return bool(self.request_states)

    def sample(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        """Sample token ids from logits.

        Args:
            logits: [B, V] float32 tensor. May already contain -inf masks from
                top-k/top-p filtering (the sampler preserves those masks).
            generators: map from *batch row index* -> torch.Generator.

        Returns:
            token_ids: [B] int64 tensor.
        """
        # Softmax once to get the baseline distribution (original probs).
        # Note: We must keep original probabilities for the state update step.
        original_probs = logits.softmax(dim=-1, dtype=torch.float32)

        B = int(logits.shape[0])
        sampled = torch.empty((B, ), device=logits.device, dtype=torch.long)

        if not self.request_states:
            probs = original_probs.clone()
            return _random_sample(probs, generators)

        enabled_indices = set(self.request_states.keys())

        # Sample baseline ONLY for indices that are not power-law enabled.
        # This avoids consuming RNG twice (baseline + power-law) for enabled rows.
        disabled_indices = [i for i in range(B) if i not in enabled_indices]
        if disabled_indices:
            probs_disabled = original_probs[disabled_indices].clone()
            gens_disabled: dict[int, torch.Generator] = {}
            for row_pos, batch_idx in enumerate(disabled_indices):
                # `generators` is keyed by the original batch row index.
                # `_random_sample` expects keys in the local row indices of
                # `probs_disabled`.
                gen = generators.get(batch_idx)
                if gen is not None:
                    gens_disabled[row_pos] = gen
            sampled_disabled = _random_sample(probs_disabled, gens_disabled)
            sampled[disabled_indices] = sampled_disabled

        # Apply power-law sampling for the enabled requests.
        # We do per-request loop to keep state updates straightforward.
        for req_idx, (target, decay, weighted_sum, total_weight) in list(
            self.request_states.items()
        ):
            # Compute adapted target probability.
            if total_weight == 0.0:
                computed_target = target
            else:
                computed_target = max(
                    0.0, min(1.0, 2.0 * target - (weighted_sum / total_weight))
                )

            req_probs = original_probs[req_idx]

            # Overwrite logits using transformed scores.
            dist = (req_probs - computed_target) * INV_WIDTH
            denom = torch.clamp(1.0 + dist * dist, min=1e-6)
            new_logits = PEAK_LOGIT_VALUE / denom

            # Preserve masked tokens.
            masked = torch.isinf(logits[req_idx]) & (logits[req_idx] < 0)
            if masked.any():
                new_logits = new_logits.masked_fill(masked, float("-inf"))

            # Sample from transformed distribution.
            # Use the same sampling mechanism as vLLM's sampler (exponential race)
            # to avoid multinomial sync and to keep RNG usage controlled.
            mod_probs = new_logits.softmax(dim=-1, dtype=torch.float32).unsqueeze(0)
            gen = generators.get(req_idx)
            if gen is None:
                idx = int(_random_sample(mod_probs, {})[0].item())
            else:
                idx = int(_random_sample(mod_probs, {0: gen})[0].item())
            sampled[req_idx] = idx

            # Update history using the ORIGINAL probability.
            new_weighted_sum = float(req_probs[idx].item()) + decay * weighted_sum
            new_total_weight = 1.0 + decay * total_weight
            self.request_states[req_idx] = (target, decay, new_weighted_sum, new_total_weight)

        return sampled


def _random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Random sample from probs (fallback path).

    Mirrors vLLM's `random_sample` behavior but keeps it local so we can safely
    sample with per-request generators.

    Note: This fallback uses exponential race + argmax, which matches vLLM's
    existing sampler and avoids torch.multinomial sync.
    """
    q = torch.empty_like(probs)
    q.exponential_()

    for i, gen in generators.items():
        q[i].exponential_(generator=gen)

    return probs.div_(q).argmax(dim=-1).view(-1)
