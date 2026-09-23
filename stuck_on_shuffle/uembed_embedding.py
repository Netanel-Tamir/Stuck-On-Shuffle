"""UEmbed (Alibaba-NLP/UEmbed-2B / -9B) counterpart of Qwen3VLEmbedder.

UEmbed is a Qwen3.5-based dense+sparse embedding model (same hybrid stack as
WeMM: every 4th decoder layer is softmax attention, the rest linear-attention).
Its shipped reference embedder (src/models/qwen35_embedding.py in the repo) is
itself a near-copy of Qwen3VLEmbedder — same system-instruction prompting with
the default "Represent the user's input.", same add_generation_prompt=True,
same Qwen3VLProcessor — so almost everything inherits unchanged. The two real
differences, both handled here:

  * The checkpoint is a BARE Qwen3_5Model (keys have no `model.` prefix and
    there is no lm_head), so UEmbedForEmbedding subclasses Qwen3_5Model
    directly instead of wrapping one the way Qwen3VLForEmbedding/
    WeMMForEmbedding do.
  * The tokenizer's post_processor appends the model's sparse special tokens
    (num_eos_tokens=16, from sparse_info.json) to EVERY sequence, and the
    DENSE embedding is read at the position num_eos_tokens BEFORE the last
    non-pad token — the repo's `pooling="last.normal"`. _pooling_last is
    overridden to apply that offset; everything downstream (normalize,
    matryoshka, the trainer's encode helpers) is agnostic to it.

Only dense embeddings are supported: the SPLADE heads in sparse_weights.pt are
frozen artifacts read off the appended positions and are irrelevant to (and
untouched by) LoRA fine-tuning of the backbone.
"""

import json
import os
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

from .qwen3_vl_embedding import (
    FPS,
    MAX_FRAMES,
    MAX_LENGTH,
    MAX_PIXELS,
    MAX_TOTAL_PIXELS,
    MIN_PIXELS,
    Qwen3VLEmbedder,
    Qwen3VLForEmbeddingOutput,
    _resolve_model_path,
    default_device,
)


class UEmbedForEmbedding(Qwen3_5Model):
    """Qwen3_5Model whose forward returns (last_hidden_state, attention_mask).

    Subclass rather than wrapper: UEmbed checkpoints store bare-Qwen3_5Model
    keys, so from_pretrained on this class matches them natively (the repo's
    own code loads them with AutoModel, which resolves to Qwen3_5Model).
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        # transformers < 5.15 reuses the rope_deltas cached by a previous
        # multimodal forward for a text-only one, which shifts its position ids.
        # UEmbed's reference code does not guard against this (WeMM's does);
        # resetting only forces a correct recompute, so it can never diverge
        # from the reference on inputs the reference handles correctly.
        self.rope_deltas = None

        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class UEmbedEmbedder(Qwen3VLEmbedder):
    """Qwen3VLEmbedder with UEmbed's model class and offset dense pooling.

    Prompting is inherited verbatim (system instruction, generation prompt):
    that IS UEmbed's official usage, not an approximation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        matryoshka_dim: int = None,
        device=None,
        **kwargs,
    ):
        device = torch.device(device) if device is not None else default_device()

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction
        self.matryoshka_dim = matryoshka_dim

        local_path = _resolve_model_path(model_name_or_path)

        # Dense pooling reads num_eos_tokens positions before the last non-pad
        # token; without sparse_info.json that offset is unknowable, and a
        # silent 0 would train and evaluate the WRONG position of an otherwise
        # healthy-looking model.
        sparse_info_path = os.path.join(local_path, "sparse_info.json")
        if not os.path.isfile(sparse_info_path):
            raise FileNotFoundError(
                f"UEmbed checkpoint {local_path!r} has no sparse_info.json — "
                "dense pooling needs its num_eos_tokens (the repo ships it; "
                "re-download the full model directory)."
            )
        with open(sparse_info_path) as f:
            self.num_eos_tokens = int(json.load(f).get("num_eos_tokens", 0))

        self.model = UEmbedForEmbedding.from_pretrained(local_path, **kwargs).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            local_path, padding_side="right"
        )
        self.model.eval()

    def _pooling_last(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """The repo's `last.normal`: the position num_eos_tokens before the last
        non-pad one — the EOS ahead of the appended sparse special tokens.

        Overrides the parent's staticmethod as an instance method (every call
        site is instance-bound). clamp(min=0) is unreachable for any sequence
        the tokenizer produced (the post_processor guarantees the appended
        tail) and only guards a malformed row from wrapping to index -1.
        """
        last_indices = (attention_mask.cumsum(dim=1) * attention_mask).argmax(dim=1)
        target_indices = (last_indices - self.num_eos_tokens).clamp(min=0)
        batch_indices = torch.arange(
            hidden_state.shape[0], device=hidden_state.device
        )
        return hidden_state[batch_indices, target_indices]
