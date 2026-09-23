"""WeMM-Embedding (tencent/WeMM-Embedding-2B / -9B) counterpart of Qwen3VLEmbedder.

WeMM is a Qwen3.5-based VLM fine-tuned for embeddings (hybrid stack: every 4th
decoder layer is standard softmax attention, the rest are linear-attention
blocks). Everything downstream-facing mirrors Qwen3VLEmbedder so training and
evaluation code can treat the two interchangeably. What makes that possible:

  * WeMM's processor IS Qwen3VLProcessor (its processor_config.json says so),
    and that processor's post-processing appends the model's `<embedding>`
    sentinel to every rendered conversation ("...<|im_end|><embedding>").
    Pooling the last non-pad position — Qwen3VLEmbedder._pooling_last — therefore
    reads exactly the position the official WeMMEmbedding.embedding() pools
    before L2-normalizing.
  * Per the model card, inputs are bare user turns rendered with
    add_generation_prompt=False and NO system instruction (task hints such as
    "Represent this image." ride along as ordinary text). default_instruction
    is therefore None here; an explicit `instruction=` is still honored as a
    system turn, which WeMM's chat template renders fine.
  * The image geometry matches (patch 16, merge 2), so the parent's pixel
    budgets and qwen_vl_utils preprocessing carry over unchanged.

WeMMForEmbedding drops WeMM's lm_head (vocab 248,078 x hidden — half a GB of
pure generation machinery at 2B) the same way Qwen3VLForEmbedding drops
Qwen3VL's: the wrapper holds a bare Qwen3_5Model under `self.model`, which is
exactly the prefix the checkpoint's `model.*` keys expect.
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5Model,
    Qwen3_5PreTrainedModel,
)
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


class WeMMForEmbedding(Qwen3_5PreTrainedModel):
    """Qwen3_5Model wrapped to return (last_hidden_state, attention_mask).

    Structural twin of Qwen3VLForEmbedding: same output class, same `self.model`
    prefix, no LM head. PEFT, gradient checkpointing and set_attn_implementation
    all reach it through the same PreTrainedModel surface.
    """

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        # transformers < 5.15 reuses the rope_deltas cached by a previous
        # multimodal forward for a text-only one, which shifts its position ids;
        # the official WeMMEmbedding.embedding() resets it the same way.
        self.model.rope_deltas = None

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class WeMMEmbedder(Qwen3VLEmbedder):
    """Qwen3VLEmbedder with WeMM's model class and the model card's prompting.

    Inherits format_model_input's content handling, _preprocess_inputs (via
    ADD_GENERATION_PROMPT=False), _pooling_last, matryoshka slicing and process()
    unchanged — see the module docstring for why each carries over.
    """

    # The card renders conversations with add_generation_prompt=False; the
    # template's generation tail ("<|im_start|>assistant\n<think>...") is chat
    # machinery the embedding path never wants.
    ADD_GENERATION_PROMPT = False

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: Optional[str] = None,  # card style: no system turn
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
        # A concrete class is loaded, so the repo's auto_map remote code
        # (modeling_wemm_embedding.WeMMEmbedding) is never imported or executed.
        self.model = WeMMForEmbedding.from_pretrained(local_path, **kwargs).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            local_path, padding_side="right"
        )
        self.model.eval()

    def format_model_input(self, *args, **kwargs):
        conversation = super().format_model_input(*args, **kwargs)
        # The card embeds bare user content — no system turn unless an
        # instruction was actually given (explicitly or as a default).
        instruction = kwargs.get("instruction") or (args[3] if len(args) > 3 else None)
        if not (instruction or self.default_instruction):
            conversation = [t for t in conversation if t["role"] != "system"]
        return conversation
