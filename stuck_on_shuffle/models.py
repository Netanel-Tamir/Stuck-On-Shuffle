"""Load one of the six binding-finetuned embedding models and embed text or images.

Every released checkpoint is a rank-4 LoRA adapter on top of a public embedding
model, with the LiNeS layer-scaling already baked into the adapter weights
(``lines.json`` in each checkpoint folder records the edit). Loading is therefore
just: load the base model -> attach the adapter with PEFT -> embed. Embeddings are
L2-normalized, so cosine similarity is a plain dot product.

    from stuck_on_shuffle import load_model

    model = load_model("qwen3vl-embedding-8b")
    t = model.embed_texts(["A red mug and a blue kettle.", "A blue mug and a red kettle."])
    i = model.embed_images(["scene.png"])
    print(t @ i.T)

Base models are fetched from the Hugging Face hub on first use (Qwen3-VL-Embedding
from Qwen, WeMM-Embedding from Tencent, UEmbed from Alibaba-NLP). To use local
copies instead, either pass ``base_model=/path/to/dir`` or set the environment
variable ``STUCK_ON_SHUFFLE_MODELS_DIR`` to one or more directories (separated like
PATH) that contain either plain model folders named after the hub repos
(``Qwen3-VL-Embedding-8B``, ``WeMM-Embedding-2B``, ...) or a Hugging Face cache
(``models--Qwen--Qwen3-VL-Embedding-8B/snapshots/<rev>``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(REPO_ROOT, "checkpoints")

# Text inputs were truncated to 512 tokens during finetuning; images are encoded
# with the models' native 8192 budget because a single image expands to many tokens.
TEXT_MAX_LENGTH = 512
IMAGE_MAX_LENGTH = 8192


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str          # "qwen3vl" | "wemm" | "uembed"
    base_model: str      # Hugging Face hub id of the frozen base
    adapter_dir: str     # folder with adapter_config.json / adapter_model.safetensors
    params: str          # rough size label, for display

    @property
    def embedder_class(self):
        if self.family == "qwen3vl":
            from .qwen3_vl_embedding import Qwen3VLEmbedder
            return Qwen3VLEmbedder
        if self.family == "wemm":
            from .wemm_embedding import WeMMEmbedder
            return WeMMEmbedder
        if self.family == "uembed":
            from .uembed_embedding import UEmbedEmbedder
            return UEmbedEmbedder
        raise ValueError(f"unknown model family {self.family!r}")


MODELS = {
    spec.name: spec
    for spec in [
        ModelSpec("qwen3vl-embedding-8b", "qwen3vl", "Qwen/Qwen3-VL-Embedding-8B",
                  os.path.join(CHECKPOINTS_DIR, "qwen3vl-embedding-8b"), "8B"),
        ModelSpec("qwen3vl-embedding-2b", "qwen3vl", "Qwen/Qwen3-VL-Embedding-2B",
                  os.path.join(CHECKPOINTS_DIR, "qwen3vl-embedding-2b"), "2B"),
        ModelSpec("wemm-embedding-9b", "wemm", "tencent/WeMM-Embedding-9B",
                  os.path.join(CHECKPOINTS_DIR, "wemm-embedding-9b"), "9B"),
        ModelSpec("wemm-embedding-2b", "wemm", "tencent/WeMM-Embedding-2B",
                  os.path.join(CHECKPOINTS_DIR, "wemm-embedding-2b"), "2B"),
        ModelSpec("uembed-9b", "uembed", "Alibaba-NLP/UEmbed-9B",
                  os.path.join(CHECKPOINTS_DIR, "uembed-9b"), "9B"),
        ModelSpec("uembed-2b", "uembed", "Alibaba-NLP/UEmbed-2B",
                  os.path.join(CHECKPOINTS_DIR, "uembed-2b"), "2B"),
    ]
}


def list_models() -> List[str]:
    return list(MODELS)


MODELS_DIR_ENV = "STUCK_ON_SHUFFLE_MODELS_DIR"


def _has_weights(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "config.json")) and any(
        f.endswith(".safetensors") for f in os.listdir(d))


def _find_local_base(hub_id: str) -> Optional[str]:
    """Look for a complete local copy of ``hub_id`` in the directories listed in
    ``STUCK_ON_SHUFFLE_MODELS_DIR`` (``os.pathsep``-separated, like PATH).

    Two layouts are recognised in each directory: a plain folder named after the
    model (``<dir>/Qwen3-VL-Embedding-8B``) and the Hugging Face cache layout
    (``<dir>/models--Qwen--Qwen3-VL-Embedding-8B/snapshots/<revision>``).
    """
    roots = [r for r in os.environ.get(MODELS_DIR_ENV, "").split(os.pathsep) if r]
    org, name = hub_id.split("/")
    for root in roots:
        plain = os.path.join(root, name)
        if os.path.isdir(plain) and _has_weights(plain):
            return plain
        snaps = os.path.join(root, f"models--{org}--{name}", "snapshots")
        if os.path.isdir(snaps):
            for rev in sorted(os.listdir(snaps)):
                cand = os.path.join(snaps, rev)
                if _has_weights(cand):
                    return cand
    return None


def _resolve_base(spec: ModelSpec, base_model: Optional[str]) -> str:
    """Local dir if one is given / configured, otherwise the hub id (downloaded on demand)."""
    if base_model:
        return base_model
    return _find_local_base(spec.base_model) or spec.base_model


def _check_adapter(spec: ModelSpec) -> dict:
    cfg_path = os.path.join(spec.adapter_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"no adapter at {spec.adapter_dir} - the checkpoints/ folder should ship with "
            f"the repository; did you clone it with the *.safetensors files?")
    if not os.path.isfile(os.path.join(spec.adapter_dir, "lines.json")):
        raise RuntimeError(
            f"{spec.adapter_dir} has no lines.json: this loader expects the released "
            f"adapters, whose LiNeS scaling is already baked in.")
    with open(cfg_path) as f:
        return json.load(f)


class BindingEmbedder:
    """A base embedding model with one of our LoRA adapters attached.

    ``embed_texts`` / ``embed_images`` return float32, L2-normalized tensors on CPU,
    one row per input, matching exactly how the models were evaluated.
    """

    def __init__(self, spec: ModelSpec, embedder, device: torch.device):
        self.spec = spec
        self.embedder = embedder
        self.device = device

    # -- properties -------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def model(self):
        return self.embedder.model

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.get_text_config().hidden_size)

    # -- encoding ---------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, items: Sequence, kind: str, instruction: Optional[str],
                batch_size: int) -> torch.Tensor:
        emb = self.embedder
        saved = emb.max_length
        emb.max_length = IMAGE_MAX_LENGTH if kind == "image" else TEXT_MAX_LENGTH
        out = []
        try:
            for i in range(0, len(items), batch_size):
                chunk = list(items[i:i + batch_size])
                convs = [emb.format_model_input(**{kind: x}, instruction=instruction)
                         for x in chunk]
                inputs = emb._preprocess_inputs(convs)
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                o = emb.model(**inputs)
                pooled = emb._pooling_last(o.last_hidden_state, o.attention_mask)
                # Normalise in float32: bf16 has ~3 significant digits, so normalising in
                # bf16 leaves norms at 1 +/- 0.004 and self-similarities visibly off 1.
                out.append(F.normalize(pooled.float(), p=2, dim=-1).cpu())
        finally:
            emb.max_length = saved
        return torch.cat(out) if out else torch.empty(0, self.embedding_dim)

    def embed_texts(self, texts: Union[str, Iterable[str]], instruction: Optional[str] = None,
                    batch_size: int = 32) -> torch.Tensor:
        """L2-normalized embeddings of one or more strings, shape (n, d).

        ``instruction`` overrides the model's default task instruction. Leave it as
        None to reproduce the paper numbers (that is how the models were trained and
        evaluated).
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(list(texts), "text", instruction, batch_size)

    def embed_images(self, images: Union[str, Image.Image, Iterable[Union[str, Image.Image]]],
                     instruction: Optional[str] = None, batch_size: int = 8) -> torch.Tensor:
        """L2-normalized embeddings of images given as paths, URLs or PIL images, shape (n, d)."""
        if isinstance(images, (str, Image.Image)):
            images = [images]
        images = [im.convert("RGB") if isinstance(im, Image.Image) else im for im in images]
        return self._encode(list(images), "image", instruction, batch_size)

    @staticmethod
    def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix between two sets of normalized embeddings."""
        return a @ b.T

    # -- base-model comparison ----------------------------------------------------
    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "disable_adapter")

    def without_adapter(self):
        """Context manager that bypasses the LoRA adapter, i.e. runs the frozen base model.

            with model.without_adapter():
                base_emb = model.embed_texts(texts)

        Handy for base-vs-finetuned comparisons without loading the weights twice.
        Not available when the model was loaded with ``merge_lora=True``.
        """
        if not self.has_adapter:
            raise RuntimeError("adapter was merged into the base weights; reload with merge_lora=False")
        return self.model.disable_adapter()


def load_model(name: str, base_model: Optional[str] = None, dtype: Optional[torch.dtype] = None,
               device: Optional[Union[str, torch.device]] = None, merge_lora: bool = False,
               verbose: bool = True) -> BindingEmbedder:
    """Load a released model by name (see ``list_models()``).

    Args:
        name: one of the keys of ``MODELS``, e.g. ``"qwen3vl-embedding-8b"``.
        base_model: optional local directory of the base model, instead of downloading
            it from the hub (or set ``STUCK_ON_SHUFFLE_MODELS_DIR``).
        dtype: weights dtype. Defaults to bfloat16 on CUDA and Apple MPS (what the models
            were trained and scored in) and float32 on CPU. Pass ``torch.float16`` on an
            MPS build that lacks bfloat16 support.
        device: ``"cuda"``, ``"mps"``, ``"cpu"`` or a ``torch.device``. Defaults to the
            best available: CUDA, then MPS, then CPU.
        merge_lora: fold the adapter into the base weights (slightly faster inference,
            identical outputs up to bf16 rounding).
    """
    from peft import PeftModel

    from .qwen3_vl_embedding import default_device, default_dtype

    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {list_models()}")
    spec = MODELS[name]
    adapter_cfg = _check_adapter(spec)
    base = _resolve_base(spec, base_model)
    dev = torch.device(device) if device is not None else default_device()
    if dtype is None:
        dtype = default_dtype(dev)

    if verbose:
        print(f"[load] {spec.name}: base {base} ({spec.family}, {dtype}, {dev})")
    embedder = spec.embedder_class(model_name_or_path=base, max_length=TEXT_MAX_LENGTH,
                                   dtype=dtype, device=dev)

    if verbose:
        print(f"[load] adapter {os.path.relpath(spec.adapter_dir, REPO_ROOT)} "
              f"(LoRA r={adapter_cfg.get('r')}, targets={sorted(adapter_cfg.get('target_modules', []))}, "
              f"LiNeS baked)")
    embedder.model = PeftModel.from_pretrained(embedder.model, spec.adapter_dir, is_trainable=False)
    if merge_lora:
        embedder.model = embedder.model.merge_and_unload()
    embedder.model.eval()

    embedder.model.to(dev)
    dev = next(embedder.model.parameters()).device
    if verbose:
        print(f"[load] ready on {dev}")
    return BindingEmbedder(spec, embedder, dev)
