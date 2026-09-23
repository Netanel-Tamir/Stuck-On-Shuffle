"""Loaders for the two Hugging Face datasets released with this repository.

* ``NetanelTamir/Stuck-On-Shuffle-SABER`` - the SABER benchmark: nine CSVs
  (``2_objects.csv`` ... ``10_objects.csv``, 200 rows each) and 9,000 PNGs under
  ``images/<n>_objects/``. Each row is one multiple-choice question: a query given
  as an image and a caption, and four candidates given as images and captions.
  ``ans_1`` is the correct candidate; ``ans_2``..``ans_4`` reuse the same objects and
  attributes with the bindings shuffled.
* ``NetanelTamir/Stuck-On-Shuffle-Finetuning-Dataset`` - the 100,000 text triplets
  used for finetuning (``finetuning_dataset.csv``): a short two/three-object query,
  a positive caption, and two hard negatives (attributes swapped / objects swapped).

Everything is fetched through ``huggingface_hub`` and cached in the standard HF
cache, so a second call is free. Images are downloaded lazily, one file at a time,
so browsing a few examples does not require the full 11 GB image set.

Offline use: if you already hold a copy of a dataset (for example from an earlier
``snapshot_download``), point the loaders at it and no network access is needed:

    export STUCK_ON_SHUFFLE_SABER_DIR=/path/to/Stuck-On-Shuffle-SABER   # has *.csv + images/
    export STUCK_ON_SHUFFLE_FINETUNING_CSV=/path/to/finetuning_dataset.csv
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

SABER_REPO = "NetanelTamir/Stuck-On-Shuffle-SABER"
FINETUNING_REPO = "NetanelTamir/Stuck-On-Shuffle-Finetuning-Dataset"

OBJECT_COUNTS = list(range(2, 11))
ANSWERS = ["ans_1", "ans_2", "ans_3", "ans_4"]
SETTINGS = ["T2T", "T2I", "I2T", "I2I"]
IMAGE_COLUMNS = ["query_img"] + [f"{a}_img" for a in ANSWERS]

_IMAGE_PATH_RE = re.compile(r"^.*?(images/\d+_objects/[^/]+\.png)$")

SABER_DIR_ENV = "STUCK_ON_SHUFFLE_SABER_DIR"
FINETUNING_CSV_ENV = "STUCK_ON_SHUFFLE_FINETUNING_CSV"


def _to_repo_path(p: str) -> str:
    """The CSVs store the paths the images had at generation time; keep only the
    ``images/<n>_objects/<file>.png`` tail, which is where they live in the repo."""
    m = _IMAGE_PATH_RE.match(str(p).replace("\\", "/"))
    return m.group(1) if m else str(p)


def _local_saber_dir() -> Optional[str]:
    """A local SABER copy named by ``STUCK_ON_SHUFFLE_SABER_DIR``, if it looks valid."""
    root = os.environ.get(SABER_DIR_ENV)
    if root and os.path.isfile(os.path.join(root, "2_objects.csv")):
        return root
    return None


def _saber_file(rel_path: str, cache_dir: Optional[str], token: Optional[str]) -> str:
    """Local path of one SABER file: from the local copy if configured, else the hub."""
    root = _local_saber_dir()
    if root is not None:
        return os.path.join(root, rel_path)
    return hf_hub_download(SABER_REPO, rel_path, repo_type="dataset", cache_dir=cache_dir,
                           token=token)


def load_saber(n_objects: Optional[Iterable[int]] = None, cache_dir: Optional[str] = None,
               token: Optional[str] = None) -> pd.DataFrame:
    """SABER as one DataFrame with an added ``n_objects`` column.

    Image columns hold repo-relative paths (``images/2_objects/0_query.png``); use
    ``saber_image`` to fetch one, or ``download_saber_images`` to get them all.
    """
    counts = list(OBJECT_COUNTS if n_objects is None else n_objects)
    frames = []
    for n in counts:
        path = _saber_file(f"{n}_objects.csv", cache_dir, token)
        df = pd.read_csv(path)
        df.insert(1, "n_objects", n)
        for c in IMAGE_COLUMNS:
            df[c] = df[c].map(_to_repo_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def saber_image(repo_path: str, cache_dir: Optional[str] = None,
                token: Optional[str] = None) -> Image.Image:
    """Download (once) and open one SABER image given its repo-relative path."""
    return Image.open(saber_image_path(repo_path, cache_dir, token)).convert("RGB")


def saber_image_path(repo_path: str, cache_dir: Optional[str] = None,
                     token: Optional[str] = None) -> str:
    """Like ``saber_image`` but returns the local file path (handy for batch encoding)."""
    return _saber_file(_to_repo_path(repo_path), cache_dir, token)


def download_saber_images(n_objects: Optional[Iterable[int]] = None, cache_dir: Optional[str] = None,
                          token: Optional[str] = None) -> str:
    """Fetch the SABER images in bulk (about 11 GB for all nine object counts).

    Returns the local snapshot root; ``os.path.join(root, row.query_img)`` is then a
    valid file path for every image column of ``load_saber``.
    """
    root = _local_saber_dir()
    if root is not None:
        return root
    counts = list(OBJECT_COUNTS if n_objects is None else n_objects)
    patterns = [f"images/{n}_objects/*" for n in counts] + ["*.csv"]
    return snapshot_download(SABER_REPO, repo_type="dataset", allow_patterns=patterns,
                             cache_dir=cache_dir, token=token)


def load_finetuning_dataset(cache_dir: Optional[str] = None,
                            token: Optional[str] = None) -> pd.DataFrame:
    """The 100k text triplets: columns id, triplet_type, query, positive,
    swapped_attributes, swapped_objects."""
    path = os.environ.get(FINETUNING_CSV_ENV)
    if not (path and os.path.isfile(path)):
        path = hf_hub_download(FINETUNING_REPO, "finetuning_dataset.csv", repo_type="dataset",
                               cache_dir=cache_dir, token=token)
    return pd.read_csv(path)


def mcq_accuracy(sims, correct_index: int = 0) -> float:
    """Fraction of rows whose highest-scoring candidate is the correct one.

    ``sims`` is an (n_rows, 4) array or tensor of query-candidate similarities.
    """
    import numpy as np
    sims = np.asarray(sims, dtype="float32")
    return float((sims.argmax(axis=1) == correct_index).mean())


# ── Large-gallery retrieval ───────────────────────────────────────────────────

_PAIR_SPLIT = re.compile(r",? and an? ")
_LEAD_ARTICLE = re.compile(r"^An? ")


def parse_bindings(caption: str) -> frozenset:
    """The attribute-object pairs of a SABER caption, as a set.

    ``"A sandy snowboard and a pink cherry."`` -> ``{("sandy", "snowboard"), ("pink", "cherry")}``.
    The attribute is always a single (possibly hyphenated) token; the rest of the phrase is
    the object.
    """
    parts = _PAIR_SPLIT.split(caption.strip().rstrip("."))
    parts[0] = _LEAD_ARTICLE.sub("", parts[0])
    pairs = []
    for p in parts:
        words = p.split()
        pairs.append((words[0], " ".join(words[1:])))
    return frozenset(pairs)


def _subset_label(row_id: int) -> str:
    return "independent" if int(row_id) < 100 else "nested"


def saber_queries(saber: pd.DataFrame) -> pd.DataFrame:
    """One row per question: n_objects, id, subset, img, txt, bindings."""
    out = saber[["n_objects", "id", "query_img", "query_txt"]].rename(
        columns={"query_img": "img", "query_txt": "txt"})
    out.insert(2, "subset", saber["id"].map(_subset_label))
    out["bindings"] = out["txt"].map(parse_bindings)
    return out.reset_index(drop=True)


def saber_gallery(saber: pd.DataFrame) -> pd.DataFrame:
    """All candidates of all questions pooled into one gallery (7,200 rows for the full
    benchmark): n_objects, id, role (ans_1..ans_4), img, txt, bindings.

    Queries are not part of the gallery, so there is nothing to self-exclude.
    """
    frames = []
    for a in ANSWERS:
        f = saber[["n_objects", "id", f"{a}_img", f"{a}_txt"]].rename(
            columns={f"{a}_img": "img", f"{a}_txt": "txt"})
        f.insert(2, "role", a)
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["bindings"] = out["txt"].map(parse_bindings)
    return out


def retrieval_positives(queries: pd.DataFrame, gallery: pd.DataFrame):
    """Boolean (n_queries, n_gallery) matrix under the subset-containment rule: a gallery
    item is a positive of a query if it contains every attribute-object binding of the
    query, possibly alongside additional pairs.

    Independent queries (id 0-99) get exactly one positive, their own ``ans_1``. Nested
    queries (id 100-199) additionally match the ``ans_1`` items of the same chain at
    larger object counts.
    """
    import numpy as np
    q = list(queries["bindings"])
    g = list(gallery["bindings"])
    pos = np.zeros((len(q), len(g)), dtype=bool)
    for i, qb in enumerate(q):
        for j, gb in enumerate(g):
            pos[i, j] = qb <= gb
    return pos


def mean_average_precision(sims, positives) -> float:
    """mAP over queries: for each query rank the gallery by similarity and average the
    precision at every positive's rank. Queries with no positive are skipped."""
    import numpy as np
    sims = np.asarray(sims, dtype="float32")
    positives = np.asarray(positives, dtype=bool)
    aps = []
    for s, p in zip(sims, positives):
        if not p.any():
            continue
        order = np.argsort(-s, kind="stable")
        hits = p[order]
        ranks = np.flatnonzero(hits) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks) + 1) / ranks)))
    return float(np.mean(aps)) if aps else float("nan")
