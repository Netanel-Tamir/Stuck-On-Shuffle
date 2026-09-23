#!/usr/bin/env python3
"""Score a released model on the SABER multiple-choice benchmark.

Reports accuracy in the four retrieval settings (T2T, T2I, I2T, I2I) per object
count and averaged over the nine object counts, exactly as in the paper. The full
run needs the 9,000 SABER images (about 11 GB, fetched once into the HF cache).

    python scripts/eval_saber_mcq.py --model qwen3vl-embedding-8b
    python scripts/eval_saber_mcq.py --model wemm-embedding-2b --n-objects 2 3 --limit 20
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from stuck_on_shuffle import (  # noqa: E402
    ANSWERS, OBJECT_COUNTS, SETTINGS, download_saber_images, list_models, load_model, load_saber,
)


def score_frame(model, df: pd.DataFrame, root: str, text_batch: int, image_batch: int) -> dict:
    """Accuracy per setting for one object count. Every item is encoded once."""
    def img_paths(col):
        return [os.path.join(root, p) for p in df[col]]

    q_txt = model.embed_texts(df["query_txt"].tolist(), batch_size=text_batch)
    q_img = model.embed_images(img_paths("query_img"), batch_size=image_batch)
    a_txt = torch.stack([model.embed_texts(df[f"{a}_txt"].tolist(), batch_size=text_batch)
                         for a in ANSWERS], dim=1)                       # (n, 4, d)
    a_img = torch.stack([model.embed_images(img_paths(f"{a}_img"), batch_size=image_batch)
                         for a in ANSWERS], dim=1)                       # (n, 4, d)
    sims = {
        "T2T": torch.einsum("nd,nkd->nk", q_txt, a_txt),
        "T2I": torch.einsum("nd,nkd->nk", q_txt, a_img),
        "I2T": torch.einsum("nd,nkd->nk", q_img, a_txt),
        "I2I": torch.einsum("nd,nkd->nk", q_img, a_img),
    }
    # ans_1 is the correct candidate. A tie at the top is broken by an unbiased draw.
    rng = np.random.default_rng(0)
    out = {}
    for s, m in sims.items():
        m = m.numpy()
        tied = m == m.max(axis=1, keepdims=True)
        picked = np.where(tied, rng.random(m.shape), -1.0).argmax(axis=1)
        out[s] = float((picked == 0).mean())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list_models(), default="qwen3vl-embedding-8b")
    p.add_argument("--base-model", default=None, help="local directory of the base model")
    p.add_argument("--n-objects", type=int, nargs="+", default=OBJECT_COUNTS,
                   help="which object counts to score (default: all nine)")
    p.add_argument("--limit", type=int, default=0, help="score only the first N rows per file")
    p.add_argument("--text-batch", type=int, default=64)
    p.add_argument("--image-batch", type=int, default=16)
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache dir for the dataset")
    p.add_argument("--out", default=None, help="optional CSV to write the per-count table to")
    args = p.parse_args()

    print(f"[data] fetching SABER csvs + images for object counts {args.n_objects}")
    root = download_saber_images(args.n_objects, cache_dir=args.cache_dir)
    df_all = load_saber(args.n_objects, cache_dir=args.cache_dir)

    model = load_model(args.model, base_model=args.base_model)

    rows = []
    for n in tqdm(args.n_objects, desc="SABER"):
        df = df_all[df_all.n_objects == n].reset_index(drop=True)
        if args.limit:
            df = df.head(args.limit)
        acc = score_frame(model, df, root, args.text_batch, args.image_batch)
        rows.append({"n_objects": n, "rows": len(df), **acc})
    table = pd.DataFrame(rows)
    mean = table[SETTINGS].mean()
    mean["overall"] = mean.mean()

    pd.set_option("display.precision", 4)
    print(f"\n{args.model} - SABER MCQ accuracy (chance = 0.25)\n")
    print(table.to_string(index=False))
    print("\nmean over object counts:")
    print(mean.to_frame().T.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
