#!/usr/bin/env python3
"""Score a released model on SABER large-gallery retrieval.

The four candidates of every question are pooled into one gallery of 7,200 items and each
of the 1,800 queries ranks the whole gallery by similarity. A gallery item is a positive
when it contains every attribute-object binding of the query (subset containment), so
independent queries (id 0-99) have exactly one positive and nested queries (id 100-199)
also match their chain's larger scenes. Reports mean average precision per modality
pairing for the independent subset, the nested subset and all queries.

    python scripts/eval_saber_retrieval.py --model qwen3vl-embedding-8b
    python scripts/eval_saber_retrieval.py --model wemm-embedding-2b --settings T2T   # no images needed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from stuck_on_shuffle import (  # noqa: E402
    SETTINGS, download_saber_images, list_models, load_model, load_saber, mean_average_precision,
    retrieval_positives, saber_gallery, saber_queries,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list_models(), default="qwen3vl-embedding-8b")
    p.add_argument("--base-model", default=None, help="local directory of the base model")
    p.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS,
                   help="modality pairings to score (default: all four)")
    p.add_argument("--text-batch", type=int, default=64)
    p.add_argument("--image-batch", type=int, default=16)
    p.add_argument("--cache-dir", default=None, help="Hugging Face cache dir for the dataset")
    p.add_argument("--out", default=None, help="optional CSV to write the table to")
    args = p.parse_args()

    saber = load_saber(cache_dir=args.cache_dir)
    queries = saber_queries(saber)
    gallery = saber_gallery(saber)
    positives = retrieval_positives(queries, gallery)
    print(f"[data] {len(queries)} queries, {len(gallery)} gallery items, "
          f"{int(positives.sum())} positive pairs")

    need_images = any("I" in s for s in args.settings)
    root = download_saber_images(cache_dir=args.cache_dir) if need_images else None

    model = load_model(args.model, base_model=args.base_model)

    # Every item is embedded once per modality and reused across the pairings.
    emb = {"q": {}, "g": {}}
    if any(s[0] == "T" for s in args.settings):
        emb["q"]["T"] = model.embed_texts(queries.txt.tolist(), batch_size=args.text_batch)
    if any(s[-1] == "T" for s in args.settings):
        emb["g"]["T"] = model.embed_texts(gallery.txt.tolist(), batch_size=args.text_batch)
    if any(s[0] == "I" for s in args.settings):
        emb["q"]["I"] = model.embed_images([os.path.join(root, x) for x in queries.img],
                                           batch_size=args.image_batch)
    if any(s[-1] == "I" for s in args.settings):
        emb["g"]["I"] = model.embed_images([os.path.join(root, x) for x in gallery.img],
                                           batch_size=args.image_batch)

    masks = {"independent": (queries.subset == "independent").values,
             "nested": (queries.subset == "nested").values,
             "all": queries.subset.notna().values}
    rows = []
    for s in args.settings:
        sims = model.similarity(emb["q"][s[0]], emb["g"][s[-1]]).numpy()
        rows.append({"setting": s, **{k: mean_average_precision(sims[m], positives[m])
                                      for k, m in masks.items()}})
    table = pd.DataFrame(rows).set_index("setting")

    pd.set_option("display.precision", 4)
    print(f"\n{args.model} - SABER large-gallery retrieval, mAP over 7,200 pooled candidates\n")
    print(table.to_string())
    if args.out:
        table.to_csv(args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
