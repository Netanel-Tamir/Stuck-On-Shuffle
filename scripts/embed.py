#!/usr/bin/env python3
"""Embed a few texts and/or images with one of the released models and print the
cosine-similarity matrix.

    python scripts/embed.py --model qwen3vl-embedding-8b \
        --text "A red mug and a blue kettle." "A blue mug and a red kettle." \
        --image scene.png

    python scripts/embed.py --list
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from stuck_on_shuffle import list_models, load_model  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list_models(), default="qwen3vl-embedding-8b")
    p.add_argument("--base-model", default=None,
                   help="local directory of the base model (default: download from the hub)")
    p.add_argument("--text", nargs="+", default=[], help="texts to embed")
    p.add_argument("--image", nargs="+", default=[], help="image paths / URLs to embed")
    p.add_argument("--instruction", default=None,
                   help="override the model's default task instruction (leave unset to "
                        "reproduce the paper setting)")
    p.add_argument("--merge-lora", action="store_true")
    p.add_argument("--list", action="store_true", help="print the available models and exit")
    args = p.parse_args()

    if args.list:
        for name in list_models():
            print(name)
        return 0
    if not (args.text or args.image):
        p.error("pass --text and/or --image")

    model = load_model(args.model, base_model=args.base_model, merge_lora=args.merge_lora)
    items, embs = [], []
    if args.text:
        items += [f"[text]  {t}" for t in args.text]
        embs.append(model.embed_texts(args.text, instruction=args.instruction))
    if args.image:
        items += [f"[image] {os.path.basename(im)}" for im in args.image]
        embs.append(model.embed_images(args.image, instruction=args.instruction))
    e = torch.cat(embs)
    print(f"\nembeddings: {tuple(e.shape)}  (mean L2 norm {e.norm(dim=-1).mean():.4f})")

    sims = model.similarity(e, e)
    w = min(max(len(s) for s in items), 60)
    print("\ncosine similarity")
    for i, label in enumerate(items):
        row = "  ".join(f"{sims[i, j]:+.3f}" for j in range(len(items)))
        print(f"  {label[:w]:<{w}}  {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
