#!/usr/bin/env python3
"""Random subsample N records from raw pool matching |reference| (IDEA-04)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="42,43,44")
    args = p.parse_args()

    raw = json.loads(Path(args.input).read_text())
    n = len(json.loads(Path(args.reference).read_text()))

    for seed in map(int, args.seeds.split(",")):
        rng = random.Random(seed)
        sample = rng.sample(raw, min(n, len(raw)))
        out = Path(args.output.replace(".json", f"_seed{seed}.json"))
        out.write_text(json.dumps(sample, indent=2))
        print(f"wrote {out} ({len(sample)} records)")


if __name__ == "__main__":
    main()
