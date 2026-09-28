from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download the official Clay v1.5 checkpoint")
    parser.add_argument("--output", default=str(project / "checkpoints" / "clay-v1.5.ckpt"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    print(f"Downloading Clay v1.5 to {output} ...")
    with urllib.request.urlopen(args.url) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(output)
    print(f"Done: {output} ({output.stat().st_size / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()

