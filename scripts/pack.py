"""Zip the repo for upload to Colab, excluding data, runs and caches."""

import zipfile
from pathlib import Path

SKIP = {"runs", "data", "__pycache__", ".git", ".venv", ".ipynb_checkpoints"}


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "asr.zip"
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part in SKIP for part in rel.parts):
                continue
            if not path.is_file() or path.name == out.name:
                continue
            zf.write(path, Path("asr") / rel)
            count += 1
    print(f"{out} ({count} files, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
