"""Build and load JSONL manifests from a LibriSpeech-style tree."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

_WS = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u00b4": "'"})
_KEEP = re.compile(r"[^0-9a-z' ]+", flags=re.UNICODE)


def normalize_text(text):
    """Lowercase, unify apostrophes, drop punctuation, collapse whitespace."""
    text = text.translate(_APOSTROPHES).lower()
    text = _KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _duration(args):
    path, utt_id = args
    try:
        return utt_id, float(sf.info(str(path)).duration)
    except Exception:
        return utt_id, None


def _read_split(split_dir):
    """Yield (utt_id, flac_path, transcript) for one LibriSpeech split."""
    for trans in sorted(Path(split_dir).rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            audio = trans.parent / f"{utt_id}.flac"
            if not audio.exists():
                audio = trans.parent / f"{utt_id}.wav"
            if not audio.exists():
                continue
            yield utt_id, audio, text


def build_manifest(root, splits, out_path, min_duration=0.3, max_duration=20.0, workers=16):
    root = Path(root)
    entries, skipped = [], 0

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"split not found: {split_dir}")
        rows = list(_read_split(split_dir))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            durations = dict(pool.map(_duration, [(p, u) for u, p, _ in rows]))

        for utt_id, audio, text in rows:
            duration = durations.get(utt_id)
            norm = normalize_text(text)
            if not duration or not norm or not (min_duration <= duration <= max_duration):
                skipped += 1
                continue
            entries.append(
                {
                    "id": utt_id,
                    "audio": str(audio.resolve().relative_to(root.resolve())).replace("\\", "/"),
                    "duration": round(duration, 3),
                    "text": norm,
                    "speaker": utt_id.split("-")[0],
                }
            )

    entries.sort(key=lambda e: (e["speaker"], e["id"]))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    hours = sum(e["duration"] for e in entries) / 3600.0
    return {"path": str(out_path), "utterances": len(entries), "hours": round(hours, 2), "skipped": skipped}


def load_manifest(path, root):
    root = Path(root)
    entries = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry["audio"] = str(root / entry["audio"])
            entries.append(entry)
    return entries
