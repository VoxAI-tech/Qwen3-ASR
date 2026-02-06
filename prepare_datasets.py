"""Download HuggingFace WebDataset repos and prepare them for Qwen3-ASR finetuning.

Usage:
  uv run python prepare_datasets.py --de --pl          # German + Polish only
  uv run python prepare_datasets.py --de --pl --en     # All three
  uv run python prepare_datasets.py --all              # All three

Produces:
  /data/razhan/qwen_data/
    audio/{de,en,pl}/*.wav
    train.jsonl              (selected languages combined)
    train_{lang_code}.jsonl  (per-language)
"""

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import numpy as np
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

DATASETS: dict[str, dict[str, str]] = {
    "de": {
        "repo": "VoxAI/bk-de-20251123",
        "lang_code": "de",
        "lang_name": "German",
    },
    "en": {
        "repo": "VoxAI/ej-au-20250918",
        "lang_code": "en",
        "lang_name": "English",
    },
    "pl": {
        "repo": "VoxAI/bk-pl-20250909-gemini",
        "lang_code": "pl",
        "lang_name": "Polish",
    },
}

OUT_ROOT = Path("/data/razhan/qwen_data")
AUDIO_DIR = OUT_ROOT / "audio"
TARGET_SR = 16_000


def list_tar_files(repo: str) -> list[str]:
    api = HfApi()
    all_files = api.list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in all_files if f.endswith(".tar"))


def process_tar(tar_path: str, lang_code: str, lang_name: str, audio_out: Path, start_idx: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with tarfile.open(tar_path) as tf:
        members = tf.getmembers()
        json_members = {m.name: m for m in members if m.name.endswith(".json")}
        wav_members = {m.name: m for m in members if m.name.endswith(".wav")}

        idx = start_idx
        for jname, jmember in sorted(json_members.items()):
            stem = jname.rsplit(".", 1)[0]
            wav_name = stem + ".wav"
            if wav_name not in wav_members:
                continue

            meta = json.load(tf.extractfile(jmember))
            transcript = meta.get("text", "") or meta.get("transcription", "")
            text = f"language {lang_name}<asr_text>{transcript}"

            out_name = f"{lang_code}_{idx:07d}.wav"
            out_path = audio_out / out_name

            if not out_path.exists():
                wav_bytes = tf.extractfile(wav_members[wav_name]).read()
                array, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

                if array.ndim > 1:
                    array = array.mean(axis=1)

                if sr != TARGET_SR:
                    import librosa
                    array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)

                sf.write(str(out_path), array, TARGET_SR)

            rows.append({"audio": str(out_path), "text": text})
            idx += 1

    return rows


def process_dataset(cfg: dict[str, str]) -> list[dict[str, str]]:
    repo = cfg["repo"]
    lang_code = cfg["lang_code"]
    lang_name = cfg["lang_name"]

    audio_out = AUDIO_DIR / lang_code
    audio_out.mkdir(parents=True, exist_ok=True)

    print(f"[{lang_code}] Listing tar files for {repo} ...")
    tar_files = list_tar_files(repo)
    print(f"[{lang_code}] Found {len(tar_files)} shards")

    rows: list[dict[str, str]] = []
    for shard_idx, tar_rel in enumerate(tar_files):
        tar_path = hf_hub_download(repo, tar_rel, repo_type="dataset")
        shard_rows = process_tar(tar_path, lang_code, lang_name, audio_out, start_idx=len(rows))
        rows.extend(shard_rows)
        print(f"[{lang_code}] Shard {shard_idx + 1}/{len(tar_files)} — {len(shard_rows)} samples (total: {len(rows)})")

    print(f"[{lang_code}] Done — {len(rows)} samples")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ASR finetuning data")
    p.add_argument("--de", action="store_true", help="Include German dataset")
    p.add_argument("--en", action="store_true", help="Include English dataset")
    p.add_argument("--pl", action="store_true", help="Include Polish dataset")
    p.add_argument("--all", action="store_true", help="Include all languages")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.all:
        selected = list(DATASETS.keys())
    else:
        selected = [k for k in DATASETS if getattr(args, k)]

    if not selected:
        print("No languages selected. Use --de, --pl, --en, or --all")
        return

    print(f"Selected languages: {selected}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    configs = [DATASETS[k] for k in selected]
    max_workers = min(len(configs), 3)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_dataset, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg = futures[future]
            lang_code = cfg["lang_code"]
            rows = future.result()
            write_jsonl(OUT_ROOT / f"train_{lang_code}.jsonl", rows)
            all_rows.extend(rows)

    write_jsonl(OUT_ROOT / "train.jsonl", all_rows)
    print(f"\nTotal: {len(all_rows)} samples in {OUT_ROOT / 'train.jsonl'}")


if __name__ == "__main__":
    main()
