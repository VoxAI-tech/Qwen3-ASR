"""Evaluate Qwen3-ASR models on VoxAI/bk-pl-dataforce-phase1 (mic channel only).

Usage:
  # Evaluate baseline
  uv run python evaluate.py --model Qwen/Qwen3-ASR-1.7B

  # Evaluate finetuned
  uv run python evaluate.py --model VoxAI/qwen-asr-pl-de

  # Compare both
  uv run python evaluate.py --model Qwen/Qwen3-ASR-1.7B VoxAI/qwen-asr-pl-de
"""

import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer

from qwen_asr import Qwen3ASRModel


def load_eval_data() -> list[dict]:
    ds = load_dataset(
        "VoxAI/bk-pl-dataforce-phase1", "utterance", split="train"
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    samples: list[dict] = []
    for ex in ds:
        if ex["channel"] != "microphone" or not ex["has_speech"]:
            continue

        texts = [t for t in ex["segments"]["transcription"] if t is not None]
        if not texts:
            continue

        reference = " ".join(texts)
        samples.append({
            "utterance_id": ex["utterance_id"],
            "audio_bytes": ex["audio"]["bytes"],
            "reference": reference,
            "duration_sec": ex["duration_sec"],
        })

    return samples


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sr


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[nc\]", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_model(model_path: str, samples: list[dict], batch_size: int = 8, device: str = "cuda:0") -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    )

    references: list[str] = []
    hypotheses: list[str] = []
    total_audio_sec = 0.0

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        audios: list[tuple[np.ndarray, int]] = []

        for s in batch:
            array, sr = decode_audio(s["audio_bytes"])
            audios.append((array, sr))
            total_audio_sec += s["duration_sec"]

        results = model.transcribe(audio=audios, language="Polish")

        for s, result in zip(batch, results):
            ref = normalize_text(s["reference"])
            hyp = normalize_text(result.text)
            references.append(ref)
            hypotheses.append(hyp)

        done = min(i + batch_size, len(samples))
        if done % 50 == 0 or done == len(samples):
            print(f"  Processed {done}/{len(samples)} utterances")

    total_wer = wer(references, hypotheses)
    total_cer = cer(references, hypotheses)

    print(f"\n  Results for {model_path}:")
    print(f"    WER:  {total_wer:.2%}")
    print(f"    CER:  {total_cer:.2%}")
    print(f"    Samples: {len(references)}")
    print(f"    Total audio: {total_audio_sec / 60:.1f} min")

    # Show some examples
    print(f"\n  Sample predictions:")
    for idx in [0, len(references) // 2, len(references) - 1]:
        print(f"    REF: {references[idx][:80]}")
        print(f"    HYP: {hypotheses[idx][:80]}")
        print()

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_path,
        "wer": total_wer,
        "cer": total_cer,
        "num_samples": len(references),
        "total_audio_sec": total_audio_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR models on Polish eval set")
    parser.add_argument("--model", nargs="+", required=True, help="Model path(s) to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    print("Loading evaluation data (mic channel, speech only)...")
    samples = load_eval_data()
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict] = []
    for model_path in args.model:
        result = evaluate_model(model_path, samples, args.batch_size, args.device)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<40} {'WER':>8} {'CER':>8}")
        print("-" * 60)
        for r in all_results:
            print(f"{r['model']:<40} {r['wer']:>7.2%} {r['cer']:>7.2%}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
