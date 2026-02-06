"""Evaluate context-biasing impact on Qwen3-ASR baseline.

Runs the baseline model twice on VoxAI/bk-pl-dataforce-phase1 (mic channel):
  1. Without context (plain transcription)
  2. With context string containing all Burger King menu items

Usage:
  uv run python evaluate_context_biasing.py
  uv run python evaluate_context_biasing.py --device cuda:1
  uv run python evaluate_context_biasing.py --model VoxAI/qwen-asr-pl-de
"""

import argparse
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from jiwer import cer, wer

from qwen_asr import Qwen3ASRModel

MENU_ITEMS: list[str] = [
    # Burgers
    "whopper", "double whopper", "whopper junior",
    "plant whopper", "plant whopper junior",
    "cheeseburger", "double cheeseburger", "plant cheeseburger",
    "hamburger", "big king", "big king XXL",
    "bacon king", "bacon king junior",
    "bacon cheese royal", "bacon cheese whopper",
    "chili cheese burger", "chicken burger",
    "crispy chicken", "chicken royal",
    "king whiskey", "bao king", "cheester",
    "ranch burger", "summer crunch", "king smart",
    # Sides
    "nuggetsy", "nuggets", "plant nuggets", "chili cheese nuggets",
    "frytki", "duże frytki", "crispy cebula",
    # Drinks
    "pepsi", "pepsi zero", "cola", "mirinda",
    "7 up", "sprite", "ice tea", "lipton", "cappuccino",
    # Desserts
    "shake", "lody", "deser",
    # Wraps / other
    "wrap",
    # Meal types
    "zestaw", "zestaw dziecięcy", "chicken lovers",
]

CONTEXT_STRING = " ".join(MENU_ITEMS)


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


def evaluate_with_context(
    model: Qwen3ASRModel,
    samples: list[dict],
    context: str,
    batch_size: int,
    label: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  {label}")
    if context:
        print(f"  Context: {context[:80]}...")
    else:
        print("  Context: (none)")
    print(f"{'='*60}")

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

        results = model.transcribe(
            audio=audios,
            language="Polish",
            context=context,
        )

        for s, result in zip(batch, results):
            ref = normalize_text(s["reference"])
            hyp = normalize_text(result.text)
            references.append(ref)
            hypotheses.append(hyp)

        done = min(i + batch_size, len(samples))
        if done % 100 == 0 or done == len(samples):
            print(f"  Processed {done}/{len(samples)} utterances")

    total_wer = wer(references, hypotheses)
    total_cer = cer(references, hypotheses)

    print(f"\n  {label}:")
    print(f"    WER:  {total_wer:.2%}")
    print(f"    CER:  {total_cer:.2%}")
    print(f"    Samples: {len(references)}")
    print(f"    Total audio: {total_audio_sec / 60:.1f} min")

    print(f"\n  Sample predictions:")
    for idx in [0, len(references) // 2, len(references) - 1]:
        print(f"    REF: {references[idx][:80]}")
        print(f"    HYP: {hypotheses[idx][:80]}")
        print()

    return {
        "label": label,
        "wer": total_wer,
        "cer": total_cer,
        "num_samples": len(references),
        "total_audio_sec": total_audio_sec,
        "context_used": bool(context),
        "references": references,
        "hypotheses": hypotheses,
    }


def write_results_md(
    model_path: str,
    results: list[dict],
    output_path: Path,
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        f"# Context Biasing Evaluation: {model_path}",
        "",
        f"**Date:** {ts}",
        f"**Dataset:** VoxAI/bk-pl-dataforce-phase1 (mic channel, speech only)",
        f"**Samples:** {results[0]['num_samples']}",
        f"**Audio:** {results[0]['total_audio_sec'] / 60:.1f} min",
        "",
        "## Results",
        "",
        "| Condition | WER | CER |",
        "|---|---|---|",
    ]

    for r in results:
        wer_str = f"{r['wer']:.2%}"
        cer_str = f"{r['cer']:.2%}"
        lines.append(f"| {r['label']} | {wer_str} | {cer_str} |")

    if len(results) == 2:
        no_ctx = results[0]
        with_ctx = results[1]
        wer_delta = with_ctx["wer"] - no_ctx["wer"]
        cer_delta = with_ctx["cer"] - no_ctx["cer"]
        wer_rel = wer_delta / no_ctx["wer"] * 100 if no_ctx["wer"] > 0 else 0
        cer_rel = cer_delta / no_ctx["cer"] * 100 if no_ctx["cer"] > 0 else 0
        direction_wer = "improvement" if wer_delta < 0 else "regression"
        direction_cer = "improvement" if cer_delta < 0 else "regression"

        lines.extend([
            "",
            "## Delta",
            "",
            f"- **WER:** {wer_delta:+.2%} ({abs(wer_rel):.1f}% relative {direction_wer})",
            f"- **CER:** {cer_delta:+.2%} ({abs(cer_rel):.1f}% relative {direction_cer})",
        ])

    lines.extend([
        "",
        "## Context String",
        "",
        "Menu items provided as system-prompt context:",
        "",
        "```",
        CONTEXT_STRING,
        "```",
        "",
        "## Sample Comparisons",
        "",
    ])

    no_ctx_refs = results[0]["references"]
    no_ctx_hyps = results[0]["hypotheses"]
    ctx_hyps = results[1]["hypotheses"] if len(results) > 1 else []

    indices = [0, len(no_ctx_refs) // 4, len(no_ctx_refs) // 2, 3 * len(no_ctx_refs) // 4, len(no_ctx_refs) - 1]
    for idx in indices:
        lines.append(f"**Utterance {idx}**")
        lines.append(f"- REF: `{no_ctx_refs[idx][:100]}`")
        lines.append(f"- No context: `{no_ctx_hyps[idx][:100]}`")
        if ctx_hyps:
            lines.append(f"- With context: `{ctx_hyps[idx][:100]}`")
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nResults written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate context biasing on Polish drive-thru data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("Loading evaluation data (mic channel, speech only)...")
    samples = load_eval_data()
    print(f"Loaded {len(samples)} samples")

    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
    )

    result_no_ctx = evaluate_with_context(
        model, samples, context="", batch_size=args.batch_size,
        label="No context",
    )
    result_with_ctx = evaluate_with_context(
        model, samples, context=CONTEXT_STRING, batch_size=args.batch_size,
        label="With menu context",
    )

    del model
    torch.cuda.empty_cache()

    all_results = [result_no_ctx, result_with_ctx]

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'WER':>8} {'CER':>8}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['label']:<25} {r['wer']:>7.2%} {r['cer']:>7.2%}")

    model_slug = args.model.replace("/", "_")
    md_path = Path(f"eval_context_biasing_{model_slug}.md")
    write_results_md(args.model, all_results, md_path)

    json_path = Path(f"eval_context_biasing_{model_slug}.json")
    json_results = [{k: v for k, v in r.items() if k not in ("references", "hypotheses")} for r in all_results]
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
