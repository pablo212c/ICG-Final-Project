
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
"""Generate natural-language explanations for photo aesthetic rankings.

Usage
-----
First produce a structured JSON with rank.py:

    python rank.py --images img1.jpg img2.jpg img3.jpg \
        --tta-views 10 --llm-input outputs/rank_llm_input.json

Then generate explanations locally with a GGUF model:

    python explain.py

Or use a local prompt-only mode (no API call) to inspect the prompt:

    python explain.py --dry-run

Default local model:

    jc-builds/Qwen3.5-9B-Q4_K_M-GGUF

Local mode uses llama-cpp-python CUDA inference with a fixed .model_cache/
folder and GPU layer offload by default.

Cloud providers remain optional:

    python explain.py --provider openai --input outputs/rank_llm_input.json
    python explain.py --provider anthropic --input outputs/rank_llm_input.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import textwrap
from pathlib import Path


DEFAULT_INPUT_PATH = "outputs/rank_llm_input.json"
DEFAULT_OUTPUT_PATH = "outputs/rank_explanation.txt"
DEFAULT_PROVIDER = "gguf"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_GGUF_REPO = "jc-builds/Qwen3.5-9B-Q4_K_M-GGUF"
DEFAULT_GGUF_PREFIX = "Qwen3.5-9B-Q4_K_M"
DEFAULT_GGUF_CACHE_DIR = ".model_cache"
DEFAULT_GGUF_N_CTX = 12288
DEFAULT_GGUF_N_GPU_LAYERS = -1
DEFAULT_GGUF_N_BATCH = 512
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_TEMPERATURE = 0.0
DEFAULT_QUANTIZATION = "4bit"
DEFAULT_GPU_MEMORY = "6GiB"
DEFAULT_CPU_MEMORY = "24GiB"
DEFAULT_OFFLOAD_FOLDER = "outputs/llm_offload"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a professional photography critic and instructor. You will receive
    a structured JSON payload that describes a set of 2-5 photos ranked by an
    aesthetic scoring model.

    Your job is to write a clear, helpful, and friendly explanation of the
    ranking results in the requested language. Follow these rules strictly:

    1. The ranking is determined ONLY by the aesthetic_score (rank_score).
       Never claim that color harmony or any single attribute changed the rank.
    2. Color harmony information is supplementary evidence for explanation.
       You may mention it to support or illustrate a point, but not as a
       cause of the ranking.
    3. Only reference attributes and scores that appear in the JSON. Do not
       invent observations the model did not compute (e.g. do not describe
       subject matter, emotions, or narrative unless an attribute covers it).
    4. When the score margin is "close", acknowledge the near-tie honestly.
    5. Structure your response as:
       a. One-sentence verdict (which photo wins and by how much).
       b. Per-photo paragraph with key strengths/weaknesses from attributes.
       c. Brief comparison highlighting the biggest attribute differences.
    6. Keep the total length under 300 words.
    7. Use the language specified in the "language" field.
    8. Distinguish the learned AADB attribute "ColorHarmony" from the
       formula-based "color_harmony" object. When discussing formula-based
       color harmony, use color_harmony.level and color_harmony.percentile as
       the authoritative qualitative description; do not contradict them.
    9. Do not turn neutral attributes into strong weaknesses. Mention a
       weakness only when the provided level is negative/strong_negative, or
       when comparison data explicitly shows a meaningful disadvantage.
    10. Use these Traditional Chinese attribute names: Balancing elements=元素平衡,
        Color harmony=色彩和諧屬性, Content=內容完整度, DoF=景深,
        Light=光線, Motion blur=動態模糊, Object emphasis=主體強調,
        Repetition=重複性, Rule of thirds=三分法, Symmetry=對稱性,
        Vivid color=鮮豔色彩.
    11. Use direct Traditional Chinese. Avoid decorative or exaggerated wording.
    12. Discuss a DIVERSE set of attributes across photos. Do not focus on
        the same attribute (especially 主體強調) in every paragraph. Each
        photo's analysis should highlight DIFFERENT strengths or weaknesses
        to give a well-rounded critique.
    13. Pick at most 2-3 most distinguishing attributes per photo. Avoid
        repeating the same attribute name across multiple paragraphs unless
        it is genuinely the single deciding factor.
""")


def _short_float(value: object) -> object:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _compact_attr_items(items: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for item in items[:limit]:
        compact.append(
            {
                "name": item.get("display_name") or item.get("name"),
                "value": _short_float(item.get("value")),
                "level": item.get("level"),
            }
        )
    return compact


def _compact_diff_items(items: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for item in items[:limit]:
        compact.append(
            {
                "name": item.get("display_name") or item.get("name"),
                "top_value": _short_float(item.get("top_value")),
                "other_value": _short_float(item.get("other_value")),
                "difference": _short_float(item.get("difference")),
            }
        )
    return compact


def _compact_payload(payload: dict) -> dict:
    metadata = payload.get("metadata", {})
    summary = payload.get("summary", {})
    compact_ranked = []
    for item in payload.get("ranked", []):
        attr_summary = item.get("attribute_summary", {})
        color = item.get("color_harmony", {})
        compact_ranked.append(
            {
                "rank": item.get("rank"),
                "filename": item.get("filename"),
                "aesthetic_score": _short_float(item.get("aesthetic_score")),
                "score_gaps": {
                    "behind_previous": _short_float(item.get("score_gaps", {}).get("behind_previous")),
                    "ahead_of_next": _short_float(item.get("score_gaps", {}).get("ahead_of_next")),
                },
                "strongest_positive_attributes": _compact_attr_items(
                    attr_summary.get("strongest_positive", []),
                ),
                "strongest_negative_attributes": _compact_attr_items(
                    attr_summary.get("strongest_negative", []),
                ),
                "color_harmony": {
                    "role": color.get("role"),
                    "score": _short_float(color.get("score")),
                    "level": color.get("level"),
                    "percentile": _short_float(color.get("percentile")),
                    "best_template": color.get("best_template"),
                },
            }
        )

    compact_comparisons = []
    for comparison in payload.get("comparisons", []):
        diffs = comparison.get("attribute_differences", {})
        compact_comparisons.append(
            {
                "top": comparison.get("top_filename"),
                "other": comparison.get("other_filename"),
                "score_margin": _short_float(comparison.get("score_margin")),
                "margin_level": comparison.get("margin_level"),
                "top_harmony_score_minus_other": _short_float(
                    comparison.get("top_harmony_score_minus_other"),
                ),
                "top_advantages": _compact_diff_items(diffs.get("top_advantages", []), limit=3),
                "top_disadvantages": _compact_diff_items(diffs.get("top_disadvantages", []), limit=3),
            }
        )

    return {
        "metadata": {
            "task": metadata.get("task"),
            "input_count": metadata.get("input_count"),
            "rank_score_field": metadata.get("rank_score_field"),
            "color_harmony_role": metadata.get("color_harmony_role"),
            "attribute_role": metadata.get("attribute_role"),
        },
        "summary": {
            "top_filename": summary.get("top_filename"),
            "top_rank_score": _short_float(summary.get("top_rank_score")),
            "runner_up_filename": summary.get("runner_up_filename"),
            "top_margin_over_runner_up": _short_float(summary.get("top_margin_over_runner_up")),
            "top_margin_level": summary.get("top_margin_level"),
        },
        "ranked": compact_ranked,
        "comparisons": compact_comparisons,
    }


def build_user_prompt(payload: dict, language: str = "zh-TW") -> str:
    trimmed = _compact_payload(payload)

    return (
        f"Language: {language}\n\n"
        f"Ranking payload:\n```json\n{json.dumps(trimmed, indent=2, ensure_ascii=False)}\n```"
    )


def _find_gguf_file(cache_dir: str | Path, filename_prefix: str) -> str | None:
    pattern = str(Path(cache_dir) / "**" / f"*{filename_prefix}*.gguf")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[0] if matches else None


def _resolve_gguf_path(
    gguf_path: str | None,
    repo_id: str,
    filename_prefix: str,
    cache_dir: str,
    local_files_only: bool,
) -> str:
    if gguf_path:
        path = Path(gguf_path)
        if not path.exists():
            raise SystemExit(f"GGUF model file not found: {path}")
        return str(path)

    existing = _find_gguf_file(cache_dir, filename_prefix)
    if existing:
        print(f"Using local GGUF model: {existing}", flush=True)
        print()
        return existing

    if local_files_only:
        raise SystemExit(
            "GGUF model is not available locally. "
            f"Expected a file matching *{filename_prefix}*.gguf under {cache_dir}."
        )

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "GGUF mode requires huggingface-hub. Install with: pip install -r requirements.txt"
        ) from exc

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"*{filename_prefix}*.gguf"],
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
    except Exception:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{filename_prefix}.gguf",
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )

    downloaded = _find_gguf_file(cache_dir, filename_prefix)
    if not downloaded:
        raise SystemExit(
            f"Downloaded repo {repo_id}, but no *{filename_prefix}*.gguf file was found."
        )
    return downloaded


def call_local_gguf(
    system: str,
    user: str,
    repo_id: str,
    filename_prefix: str,
    cache_dir: str,
    gguf_path: str | None,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    n_ctx: int,
    n_gpu_layers: int,
    n_batch: int,
) -> str:
    """Run a local GGUF model through llama.cpp with GPU layer offload."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "GGUF mode requires llama-cpp-python. Install with: pip install -r requirements.txt"
        ) from exc

    model_path = _resolve_gguf_path(
        gguf_path=gguf_path,
        repo_id=repo_id,
        filename_prefix=filename_prefix,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    n_threads = os.cpu_count() or 8
    print(
        "Local GGUF loading: "
        f"repo={repo_id}, file={model_path}, n_ctx={n_ctx}, "
        f"n_gpu_layers={n_gpu_layers}, n_batch={n_batch}",
        flush=True,
    )
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
    )
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    return str(response["choices"][0]["message"]["content"]).strip()


def call_local_hf(
    system: str,
    user: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    quantization: str,
    gpu_memory: str,
    cpu_memory: str,
    offload_folder: str,
) -> str:
    """Run a local Hugging Face causal language model; no API key required."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Local LLM mode requires transformers and accelerate. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
        "offload_folder": offload_folder,
        "offload_state_dict": True,
    }
    if torch.cuda.is_available():
        model_kwargs["max_memory"] = {0: gpu_memory, "cpu": cpu_memory}

    if quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "4-bit local LLM mode requires bitsandbytes. "
                "Install with: pip install -r requirements.txt"
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else "auto"

    print(
        "Local LLM loading: "
        f"model={model_name}, quantization={quantization}, "
        f"gpu_memory_cap={gpu_memory if torch.cuda.is_available() else 'cpu-only'}, "
        f"cpu_memory_cap={cpu_memory}, offload_folder={offload_folder}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([prompt], return_tensors="pt")
    input_device = model.get_input_embeddings().weight.device
    model_inputs = {key: value.to(input_device) for key, value in model_inputs.items()}

    generation_args = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_args.update({"do_sample": True, "temperature": temperature})
    else:
        generation_args.update({"do_sample": False})

    with torch.no_grad():
        generated = model.generate(**generation_args)

    output_ids = generated[0][model_inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def call_openai(system: str, user: str, model: str, api_key: str, base_url: str | None) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    try:
        import openai
    except ImportError:
        raise SystemExit(
            "openai package is required. Install with: pip install openai"
        )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def call_anthropic(system: str, user: str, model: str, api_key: str) -> str:
    """Call the Anthropic Messages API."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "anthropic package is required. Install with: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.4,
    )
    return response.content[0].text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM explanations for photo ranking results.",
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT_PATH,
        help=f"Path to the structured JSON from rank.py --llm-input. Default: {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH,
        help=f"Path to write the explanation text. Default: {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--language", default="zh-TW",
        help="Language for the explanation (default: zh-TW).",
    )
    parser.add_argument(
        "--provider", default=DEFAULT_PROVIDER,
        choices=["gguf", "local", "openai", "anthropic"],
        help="LLM provider. Default is local GGUF inference.",
    )
    parser.add_argument(
        "--model", default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum number of generated tokens. Default: {DEFAULT_MAX_NEW_TOKENS}.",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature. Default: {DEFAULT_TEMPERATURE} for deterministic decoding.",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Use only already downloaded local model files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    user_prompt = build_user_prompt(payload, language=args.language)

    if args.dry_run:
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print("\n=== USER PROMPT ===")
        print(user_prompt)
        return

    if args.provider == "gguf":
        explanation = call_local_gguf(
            SYSTEM_PROMPT,
            user_prompt,
            repo_id=DEFAULT_GGUF_REPO,
            filename_prefix=DEFAULT_GGUF_PREFIX,
            cache_dir=DEFAULT_GGUF_CACHE_DIR,
            gguf_path=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            local_files_only=args.local_files_only,
            n_ctx=DEFAULT_GGUF_N_CTX,
            n_gpu_layers=DEFAULT_GGUF_N_GPU_LAYERS,
            n_batch=DEFAULT_GGUF_N_BATCH,
        )
    elif args.provider == "local":
        model = args.model or os.environ.get("LLM_MODEL", DEFAULT_LOCAL_MODEL)
        explanation = call_local_hf(
            SYSTEM_PROMPT,
            user_prompt,
            model_name=model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            local_files_only=args.local_files_only,
            quantization=DEFAULT_QUANTIZATION,
            gpu_memory=DEFAULT_GPU_MEMORY,
            cpu_memory=DEFAULT_CPU_MEMORY,
            offload_folder=DEFAULT_OFFLOAD_FOLDER,
        )
    elif args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY environment variable.")
        model = args.model or os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
        explanation = call_anthropic(SYSTEM_PROMPT, user_prompt, model, api_key)
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY environment variable.")
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        model = args.model or os.environ.get("LLM_MODEL", "gpt-4o")
        explanation = call_openai(SYSTEM_PROMPT, user_prompt, model, api_key, base_url)

    print('\n')
    print(explanation)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(explanation, encoding="utf-8")
        print(f"\nWrote explanation to {output_path}")


if __name__ == "__main__":
    main()
