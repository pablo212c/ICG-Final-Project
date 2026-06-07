# ICG Photo Ranking

## Instruction

```bash
python app.py
```

Open `http://127.0.0.1:7860`, upload 2-5 images, then click **Rank Photos**. The UI shows the selected image, ranking table, color harmony features, and a fast explanation without waiting for the 9B LLM.

Stop the UI:

- If you started it directly with `python app.py` in a terminal, press `Ctrl+C` in that same terminal.
- If the terminal was closed or port `7860` is still occupied, find and stop the process in PowerShell:

```powershell
netstat -ano | findstr :7860
Stop-Process -Id <PID>
```

```bash
python scripts/rank.py --demo all
python scripts/explain.py
```

Run subset of photos:

```bash
python scripts/rank.py --demo img1 img3 img5
python scripts/explain.py
```

Run your own photos:

You NEED to put photos in `img/`.

```bash
python scripts/rank.py --images imgA.jpg imgB.jpg
python scripts/rank.py --images imgC.jpg imgD.jpg
python scripts/explain.py
```

`rank.py` writes `outputs/rank_llm_input.json` automatically. `explain.py` reads that file and writes the full local LLM explanation to `outputs/rank_explanation.txt`.

Your own photos do not need true labels. `demo_img/ground_truth.csv` is only for showing that the prepared demo ranking matches AADB labels.

Before a live demo, run `python scripts/explain.py` once so the GGUF model is already downloaded into `.model_cache/`. During the demo, use this if you want to avoid downloading:

```bash
python scripts/explain.py --local-files-only
```

Photo aesthetic ranking prototype for AADB. The current repository contains only the code needed to train, test, and run ranking inference. Datasets, checkpoints, predictions, PDFs, and MATLAB reference files are intentionally ignored by git.

## Current Status

Implemented:

- AADB train/validation/test loader
- ConvNeXt-Small aesthetic score model
- 11 AADB attribute prediction head
- score loss + attribute loss + pairwise ranking loss + small listwise top-ranking loss
- EMA checkpoint saving
- test metrics: MSE, MAE, Spearman, Kendall, attribute MAE
- sampled group metrics: group Spearman, group Kendall, pairwise accuracy, top1 accuracy
- ranking inference for 2-5 input images
- color harmony structured features for downstream explanation
- JSON payload suitable for a future LLM explanation step
- local GGUF LLM explanation with Qwen3.5-9B-Q4_K_M
- deeper color harmony ablation

## Repository Layout

```text
app.py                      Gradio UI (entry point at project root)
scripts/train.py            train/test the model
scripts/rank.py             rank 2-5 images and export explanation-ready JSON
scripts/explain.py          generate natural-language explanation from rank JSON
src/                        dataset, model, metrics, inference, transforms, color harmony
configs/ema_s.json          main training config
configs/test_tta.json       final test config
configs/rank_example.json   example ranking config
requirements.txt            Python dependencies
```

## Local Data Layout

These folders are required locally but should not be pushed:

```text
datasetImages_warp256/
imgListFiles_label/imgListFiles_label/
```

The split is defined by AADB label txt files:

```text
train       8458 images
validation   500 images
test        1000 images
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

Default training uses `configs/ema_s.json`.

```bash
python scripts/train.py
```

Useful overrides:

```bash
python scripts/train.py --epochs 40
python scripts/train.py --batch-size 8
python scripts/train.py --checkpoint-policy auto
```

The default checkpoint path is:

```text
checkpoints/best.pt
```

## Test

Default test uses `configs/test_tta.json`.

```bash
python scripts/train.py --test
```

For 5-image group ranking, the config uses:

```text
group_size = 5
group_count = 1000
seed = 42
```

The same checkpoint, seed, group size, and group count produce deterministic group metrics.

## Rank Images

Put your own photos in `img/`, then rank by filename:

```bash
python scripts/rank.py --images imgA.jpg imgB.jpg
```

Your own photos do not need true labels. You can also pass relative or absolute paths directly:

```bash
python scripts/rank.py --images other_folder/my_photo1.jpg other_folder/my_photo2.jpg
python scripts/explain.py
```

Print JSON:

```bash
python scripts/rank.py --config configs/rank_example.json --json
```

Write downstream LLM input without running an LLM:

```bash
python scripts/rank.py --images imgA.jpg imgB.jpg
```

Run the locally prepared demo group:

```bash
python scripts/rank.py --demo all
python scripts/explain.py
```

Run a selected subset of 2-5 demo images:

```bash
python scripts/rank.py --demo img1 img2 img5
python scripts/explain.py
```

Generate a local LLM explanation without an API key:

```bash
python scripts/explain.py
```

Default input: `outputs/rank_llm_input.json`
Default output: `outputs/rank_explanation.txt`

Default local model:

```text
jc-builds/Qwen3.5-9B-Q4_K_M-GGUF
```

The program first searches for an existing GGUF file in `.model_cache/`. It downloads from Hugging Face into `.model_cache/` only when no local file is found. It runs through `llama-cpp-python` with CUDA GGUF inference, `Q4_K_M` quantization, `n_gpu_layers=-1`, and `n_ctx=12288` by default. This avoids Transformers CPU offload and is better for live demos on a 12GB GPU.

Run only from already downloaded model files:

```bash
python scripts/explain.py --local-files-only
```

Fallback to the previous Hugging Face Transformers 4B backend:

```bash
python scripts/explain.py --provider local
```

Inspect the prompt without loading a model:

```bash
python scripts/explain.py --dry-run
```

Cloud providers remain optional:

```bash
python scripts/explain.py --provider openai --input outputs/rank_llm_input.json
python scripts/explain.py --provider anthropic --input outputs/rank_llm_input.json
```

Cloud mode requires the corresponding package and API key.

Ranking uses only:

```text
rank_score = aesthetic_score
```

Color harmony and attributes are exported as explanation features. They do not change the ranking score.

## Metrics

Task-focused ranking:

```text
pairwise_accuracy : pair order correctness in sampled groups
top1_accuracy     : whether the best image in a sampled group is selected
```

Because the final use case is selecting the best image from 2-5 candidates, `top1_accuracy` is the most important project metric.

## AADB Test Comparison

Paper baseline: Kong et al., *Photo Aesthetics Ranking Network with Attributes and Content Adaptation*, ECCV 2016, AADB Table 2. The paper reports Spearman's rank correlation on AADB; it does not report our extra group-ranking metrics.

Our model is trained using `configs/ema_s.json`. The final test results are evaluated using test-time augmentation (TTA) via `configs/test_tta.json` on `checkpoints/best.pt`.

| Method | Spearman↑ | Kendall↑ | MSE↓ | MAE↓ | Attr MAE↓ | Pairwise Acc↑ | Top1 Acc↑ | Group Spearman↑ | Group Kendall↑ | Group Rank Score↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reg+Rank (within- & cross-rater) | 0.6515 | 0.4849 | 0.0239 | 0.1219 | - | - | - | - | - | - |
| Reg+Rank+Att | 0.6656 | 0.4886 | 0.0291 | 0.1386 | - | - | - | - | - | - |
| Reg+Rank+Cont | 0.6737 | 0.4998 | **0.0185** | 0.1112 | - | - | - | - | - | - |
| Reg+Rank+Att+Cont | **0.6782** | **0.5028** | 0.0196 | 0.1131 | - | - | - | - | - | - |
| Ours | 0.6660 | 0.4996 | 0.0199 | **0.1095** | **0.1762** | **0.7495** | **0.6050** | **0.5679** | **0.4833** | **0.6772** |

Notes:

- Higher is better for Spearman, Kendall, pairwise accuracy, top1 accuracy, group Spearman, group Kendall, and group rank score.
- Lower is better for MSE, MAE, and attribute MAE.
