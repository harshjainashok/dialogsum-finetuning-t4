# Fine-Tuning LLMs for Dialogue Summarisation

Fine-tuning **Qwen2.5-0.5B-Instruct** and **Llama-3.2-3B-Instruct** on the [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) dataset using LoRA, Quantization (zero-shot baseline), and QLoRA — all on a Google Colab T4 GPU (Free Tier).

## Results (Full Test Set, n=1,500)

| Method | Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|--------|-------|---------|---------|---------|-------------|
| Zero-shot | Llama-3.2-3B | 0.2729 | 0.0834 | 0.2054 | 0.8695 |
| Zero-shot | Qwen-0.5B | 0.2434 | 0.0647 | 0.1862 | 0.8711 |
| LoRA-16 | Llama-3.2-3B | **0.4855** | 0.2198 | 0.4015 | 0.9225 |
| LoRA-16 | Qwen-0.5B | 0.4064 | 0.1456 | 0.3218 | 0.9112 |
| QLoRA | Llama-3.2-3B | 0.4849 | 0.2213 | 0.4008 | 0.9220 |
| QLoRA | Qwen-0.5B | 0.4225 | 0.1520 | 0.3340 | 0.9130 |

## Repository Structure

```
├── AI_Assignment_T4_v1.ipynb     # Main Colab notebook (end-to-end reproducible)
└── outputs/
    ├── QLORA_RESULTS.json         # QLoRA validation ROUGE results
    ├── LORA16_RESULTS.json        # LoRA-16 validation ROUGE results
    ├── GRID_RESULTS.json          # Hyperparameter grid results (Configs A/B/C)
    ├── plots/                     # Training vs validation loss curves (PNG)
    ├── predictions/               # Test & validation prediction JSONL/CSV files
    ├── grid/                      # Grid search checkpoints (Llama)
    └── grid_qwen/                 # Grid search checkpoints (Qwen)
```

## Setup

Open `AI_Assignment_T4_v1.ipynb` in Google Colab (T4 GPU, Free Tier) and run all cells.

**Seed:** 3407 | **Hardware:** T4 GPU (15 GB VRAM) | **Library:** [Unsloth](https://github.com/unslothai/unsloth)
