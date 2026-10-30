**English → Hindi Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma**

Foundations of Generative AI (ICT4403) - Assignment 3

**Name:** Tejas K Rao

**Registration No.:** 230911009

**1. Problem statement and objective**

Build and compare four machine-translation systems for **English → Hindi**, two encoder-decoder and two decoder-only:

| System | Architecture | Adaptation |
|---|---|---|
| Basic Transformer | encoder-decoder | trained from scratch |
| mT5-small | encoder-decoder | fine-tuned |
| Llama-3.2-1B-Instruct | decoder-only | few-shot prompting and LoRA fine-tuning |
| Gemma-3-1B-it | decoder-only | few-shot prompting and LoRA fine-tuning |

All systems translate the same held-out test sentences. They are compared on translation quality (BLEU, chrF) and computational cost (training time, inference time, parameter count). The translations are then examined for bias, hallucination, cultural misrepresentation and sensitive-content errors, and mitigations are proposed.

**2. Dataset**

- **Source:** IIT Bombay English-Hindi Parallel Corpus, <https://huggingface.co/datasets/cfilt/iitb-english-hindi> (original project page: <https://www.cfilt.iitb.ac.in/iitb_parallel/>). About 1.6 million sentence pairs from government, news and web sources.
- **Dataset actually used:** `cfilt/iitb-english-hindi` (the OPUS-100 fallback was not needed).
- **Subset used (from `results/dataset_stats.json`):** 50,000 training pairs, 500 validation pairs and 500 test pairs. Average sentence length is 14.92 English words and 16.27 Hindi words; the training subset has 66,732 distinct English and 70,703 distinct Hindi word types.
- **Pre-processing:** whitespace normalisation; keep pairs with 3-40 words per side, plausible length ratio, mostly-Devanagari Hindi and Devanagari-free English; exact-duplicate removal; any training pair whose English sentence appears in validation or test is removed (no leakage). Random subset with seed 42 because of free-GPU limits.
- Splits are saved in `data/*.tsv`.

**3. Models implemented**

- **Basic Transformer (scratch):** 3 encoder + 3 decoder layers, d_model 256, 4 heads, feed-forward 1024, pre-layer-norm, tied embeddings, joint SentencePiece BPE vocabulary (8,000). Cross-entropy with label smoothing 0.1, AdamW with warm-up and inverse-square-root decay, 15 epochs, best-validation checkpoint.
- **mT5-small:** `google/mt5-small` fine-tuned with the prefix `translate English to Hindi:`, Adafactor (lr 1e-3), fp32, 2 epochs.
- **Llama-3.2-1B-Instruct and Gemma-3-1B-it:** (a) 3-shot prompting with fixed examples from the validation set; (b) LoRA fine-tuning (rank 16, alpha 32, all attention and MLP projections) on 10,000 training pairs for 1 epoch, loss on the Hindi side only. Same prompt format for both models.
- **Hugging Face repositories actually used:** `google/mt5-small`, `unsloth/Llama-3.2-1B-Instruct` and `unsloth/gemma-3-1b-it`. The two `unsloth/...` repositories are ungated re-uploads of the Llama 3.2 1B Instruct and Gemma 3 1B instruction-tuned models, so the official `meta-llama/` and `google/` repositories were not loaded in this run.
- Decoding for the main comparison: greedy, identical test sentences for every system.

**4. Execution steps and dependencies**

1. Open `FGAI_Assignment3_EN_HI_Machine_Translation.ipynb` on Kaggle (or Google Colab) with a **GPU (T4)** runtime and Internet enabled.
2. **Run all** (about 3 hours on a T4 in my run). A Hugging Face login is optional because the notebook falls back to the ungated `unsloth/...` copies of Llama and Gemma.
3. Results are written to `results/`, `plots/` and `samples/`; the splits are in `data/`.

Dependencies: see `requirements.txt` (PyTorch, transformers, peft, datasets, sentencepiece, sacrebleu, pandas, matplotlib).

Hardware used: Tesla T4 GPU (Kaggle).

**5. Results obtained**

**Main comparison (test set: 500 sentences, greedy decoding, Tesla T4)**

| System | Architecture | BLEU | chrF | Params total (M) | Params trainable (M) | Train time (s) | Inference, test set (s) | ms / sentence |
|---|---|---|---|---|---|---|---|---|
| Transformer (scratch) | enc-dec | 2.76 | 21.66 | 7.6 | 7.58 | 584 | 4.9 | 10 |
| mT5-small (fine-tuned) | enc-dec | 5.89 | 25.71 | 300.2 | 300.18 | 3377 | 18.4 | 37 |
| Llama-3.2-1B (few-shot) | decoder-only | 8.97 | 33.39 | 1235.8 | 0.00 | 0 | 97.4 | 195 |
| Llama-3.2-1B (LoRA) | decoder-only | 9.80 | 35.39 | 1247.1 | 11.27 | 380 | 144.6 | 289 |
| Gemma-3-1B (few-shot) | decoder-only | 12.00 | 38.10 | 999.9 | 0.00 | 0 | 149.5 | 299 |
| Gemma-3-1B (LoRA) | decoder-only | 13.21 | 39.04 | 1012.9 | 13.05 | 1474 | 175.8 | 352 |

**Temperature sweep (BLEU / chrF on first 100 test sentences)**

| System | greedy | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|
| Transformer (scratch) | 2.41 / 21.88 | 2.88 / 21.8 | 1.26 / 20.09 | 0.74 / 17.51 |
| mT5-small (fine-tuned) | 4.64 / 24.25 | 4.31 / 24.76 | 2.5 / 22.44 | 1.68 / 21.29 |
| Llama-3.2-1B (few-shot) | 10.11 / 33.35 | 7.52 / 31.08 | 5.75 / 28.88 | 1.87 / 20.32 |
| Llama-3.2-1B (LoRA) | 8.53 / 33.68 | 9.26 / 33.33 | 5.44 / 26.79 | 1.62 / 19.66 |
| Gemma-3-1B (few-shot) | 10.93 / 35.51 | 11.51 / 35.56 | 11.81 / 35.44 | 10.34 / 34.26 |
| Gemma-3-1B (LoRA) | 12.29 / 36.53 | 11.12 / 35.69 | 8.25 / 31.51 | 5.32 / 23.92 |

**Red-flag rates (% of test sentences)**

| System | latin |
