### Main comparison (test set: 500 sentences, greedy decoding, Tesla T4)

| System | Architecture | BLEU | chrF | Params total (M) | Params trainable (M) | Train time (s) | Inference, test set (s) | ms / sentence |
|---|---|---|---|---|---|---|---|---|
| Transformer (scratch) | enc-dec | 2.76 | 21.66 | 7.6 | 7.58 | 584 | 4.9 | 10 |
| mT5-small (fine-tuned) | enc-dec | 5.89 | 25.71 | 300.2 | 300.18 | 3377 | 18.4 | 37 |
| Llama-3.2-1B (few-shot) | decoder-only | 8.97 | 33.39 | 1235.8 | 0.00 | 0 | 97.4 | 195 |
| Llama-3.2-1B (LoRA) | decoder-only | 9.80 | 35.39 | 1247.1 | 11.27 | 380 | 144.6 | 289 |
| Gemma-3-1B (few-shot) | decoder-only | 12.00 | 38.10 | 999.9 | 0.00 | 0 | 149.5 | 299 |
| Gemma-3-1B (LoRA) | decoder-only | 13.21 | 39.04 | 1012.9 | 13.05 | 1474 | 175.8 | 352 |

### Temperature sweep (BLEU / chrF on first 100 test sentences)

| System | greedy | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|
| Transformer (scratch) | 2.41 / 21.88 | 2.88 / 21.8 | 1.26 / 20.09 | 0.74 / 17.51 |
| mT5-small (fine-tuned) | 4.64 / 24.25 | 4.31 / 24.76 | 2.5 / 22.44 | 1.68 / 21.29 |
| Llama-3.2-1B (few-shot) | 10.11 / 33.35 | 7.52 / 31.08 | 5.75 / 28.88 | 1.87 / 20.32 |
| Llama-3.2-1B (LoRA) | 8.53 / 33.68 | 9.26 / 33.33 | 5.44 / 26.79 | 1.62 / 19.66 |
| Gemma-3-1B (few-shot) | 10.93 / 35.51 | 11.51 / 35.56 | 11.81 / 35.44 | 10.34 / 34.26 |
| Gemma-3-1B (LoRA) | 12.29 / 36.53 | 11.12 / 35.69 | 8.25 / 31.51 | 5.32 / 23.92 |

### Red-flag rates (% of test sentences)

| System | latin | too_long | too_short | number_mismatch | repetition | empty |
|---|---|---|---|---|---|---|
| Transformer (scratch) | 0.0 | 0.2 | 0.4 | 11.0 | 35.0 | 0.0 |
| mT5-small (fine-tuned) | 0.2 | 1.0 | 0.4 | 10.4 | 24.8 | 0.0 |
| Llama-3.2-1B (few-shot) | 0.4 | 0.4 | 0.0 | 3.2 | 6.8 | 0.0 |
| Llama-3.2-1B (LoRA) | 0.2 | 1.4 | 0.2 | 1.6 | 9.2 | 0.0 |
| Gemma-3-1B (few-shot) | 4.6 | 0.0 | 0.2 | 2.0 | 4.2 | 0.0 |
| Gemma-3-1B (LoRA) | 2.4 | 0.4 | 0.8 | 2.6 | 6.8 | 0.2 |

### Models actually used

- Llama-3.2-1B (few-shot): unsloth/Llama-3.2-1B-Instruct
- Llama-3.2-1B (LoRA): unsloth/Llama-3.2-1B-Instruct
- Gemma-3-1B (few-shot): unsloth/gemma-3-1b-it
- Gemma-3-1B (LoRA): unsloth/gemma-3-1b-it

Dataset: cfilt/iitb-english-hindi (train 50000, val 500, test 500).
