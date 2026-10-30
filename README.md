# English → Hindi Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma

Foundations of Generative AI (ICT4403) - Assignment 3

**Name:** TODO your name
**Registration No.:** TODO your registration number

> Everything marked **TODO** must be replaced after you run the notebook. Delete this note and all TODO markers before submitting.

## 1. Problem statement and objective
Build and compare four machine-translation systems for **English → Hindi**, two encoder-decoder and two decoder-only:

| System | Architecture | Adaptation |
|---|---|---|
| Basic Transformer | encoder-decoder | trained from scratch |
| mT5-small | encoder-decoder | fine-tuned |
| Llama-3.2-1B-Instruct | decoder-only | few-shot prompting and LoRA fine-tuning |
| Gemma-3-1B-it | decoder-only | few-shot prompting and LoRA fine-tuning |

All systems translate the same held-out test sentences. They are compared on translation quality (BLEU, chrF) and computational cost (training time, inference time, parameter count). The translations are then examined for bias, hallucination, cultural misrepresentation and sensitive-content errors, and mitigations are proposed.

## 2. Dataset
- **Source:** IIT Bombay English-Hindi Parallel Corpus, <https://huggingface.co/datasets/cfilt/iitb-english-hindi> (original project page: <https://www.cfilt.iitb.ac.in/iitb_parallel/>). About 1.6 million sentence pairs from government, news and web sources.
- **TODO:** if the notebook printed that it used the fallback `Helsinki-NLP/opus-100` (`en-hi`), replace the link above with <https://huggingface.co/datasets/Helsinki-NLP/opus-100> and say so here.
- **Subset used (from `results/dataset_stats.json`):** TODO train / validation / test pair counts, average English and Hindi sentence length.
- **Pre-processing:** whitespace normalisation; keep pairs with 3-40 words per side, plausible length ratio, mostly-Devanagari Hindi and Devanagari-free English; exact-duplicate removal; any training pair whose English sentence appears in validation or test is removed (no leakage). Random subset with seed 42 because of free-GPU limits.
- Splits are saved in `data/*.tsv`.

## 3. Models implemented
- **Basic Transformer (scratch):** 3 encoder + 3 decoder layers, d_model 256, 4 heads, feed-forward 1024, pre-layer-norm, tied embeddings, joint SentencePiece BPE vocabulary (8,000). Cross-entropy with label smoothing 0.1, AdamW with warm-up and inverse-square-root decay, 15 epochs, best-validation checkpoint.
- **mT5-small:** `google/mt5-small` fine-tuned with the prefix `translate English to Hindi:`, Adafactor (lr 1e-3), fp32, 2 epochs.
- **Llama-3.2-1B-Instruct and Gemma-3-1B-it:** (a) 3-shot prompting with fixed examples from the validation set; (b) LoRA fine-tuning (rank 16, alpha 32, all attention and MLP projections) on 10,000 training pairs for 1 epoch, loss on the Hindi side only. Same prompt format for both models.
- **TODO:** state the exact Hugging Face repositories used (printed in the notebook and listed in `results/README_results_snippet.md`; Llama and Gemma may have fallen back to the `unsloth/...` copies of the same weights).
- Decoding for the main comparison: greedy, identical test sentences for every system.

## 4. Execution steps and dependencies
1. Create a free Hugging Face account and token, and accept the licences for `meta-llama/Llama-3.2-1B-Instruct` and `google/gemma-3-1b-it`.
2. Open `FGAI_Assignment3_EN_HI_Machine_Translation.ipynb` in Google Colab with a **T4 GPU** runtime and **Run all** (about 2 hours). Paste your Hugging Face token when prompted.
3. Results are written to `results/`, `plots/` and `samples/`.

Dependencies: see `requirements.txt` (PyTorch, transformers, peft, datasets, sentencepiece, sacrebleu, pandas, matplotlib).
Hardware used: TODO (printed in the results table, e.g. Tesla T4).

## 5. Results obtained
TODO: paste the contents of `results/README_results_snippet.md` here (main comparison table, temperature sweep, red-flag rates, models used). Figures: `plots/02_system_comparison.png`, `plots/03_training_curves.png`, `plots/04_temperature_sweep.png`. Full translations of the test set are in `samples/`.

## 6. Model comparison and evaluation
TODO: write 2-4 short paragraphs in your own words, backed by the table above:
- Which system has the best BLEU/chrF, and by how much? Which differences are large enough to matter (single run, 500 test sentences, one reference)?
- Encoder-decoder vs decoder-only: compare quality against cost (parameters, trainable parameters, training time, inference time) and explain the gaps through model size, pretraining, tokenization (how many tokens each tokenizer needs for Hindi) and adaptation method.
- Few-shot vs LoRA for the same LLM: what did fine-tuning add, and at what cost?
- From-scratch Transformer vs pretrained models: what does pretraining buy with this little data?

## 7. Sampling-temperature analysis
TODO: using `results/temperature_sweep.csv` and `plots/04_temperature_sweep.png`, explain how BLEU/chrF change from greedy to T = 0.3, 0.7, 1.0 for each system, quote one or two example outputs, and explain why randomness usually hurts translation accuracy.

## 8. Responsible and ethical AI analysis
TODO: using `samples/probe_translations.txt`, `results/red_flag_rates.csv` and the flagged outputs printed in the notebook, discuss with **real examples from your outputs**:
- **Bias:** gender and occupation assumptions in the pronoun and noun forms chosen.
- **Hallucination:** content added, dropped or repeated.
- **Cultural misrepresentation:** idioms translated literally, register (आप / तुम), religious and cultural terms.
- **Sensitive content:** medical dosage, legal and violent sentences, numbers and dates.
- **Mitigations:** human review for high-stakes text, quality-estimation flags, glossaries, gender-alternative outputs, bias test suites, disclosure of machine translation, bilingual reviewer before deployment, monitoring and feedback channels.

## 9. Conclusion
TODO: name the best-performing system and justify it with your numbers (quality and cost), and say what you would recommend for real use and why.

## 10. Limitations
Single run per system; a 500-sentence test subset with one reference per sentence; small models and a small training subset because of free-GPU limits; LoRA models trained on 10,000 pairs while the from-scratch Transformer and mT5 used the full 50,000-pair subset; BLEU/chrF do not capture adequacy, bias or safety.

## Repository layout
`FGAI_Assignment3_EN_HI_Machine_Translation.ipynb` (code and outputs), `data/` (splits), `results/` (metrics, tables, per-system JSON), `plots/`, `samples/` (translations and probe outputs), `requirements.txt`.
