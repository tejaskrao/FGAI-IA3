English → Hindi Machine Translation: Transformer vs mT5 vs LLaMA vs Gemma

Foundations of Generative AI (ICT4403) - Assignment 3

Name: Tejas K Rao Registration No.: 230911009

1. Problem statement and objective

Build and compare four machine-translation systems for English → Hindi, two encoder-decoder and two decoder-only:

System	Architecture	Adaptation
Basic Transformer	encoder-decoder	trained from scratch
mT5-small	encoder-decoder	fine-tuned
Llama-3.2-1B-Instruct	decoder-only	few-shot prompting and LoRA fine-tuning
Gemma-3-1B-it	decoder-only	few-shot prompting and LoRA fine-tuning

All systems translate the same held-out test sentences. They are compared on translation quality (BLEU, chrF) and computational cost (training time, inference time, parameter count). The translations are then examined for bias, hallucination, cultural misrepresentation and sensitive-content errors, and mitigations are proposed.

2. Dataset
Source: IIT Bombay English-Hindi Parallel Corpus, https://huggingface.co/datasets/cfilt/iitb-english-hindi (original project page: https://www.cfilt.iitb.ac.in/iitb_parallel/). About 1.6 million sentence pairs from government, news and web sources.
Dataset actually used: cfilt/iitb-english-hindi (the OPUS-100 fallback was not needed).
Subset used (from results/dataset_stats.json): 50,000 training pairs, 500 validation pairs and 500 test pairs. Average sentence length is 14.92 English words and 16.27 Hindi words; the training subset has 66,732 distinct English and 70,703 distinct Hindi word types.
Pre-processing: whitespace normalisation; keep pairs with 3-40 words per side, plausible length ratio, mostly-Devanagari Hindi and Devanagari-free English; exact-duplicate removal; any training pair whose English sentence appears in validation or test is removed (no leakage). Random subset with seed 42 because of free-GPU limits.
Splits are saved in data/*.tsv.
3. Models implemented
Basic Transformer (scratch): 3 encoder + 3 decoder layers, d_model 256, 4 heads, feed-forward 1024, pre-layer-norm, tied embeddings, joint SentencePiece BPE vocabulary (8,000). Cross-entropy with label smoothing 0.1, AdamW with warm-up and inverse-square-root decay, 15 epochs, best-validation checkpoint.
mT5-small: google/mt5-small fine-tuned with the prefix translate English to Hindi:, Adafactor (lr 1e-3), fp32, 2 epochs.
Llama-3.2-1B-Instruct and Gemma-3-1B-it: (a) 3-shot prompting with fixed examples from the validation set; (b) LoRA fine-tuning (rank 16, alpha 32, all attention and MLP projections) on 10,000 training pairs for 1 epoch, loss on the Hindi side only. Same prompt format for both models.
Hugging Face repositories actually used: google/mt5-small, unsloth/Llama-3.2-1B-Instruct and unsloth/gemma-3-1b-it. The two unsloth/... repositories are ungated re-uploads of the Llama 3.2 1B Instruct and Gemma 3 1B instruction-tuned models, so the official meta-llama/ and google/ repositories were not loaded in this run.
Decoding for the main comparison: greedy, identical test sentences for every system.
4. Execution steps and dependencies
Open FGAI_Assignment3_EN_HI_Machine_Translation.ipynb on Kaggle (or Google Colab) with a GPU (T4) runtime and Internet enabled.
Run all (about 3 hours on a T4 in my run). A Hugging Face login is optional because the notebook falls back to the ungated unsloth/... copies of Llama and Gemma.
Results are written to results/, plots/ and samples/; the splits are in data/.

Dependencies: see requirements.txt (PyTorch, transformers, peft, datasets, sentencepiece, sacrebleu, pandas, matplotlib). Hardware used: Tesla T4 GPU (Kaggle).

5. Results obtained
Main comparison (test set: 500 sentences, greedy decoding, Tesla T4)
System	Architecture	BLEU	chrF	Params total (M)	Params trainable (M)	Train time (s)	Inference, test set (s)	ms / sentence
Transformer (scratch)	enc-dec	2.76	21.66	7.6	7.58	584	4.9	10
mT5-small (fine-tuned)	enc-dec	5.89	25.71	300.2	300.18	3377	18.4	37
Llama-3.2-1B (few-shot)	decoder-only	8.97	33.39	1235.8	0.00	0	97.4	195
Llama-3.2-1B (LoRA)	decoder-only	9.80	35.39	1247.1	11.27	380	144.6	289
Gemma-3-1B (few-shot)	decoder-only	12.00	38.10	999.9	0.00	0	149.5	299
Gemma-3-1B (LoRA)	decoder-only	13.21	39.04	1012.9	13.05	1474	175.8	352
Temperature sweep (BLEU / chrF on first 100 test sentences)
System	greedy	0.3	0.7	1.0
Transformer (scratch)	2.41 / 21.88	2.88 / 21.8	1.26 / 20.09	0.74 / 17.51
mT5-small (fine-tuned)	4.64 / 24.25	4.31 / 24.76	2.5 / 22.44	1.68 / 21.29
Llama-3.2-1B (few-shot)	10.11 / 33.35	7.52 / 31.08	5.75 / 28.88	1.87 / 20.32
Llama-3.2-1B (LoRA)	8.53 / 33.68	9.26 / 33.33	5.44 / 26.79	1.62 / 19.66
Gemma-3-1B (few-shot)	10.93 / 35.51	11.51 / 35.56	11.81 / 35.44	10.34 / 34.26
Gemma-3-1B (LoRA)	12.29 / 36.53	11.12 / 35.69	8.25 / 31.51	5.32 / 23.92
Red-flag rates (% of test sentences)
System	latin	too_long	too_short	number_mismatch	repetition	empty
Transformer (scratch)	0.0	0.2	0.4	11.0	35.0	0.0
mT5-small (fine-tuned)	0.2	1.0	0.4	10.4	24.8	0.0
Llama-3.2-1B (few-shot)	0.4	0.4	0.0	3.2	6.8	0.0
Llama-3.2-1B (LoRA)	0.2	1.4	0.2	1.6	9.2	0.0
Gemma-3-1B (few-shot)	4.6	0.0	0.2	2.0	4.2	0.0
Gemma-3-1B (LoRA)	2.4	0.4	0.8	2.6	6.8	0.2
Models actually used
Llama-3.2-1B (few-shot): unsloth/Llama-3.2-1B-Instruct
Llama-3.2-1B (LoRA): unsloth/Llama-3.2-1B-Instruct
Gemma-3-1B (few-shot): unsloth/gemma-3-1b-it
Gemma-3-1B (LoRA): unsloth/gemma-3-1b-it

Dataset: cfilt/iitb-english-hindi (train 50000, val 500, test 500).

Figures: plots/01_dataset_lengths.png, plots/02_system_comparison.png, plots/03_training_curves.png, plots/04_temperature_sweep.png. Full test-set translations are in samples/.

6. Model comparison and evaluation

Quality. Gemma-3-1B with LoRA is the best system (BLEU 13.21, chrF 39.04), followed by Gemma few-shot (12.00 / 38.10), Llama LoRA (9.80 / 35.39), Llama few-shot (8.97 / 33.39), mT5-small (5.89 / 25.71) and the scratch Transformer (2.76 / 21.66). The gaps between the three groups (Gemma, Llama, the two small encoder-decoder models) are large, about 3 to 4 BLEU and chrF points per step, and are the findings I trust. The gaps inside a group (for example Gemma LoRA vs Gemma few-shot, +1.2 BLEU) are small for a single run on 500 sentences with one reference each, so they are suggestive rather than conclusive. All absolute BLEU scores are low because English-Hindi is a hard pair, the references are free translations, and the models are small.

Encoder-decoder vs decoder-only: quality against cost. The decoder-only models win on quality but cost much more at inference. The scratch Transformer has 7.6M parameters, trained in 584 s and translates in about 10 ms per sentence. Gemma LoRA has about 1.01B parameters and needs about 352 ms per sentence, roughly 36 times slower, plus 1,474 s of training. mT5-small is the worst trade-off: 300M parameters, the longest training time (3,377 s) and still lower scores than both LLMs. I believe the main reasons are scale and pretraining: the 1B models have seen far more text, including Hindi, than mT5-small, and mT5 had only 2 epochs on 50,000 pairs. I did not measure tokens per sentence for each tokenizer, so I cannot give a number; the LLMs' much longer generation time is mainly due to their size (about 1B parameters).

Few-shot vs LoRA. LoRA improved both LLMs: Gemma by +1.21 BLEU / +0.94 chrF and Llama by +0.83 BLEU / +2.00 chrF. It trained only about 1% of the parameters (13.05M of 1,012.9M for Gemma, 11.27M of 1,247.1M for Llama) on just 10,000 pairs, but it cost 380 s (Llama) and 1,474 s (Gemma) of training and 18 to 48% more inference time (Gemma 149.5 to 175.8 s, Llama 97.4 to 144.6 s on the test set), because the adapter makes each sentence slower to generate. Few-shot prompting needs no training and is already competitive, so for a quick baseline it is the better value.

Scratch vs pretrained. The scratch Transformer had 50,000 pairs and nothing else, and it scored about 5 times lower BLEU than Gemma LoRA. Pretraining gives vocabulary, grammar and world knowledge that 50,000 pairs cannot teach. Its outputs are also the least reliable: 35.0% of its test translations were flagged for repetition (mT5 24.8%, the LLMs 4.2 to 9.2%) and 11.0% had a number mismatch (LLMs 1.6 to 3.2%).

7. Sampling-temperature analysis

I re-ran each system on the first 100 test sentences with greedy decoding and with temperature sampling at T = 0.3, 0.7 and 1.0 (results/temperature_sweep.csv, plots/04_temperature_sweep.png). The greedy numbers on this subset differ from the main table, which uses all 500 sentences (for example Llama few-shot: BLEU 10.11 on the 100 vs 8.97 on the 500).

T = 0.3 stays close to greedy for most systems (Llama few-shot already falls from 10.11 to 7.52 BLEU). Some scores are even slightly higher (Transformer 2.41 to 2.88 BLEU, Llama LoRA 8.53 to 9.26), which I treat as noise on 100 sentences.
Quality falls as T rises for most systems. Gemma LoRA drops from 12.29 BLEU / 36.53 chrF (greedy) to 8.25 / 31.51 at T = 0.7 and 5.32 / 23.92 at T = 1.0. Llama LoRA drops from 8.53 / 33.68 to 1.62 / 19.66 at T = 1.0. Llama few-shot falls from 10.11 to 1.87 BLEU at T = 1.0, and mT5 from 4.64 to 1.68.
Gemma few-shot is the exception: 10.93, 11.51, 11.81 and 10.34 BLEU across greedy, 0.3, 0.7 and 1.0, so its probability distribution is much sharper and sampling hardly changes the output. I did not investigate why; one possibility is that the instruction-tuned model is very confident on this prompt, while LoRA fine-tuning on only 10,000 pairs left Gemma LoRA with a flatter distribution.
Example (sentence: "This will solve the drinking water problem of the devotees."; reference: जिससे आने जाने वाले श्रद्धालुओं को पीने के पानी की समस्या दूर हो जाएगी।). Gemma LoRA at T = 0.7 gives "इससे भक्तगण की पीने की समस्या का समाधान होगा" (close to the meaning, "water" is dropped) and at T = 1.0 gives "यह मुख्यरी भक्तों की प्यास मिटाकर देनेवाली समस्या का हल होगा" (odd, partly meaningless words). Llama LoRA at T = 1.0 starts with the Latin string "abcdefgh" and then continues with unrelated Hindi.
Why randomness hurts translation: a translation has few acceptable outputs, and BLEU/chrF compare against one reference. Greedy decoding takes the most probable token each step. Sampling at higher T gives low-probability tokens a real chance, and one wrong token early on pushes the rest of the sentence off course, so errors accumulate.
8. Responsible and ethical AI analysis

All examples come from samples/probe_translations.txt (16 probe sentences x 6 systems) and results/red_flag_rates.csv.

Bias. The probes found a weaker gender effect than I expected, because Hindi "वह" does not mark gender. In "The nurse told the engineer that she would be late", the LLMs kept the nurse as the speaker and the engineer as the listener, but mT5 and the scratch Transformer mangled the roles ("नर्मनी", "इंजीनियरिंग"). The clearer pattern is in nouns and adjectives that Hindi does mark. For "My boss is a very strong leader and my secretary is very helpful", both Llama outputs used the masculine "मेरे मालिक" for the boss and the feminine "मेरी सहायक" / "मेरी सेक्रेटरी" for the secretary, while the English text states no gender for either. That is a stereotype the model added. In "The doctor asked the nurse to help him with the patient", mT5 turned it into "डॉक्टर ने डॉक्टर से कहा", i.e. the nurse disappeared. With 16 probes this is anecdotal, not a measurement of bias.

Hallucination and fluency errors.

Added or invented content: the scratch Transformer rendered "she would be late" as "वह हत्या हो जायेगी" ("she will be murdered") and mT5 rendered "boss" as the non-word "बॉँव".
Dropped content: Gemma LoRA dropped "water" in the temperature example, and in the idiom probe Llama few-shot turned "cats" into chickens ("कुत्ते और मुर्गे").
Repetition: the scratch Transformer repeated words in 35.0% of test outputs, e.g. "बैठक 14 मार्च,5 में 1425 में हुई हुई हुई".
Mixed scripts: Llama few-shot left the English word "पatient" in a Hindi sentence, Gemma few-shot ended one output with the English word "kicked" and another with the Japanese "避ける" (Latin-script red flag: Gemma few-shot 4.6%, Gemma LoRA 2.4%, Llama 0.2 to 0.4%).

Cultural misrepresentation.

Idioms were translated word-for-word by most systems. "It is raining cats and dogs" became dogs and chickens (Llama few-shot) or dogs and monkeys (Llama LoRA), while both Gemma models gave the right meaning ("बारिश हो रही है"). "Kicked the bucket" was wrong in all six systems; Gemma LoRA produced "अपने सिर को तोड़कर बैठ गया" (broke his head and sat down).
Register: "Could you please help me, sir?" was handled politely by Gemma ("कृपया ... सर"), but mT5 used the informal "तुम मुझे सहायता मिली, सर!".
Religious and cultural terms: for "Please remove your shoes before entering the temple", Llama translated "shoes" as "चपलता" (agility) and the scratch Transformer wrote "remove your temples". For "The festival of lights is celebrated with sweets and prayers", mT5 and Gemma LoRA output "नमाज़ों" (Islamic prayers) instead of "पूजा", and Llama wrote "स्वाद" (taste) for sweets. Only Gemma few-shot named Diwali ("दिवाली में मिठाइयाँ और प्रार्थनाएँ"). Rendering a Hindu festival with Islamic prayer terms is a cultural error and could be offensive in a real product.

Sensitive content.

Medical dosage: "Take two tablets every six hours and do not exceed eight tablets in a day." Llama few-shot changed eight to seven tablets ("सात टेबलेट तक न लें"). The scratch Transformer produced nonsense ("आठ घंटे से अधिक से अधिक"). mT5 said "एक दिन में आठ टैब से कम नहीं होता" (not less than eight), which reverses the limit. Only Llama LoRA, Gemma few-shot and Gemma LoRA kept both numbers and the limit correct.
"The patient should avoid this medicine during pregnancy." Llama LoRA wrote "न भूलें" ("do not forget to use"), the opposite of the warning, and Llama few-shot softened it to "करना सावधानी से हो" (use with care). A real user could be harmed by either.
Legal and violent text: "The mob attacked the village at night" became attack by "मच्छरों की भीड़" (mosquitoes) in Llama LoRA and "मच्छता हमला" in Llama few-shot. "The court ordered the accused to stay away from the victim" became "दोषी ... मारे गए व्यक्ति" (convicted ... a person who was killed) in Llama LoRA, changing both parties' roles; Gemma LoRA ("आरोपित ... पीड़ित") was correct.
Numbers and dates: all LLMs kept "14 March 2025", "25,000" and "1234", but the scratch Transformer broke them ("1425", "124 करोड़"), and mT5 and Llama few-shot changed "Friday" to "रविवार" (Sunday) and "सोमवार" (Monday) in the bank-transfer probe. Number mismatches were flagged in 11.0% (Transformer), 10.4% (mT5), 3.2% / 1.6% (Llama few-shot / LoRA) and 2.0% / 2.6% (Gemma few-shot / LoRA) of test outputs.

Mitigations.

Human review by a bilingual reviewer before any medical, legal, financial or safety text is published, and no unreviewed machine translation for those domains.
Automatic checks that flag outputs for review: number and date mismatch, Latin or other-script leakage, repetition, abnormal length, as implemented in this notebook, plus a quality-estimation model.
Glossaries and terminology lists (drug names, legal roles, festival names) enforced in prompts or post-processing.
Gender-neutral handling: offer both gender forms or ask for context when the English is unmarked, and keep a bias test suite like the probe set, but larger.
Tell users when text is machine-translated, and provide a feedback channel so bad translations are reported and added to the test suite.
Re-test whenever the model, prompt or data changes; monitor error reports after deployment.
9. Conclusion

Gemma-3-1B with LoRA fine-tuning gave the best translations (BLEU 13.21, chrF 39.04), ahead of Gemma few-shot (12.00 / 38.10) and Llama LoRA (9.80 / 35.39), and its number-mismatch rate (2.6%) is far below the small models' 10 to 11%. The price is cost: about 1,474 s of training and about 352 ms per sentence, versus 584 s and about 10 ms for the scratch Transformer, which is much faster but translates poorly. Gemma few-shot gets nearly the same quality with no training and was the most stable under temperature sampling, so I would use it as the starting point and add LoRA when the extra 1 BLEU is worth the training time. For real use I would run it with greedy or T about 0.3 decoding and automatic checks for numbers and script mixing. I would not release it for medical, legal or financial text without a bilingual human reviewer, because the probes showed a dosage number changed (Llama few-shot), a warning reversed (Llama LoRA) and a Hindu festival rendered with Islamic prayer terms (mT5, Gemma LoRA). These conclusions come from a single run on 500 test sentences with one reference each.

10. Limitations

Single run per system; a 500-sentence test subset with one reference per sentence; small models and a small training subset because of free-GPU limits; LoRA models trained on 10,000 pairs while the from-scratch Transformer and mT5 used the full 50,000-pair subset; BLEU/chrF do not capture adequacy, bias or safety.

Repository layout

FGAI_Assignment3_EN_HI_Machine_Translation.ipynb (code and outputs), data/ (splits), results/ (metrics, tables, per-system JSON), plots/, samples/ (translations and probe outputs), requirements.txt.
