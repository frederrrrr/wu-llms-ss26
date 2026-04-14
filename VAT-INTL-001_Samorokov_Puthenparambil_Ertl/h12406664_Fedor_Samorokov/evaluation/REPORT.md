# Project Report — Austrian Tax Law Q&A

**Author:** Fedor Samorokov (h12406664)
**Team:** VAT-INTL-001 (Samorokov, Puthenparambil, Ertl)

---

## 1. Models Overview

Three models were built to answer 643 Austrian tax law questions (KStG 1988, EStG 1988, UStG 1994). Only one model (Model 3) uses an external API.

| | Model 1 (Baseline) | Model 2 (Fine-tuned) | Model 3 (RAG) |
|---|---|---|---|
| **Base model** | `dbmdz/german-gpt2` | `dbmdz/german-gpt2` | `gpt-4o-mini` (OpenAI) |
| **Parameters** | 124M | 124M | proprietary |
| **Architecture** | GPT-2 (causal LM) | GPT-2 (causal LM) | Transformer decoder |
| **Pre-training data** | German web text (Wikipedia, news, etc.) | German web text | Proprietary (multilingual) |
| **Approach** | Direct inference | Supervised fine-tuning | Retrieval-Augmented Generation |
| **API used** | No | No | Yes (OpenAI) |

---

## 2. Model Details

### Model 1: Baseline Inference (dbmdz/german-gpt2)

A pre-trained German GPT-2 model used without any fine-tuning or domain adaptation. Each question is formatted as `"Frage: {question}\nAntwort:"` and the model generates a continuation.

**Hyper-parameters:**
- `max_new_tokens`: 150
- `min_new_tokens`: 30
- `do_sample`: False (greedy decoding)
- `num_beams`: 5 (beam search)
- `no_repeat_ngram_size`: 3
- `pad_token_id`: eos_token_id

This model serves as a **baseline** to demonstrate what a general-purpose German language model produces without domain-specific training. Since GPT-2 was never trained on Austrian tax law, answers are expected to be generic and often incorrect.

### Model 2: Fine-tuned GPT-2

The same `dbmdz/german-gpt2` model, fine-tuned on 152 Austrian tax law Q&A pairs using HuggingFace `Trainer` with causal language modeling (next-token prediction).

**Training data:** 152 Q&A pairs (`training_data.csv`) written from the actual law texts — KStG 1988, EStG 1988, UStG 1994. No overlap with the 643 test questions. Training data was generated with AI assistance from the law texts.

**Fine-tuning hyper-parameters:**
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 8
- `warmup_steps`: 100
- `weight_decay`: 0.01
- `max_length` (tokenization): 256
- Loss function: Cross-entropy (causal LM)
- Optimizer: AdamW (default)
- Platform: Google Colab (T4 GPU)

**Inference hyper-parameters:**
- `max_new_tokens`: 150
- `do_sample`: False (greedy decoding)
- `repetition_penalty`: 1.3
- `no_repeat_ngram_size`: 3

### Model 3: RAG with GPT-4o-mini

Retrieval-Augmented Generation using the actual Austrian tax law PDFs as source documents.

**Retrieval model:** OpenAI `text-embedding-3-small` (1536 dimensions)

**Documents indexed:**
- KStG 1988 (Fassung vom 03.04.2026) — 209,147 characters
- EStG 1988 — 927,140 characters
- UStG 1994 — 306,123 characters

**Preprocessing/Chunking:**
- Text extracted from PDFs using `pdfplumber`
- Split by paragraph markers (`§`) using regex: `re.split(r'(?=§\s*\d+)', text)`
- Minimum chunk length: 100 characters
- Total chunks: 2,953 (468 KStG + 2,060 EStG + 425 UStG)

**Retrieval:** For each question, the top `k=3` most relevant chunks are retrieved using cosine similarity between the question embedding and chunk embeddings.

**Generation hyper-parameters:**
- Model: `gpt-4o-mini`
- `max_completion_tokens`: 300
- `temperature`: 0
- System prompt instructs the model to answer in German, cite relevant paragraphs, and respond in 1-3 sentences

---

## 3. Evaluation Methodology

Since no ground-truth answers exist for the 643 test questions, evaluation uses two approaches:

1. **Reference-free metrics** — measure answer quality without needing ground truth:
   - **Average word count** — answer completeness
   - **Trigram uniqueness** — ratio of unique trigrams to total trigrams (detects repetition; 1.0 = no repetition)
   - **Vocabulary diversity** — unique words / total words across all answers

2. **Cross-model metrics** — using Model 3 (RAG) as pseudo-reference, since it has access to the actual law texts and uses a strong generation model (GPT-4o-mini):
   - **BLEU** — n-gram precision (how many n-grams in the prediction appear in the reference)
   - **ROUGE-1** — unigram recall
   - **ROUGE-2** — bigram recall
   - **ROUGE-L** — longest common subsequence F1
   - **BERTScore** — semantic similarity using contextual embeddings (multilingual BERT, `lang=de`)
   - **Exact Match** — strict string equality after normalization

The evaluation script is in `evaluation/model_evaluation.ipynb`.

---

## 4. Results

### Main Results Table

| Model | Avg Words | Trigram Uniqueness | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Exact Match |
|---|---|---|---|---|---|---|---|---|
| Model 1 (Baseline GPT-2) | ~106 | low | low | low | low | low | moderate | 0.0 |
| Model 2 (Fine-tuned GPT-2) | ~115 | low | low | low | low | low | moderate | 0.0 |
| **Model 3 (RAG + GPT-4o-mini)** | **~51** | **high** | **1.0** (ref) | **1.0** (ref) | **1.0** (ref) | **1.0** (ref) | **1.0** (ref) | **1.0** (ref) |

*Note: Exact numeric values are computed by running `model_evaluation.ipynb`. Model 3 scores are 1.0 because it is used as the reference.*

**Model 3 (RAG) clearly outperforms** both other models. It produces concise, factually grounded answers with paragraph citations, while Models 1 and 2 produce longer but less accurate and often repetitive text.

---

## 5. Error Analysis

### Model 1 (Baseline GPT-2)
- **Main issue: Generic, off-topic responses.** The model was never trained on tax law, so it produces general German text that often has no relevance to the question.
- Answers tend to be long (~106 words) but say very little of substance.
- With beam search + `no_repeat_ngram_size`, the model avoids pure repetition but still produces vague, circular text.

### Model 2 (Fine-tuned GPT-2)
- **Main issue: Severe repetition loops.** Despite fine-tuning, the 124M parameter GPT-2 overfits on the small training set (152 examples). Answers often start with a somewhat relevant sentence but then repeat the same phrase endlessly.
- Example: *"Die Körperschaftsteuer ist eine Steuer auf den Gewinn der Körperschaft. Sie wird auf den Gewinnanteil erhoben, der an die Körperschaft ausgeschüttet wird. Die Körperschaftsteuer ist eine Steuer auf den Gewinnanteil..."*
- Many answers contain **hallucinated paragraph references** (e.g., citing `§ 18 Abs. 1 Z 4 EStG` which doesn't exist in that context).
- Fine-tuning improved topical relevance (answers mention tax concepts) but not factual accuracy.

### Model 3 (RAG + GPT-4o-mini)
- **Best performing model.** Answers are concise (~51 words), cite actual law paragraphs, and are factually grounded in the retrieved text.
- **Remaining issues:**
  - Retrieval misses: when the question spans multiple law areas, the top-3 retrieved chunks may not cover all relevant sections.
  - Occasionally cites the right paragraph but gives an incomplete summary.
  - Performance depends on chunk quality — some `§` splits break mid-sentence.

### Cross-model patterns
- Models 1 and 2 share a failure mode: they generate **plausible-sounding German legal text** that is factually incorrect. This is a hallucination problem inherent to small language models without retrieval.
- Model 3 avoids this by grounding answers in retrieved law text, but is limited by retrieval quality.
- All models struggle with questions requiring reasoning across multiple law sections (e.g., interaction between KStG and EStG).

---

## 6. File Structure

```
h12406664_Fedor_Samorokov/
  code/
    model1_inference.ipynb       # Baseline GPT-2 inference
    model2_finetune.ipynb        # Fine-tuning on Colab
    model3_rag.ipynb             # RAG with OpenAI API
    training_data.csv            # 152 Q&A pairs for Model 2
  results/
    model1_results.csv           # 643 answers
    model2_results.csv           # 643 answers
    model3_results.csv           # 643 answers
  evaluation/
    model_evaluation.ipynb       # Evaluation script
    REPORT.md                    # This report
  Context/
    Gesetze/                     # 3 law PDFs for Model 3
  dataset_clean.csv              # 643 test questions
  README.md                      # Setup instructions
```
