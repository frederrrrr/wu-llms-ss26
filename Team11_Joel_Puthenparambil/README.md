# Team11 — Joel Puthenparambil
## WU LLM Course SS26 — Austrian Tax Law Q&A

**Task:** Given a German-language Austrian tax law question, generate a 1–4 sentence answer in German citing the relevant statute (e.g. `§ 7 Abs 1 KStG`).

**Dataset:** `dataset_clean.csv` — 643 questions, used only for inference (test set).

---

## Models

### Model 1 — Inference Only

| Property | Value |
|----------|-------|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Parameters | 7 billion |
| Quantization | 4-bit (bitsandbytes `BitsAndBytesConfig`) |
| Sampling | `do_sample=True`, `temperature=0.7`, `top_p=0.9`, `max_new_tokens=200` |
| Platform | Google Colab (free T4 GPU) |
| Notebook | `code/model1_inference.ipynb` |
| Output | `results/model1_inference.csv` |

**Pre-training data:** Mistral-7B-Instruct-v0.2 was pre-trained by Mistral AI on a large multilingual web corpus and instruction-tuned on supervised fine-tuning (SFT) and RLHF data. The exact training set is not publicly disclosed.

**Approach:** The model is loaded in 4-bit quantization and prompted zero-shot with a shared system prompt instructing it to answer in German and cite Austrian statutes. No fine-tuning or retrieval.

**System prompt (same across all models):**
```
Beantworte die folgende Frage zum österreichischen Steuerrecht auf Deutsch.
Antworte in maximal 1–4 Sätzen.
Nenne die einschlägige Rechtsnorm (z.B. § 7 Abs 1 KStG).
Halluziniere keine Paragraphen. Wenn unklar, formuliere vorsichtig.
```

---

### Model 2 — Fine-Tuned

| Property | Value |
|----------|-------|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Parameters | 1.1 billion |
| Method | LoRA (PEFT), fp16, no quantization |
| LoRA rank | r=16, alpha=32, target modules: q_proj, v_proj |
| Epochs | 3 |
| Batch size | 4 |
| Learning rate | 2e-4 (AdamW default) |
| Training examples | 758 |
| Platform | Lightning.ai (free T4 GPU) |
| Notebook | `code/model2_finetune.ipynb` |
| Output | `results/model2_finetuned.csv` |

**Pre-training data:** TinyLlama-1.1B-Chat-v1.0 was pre-trained on ~3 trillion tokens from SlimPajama and StarCoder, then instruction-tuned on UltraChat and UltraFeedback by the TinyLlama team.

**Fine-tuning data:** 758 template-generated German Q&A pairs created from Austrian statute text scraped from RIS (ris.bka.gv.at). Statutes: EStG 1988, KStG 1988, UStG 1994, GrEStG 1987, BAO. The `dataset_clean.csv` test set was NOT used for training.

**Inference parameters:** `repetition_penalty=1.3`, `do_sample=True`, `temperature=0.5`, `top_p=0.9`, `max_new_tokens=200`.

---

### Model 3 — RAG (Retrieval-Augmented Generation)

| Property | Value |
|----------|-------|
| Retrieval model | `paraphrase-multilingual-mpnet-base-v2` |
| Index | FAISS IndexFlatIP (cosine similarity) |
| Chunk size | ~500 characters |
| Top-K passages | 3 |
| Generator | `gemini-2.5-flash-lite` (Google AI Studio) |
| RAG corpus | EStG 1988, KStG 1988, UStG 1994, GrEStG 1987, BAO |
| Platform | Google Colab (CPU) |
| Notebook | `code/model3_rag.ipynb` |
| Output | `results/model3_rag.csv` |

**Pre-training data:** `paraphrase-multilingual-mpnet-base-v2` was pre-trained by the sentence-transformers team on multilingual paraphrase data (50+ languages). Gemini 2.5 Flash-Lite is a Google proprietary model; training data is not publicly disclosed.

**RAG pipeline:**
1. Scrape Austrian statute text from RIS (ris.bka.gv.at) by law ID
2. Split into ~500-character passages
3. Embed all passages with `paraphrase-multilingual-mpnet-base-v2`
4. At inference time: embed the question, retrieve top-3 passages via cosine similarity (FAISS)
5. Pass question + 3 retrieved passages to Gemini 2.5 Flash-Lite for generation

---

## Evaluation

### Methodology

Reference answers come from the course dataset (golden-standard answers created by the course team). All three model outputs are evaluated against these using **BERTScore** (`xlm-roberta-base`, German).

BERTScore measures semantic similarity using contextual embeddings, making it well-suited for German legal text where paraphrasing is common and exact n-gram matches are rare.

Evaluation notebook: `code/evaluate.ipynb` — Results: `results/evaluation_report.csv`

### Results

| Model | BERTScore Precision | BERTScore Recall | BERTScore F1 |
|-------|---------------------|------------------|--------------|
| Model 1 — Mistral-7B (inference) | 0.8372 | 0.8544 | 0.8454 |
| Model 2 — TinyLlama (fine-tuned) | 0.8306 | 0.8193 | 0.8246 |
| Model 3 — RAG + Gemini | 0.8515 | 0.8688 | **0.8597** |

**Model 3 performs best.** The RAG approach scores highest on all three BERTScore dimensions, followed by Model 1, then Model 2.

---

## Error Analysis

**Model 1 (Mistral-7B):** Generally produces grammatically correct German with plausible statute citations. Occasionally cites paragraphs that do not exist or conflates §-numbers across laws (hallucination). Answers tend to be longer than necessary.

**Model 2 (TinyLlama fine-tuned):** The 1.1B parameter base model is small for this domain. Answers sometimes drift off-topic or repeat phrases. Fine-tuning on 758 template-generated pairs improved citation format consistency but did not fully prevent hallucination. The main quality bottleneck is model capacity — the LoRA configuration is standard but the base model is too small to reliably handle Austrian legal terminology.

**Model 3 (RAG + Gemini):** Highest answer quality overall. Retrieved statute passages ground the generator in real legal text, strongly reducing hallucination. Failures occur mainly when the relevant statute is not among the five scraped laws (e.g. niche sub-regulations), causing the retriever to return loosely related passages and the generator to produce a vague answer.

**Common pattern across all models:** Questions that require reasoning across multiple statutes (e.g. combining EStG and BAO provisions) are harder for all three models. Models 1 and 2 tend to cite only one statute in such cases; Model 3 handles it better when both laws are in the index.

---

## Repository Structure

```
Team11_Joel_Puthenparambil/
├── code/
│   ├── model1_inference.ipynb   # Mistral-7B 4-bit inference
│   ├── model2_finetune.ipynb    # TinyLlama LoRA fine-tuning + inference
│   ├── model3_rag.ipynb         # FAISS retrieval + Gemini generation
│   └── evaluate.ipynb           # BERTScore evaluation
└── results/
    ├── model1_inference.csv     # 643 rows
    ├── model2_finetuned.csv     # 643 rows
    ├── model3_rag.csv           # 643 rows
    └── evaluation_report.csv   # BERTScore results for all 3 models
```

---

## Reproduction

All notebooks are self-contained and run on Google Colab free tier (T4 GPU for Models 1 and 2, CPU for Model 3). Upload `dataset_clean.csv` when prompted.

**Model 3 only:** requires a Google AI Studio API key with billing enabled (free tier: 20 requests/day, insufficient for 643 questions).
