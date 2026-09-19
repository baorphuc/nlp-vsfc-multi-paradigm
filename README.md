# Vietnamese Student Feedback: Multi-Paradigm NLP Comparison

Comparing three NLP paradigms — Classic ML, fine-tuned Transformer, and zero-shot LLM prompting — on a multi-task Vietnamese text classification problem (sentiment + topic), using the UIT-VSFC corpus.

## Overview

University student feedback contains two signals worth extracting automatically: **sentiment** (negative / neutral / positive) and **topic** (lecturer / training_program / facility / others). Rather than building a single model, this project asks a comparative question: *given the same data and the same task, how do three fundamentally different NLP approaches trade off against each other?*

| Approach | Training required | Representation |
|---|---|---|
| TF-IDF + SVM | Yes (seconds) | Word frequency |
| PhoBERT (multi-task fine-tune) | Yes (GPU, ~minutes) | Contextual embedding |
| Gemini (zero-shot prompting) | No | Pretrained LLM reasoning |

## Key findings

- **PhoBERT wins overall** — Macro-F1 0.83 (sentiment) / 0.81 (topic) on the full 3,166-sentence test set, a consistent improvement over SVM across every class, largest on the rare `neutral` class (+0.21 F1).
- **SVM and Gemini land at nearly the same Macro-F1** (~0.70–0.77) on a stratified 400-sentence subset — despite opposite philosophies: SVM trains directly on the data but only sees word frequency, while Gemini never sees the dataset at all. The gap between "must train on labeled data" and "zero-shot" is smaller than expected.
- **Gemini beats PhoBERT on recall for the `neutral` class** (0.67 vs. 0.43), despite lower overall F1. PhoBERT, fine-tuned on a training set where `neutral` is only 4.3% of the data, learns a bias toward *not* predicting it. Gemini, with no exposure to that class distribution, makes the call purely on semantics — a concrete illustration of how fine-tuning on imbalanced data can bake in the imbalance itself.
- **51.5% of `neutral` sentences and 51% of `others` sentences are misclassified** by PhoBERT — traced back to manual analysis of 46 hand-annotated examples showing these are largely *neutral descriptive statements* ("materials were provided") that models confuse with positive sentiment due to surface-level positive words.
- Manual annotation also surfaced a labeling pattern: sentences phrased as **constructive suggestions** ("more equipment is needed") are consistently labeled `negative` even with zero negative vocabulary — sentiment here is implicit, tied to "desire for change" rather than word choice.

## Repository structure

```
notebooks/
  00_download_data.ipynb      Load UIT-VSFC from HuggingFace
  01_eda.ipynb                 Class distribution, sentence length analysis
  02_manual_analysis.ipynb     46 hand-annotated examples, linguistic patterns
  03_phobert_train.ipynb       PhoBERT multi-task fine-tuning (GPU)
  04_classic_ml.ipynb          TF-IDF + SVM baseline
  05_gemini_llm.ipynb          Zero-shot prompting with Gemini API

data/processed/
  manual_sample_40_50.csv      46 manually annotated sentences
  full_comparison.csv          SVM vs PhoBERT, per-class metrics (full test set)
  three_way_comparison.csv     SVM vs PhoBERT vs Gemini (400-sentence subset)

results/
  f1_comparison_chart.png
  confusion_matrices.png
  final_metrics.json

docs/
  BAO_CAO_DO_AN_NLP.md          Full report (Vietnamese)
```

## Method

**Data**: [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) — 16,175 sentences, single-label sentiment and topic annotations. Heavily imbalanced: `neutral` (4.3%) and `facility` (4.4%) are rare classes.

**Preprocessing**: Vietnamese word segmentation via `pyvi` (required for PhoBERT/TF-IDF, since Vietnamese compound words change meaning if split at the syllable level).

**Models**:
- `LinearSVC` on TF-IDF (unigram + bigram, 10k features), trained independently per task
- `vinai/phobert-base` backbone with two linear heads (sentiment, topic), joint loss, fine-tuned on Colab T4
- `gemini-3.1-flash-lite`, zero-shot, JSON-structured prompt with task definitions

**Evaluation**: Macro-F1 as the primary metric (not accuracy), chosen specifically because of the class imbalance — accuracy alone rewards a model for ignoring rare classes.

## Limitations

- Single-label topic annotation doesn't capture sentences that discuss multiple aspects at once (e.g. "spacious classroom, enthusiastic lecturer" — labeled only `facility`).
- The `others` topic class has no distinct vocabulary signature and shows signs of inconsistent original labeling.
- Gemini evaluation is limited to a 400-sentence stratified subset due to free-tier API rate limits.

## Stack

Python · PyTorch · Transformers · scikit-learn · pyvi · Gemini API · pandas · matplotlib