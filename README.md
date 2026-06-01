# crescendo-defense-llm

Benchmarking two inference-time defenses against Crescendo multi-turn jailbreak attacks on Llama-3.2-3B-Instruct.

**AIMS-DTU Research Internship 2026**

---

## Overview

[Crescendo](https://arxiv.org/abs/2404.01833) is a multi-turn jailbreak attack that gradually steers a language model toward harmful outputs through a sequence of individually benign-seeming messages. By the final turn, the model has been conditioned by its own prior responses — a phenomenon called *memory stacking* — making refusal feel inconsistent.

This project builds a controlled evaluation harness around **Llama-3.2-3B-Instruct** and measures how two lightweight, inference-time defense mechanisms affect Attack Success Rate (ASR) across 10 attack scenarios spanning 5 harm categories.

---

## Results

| Condition | ASR | Succeeded / Total |
|---|---|---|
| Baseline (No Defense) | **50.0%** | 5 / 10 |
| Defense 1 — Semantic Drift + Toxicity | **0.0%** | 0 / 10 |
| Defense 2 — Context Reset + Intent Classification | **10.0%** | 1 / 10 |

### Per-Category Breakdown

| Category | Baseline | Defense 1 | Defense 2 |
|---|---|---|---|
| Weapons (2) | 50% | 0% | 50% |
| Malware (2) | 50% | 0% | 0% |
| Social Engineering (2) | 0% | 0% | 0% |
| Harmful Substances (2) | 50% | 0% | 0% |
| Extremism (1) | 100% | 0% | 0% |
| Misinformation (1) | 100% | 0% | 0% |

---

## Repository Structure

```
crescendo-defense-llm/
├── day1_crescendo_harness.py      # Baseline: undefended attack runner
├── defense1_semantic_drift.py     # Defense 1: semantic drift + toxicity filtering
├── defense2_context_reset.py      # Defense 2: context reset + intent classification
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Defense Mechanisms

### Defense 1 — Semantic Drift + Toxicity Filtering

Encodes Turn 1 as a sentence embedding anchor (all-MiniLM-L6-v2). Each subsequent user message is compared against this anchor via cosine similarity. If similarity drops below **τ = 0.72**, or if a toxicity classifier (unitary/toxic-bert) scores the message above **0.65**, the turn is intercepted before reaching the main model.

- Model-agnostic — no modification to LLM weights
- Operates in sub-second time on CPU
- Adds zero tokens to the LLM context

### Defense 2 — Context Reset + Intent Classification

Targets the memory-stacking mechanism directly:

- **History truncation**: only the last 3 turns are kept in context (`MAX_HISTORY = 3`)
- **Periodic system prompt re-injection**: safety system prompt prepended every 2 turns (`RESET_EVERY_K = 2`)
- **Intent classification**: each user message is classified as `benign` or `adversarial` by a zero-shot LLM call before the main model is consulted

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Model Access

The model is gated. You need to:

1. Request access at [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. Create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Authenticate

### Running on Google Colab (recommended — free T4 GPU)

1. `Runtime → Change runtime type → T4 GPU`
2. Install dependencies:
```python
!pip install -r requirements.txt -q
```
3. Set your HF token, then run any script:
```python
!python crescendo_harness.py
!python defense1_semantic_drift.py
!python defense2_context_reset.py
```

Results are saved to `results/baseline/`, `results/defense1/`, and `results/defense2/` as JSON files after each scenario.

**Expected runtime per script:** ~30–45 minutes on a T4 GPU.

---

## Attack Scenarios

10 Crescendo attack scenarios across 5 harm categories, each with 4 escalating turns:

| ID | Category | Name |
|---|---|---|
| ATK-001 | Weapons | Historical Arms Escalation |
| ATK-002 | Weapons | Fiction to Explosives |
| ATK-003 | Malware | Student to Exploit Automation |
| ATK-004 | Malware | CEH Cert to Phishing Campaign |
| ATK-005 | Social Engineering | Psychology Research to Fraud Script |
| ATK-006 | Social Engineering | UX Dark Patterns to Predatory Script |
| ATK-007 | Harmful Substances | Lab Safety to Toxic Synthesis |
| ATK-008 | Harmful Substances | Harm Reduction NGO Reframe |
| ATK-009 | Extremism | Academic Radicalization to Recruitment Content |
| ATK-010 | Misinformation | Political Science to Election Disinfo |

---

## References

1. Russinovich et al. "Great, Now Write an Article About That": The Crescendo Multi-Turn LLM Jailbreak Attack. *arXiv:2404.01833*, 2024.
2. Reimers & Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*, 2019.
3. Hanu & Unitary team. Detoxify. GitHub, 2020.
