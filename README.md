# Memorability Ranker

A neural architecture for **clause-level memorability prediction** from narrative context.

Memorability Ranker estimates the probability that an individual clause will be recalled by humans using contextual language representations learned by **ModernBERT**, enhanced with **Clause Attention**, **Memory Projection**, and a learnable **Semantic Memory Module**.

Unlike approaches that rely on spaced-repetition logs or handcrafted heuristics, this model directly learns from **human narrative recall behavior**, making it suitable for applications where understanding intrinsic information retention is important.

---

## Motivation

Not all information is equally memorable.

Educational notes, articles, and narratives contain concepts that naturally vary in how well people remember them. Existing NLP systems typically treat every sentence with equal importance, despite decades of cognitive research showing that memory retention differs substantially across pieces of information.

Memorability Ranker aims to estimate this **intrinsic memorability** at the clause level, enabling downstream systems to adapt content based on predicted human recall.

Potential applications include:

-  Educational note optimization
-  Memorability-aware text rewriting
-  Flashcard generation and prioritization
-  Narrative analysis
-  Information saliency estimation
-  Human-centered NLP systems

---

## Model Architecture

```
Clause + Context
        │
        ▼
Tokenizer
        │
        ▼
ModernBERT Encoder
        │
        ▼
Clause Attention
        │
        ▼
Memory Projection
        │
        ▼
Semantic Memory Module
        │
        ▼
Regression Head
        │
        ▼
Memorability Score
```

### Components

| Module | Purpose |
|---------|---------|
| ModernBERT | Generates contextual representations for the clause and surrounding narrative. |
| Clause Attention | Learns which tokens contribute most to memorability prediction. |
| Memory Projection | Compresses contextual representations into a compact memory representation. |
| Semantic Memory Module | Retrieves and integrates semantically related memory patterns learned during training. |
| Regression Head | Predicts a continuous memorability score between 0 and 1. |

---

## Training Pipeline

1. Segment narratives into clauses.
2. Construct contextual windows around each target clause.
3. Tokenize the clause-context pair.
4. Encode the input using ModernBERT.
5. Aggregate informative tokens using Clause Attention.
6. Compress representations through Memory Projection.
7. Enhance representations using the Semantic Memory Module.
8. Train a regression head to predict normalized human recall probability.

---

## Dataset Format

Each training example follows the structure

```json
{
    "paragraph": "... surrounding narrative context ...",
    "target_clause": "... target clause ...",
    "memorability": 0.73
}
```

where

- **paragraph** – local narrative context surrounding the clause
- **target_clause** – clause whose memorability is predicted
- **memorability** – normalized human recall probability (0–1)

---

## Repository Structure

```
memorability-ranker
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dataset/
│   ├── dataset.py
│   └── tokenizer.py
│
├── models/
│   ├── modernbert.py
│   ├── clause_attention.py
│   ├── memory_projection.py
│   ├── semantic_memory.py
│   └── memorability_ranker.py
│
├── trainer/
│   ├── trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── collate.py
│
├── train.py
├── evaluate.py
├── cross_validate.py
├── config.py
├── README.md
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/<username>/memorability-ranker.git
cd memorability-ranker

uv sync
```

---

## Training

```bash
python train.py
```

---

## Evaluation

```bash
python evaluate.py
```

---

## Research Status

This repository is an active research implementation exploring clause-level memorability prediction from narrative text. The architecture and training procedure may evolve as additional experiments are conducted.

---

## License

Released under the MIT License.