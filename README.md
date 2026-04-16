# Robot Replanning Under Failure — Executive Function Benchmark

> A Kaggle Community Benchmark evaluating executive functions in large language models using robotics-inspired failure recovery scenarios.

**Submitted to:** [Measuring Progress Toward AGI — Kaggle x Google DeepMind Hackathon 2026](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
**Track:** Executive Functions  
**Author:** Kamalaksh Dessai  
**Benchmark:** [kaggle.com/benchmarks/denu77/robot-replanning-executive-functions-benchmark](https://www.kaggle.com/benchmarks/denu77/robot-replanning-executive-functions-benchmark)  
**Task:** [kaggle.com/benchmarks/tasks/denu77/robot-replanning-executive-functions](https://www.kaggle.com/benchmarks/tasks/denu77/robot-replanning-executive-functions)

---

## What This Benchmark Tests

Standard AI benchmarks test what models *know*. This benchmark tests how models *think under pressure*.

When a robot task fails mid-execution, can a language model:
- **Suppress already-completed steps** without repeating them? *(inhibitory control)*
- **Adapt to new hard constraints** introduced by the failure? *(cognitive flexibility)*
- **Produce a minimal, efficient recovery plan?** *(executive planning)*

These are the executive functions defined in Google DeepMind's *Measuring Progress Toward AGI: A Cognitive Framework* — and they are almost never isolated in existing evaluations.

---

## The Key Finding

> **Larger models score *lower* than smaller models.**

Claude Sonnet 4.6, Gemini 2.5 Flash, and DeepSeek R1 all scored **0.00**.  
Claude Haiku 4.5, GPT-5.4 mini, and Gemini 3.1 Flash-Lite scored **0.09**.

Why? Larger models over-explain. They add qualifications, enumerate edge cases, pad responses with unnecessary steps — and fail the conciseness and efficiency checks. This reveals a real executive function deficit: **reduced inhibitory control over output generation**. Standard benchmarks reward thoroughness. This one penalizes it when it is unnecessary.

---

## Results Across 13 Models

| Model | Score | Tier |
|-------|-------|------|
| Claude Haiku 4.5 | 0.09 | Top |
| Gemini 3.1 Flash-Lite Preview | 0.09 | Top |
| GPT-5.4 | 0.09 | Top |
| GPT-5.4 mini | 0.09 | Top |
| Claude Opus 4.6 | 0.07 | Mid |
| Gemini 3.1 Pro Preview | 0.07 | Mid |
| Gemini 2.5 Flash | 0.00 | Failed |
| Claude Sonnet 4.6 | 0.00 | Failed |
| DeepSeek R1 | 0.00 | Failed |
| DeepSeek V3.2 | 0.00 | Failed |
| Gemini 2.0 Flash | 0.00 | Failed |
| Gemma 4 26B A4B | 0.00 | Failed |
| GPT-5.4 nano | 0.00 | Failed |

---

## How the Benchmark Works

### The Scenario Format

Each scenario gives the model:
- **Original task** — what the robot was trying to do
- **Completed steps** — what has already been done (must NOT be repeated)
- **Failure** — what went wrong
- **Constraint** — a hard rule the new plan must respect (e.g. "carry one object at a time")

The model must produce a numbered recovery plan that is correct, concise, and constraint-compliant.

### Difficulty Tiers

| Tier | Count | Weight | Description |
|------|-------|--------|-------------|
| easy | 3 | 1.0x | Single object, single constraint |
| medium | 5 | 2.0x | Constraint traps, partial completion |
| hard | 6 | 3.0x | Multiple constraints, trick answers |
| very_hard | 3 | 4.0x | Simultaneous conflicting constraints |
| trick | 2 | 5.0x | Task already done or physically impossible |

The two **trick scenarios** are the most revealing. In scenario 16, the task is already complete — the correct answer is zero steps. In scenario 20, an object is physically immovable — the correct answer is to recognize the subtask is impossible. Models that generate unnecessary steps fail here.

### Scoring System

Each response is evaluated on **9 checks**:

**4 rule-based checks:**
1. Response contains a numbered list
2. No already-completed step is repeated
3. Step count does not exceed efficiency threshold (`max_steps + 1`)
4. No explicit constraint violation (e.g. "carry both", "pick up both")

**5 LLM judge criteria (each scored independently):**
5. No repetition of completed steps
6. Failure is directly addressed with a concrete alternative
7. Stated constraint is strictly respected
8. Steps follow logically coherent physical order
9. Response complexity matches difficulty tier

**Score calculation:**
```
accuracy = checks_passed / 9

if accuracy == 1.0:
    score = difficulty_weight + speed_bonus + verbosity_bonus
elif accuracy >= 8/9:
    score = speed_bonus + verbosity_bonus   # partial credit
else:
    score = 0.0

normalized_score = score / 5.5  # max possible = trick(5) + speed(0.3) + verbosity(0.2)
```

**Speed bonus:** <3s = +0.3 | <10s = +0.2 | <30s = +0.1  
**Verbosity bonus:** <80 words = +0.2 | <150 words = +0.1

---

## Dataset

Hand-crafted by the author — no external data sources. 20 scenarios stored as a pandas DataFrame.

| Column | Type | Description |
|--------|------|-------------|
| id | int | Scenario identifier (1–20) |
| task | str | Original robot task description |
| completed | list[str] | Steps already completed — must not be repeated |
| failure | str | The failure event that occurred |
| constraint | str | Hard rule the new plan must respect |
| bad_steps | list[str] | Specific step strings to check for repetition |
| max_steps | int | Maximum acceptable steps for an efficient replan |
| difficulty | str | easy / medium / hard / very_hard / trick |

---

## Running the Benchmark

### Requirements
- Kaggle account (free)
- kaggle-benchmarks SDK (Apache 2.0)

### Setup
```bash
pip install kaggle-benchmarks
```

### Run
Open the task notebook on Kaggle and run all cells. The benchmark uses `kbench.llm` as a placeholder — Kaggle automatically substitutes each model when you use the "Add Models" button on the task page. No API keys needed — Kaggle provides free model quota.

```python
import kaggle_benchmarks as kbench
import pandas as pd
import time

# Full code in benchmark_task.py
robot_replan.evaluate(evaluation_data=df, llm=[kbench.llm])
```

---

## Project Structure

```
├── README.md                  # This file
├── benchmark_task.py          # Full benchmark code
└── scenarios.json             # All 20 scenarios as JSON
```

---

## Cognitive Framework Alignment

This benchmark directly targets the Executive Functions track from Google DeepMind's AGI cognitive framework:

| Cognitive Function | How It Is Tested |
|-------------------|------------------|
| Inhibitory control | Model must not repeat completed steps; must suppress over-generation in trick scenarios |
| Cognitive flexibility | Model must adapt plan when failure invalidates original approach |
| Planning | Model must produce ordered, logically coherent steps toward a goal |
| Working memory | Model must track completed state, active constraints, and failure context simultaneously |

---

## References

- Google DeepMind: *Measuring Progress Toward AGI: A Cognitive Framework* (2024)
- Miyake, A. et al. (2000): The Unity and Diversity of Executive Functions. *Cognitive Psychology*, 41(1), 49–100
- Luria, A.R. (1966): *Higher Cortical Functions in Man*. Basic Books
- Kaggle Benchmarks SDK: https://github.com/Kaggle/kaggle-benchmarks (Apache 2.0)

---

## License

CC0 — as required by competition rules. Free to use, modify, and build upon.

---

## Citation

```
Dessai, K. (2026). Robot Replanning Under Failure: A Weighted Executive Function Benchmark.
Kaggle x Google DeepMind Measuring AGI Hackathon.
https://www.kaggle.com/benchmarks/denu77/robot-replanning-executive-functions-benchmark
```
