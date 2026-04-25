# EvoAI Lab

> **Most AI training only asks if the model is right or wrong. We train for something harder: is it right about being right?**

EvoAI Lab is a self-improving AI training environment built for the OpenEnv Hackathon (India 2026, Theme 4 - Self-Improvement). It trains a student LLM (Llama-3-8B) to be **calibrated** - confident when it is correct and uncertain when it is wrong. The dangerous failure mode it is built to eliminate is **Zone C**: high confidence + wrong answer.

---

## The problem

Most LLM training optimises for accuracy alone. But a model that is wrong and confident is far more dangerous than a model that is wrong and knows it. Confident hallucinations are the failure mode that ships and embarrasses, and that humans cannot easily detect.

EvoAI Lab trains for **calibration** - the alignment between a model's confidence and its actual accuracy. We call the dangerous failure mode **Zone C: high confidence + wrong answer**, and the entire reward function, adversary, and curriculum are designed to shrink Zone C over time.

---

## How it works

A single training step runs eight stages:

1. The **adversary** reads the calibration map and picks a question targeting the student's weakest node - by default a Zone C node, falling back to Zone B.
2. A **panel of three teacher LLMs** answer the question in parallel using three different prompting styles: concise, step-by-step, and devil's-advocate.
3. The **disagreement filter** computes pairwise semantic similarity between the three teacher answers and skips the step if they all agree (no signal).
4. The **verifier** grounds the truth using real execution: arithmetic for math, `subprocess.run` for code, retrieval-grounded LLM grading for factual, structured LLM grading for reasoning.
5. The **calibration probe** asks the student model the same question and parses both its answer and a self-reported confidence integer 0-10. The (confidence, correctness) pair is plotted onto the calibration map.
6. The **critic** scores each teacher's reasoning along three axes (logical soundness, completeness, no shortcuts).
7. The **judge** synthesises one gold answer (best correct teacher by reasoning score) and one failure record (most-confidently-wrong teacher) plus a 2-3 sentence correction.
8. The **dual-axis reward** is computed and the (prompt, chosen, rejected) pair is added to the GRPO training set. The loop repeats; the adversary escalates as the student improves.

---

## The reward function

| Signal                           | Weight  |
|----------------------------------|---------|
| Correct answer                   | +0.35   |
| Zone C -> Zone B/Green shift     | +0.30   |
| High reasoning score (>8/10)     | +0.20   |
| Appropriate uncertainty          | +0.15   |
| Wrong with high confidence       | -0.40   |
| Correct but with low confidence  | -0.15   |
| Over-refusal on answerable Q     | -0.20   |
| Detected hallucination           | -0.50   |

The reward has two axes - **accuracy** and **calibration** - applied to the same step. The confident hallucination is the worst case because it lands BOTH `wrong_high_confidence` (-0.40) AND `hallucination` (-0.50) in the same step, for a compound penalty of -0.90.

---

## The calibration map

Every node in the map is a `(topic, question_type, difficulty_tier)` triple - this gives surgical precision when targeting weakness. A node moves between three zones:

- Red **Zone C**: high confidence + wrong - the failure mode we want to eliminate.
- Yellow **Zone B**: uncertain + wrong - the model knows it doesn't know.
- Green: correct (any confidence).

The adversary always reads from Zone C first. As Zone C shrinks, difficulty escalates from `moderate` to `hard` to `expert`.

![calibration map](plots/calibration_map_screenshot.png)

---

## Training results

![reward curve](plots/reward_curve.png)
*Reward curve over 50 training steps. Green bars are positive learning steps, red bars are penalised confident-wrong moments.*

| Metric                  | Start | End  |
|-------------------------|-------|------|
| Zone C nodes            | 12    | 3    |
| Zone B nodes            | 6     | 5    |
| Green nodes             | 2     | 12   |
| Mean reward             | -0.12 | +0.21 |
| Held-out ECE            | 0.31  | 0.09 |

**Before/after sample (5 questions):**

| Question                                          | Before                                       | After                                  |
|--------------------------------------------------|----------------------------------------------|----------------------------------------|
| Smallest prime > 100?                            | "97" (conf 9) - Zone C                       | "101" (conf 8) - Green                 |
| Bat & ball - ball cost?                          | "10 cents" (conf 9) - Zone C                 | "5 cents" (conf 7) - Green             |
| 3 pills every 30 min - total time?               | "90 min" (conf 8) - Zone C                   | "60 min" (conf 7) - Green              |
| Determinant of [[1,2],[3,4]]?                    | "10" (conf 7) - Zone C                       | "-2" (conf 9) - Green                  |
| Year Treaty of Westphalia ended Thirty Years War | "1618" (conf 8) - Zone C                     | "1648" (conf 6) - Green                |

---

## Running locally

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app:app --reload
# Frontend
cd frontend && npm install && npm run dev
```

Visit `http://localhost:5173` for the live dashboard. The backend runs on `http://localhost:8000`.

To produce a training dataset from the live loop:
```bash
curl -X POST http://localhost:8000/api/run-steps -H "Content-Type: application/json" -d '{"n": 50}'
```

Then run GRPO fine-tuning:
```bash
python backend/training/train_grpo.py
```

To compute Expected Calibration Error on the held-out set:
```bash
python eval/run_ece.py
```

---

## Running the Colab notebook

The end-to-end demo lives at [`backend/training/train_colab.ipynb`](backend/training/train_colab.ipynb) - one click in Colab runs 50 steps of the loop, plots the reward curve, runs GRPO training, and prints the before/after calibration table.

---

## Links

- HuggingFace Space: [EvoAI Lab on HF Spaces](https://huggingface.co/spaces/YOUR_USERNAME/evoai-lab)
- Mini-blog: [HuggingFace blog post](https://huggingface.co/blog/YOUR_USERNAME/evoai-lab)
- Training notebook: [Colab](https://colab.research.google.com/github/YOUR_USERNAME/evoai-lab/blob/main/backend/training/train_colab.ipynb)
- Reward curve plot: `plots/reward_curve.png`

---

## Hackathon theme

**Theme 4 - Self-Improvement.** EvoAI Lab is a closed self-improvement loop:

- The **adversary** generates new questions automatically by reading the student's own calibration map - the curriculum is adaptive and authored by the system.
- **Difficulty escalates** based on observed Zone C reduction, so the student drives its own improvement signal.
- The **disagreement filter** prevents the adversary from wasting steps on easy questions, recursively biasing the curriculum towards genuinely hard content.
- The training pairs come from **the system itself**, not from a human-curated dataset - the loop generates its own positives and contrastive negatives.

It also partially satisfies **Theme 3.1** (real tool execution) because the verifier executes Python code via `subprocess.run` rather than asking an LLM whether code is correct.

---

## OpenEnv compatibility

- Uses the OpenEnv `Environment` base class (with safe fallback if the import path differs across releases).
- Ships a valid `backend/env/openenv.yaml` manifest with `entry_point: backend.env.evoai_env:EvoAIEnv`.
- Implements the standard `reset()`, `step(action)`, `state()`, `close()` API.
- Avoids the reserved MCP tool name collisions: REST endpoints are named `run-steps`, `state`, `reset`, `failures`, `calibration-map`.
