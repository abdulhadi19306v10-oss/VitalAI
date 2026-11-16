# VitalAI — Exam & Viva Revision Guide

> **Purpose:** Complete study sheet covering every AI concept, mathematical derivation, dataset discovery, code module, and expected viva question for **VitalAI: Intelligent Disease Prediction & Triage System**.

---

## 1. 30-Second Elevator Pitch (How to Open Your Viva)

> "VitalAI is a dual-AI medical decision-support system that combines **Machine Learning (Self-Written Multinomial Naive Bayes in NumPy)** with **Logic-Based Reasoning (a Forward-Chaining First-Order Logic Expert System)** behind a **FastAPI** backend.
> First, Multinomial Naive Bayes takes a binary vector of patient symptoms and outputs a calibrated probability distribution over 41 diseases along with uncertainty metrics and symptom contributions.
> Second, our Forward-Chaining Expert System takes *both* those ML probabilities and the raw symptoms, fires First-Order Logic rules across multiple reasoning hops to catch emergency red flags, handle low-confidence differentials, apply infection precautions, and produce a full inference proof trace explaining *why* it reached that triage urgency and specialist recommendation."

---

## 2. End-to-End Architecture & Data Flow

```text
Patient Symptoms (e.g., ["Chest Pain", "breathlessness", "sweating"])
   │
   ▼
[1] FastAPI Input Validation & Normalization (backend/api/schemas.py)
   │  • Normalizes keys ("Chest Pain" -> "chest_pain"), deduplicates
   │  • Rejects empty input (HTTP 422) or unknown symptoms (HTTP 400)
   │  • Builds 131-dim binary vector x in {0, 1}^131
   ▼
[2] AI Component 1 — Multinomial Naive Bayes (backend/ml/naive_bayes.py)
   │  • Computes log P(c_k) + sum(x_i * log P(s_i | c_k))
   │  • Log-sum-exp normalization -> posterior probabilities P(disease | x)
   │  • Computes Shannon entropy, top-1/top-2 margin, symptom lift, follow-up questions
   ▼
[3] AI Component 2 — Forward-Chaining FOL Expert System (backend/logic/expert_system.py)
   │  • Asserts HasSymptom(s), PredictedDisease(d1, p1), CandidateDisease(dk, pk), ConfidenceMargin(m)
   │  • Chains rules R1..R14 in WorkingMemory until fixpoint
   │  • Checks red flags, low-confidence differential safety, infection control
   │  • Outputs FinalUrgency, FinalSpecialist, Actions, Precautions & Inference Proof Trace
   ▼
[4] Structured JSON Response + Medical Disclaimer (backend/api/app.py)
```

---

## 3. Codebase Map (Which File Does What & Team Ownership)

| File | Responsibility | Viva Owner |
| :--- | :--- | :--- |
| [data_loader.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/ml/data_loader.py) | CSV cleaning, `Unnamed` stripping, `fluid_overload.1` merge, deduplication, stratified K-fold | Member 1 (AI/ML Lead) |
| [naive_bayes.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/ml/naive_bayes.py) | Pure-NumPy `MultinomialNaiveBayes` & `BernoulliNaiveBayes`, explainability, `.npz` save/load | Member 1 (AI/ML Lead) |
| [knn_baseline.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/ml/knn_baseline.py) | Pure-NumPy `CosineKNNClassifier` baseline for rubric comparison | Member 1 (AI/ML Lead) |
| [evaluate_model.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/ml/evaluate_model.py) | Deduplicated 5-fold CV, $\alpha$ sweep, dropout/noise stress tests, confusion matrix & plots | Member 1 (AI/ML Lead) |
| [train_model.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/ml/train_model.py) | Training CLI + overfitting audit gate (`PASS_LOW_MISTAKE_NO_OVERFIT`) | Member 1 (AI/ML Lead) |
| [knowledge_base.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/logic/knowledge_base.py) | Clinical ontology (`DISEASE_KB`) covering all 41 diseases & `URGENCY_RANK` lattice | Member 2 (Logic Engineer) |
| [expert_system.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/logic/expert_system.py) | `WorkingMemory`, `Fact`, `Rule`, 14 forward-chaining FOL rules & `InferenceStep` trace | Member 2 (Logic Engineer) |
| [schemas.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/api/schemas.py) | Pydantic v2 validation, normalization, strict error handling | Member 3 (Backend/API) |
| [app.py](file:///C:/Users/Abdul%20Hadi/Desktop/4th%20sem/AI%20lab/mid%20project/backend/api/app.py) | FastAPI endpoints: `/health`, `/symptoms`, `/predict`, `/metrics` | Member 3 (Backend/API) |

---

## 4. Deep Dive: Dataset Findings (Impressive Viva Talking Points)

When inspecting the Kaggle **"Disease Prediction Using Machine Learning"** dataset (`Training.csv` and `Testing.csv`), our pipeline discovered **three real data-engineering issues** that standard Kaggle tutorials miss:

1. **Trailing Empty Column (`Unnamed: 133`):**
   - Caused by a stray trailing comma in the CSV header. `_drop_empty_unnamed_columns()` verifies it is completely empty/NaN and drops it safely.
2. **Duplicate Broken Symptom Column (`fluid_overload` vs `fluid_overload.1`):**
   - In the raw Kaggle CSV, two columns were named `fluid_overload`, so pandas renamed the second to `fluid_overload.1`.
   - Inspection proved `fluid_overload` was **all zeros (`0` everywhere)** while `fluid_overload.1` held all `114` positive `Alcoholic hepatitis` training rows (`1` in test).
   - `data_loader.py` merges `col` and `col.<n>` via logical OR (`np.maximum`), turning 132 raw columns into **131 clean active symptom features** so users selecting `"Fluid Overload"` get the real signal.
3. **Massive Duplicate Leakage (`93.82%` Duplicates + `100%` Train-Test Overlap):**
   - `Training.csv` has **4,920 rows** (120 per disease across 41 diseases), of which **4,616 rows (`93.82%`) are exact duplicates**, leaving **304 unique symptom vectors** (`5–10` per disease).
   - Worse, **all 41 rows (`41 / 41`) in Kaggle's official `Testing.csv` appear verbatim inside `Training.csv`**.
   - **Why this matters:** A naive random split or testing only on `Testing.csv` gives fake 100% accuracy and hides overfitting. Our pipeline fixes this by running **Deduplicated Stratified 5-Fold CV** on the 304 unique patterns and **Symptom-Dropout + Random Noise Stress Tests**.

---

## 5. Deep Dive: AI Component 1 — Multinomial Naive Bayes (Math & Code)

### Why Multinomial Naive Bayes?
1. **Probabilistic ranking + uncertainty:** Outputs calibrated posteriors across all 41 diseases (not just a single hard class), which feeds the Expert System's differential-diagnosis rules.
2. **Forgiving on partial patient symptoms:** Multinomial NB scores **present** symptoms ($x_i = 1$) without aggressively penalizing every unclicked symptom, making it strong when a patient reports only 2–4 out of 8 textbook symptoms.
3. **Fast, interpretable, and self-written:** Implemented from scratch in NumPy (`fit` takes ~4 ms, zero heavy black-box weights).

### Exact Equations Implemented in `naive_bayes.py`
1. **Class Prior (in log space):**
   $$\log P(c_k) = \log\left(\frac{N_k}{N}\right)$$
   *(All 41 diseases have 120 training rows, so priors are balanced.)*

2. **Laplace-Smoothed Conditional Probability ($\alpha = 0.1$):**
   $$P(s_i \mid c_k) = \frac{\text{count}(s_i, c_k) + \alpha}{\sum_{j=1}^{M} \text{count}(s_j, c_k) + \alpha M}$$
   - **Why $\alpha$ matters (Regularization against Overfitting):** If $\alpha = 0$, a single symptom with zero training occurrences for disease $c_k$ forces $P(s_i \mid c_k) = 0$ and wipes out the entire probability ($\log(0) = -\infty$). Conversely, if $\alpha$ is too tiny (`0.01`), noisy symptoms receive overly harsh negative log-penalties; if $\alpha$ is too large ($\ge 0.5$), conditional probabilities flatten toward uniform and the model **underfits** (CV error climbs to `2.3%` at $\alpha=0.5$ and `11.2%` at $\alpha=5.0$).
   - **Why $\alpha = 0.1$ was chosen:** Keeps **100% clean deduplicated 5-fold CV accuracy (`0.0` gap)** while improving heavy-noise resilience (`89.61%` vs `89.08%` at $\alpha=0.01$ under 30% dropout + 5% random noise).

3. **Joint Log-Likelihood & Numerically Stable Posterior (Log-Sum-Exp):**
   $$\ell_k(x) = \log P(c_k) + \sum_{i=1}^{M} x_i \log P(s_i \mid c_k)$$
   $$P(c_k \mid x) = \frac{\exp(\ell_k(x) - \ell_{\max})}{\sum_{j=1}^{C} \exp(\ell_j(x) - \ell_{\max})}$$
   Subtracting $\ell_{\max} = \max_j \ell_j(x)$ prevents floating-point underflow/overflow when exponentiating negative log-scores.

4. **Explainability & Active Follow-Up Features:**
   - **Normalized Shannon Entropy:** $H = -\frac{\sum_k p_k \ln p_k}{\ln C} \in [0, 1]$ measures total uncertainty across the 41 diseases.
   - **Symptom Contribution Lift:** Compares $\log P(s_i \mid c_1)$ against the mean across other classes so the UI/API shows *which* selected symptoms pushed the top prediction.
   - **Discriminating Follow-Up Symptoms:** Ranks absent symptoms ($x_i = 0$) by $|\log P(s_i \mid c_1) - \log P(s_i \mid c_2)|$ to tell the user which additional check best separates the top 2 diseases.

### Model Comparison & Anti-Overfitting Results (Observed Numbers)

| Model | Train Acc (`4,920`) | Test Acc (`41`) | Deduplicated 5-Fold CV (`304` unique) | Gen. Gap | Noisy CV (`20%` drop + `2%` noise) | Perturbed Test (`20%` drop + `1%` noise) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MultinomialNB(alpha=0.1)` (Saved)** | `100.00%` | `100.00%` | **`100.00%`** (`log_loss=0.0121`) | `0.000` | **`97.08%`** | `98.05%` |
| `BernoulliNB(alpha=0.25)` | `100.00%` | `100.00%` | `100.00%` (`log_loss=0.0013`) | `0.000` | `96.76%` | `98.54%` |
| `CosineKNN(k=5)` | `99.88%` | `100.00%` | `100.00%` (`log_loss=0.0208`) | `-0.002` | `96.39%` | `98.54%` |

*(Note: At 40% symptom dropout + 1% noise on test, `MultinomialNB(alpha=0.05)`/`CosineKNN` scored `93.66%` vs `BernoulliNB`'s `91.22%`, because Bernoulli penalizes missing symptoms while Multinomial only scores active ones.)*

---

## 6. Deep Dive: AI Component 2 — Forward-Chaining Expert System

### How Forward Chaining Works in `expert_system.py`
- **Knowledge Representation:** First-Order Logic predicates stored as `Fact(predicate, args)` inside `WorkingMemory` (set-backed so duplicate facts cannot trigger infinite loops).
- **Inference Mechanism:** Data-driven **Forward Chaining**:
  1. Assert initial patient facts (`HasSymptom(s)`, `SymptomCount(n)`) and ML facts (`PredictedDisease(d1, p1)`, `CandidateDisease(dk, pk)`, `ConfidenceMargin(m)`, `NormalizedEntropy(h)`).
  2. Loop over rules `R1`..`R14`: whenever a rule's antecedent matches facts in `WorkingMemory`, fire its consequent to assert new derived facts and append an `InferenceStep` to `inference_trace`.
  3. Repeat until an iteration asserts zero new facts (**fixpoint termination**).

### How the Two AI Components Meaningfully Integrate (4/4 Rubric Marks)
Instead of a simple 1-to-1 dictionary lookup (`disease -> urgency`), the Expert System performs **3-hop reasoning over both raw symptoms and ML probability structure**:
1. **Hop 1 — Ontology & Infection Tagging (`R1_DiseaseTaxonomy`, `R11_InfectionControl`):** Maps ML `PredictedDisease(d1, p1)` to category, base urgency, specialist, and `ContagiousDisease(d)` (asserting isolation precautions when $p_1 \ge 0.45$).
2. **Hop 2 — Clinical Red-Flag Override (`R2`–`R7`):** Checks raw symptom conjunctions independently of ML top-1! Example: even if Naive Bayes predicts `GERD` (`Low` urgency), `HasSymptom('chest_pain')` $\land$ `HasSymptom('sweating')` fires `R2_CardiacRedFlag` $\implies$ `EscalatedUrgency('Critical')` + `OverrideSpecialist('Cardiologist / Emergency Medicine')`.
3. **Hop 2 $\to$ Hop 3 — Uncertainty-Driven Differential Safety (`R8` $\to$ `R10` $\to$ `R12` $\to$ `R14`):**
   - `R8_LowConfidenceDifferential` fires if $p_1 < 0.65$ or margin $p_1 - p_2 < 0.20$.
   - `R10_HighUrgencyDifferentialSafety` checks if `DifferentialAlert` is active AND any plausible runner-up `CandidateDisease(d2, p2)` ($p_2 \ge 0.15$) has higher urgency than top-1 (e.g., `GERD` at `42%` vs `Heart attack` at `31%`), safely escalating final urgency!
   - `R12_ResolveFinalUrgency` resolves urgency via a strict monotonic lattice (`Low < Moderate < High < Critical`), and `R14_CriticalEmergencyAction` fires when `FinalUrgency('Critical')` is reached.

---

## 7. Likely Viva Questions & Model Answers

1. **Q: Why use two AI techniques instead of just Naive Bayes?**
   - *Answer:* ML (Naive Bayes) excels at statistical pattern recognition across 131 noisy symptom combinations, but it has no concept of clinical safety or medical consequences (for example, missing a subtle heart attack because reflux symptoms looked statistically close). Rule-based First-Order Logic guarantees deterministic safety overrides (red-flag rules), handles low-confidence differential escalation, enforces infection precautions, and gives a transparent **Inference Trace**.

2. **Q: Why didn't you evaluate only on `Testing.csv`? Doesn't 100% accuracy mean overfitting?**
   - *Answer:* Overfitting means a large gap between training accuracy and *unseen* validation accuracy. On this Kaggle dataset, `93.82%` of training rows are duplicates and **41/41 `Testing.csv` rows appear verbatim in `Training.csv`**. We proved our model isn't merely memorizing duplicates by deduplicating to the 304 unique patterns, running 5-fold stratified CV (`100%` accuracy, `0.0` gap), and stress-testing with 20–40% missing symptoms + random noise (`97.08%` noisy CV, `98.05%` perturbed test).

3. **Q: What happens if a patient sends zero symptoms or an unrecognized symptom?**
   - *Answer:* `PredictRequest` in `schemas.py` normalizes strings and raises a Pydantic validation error (`HTTP 422`) when zero symptoms remain. Unknown symptom keys are caught in `/predict` and return `HTTP 400` listing the unrecognized keys.

4. **Q: Why did Multinomial Naive Bayes beat Cosine KNN and Bernoulli Naive Bayes under partial/noisy symptoms?**
   - *Answer:* Under 20% dropout + 2% noise CV across 304 unique patterns, `MultinomialNB` scored `97.08%` vs `BernoulliNB` (`96.76%`) and `CosineKNN` (`96.39%`). Multinomial NB focuses probability mass on *observed* symptoms without harshly penalizing every absent symptom when patients under-report, and Laplace smoothing ($\alpha=0.1$) avoids brittle zero-probability lockouts.

5. **Q: What are the limitations and ethical considerations of VitalAI?**
   - *Answer:* (1) **Synthetic training distribution:** All 4,920 Kaggle rows come from 304 idealized symptom patterns without age, sex, vitals, lab results, or comorbidities. (2) **Conditional independence assumption:** Naive Bayes assumes symptoms are independent given the disease, whereas clinical symptoms often co-occur. (3) **Clinical safety & misuse:** VitalAI is a triage decision-support prototype, not a diagnostic replacement for a doctor; every API response carries an explicit disclaimer, and the Expert System defaults to conservative safety escalation whenever uncertainty or red flags exist.

---

## 8. Quick Verification Commands (For Live Demo)

```powershell
# 1. Run all 21 automated unit, rule-validation, and API tests
python -m pytest -v tests/

# 2. Re-run training + anti-overfitting audit (creates plots & JSON report)
python -m backend.ml.train_model --data-dir data --model-type multinomial --alpha 0.1

# 3. Start the FastAPI backend server
uvicorn backend.api.app:app --reload
# Open interactive Swagger UI at: http://127.0.0.1:8000/docs
```
