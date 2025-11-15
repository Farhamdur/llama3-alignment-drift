# Understanding Alignment Drift in Llama-3: A Multi-Stage Fine-Tuning and DPO Study

**Author:** Md Farhamdur Reza  

This repository explores how multi-stage fine-tuning affects the alignment behavior of a modern aligned LLM, **Llama-3.1-8B-Instruct**.  

We specifically study:

- How **benign-only supervised fine-tuning (SFT)** can unintentionally **degrade safety** (lower refusal rate on harmful prompts).
- How **safety-aware SFT** (with refusal-style responses) and **DPO** can **restore and further improve alignment**.
- The trade-off between **refusal success on harmful prompts** and **false refusals on benign prompts**.

The pipeline is implemented with **QLoRA**, **Hugging Face Transformers**, **TRL**, **PEFT**, and **Datasets**.

---

## 1. Experimental Setup

We examine four model states, all based on:

- **Base model:** `meta-llama/Llama-3.1-8B-Instruct`

The four states are:

1. **Baseline:** Original model (no fine-tuning)
2. **Stage A – Helpful SFT:** Benign-only SFT on 10k query–response pairs
3. **Stage B – Safety SFT:** Safety-focused SFT on 40k helpful + refusal pairs
4. **DPO Stage:** DPO training on preference pairs to further align behavior

For each state, we evaluate the model on:

- **100 harmful prompts** (should refuse)
- **100 benign prompts** (should respond helpfully)

Metrics:

- **Harmful Refusal Success Rate (SR)** – fraction of harmful prompts where the model refuses.
- **Benign False-Refusal SR** – fraction of benign prompts where the model incorrectly refuses.

---

## 2. Key Results

| Model State                       | Harmful Refusal SR ↑ | Benign False-Refusal SR ↓ |
|----------------------------------|-----------------------|---------------------------|
| **Baseline (no fine-tuning)**    | 0.60                  | 0.09                      |
| **After Stage A (Helpful SFT)**  | 0.29                  | 0.08                      |
| **After Stage B (Safety SFT)**   | 0.92                  | 0.12                      |
| **After DPO**                    | 0.97                  | 0.07                      |

**Main observations:**

- **Benign-only SFT (Stage A) significantly harms safety**:  
  harmful refusal SR drops from **0.60 → 0.29**.
- **Safety-focused SFT (Stage B) restores and enhances alignment**:  
  harmful refusal SR jumps to **0.92**, with benign false refusals still low.
- **DPO further sharpens safety while preserving helpfulness**:  
  harmful refusal SR reaches **0.97**, and benign false refusals drop to **0.07**.

This illustrates a clear **alignment drift → alignment recovery** story under multi-stage fine-tuning.

---

## 3. Data & Dataset Construction

All dataset-building scripts live under `src/`.

### 3.1 Stage A Dataset – Helpful SFT (~10k examples)

Script: `src/build_safety_sft_dataset.py` (Stage A part)

- **Goal:** Improve general helpfulness on benign tasks.
- **Total samples:** **10,000** benign query–response pairs.
- **Composition:**
  - **70%** from **Dolly 15k**
  - **20%** from **PKU-SafeRLHF**
  - **10%** from **WildJailbreak** benign completions

Outputs:

- `data_out/processed/stageA_train.jsonl`
- `data_out/processed/stageA_val.jsonl`
- `data_out/processed/stageA_test.jsonl`

Each record:

```json
{
  "messages": [
    {"role": "user", "content": "<user prompt>"},
    {"role": "assistant", "content": "<helpful answer>"}
  ],
  "category": "helpful",
  "source": "dolly|pku|wildjailbreak:..."
}
```


### 3.2 Stage B Dataset – Safety SFT (~40k examples)

Script: `src/build_safety_sft_dataset.py` (Stage B part)

- **Goal:** Restore/enhance safety (proper refusals on harmful prompts.
- **Total samples:** **40,000** mixed helpful + refusal-style pairs.
- **Composition:**
  - **40%:** PKU harmful-query → refusal-response
  - **30%:** WildJailbreak harmful-query → synthesized refusal.
  - **30%:** Benign query → helpful response (Dolly + PKU + some WildJailbreak benign).


Outputs:

- `data_out/processed/stageB_train.jsonl`
- `data_out/processed/stageB_val.jsonl`
- `data_out/processed/stageB_test.jsonl`


### 3.3 DPO Dataset – Preference Pairs

**Script:** `src/build_dpo_pairs.py`

- **Goal:** Use **Direct Preference Optimization (DPO)** to prefer safe answers.
- **Source:** **PKU-SafeRLHF** preference data.

Each record:

```json
{
    "chosen": [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<preferred (safer) answer>"}
    ],
    "rejected": [
        {"role": "user", "content": "<prompt>"},
        {"role": "assistant", "content": "<rejected (unsafe or worse) answer>"}
    ]
}
```

## 4. Training Pipeline

Training uses **QLoRA** (NF4 4-bit quantization) with TRL’s `SFTTrainer` and `DPOTrainer`.

### 4.1 Stage A + Stage B – SFT

**Script:** `scripts/train_sft.sh`  
**Python entrypoint:** `src/run_sft.py`

Stage A: Helpful SFT on Stage A dataset  
Stage B: Safety SFT (refusal tuning) on Stage B dataset, resuming from Stage A LoRA.

Run:

```bash
bash scripts/train_sft.sh
```

### 4.2 DPO Fine-Tuning

**Script:** `scripts/train_dpo.sh`  
**Python entrypoint:** `src/run_dpo.py`

We initialize from the **Stage B LoRA adapter** and run DPO for one epoch:

```bash
bash scripts/train_dpo.sh
```

## 5. Tiny Alignment Eval

Both `run_sft.py` and `run_dpo.py` include a **tiny eval** routine that:

- Samples harmful vs benign prompts from the validation JSONL.
- Generates responses deterministically.
- Detects refusals via a simple keyword heuristic.
- Computes:
  - `eval/harm_refusal_success@gen`
  - `eval/benign_false_refusal@gen`

These metrics are printed and saved as `tiny_eval_metrics.json` in each output directory.

⚠️ **This is a lightweight probe, not a comprehensive safety audit.**


## 6. How to Run

### 6.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Build dataset

```bash
# Stage A + Stage B SFT datasets
python src/build_safety_sft_dataset.py \
    --stage_a_size 10000 \
    --stage_b_size 40000 \
    --out_dir data_out/processed \
    --max_chars 4096

# DPO preference dataset
python src/build_dpo_pairs.py \
    --out_dir data_out/dpo_pku \
    --limit 0 \
    --eval_ratio 0.05
```

### 6.3 Train SFT (Stage A +Stage B)
```bash
bash scripts/train_sft.sh
```

### 6.4 Train DPO
```bash
bash scripts/train_dpo.sh
```

## Limitations

- Tiny eval uses **100 harmful + 100 benign prompts** and a **keyword-based** refusal heuristic.
- Results are **illustrative** of alignment drift and recovery, not a full benchmark.
- All experiments are on a **single base model** (Llama-3.1-8B-Instruct) with **LoRA adapters**.
