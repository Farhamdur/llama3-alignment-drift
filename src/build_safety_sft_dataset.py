import os, json, random, re, hashlib, argparse, hashlib
from typing import Dict, Any, List, Tuple
from datasets import load_dataset, Dataset
random.seed(13)

REFUSAL = ("I can’t help with requests that could cause harm or violate safety policies. "
           "This includes content that enables physical harm, illegal activity, or privacy violations. "
           "If you’d like, I can suggest safe, high-level resources or alternatives instead.")

def synthesize_refusal(prompt: str) -> str:
    return REFUSAL

def _hash_pair(user: str, asst: str) -> str:
    return hashlib.sha256((user.strip().lower()+"|||"+asst.strip().lower()).encode()).hexdigest()

def make_pair(user: str, assistant: str, category: str, source: str) -> Dict[str,Any]:
    return {
        "messages": [
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "category": category,
        "source": source,
    }

def dedup_records(records: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    seen = set()
    out = []
    for ex in records:
        u = ex["messages"][0]["content"]
        a = ex["messages"][1]["content"]
        h = _hash_pair(u, a)
        if h in seen:
            continue
        seen.add(h)
        out.append(ex)
    return out

def split_stratified(records: List[Dict[str,Any]], val_ratio=0.05, test_ratio=0.05):
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    train, val, test = [], [], []
    for cat, arr in by_cat.items():
        random.shuffle(arr)
        n = len(arr)
        n_val = max(1, int(n*val_ratio)) if n>10 else (1 if n>=3 else 0)
        n_test = max(1, int(n*test_ratio)) if n>10 else (1 if n>=3 else 0)
        n_train = max(0, n - n_val - n_test)
        train += arr[:n_train]
        val += arr[n_train:n_train+n_val]
        test += arr[n_train+n_val:]
    return train, val, test

def cap_len(text: str, max_chars: int) -> str:
    return text[:max_chars]

# ----------------------------- DOLLY -----------------------------
def load_dolly(n: int = 15000, max_chars: int = 4096) -> List[Dict[str,Any]]:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    recs = []
    for x in ds:
        prompt = (x.get("instruction","") or "")
        ctx = x.get("context","")
        if ctx:
            prompt += f"\n\nContext:\n{ctx}"
        ans = (x.get("response","") or "")
        prompt, ans = cap_len(prompt, max_chars), cap_len(ans, max_chars)
        if prompt.strip() and ans.strip():
            recs.append(make_pair(prompt, ans, "helpful", "dolly"))
    random.shuffle(recs)
    return recs[:min(n, len(recs))]

# ------------------------------ PKU ------------------------------
def load_pku(max_chars: int = 4096) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
    helpful, refusals = [], []
    seen = set()
    for row in ds:
        q = row.get("question") or row.get("prompt") or ""
        if not q or not q.strip():
            continue
        # Different published variants use 'answer_0/1' or 'response_0/1' keys.
        r0 = row.get("response_0", row.get("answer_0",""))
        r1 = row.get("response_1", row.get("answer_1",""))
        s0 = bool(row.get("is_response_0_safe", row.get("is_answer_0_safe", False)))
        s1 = bool(row.get("is_response_1_safe", row.get("is_answer_1_safe", False)))
        safer_id  = row.get("safer_response_id", row.get("safer_answer_id", None))
        better_id = row.get("better_response_id", row.get("better_answer_id", None))

        def pick(idx):
            return (r0, r1)[idx], (s0, s1)[idx]

        chosen = None
        if safer_id in (0,1):
            ans, safe = pick(safer_id)
            if safe: chosen = ans
        if chosen is None and better_id in (0,1):
            ans, safe = pick(better_id)
            if safe: chosen = ans
        if chosen is None:
            if s0 and not s1: chosen = r0
            elif s1 and not s0: chosen = r1
            elif s0 and s1: chosen = r0 if len(r0)>=len(r1) else r1

        if chosen is None:
            a = synthesize_refusal(q)
            h = _hash_pair(q, a)
            if h not in seen:
                seen.add(h)
                refusals.append(make_pair(cap_len(q,max_chars), cap_len(a,max_chars), "refusal", "pku"))
            continue

        h = _hash_pair(q, chosen)
        if h in seen: 
            continue
        seen.add(h)
        helpful.append(make_pair(cap_len(q,max_chars), cap_len(chosen,max_chars), "helpful", "pku"))
    return helpful, refusals

# -------------------------- WildJailbreak -------------------------
def load_wildjailbreak(max_chars: int = 4096):
    ds = load_dataset("allenai/wildjailbreak", 
                      name="train", 
                      split="train", 
                      delimiter="\t", 
                      keep_default_na=False)
    buckets = {
        "vanilla_benign": [],
        "vanilla_harmful": [],
        "adversarial_benign": [],
        "adversarial_harmful": [],
    }
    for r in ds:
        t = r.get("data_type") or ""
        if "vanilla" in t:
            prompt = r.get("vanilla") or ""
        else:
            prompt = r.get("adversarial") or ""
        prompt = cap_len(prompt, max_chars)
        completion = (r.get("completion") or "").strip()
        completion = cap_len(completion, max_chars) if completion else completion
        if t in buckets:
            buckets[t].append({"prompt": prompt, "completion": completion, "type": t})
    return buckets

# ----------------------------- Builders -----------------------------
def build_stage_a(dolly_n=4000, pku_help_n=2000, wj_benign_n=1000, max_chars=4096):
    dolly = load_dolly(dolly_n, max_chars=max_chars)
    pku_helpful, _ = load_pku(max_chars=max_chars)
    random.shuffle(pku_helpful); pku_helpful = pku_helpful[:pku_help_n]

    wj = load_wildjailbreak(max_chars=max_chars)
    benign = []
    for t in ("vanilla_benign","adversarial_benign"):
        for r in wj[t]:
            a = r.get("completion","")
            if a:
                benign.append(make_pair(r["prompt"], a, "helpful", f"wildjailbreak:{t}"))
    random.shuffle(benign); benign = benign[:wj_benign_n]

    records = dedup_records(dolly + pku_helpful + benign)
    return split_stratified(records, val_ratio=0.05, test_ratio=0.05)

def build_stage_b(pku_refusal_n=2000, wj_harm_refusal_n=1000, benign_backfill_n=2000, max_chars=4096):
    _, pku_refusals = load_pku(max_chars=max_chars)
    random.shuffle(pku_refusals); pku_refusals = pku_refusals[:pku_refusal_n]

    wj = load_wildjailbreak(max_chars=max_chars)
    wj_harm = []
    for t in ("vanilla_harmful","adversarial_harmful"):
        for r in wj[t]:
            wj_harm.append(make_pair(r["prompt"], synthesize_refusal(r["prompt"]), "refusal", f"wildjailbreak:{t}"))
    random.shuffle(wj_harm); wj_harm = wj_harm[:wj_harm_refusal_n]

    # benign backfill: Dolly + PKU-helpful + small WJ-benign
    dolly = load_dolly(n=benign_backfill_n//2, max_chars=max_chars)
    pku_helpful, _ = load_pku(max_chars=max_chars)
    random.shuffle(pku_helpful); pku_helpful = pku_helpful[:max(0, benign_backfill_n - len(dolly))]

    wj_benign = []
    for t in ("vanilla_benign","adversarial_benign"):
        for r in wj[t]:
            a = r.get("completion","")
            if a:
                wj_benign.append(make_pair(r["prompt"], a, "helpful", f"wildjailbreak:{t}"))
    random.shuffle(wj_benign); wj_benign = wj_benign[:min(500, benign_backfill_n//4)]

    records = dedup_records(pku_refusals + wj_harm + dolly + pku_helpful + wj_benign)
    return split_stratified(records, val_ratio=0.05, test_ratio=0.05)

def dump_jsonl(path: str, rows: List[Dict[str,Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a_size", type=int, default=10000, help="Approx total examples for Stage A")
    ap.add_argument("--stage_b_size", type=int, default=40000, help="Approx total examples for Stage B")
    ap.add_argument("--out_dir", type=str, default="data_out/processed", help="Output dir")
    ap.add_argument("--max_chars", type=int, default=4096)
    args = ap.parse_args()

    # Stage A mix: 70% Dolly, 20% PKU-helpful, 10% WJ-benign
    dolly_A = int(args.stage_a_size * 0.70)
    pku_help_A = int(args.stage_a_size * 0.20)
    wj_benign_A = max(500, args.stage_a_size - dolly_A - pku_help_A)
    trainA, valA, testA = build_stage_a(dolly_n=dolly_A, pku_help_n=pku_help_A, wj_benign_n=wj_benign_A, max_chars=args.max_chars)

    # Stage B mix: 40% PKU-refusal, 30% WJ-harm->refusal, 30% benign backfill
    pku_ref_B = int(args.stage_b_size * 0.40)
    wj_harm_B = int(args.stage_b_size * 0.30)
    benign_backfill_B = args.stage_b_size - pku_ref_B - wj_harm_B
    trainB, valB, testB = build_stage_b(pku_refusal_n=pku_ref_B, wj_harm_refusal_n=wj_harm_B, benign_backfill_n=benign_backfill_B, max_chars=args.max_chars)

    out_dir = args.out_dir
    dump_jsonl(os.path.join(out_dir, "stageA_train.jsonl"), trainA)
    dump_jsonl(os.path.join(out_dir, "stageA_val.jsonl"),   valA)
    dump_jsonl(os.path.join(out_dir, "stageA_test.jsonl"),  testA)

    dump_jsonl(os.path.join(out_dir, "stageB_train.jsonl"), trainB)
    dump_jsonl(os.path.join(out_dir, "stageB_val.jsonl"),   valB)
    dump_jsonl(os.path.join(out_dir, "stageB_test.jsonl"),  testB)

    def stats(rows):
        from collections import Counter
        return {"total": len(rows), "by_category": dict(Counter(r["category"] for r in rows))}

    print("Stage A train:", stats(trainA))
    print("Stage A val:  ", stats(valA))
    print("Stage A test: ", stats(testA))
    print("Stage B train:", stats(trainB))
    print("Stage B val:  ", stats(valB))
    print("Stage B test: ", stats(testB))

if __name__ == "__main__":
    main()
