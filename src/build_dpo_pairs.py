#!/usr/bin/env python3
# build_pku_only_dpo.py
import os, re, json, argparse, random
from typing import List, Dict, Any
from collections import Counter
from datasets import load_dataset

random.seed(13)

REFUSAL_TPLS = [
  "I can’t help with that because it could cause harm or violate safety policies. I can offer high-level, non-operational information instead.",
  "I’m not able to assist with requests that enable harm or illegal activity. If you’d like, I can provide safer, policy-aligned guidance.",
  "I can’t provide those instructions. If your goal is legitimate and safety-focused, I can suggest high-level alternatives that avoid risky detail.",
  "I’m unable to help with potentially dangerous or unlawful content. I can share general safety principles or point to responsible resources.",
]
SAFE_ALTS = {
  "weapons":"I can discuss emergency preparedness and risk reduction without technical steps.",
  "cybercrime":"I can help secure your own systems (password hygiene, patching, backups) at a high level.",
  "bio_chem":"I can share lab safety principles (PPE, SOPs, hazard communication) at a non-operational level.",
  "privacy":"I can discuss privacy-first design and data minimization best practices.",
  "violence":"If there’s risk of harm, contact local authorities. I can share de-escalation resources.",
  "generic":"I can provide safe, high-level information or alternatives that avoid harmful detail.",
}

REFUSAL = ("I can’t help with requests that could cause harm or violate safety policies. "
           "This includes content that enables physical harm, illegal activity, or privacy violations. "
           "If you’d like, I can suggest safe, high-level resources or alternatives instead.")


def infer_harm_category(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(wifi|hack|exploit|breach|ddos|ransom|malware|keylogger)\b", t): return "cybercrime"
    if re.search(r"\b(explosive|bomb|ignite|grenade|napalm)\b", t): return "weapons"
    if re.search(r"\b(toxic|poison|sarin|ricin|culture|bioreactor|pathogen)\b", t): return "bio_chem"
    if re.search(r"\b(doxx|dox|bypass 2fa|steal data|spy|stalkerware)\b", t): return "privacy"
    if re.search(r"\b(kill|harm|assault|kidnap)\b", t): return "violence"
    return "generic"
def synth_refusal(prompt: str, cat: str | None = None) -> str:
    # cat = cat or infer_harm_category(prompt)
    # return random.choice(REFUSAL_TPLS) + " " + SAFE_ALTS.get(cat, SAFE_ALTS["generic"])
    return REFUSAL

def _mk_pair(prompt, chosen, rejected, bucket, harm_category="generic"):
    return {
        "chosen": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": chosen.strip()}
        ],
        "rejected": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": rejected.strip()}
        ],
        # "prompt": prompt.strip(),
        # "chosen": chosen.strip(),
        # "rejected": rejected.strip(),
        # "tags": {"bucket": bucket, "harm_category": harm_category, "source": "PKU"}
    }

def load_pku_pairs(limit: int | None = None,
                   strict_unsafe_refusal: bool = True,
                   include_both_safe_pairs: bool = True) -> List[Dict[str,Any]]:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
    out = []
    for r in ds:
        prompt = (r.get("question") or r.get("prompt") or "").strip()
        if not prompt: continue
        r0 = (r.get("response_0", r.get("answer_0","")) or "").strip()
        r1 = (r.get("response_1", r.get("answer_1","")) or "").strip()
        if not r0 or not r1: continue
        s0 = bool(r.get("is_response_0_safe", r.get("is_answer_0_safe", False)))
        s1 = bool(r.get("is_response_1_safe", r.get("is_answer_1_safe", False)))
        safer_id  = r.get("safer_response_id", r.get("safer_answer_id", None))
        better_id = r.get("better_response_id", r.get("better_answer_id", None))

        def pick(i): return (r0, r1)[i]

        # 1) one safe, one unsafe → choose safe
        if s0 ^ s1:
            chosen, rejected = (r0, r1) if s0 else (r1, r0)
            out.append(_mk_pair(prompt, chosen, rejected, bucket="safety",
                                harm_category=infer_harm_category(prompt)))
        # 2) both safe → optional benign/helpfulness pair
        elif s0 and s1:
            if not include_both_safe_pairs:
                pass
            else:
                if better_id in (0,1):
                    chosen, rejected = pick(better_id), pick(1-better_id)
                elif safer_id in (0,1):
                    chosen, rejected = pick(safer_id), pick(1-safer_id)
                else:
                    chosen, rejected = (r0, r1) if len(r0)>=len(r1) else (r1, r0)
                out.append(_mk_pair(prompt, chosen, rejected, bucket="benign"))
        # 3) both unsafe → chosen = synthesized refusal (safety-first)
        else:
            if not strict_unsafe_refusal:
                # skip to avoid rewarding unsafe
                pass
            else:
                if better_id in (0,1):
                    rejected = pick(better_id)
                elif safer_id in (0,1):
                    rejected = pick(safer_id)
                else:
                    rejected = r0 if len(r0)>=len(r1) else r1
                cat = infer_harm_category(prompt)
                chosen = synth_refusal(prompt, cat)
                out.append(_mk_pair(prompt, chosen, rejected, bucket="safety",
                                    harm_category=cat))

        if limit and len(out) >= limit: break
    return out

def dedup_by_prompt(pairs: List[Dict[str,Any]], prefer_safety=True) -> List[Dict[str,Any]]:
    keep = {}
    for p in pairs:
        k = p["prompt"].strip().lower()
        q = keep.get(k)
        if q is None: keep[k] = p
        else:
            if prefer_safety and p["tags"]["bucket"] == "safety" and q["tags"]["bucket"] != "safety":
                keep[k] = p
    return list(keep.values())

def split_train_eval(pairs, eval_ratio=0.05):
    random.shuffle(pairs)
    n_eval = max(1000, int(len(pairs)*eval_ratio))
    return pairs[n_eval:], pairs[:n_eval]

def stats(pairs):
    bc = Counter(p["tags"]["bucket"] for p in pairs)
    hc = Counter(p["tags"]["harm_category"] for p in pairs)
    return {"total": len(pairs), "by_bucket": dict(bc), "by_harm_category": dict(hc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data_out/dpo_pku")
    ap.add_argument("--limit", type=int, default=0, help="soft cap; 0 = all")
    ap.add_argument("--strict_unsafe_refusal", action="store_true")
    ap.add_argument("--no_both_safe_pairs", action="store_true")
    ap.add_argument("--eval_ratio", type=float, default=0.05)
    args = ap.parse_args()

    pairs = load_pku_pairs(
        limit=(args.limit or None),
        strict_unsafe_refusal=args.strict_unsafe_refusal or True,
        include_both_safe_pairs=not args.no_both_safe_pairs
    )
    # pairs = dedup_by_prompt(pairs, prefer_safety=True) # To remove the same query
    train, eval_ = split_train_eval(pairs, eval_ratio=args.eval_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,"dpo_train.jsonl"),"w",encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(os.path.join(args.out_dir,"dpo_eval.jsonl"),"w",encoding="utf-8") as f:
        for r in eval_: f.write(json.dumps(r, ensure_ascii=False)+"\n")

    # manifest = {"counts":{"train":stats(train),"eval":stats(eval_)},
    #             "notes":"PKU-only DPO; safety-first pairing policy as described."}
    # with open(os.path.join(args.out_dir,"manifest.json"),"w",encoding="utf-8") as f:
    #     json.dump(manifest, f, indent=2, ensure_ascii=False)

    # print("Train:", stats(train))
    # print("Eval :", stats(eval_))
    # print("Wrote:", args.out_dir)

if __name__ == "__main__":
    main()
