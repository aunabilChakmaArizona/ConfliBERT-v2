#!/usr/bin/env python
"""
E5b: zero-shot LLM baseline on the CrisisWatch trend task via OpenRouter.

Codes every dev/test example (the same inputs the encoders see in --input-mode
full) into unchanged / deteriorated / improved. Months after the model's training
cutoff make this a contamination-free comparison.

Usage:  python src/eval/llm_cw_baseline.py --limit 10   # smoke, then full
Output: appends to analysis/data/cw_llm_preds.csv
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_ipe_baseline import API_URL, get_key

CLASSES = ["unchanged", "deteriorated", "improved"]

RUBRIC = """You are an analyst replicating Crisis Group's CrisisWatch conflict \
tracker. Below are a country's recent monthly conflict-situation entries (oldest \
first). Judge the situation TREND IN THE FINAL (most recent) MONTH relative to \
the month before it, exactly as CrisisWatch would: did the overall situation \
deteriorate, improve, or remain broadly unchanged? Weigh the whole entry \
(escalation, casualties, political breakdown vs ceasefires, agreements, calming).

Respond with ONLY a JSON object in exactly this form:
{"trend": "unchanged" | "deteriorated" | "improved"}"""


def call(session, key, model, text, use_reasoning):
    body = {"model": model, "max_tokens": 1200,
            "messages": [{"role": "user",
                          "content": f"{RUBRIC}\n\n<entries>\n{text}\n</entries>"}]}
    if use_reasoning:
        body["reasoning"] = {"effort": "low"}
    for a in range(5):
        r = session.post(API_URL, json=body, timeout=120,
                         headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 400 and use_reasoning and "reasoning" in r.text:
            body.pop("reasoning"); use_reasoning = False; continue
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** a); continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            time.sleep(2 ** a); continue
        return data, use_reasoning
    raise RuntimeError(f"gave up: {r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", default="external/crisiswatch/task.jsonl")
    ap.add_argument("--model", default="openai/gpt-5.6-luna")
    ap.add_argument("--out", default="analysis/data/cw_llm_preds.csv")
    ap.add_argument("--key-file", default=".openrouter_key")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--price-in", type=float, default=0.10)
    ap.add_argument("--price-out", type=float, default=1.60)
    ap.add_argument("--max-cost", type=float, default=1.5)
    args = ap.parse_args()

    key = get_key(args.key_file)
    rows = [json.loads(l) for l in open(args.task_file, encoding="utf-8")]
    rows = [r for r in rows if r["split"] in ("dev", "test")]
    if args.limit:
        rows = rows[:args.limit]

    header = ["model", "split", "country", "year", "month", "gold", "pred",
              "parse_error", "input_tokens", "output_tokens"]
    new = not os.path.exists(args.out)
    done = set()
    if not new:
        with open(args.out, newline="") as fh:
            done = {(r["model"], r["country"], r["year"], r["month"])
                    for r in csv.DictReader(fh)}

    session = requests.Session()
    use_reasoning = True
    tot_in = tot_out = n = 0
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for r in rows:
            k = (args.model, r["country"], str(r["year"]), str(r["month"]))
            if k in done:
                continue
            cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
            if cost >= args.max_cost:
                print(f"STOPPING at --max-cost after {n} examples")
                break
            text = (r["text_history"] + "\n" + r["text_current"]).strip()
            data, use_reasoning = call(session, key, args.model, text, use_reasoning)
            usage = data.get("usage", {})
            tot_in += usage.get("prompt_tokens", 0)
            tot_out += usage.get("completion_tokens", 0)
            content = data["choices"][0]["message"]["content"] or ""
            m = re.search(r'"trend"\s*:\s*"(unchanged|deteriorated|improved)"', content)
            pred, perr = (m.group(1), 0) if m else ("", 1)
            w.writerow([args.model, r["split"], r["country"], r["year"], r["month"],
                        r["label"], pred, perr,
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)])
            fh.flush()
            n += 1
            if n % 50 == 0:
                cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
                print(f"{n} done  est ${cost:.3f}", flush=True)
            time.sleep(0.25)
    cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
    print(f"CW_LLM_DONE model={args.model} n={n} cost=${cost:.3f}")


if __name__ == "__main__":
    main()
