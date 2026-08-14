#!/usr/bin/env python
"""
E5: LLM baseline on the IndiaPoliceEvents corpus via OpenRouter (paper experiment).

Codes every document with a commercial LLM (no fine-tuning, zero-shot rubric drawn
from the IPE annotation questions), producing the same per-doc label matrix as the
cross-fitted encoders in ipe_crossfit.py, plus token usage for cost accounting.

Auth: OPENROUTER_API_KEY env var, or a key file (default .openrouter_key in repo root).

Usage:
  python src/eval/llm_ipe_baseline.py --limit 10          # smoke run, prints cost/doc
  python src/eval/llm_ipe_baseline.py                     # full corpus (1,257 docs)
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipe_crossfit import load_corpus, LABELS

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Doc-level questions from Halterman et al. (2021), Table 1.
RUBRIC = """You are coding news articles for a political-science event dataset on \
police activity during the 2002 Gujarat violence in India. Read the article and \
answer five yes/no questions ABOUT POLICE ACTIONS ONLY (police, security forces, \
army acting in a policing role). Base every answer strictly on what the article \
states; do not infer events the text does not report.

KILL: Did police kill someone? (e.g. police firing killed rioters)
ARREST: Did police arrest, detain, or round up anyone?
FAIL: Did police fail to act, fail to intervene, stand by, or act ineffectively \
while violence occurred?
FORCE: Did police use force or violence other than killing? (lathi charge, tear \
gas, firing without reported deaths, beatings)
ANY_ACTION: Did police take any action at all? (includes patrolling, imposing \
curfew, escorting, investigating, plus any of the above)

Respond with ONLY a JSON object, no other text, in exactly this form:
{"KILL": true/false, "ARREST": true/false, "FAIL": true/false, "FORCE": true/false, "ANY_ACTION": true/false}"""

SCHEMA = {
    "type": "object",
    "properties": {L: {"type": "boolean"} for L in LABELS},
    "required": list(LABELS),
    "additionalProperties": False,
}


def get_key(key_file):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and os.path.exists(key_file):
        key = open(key_file).read().strip()
    if not key:
        sys.exit(f"No OPENROUTER_API_KEY in env and no key file at {key_file}")
    return key


def parse_labels(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    obj = json.loads(m.group(0))
    return [int(bool(obj[L])) for L in LABELS]


def call(session, key, model, doc_text, use_schema, use_reasoning):
    body = {
        "model": model,
        "max_tokens": 2000,
        "messages": [{"role": "user",
                      "content": f"{RUBRIC}\n\n<article>\n{doc_text}\n</article>"}],
    }
    if use_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "police_actions", "strict": True, "schema": SCHEMA},
        }
    if use_reasoning:
        body["reasoning"] = {"effort": "low"}
    for attempt in range(5):
        r = session.post(API_URL, json=body, timeout=120,
                         headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 400 and use_schema and "response_format" in r.text:
            body.pop("response_format"); use_schema = False; continue
        if r.status_code == 400 and use_reasoning and "reasoning" in r.text:
            body.pop("reasoning"); use_reasoning = False; continue
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt); continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:                      # OpenRouter can 200 with an error body
            time.sleep(2 ** attempt); continue
        return data, use_schema, use_reasoning
    raise RuntimeError(f"gave up after retries: {r.status_code} {r.text[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="external/IndiaPoliceEvents/data/final")
    ap.add_argument("--model", default="openai/gpt-5.6-luna")
    ap.add_argument("--out", default="analysis/data/ipe_llm_preds.csv")
    ap.add_argument("--key-file", default=".openrouter_key")
    ap.add_argument("--limit", type=int, default=0, help="cap docs for a smoke run")
    ap.add_argument("--price-in", type=float, default=0.10, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=1.60, help="$ per 1M output tokens")
    ap.add_argument("--max-cost", type=float, default=2.0, help="hard $ stop")
    args = ap.parse_args()

    key = get_key(args.key_file)
    ids, texts, y, dates = load_corpus(args.corpus)
    if args.limit:
        ids, texts, y, dates = (ids[:args.limit], texts[:args.limit],
                                y[:args.limit], dates[:args.limit])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = (["model", "doc_id", "date"] + [f"gold_{L}" for L in LABELS]
              + [f"pred_{L}" for L in LABELS]
              + ["parse_error", "input_tokens", "output_tokens"])
    new = not os.path.exists(args.out)
    done = set()
    if not new:
        with open(args.out, newline="") as fh:
            done = {(r["model"], r["doc_id"]) for r in csv.DictReader(fh)}

    session = requests.Session()
    use_schema, use_reasoning = True, True
    tot_in = tot_out = n_done = 0
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for i, doc_id in enumerate(ids):
            if (args.model, doc_id) in done:
                continue
            cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
            if cost >= args.max_cost:
                print(f"STOPPING: cost ${cost:.2f} hit --max-cost after {n_done} docs")
                break
            data, use_schema, use_reasoning = call(
                session, key, args.model, texts[i], use_schema, use_reasoning)
            usage = data.get("usage", {})
            tot_in += usage.get("prompt_tokens", 0)
            tot_out += usage.get("completion_tokens", 0)
            text = data["choices"][0]["message"]["content"] or ""
            try:
                preds, perr = parse_labels(text), 0
            except (ValueError, KeyError, json.JSONDecodeError):
                preds, perr = [""] * len(LABELS), 1
            w.writerow([args.model, doc_id, dates[i]] + y[i].tolist() + preds
                       + [perr, usage.get("prompt_tokens", 0),
                          usage.get("completion_tokens", 0)])
            fh.flush()
            n_done += 1
            if n_done % 25 == 0:
                cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
                print(f"{n_done} docs  in={tot_in} out={tot_out}  est ${cost:.3f}",
                      flush=True)
            time.sleep(0.25)

    cost = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
    per_doc = cost / n_done if n_done else 0
    print(f"LLM_BASELINE_DONE model={args.model} docs={n_done} "
          f"input_tokens={tot_in} output_tokens={tot_out} cost=${cost:.3f} "
          f"(${per_doc*1000:.2f} per 1k docs; full 1,257-doc corpus ≈ "
          f"${per_doc*1257:.2f})")


if __name__ == "__main__":
    main()
