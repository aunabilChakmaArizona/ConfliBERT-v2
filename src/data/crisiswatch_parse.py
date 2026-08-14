#!/usr/bin/env python
"""
Parse CrisisWatch regional bulletin PDFs into country-month entries with trend
ratings, for the paper's document-level trend-assessment task.

Entry format in the PDFs: each country entry is marked by a RUN of symbol-font
glyphs (private-use area, Wingdings encodings): a trend arrow plus optional flag
icons, followed by the country name, a bold one-line summary, and the narrative.
Glyph mapping (spot-validated and cross-checked against the global-overview
PDFs' explicit "Deteriorated/Improved Situations" lists via --validate):

  0xf0c6 arrow -> unchanged     0xf0c7 arrow -> improved
  0xf0c8 arrow -> deteriorated  0xf04d bomb  -> conflict risk alert (flag)

Output: external/crisiswatch/entries.jsonl with
  {country, year, month, region, status, flags, text, source_file}

Usage:  python src/data/crisiswatch_parse.py [--pdf-dir external/crisiswatch/pdfs]
        python src/data/crisiswatch_parse.py --validate
"""
from __future__ import annotations
import argparse, collections, json, os, re, sys
from pypdf import PdfReader

G_UNCHANGED, G_IMPROVED, G_DETERIORATED = chr(0xF0C6), chr(0xF0C7), chr(0xF0C8)
GLYPH_STATUS = {G_UNCHANGED: "unchanged", G_IMPROVED: "improved",
                G_DETERIORATED: "deteriorated"}
PUA_LO, PUA_HI = chr(0xE000), chr(0xF8FF)
RUN_RE = re.compile("[" + PUA_LO + "-" + PUA_HI + "](?:[\\s" + PUA_LO + "-" + PUA_HI + "])*")

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"])}
MONTHS.update({m[:3].lower(): v for m, v in list(MONTHS.items())})
MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

REGIONS = {"africa": "Africa", "asia": "Asia", "europe": "Europe/Central Asia",
           "latam": "Latin America", "latin": "Latin America",
           "mena": "MENA", "middle": "MENA", "northamerica": "North America"}
REGIONS_EXTRA = {"eca": "Europe/Central Asia",
                 "europe and central": "Europe/Central Asia"}

COUNTRY_RE = re.compile(
    r"([A-Z][\w'’./()À-ɏ-]*(?: [A-Z][\w'’./()À-ɏ-]*){0,6}?)"
    r"(?:\s{2,}|\n)")


def file_region(low):
    for k, v in {**REGIONS_EXTRA, **REGIONS}.items():
        if k in low:
            return v
    return None


def file_meta(name):
    """(year, month, region) from a bulletin filename, else None."""
    low = name.lower().replace("%20", " ").replace("_", " ").replace("-", " ")
    if "overview" in low or "monitoring" in low or " cif " in low:
        return None
    m = re.search(rf"\b({MONTH_ALT})\b", low)
    y = re.search(r"\b(20\d\d)\b", low)
    reg = file_region(low)
    if m and y and reg:
        return int(y.group(1)), MONTHS[m.group(1)], reg
    return None


def header_meta(path, name):
    """(year, month, region) from the PDF's first-page header, for files whose
    name carries no date (the CW<issue>_<REGION>.pdf family)."""
    low = name.lower().replace("_", " ").replace("-", " ")
    if "overview" in low or "monitoring" in low or " cif " in low:
        return None
    reg = file_region(low)
    if not reg:
        return None
    try:
        first = PdfReader(path).pages[0].extract_text() or ""
    except Exception:
        return None
    h = re.search(rf"CrisisWatch\s+(20\d\d)\s*[–—-]\s*({MONTH_ALT})\b",
                  first[:600], re.I)
    if h:
        return int(h.group(1)), MONTHS[h.group(2).lower()], reg
    h = re.search(rf"({MONTH_ALT})\s+(20\d\d)", first[:600], re.I)
    if h:
        return int(h.group(2)), MONTHS[h.group(1).lower()], reg
    return None


def parse_bulletin(path, year, month, region):
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    hits = list(RUN_RE.finditer(text))
    entries = []
    for i, h in enumerate(hits):
        run = [c for c in h.group(0) if PUA_LO <= c <= PUA_HI]
        arrows = [c for c in run if c in GLYPH_STATUS]
        flags = sorted({hex(ord(c)) for c in run
                        if c not in GLYPH_STATUS and ord(c) != 0xF020})
        status = GLYPH_STATUS[arrows[0]] if arrows else "no-arrow"
        chunk = text[h.end(): hits[i + 1].start() if i + 1 < len(hits) else len(text)]
        chunk = chunk.strip()
        m = COUNTRY_RE.match(chunk)
        if not m:
            continue
        country = m.group(1).strip()
        body = chunk[m.end():].strip()
        body = re.sub(r"\n[A-Z][A-Za-z &]{2,40}\n?\s*$", "", body).strip()
        entries.append({
            "country": country, "year": year, "month": month, "region": region,
            "status": status, "flags": flags,
            "text": re.sub(r"\s+", " ", body),
            "source_file": os.path.basename(path),
        })
    return entries


def parse_overview_lists(path):
    """'Deteriorated/Improved Situations' country lists from an overview PDF."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages[:3])
    out = {}
    for key, pat in [("deteriorated",
                      r"Deteriorated Situations?(.{0,600}?)(?:Improved|Conflict|Resolution|Outlook)"),
                     ("improved",
                      r"Improved Situations?(.{0,600}?)(?:Deteriorated|Conflict|Resolution|Outlook)")]:
        m = re.search(pat, text, re.S | re.I)
        out[key] = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default="external/crisiswatch/pdfs")
    ap.add_argument("--out", default="external/crisiswatch/entries.jsonl")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    files = sorted(os.listdir(args.pdf_dir))
    all_entries, skipped = [], []
    for f in files:
        meta = file_meta(f) or header_meta(os.path.join(args.pdf_dir, f), f)
        if not meta:
            skipped.append(f)
            continue
        try:
            all_entries += parse_bulletin(os.path.join(args.pdf_dir, f), *meta)
        except Exception as e:
            print(f"PARSE-FAIL {f}: {e}", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fh:
        for e in all_entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    n_by = collections.Counter(e["status"] for e in all_entries)
    ym = sorted({(e["year"], e["month"]) for e in all_entries})
    print(f"entries: {len(all_entries)}  months: {len(ym)} "
          f"({ym[0]}..{ym[-1]})  status: {dict(n_by)}")
    print(f"countries: {len({e['country'] for e in all_entries})}  "
          f"non-bulletin files skipped: {len(skipped)}")

    if args.validate:
        idx = collections.defaultdict(dict)
        for e in all_entries:
            idx[(e["year"], e["month"])][e["country"]] = e["status"]
        n_checks = n_mismatch = 0
        for f in files:
            if "overview" not in f.lower():
                continue
            low = f.lower().replace("-", " ").replace("_", " ")
            m = re.search(rf"\b({MONTH_ALT})\b", low)
            y = re.search(r"\b(20\d\d)\b", low)
            if not (m and y):
                continue
            key = (int(y.group(1)), MONTHS[m.group(1)])
            if key not in idx:
                continue
            lists = parse_overview_lists(os.path.join(args.pdf_dir, f))
            for status in ("deteriorated", "improved"):
                for c, s in idx[key].items():
                    n_checks += 1
                    in_list = c.split("(")[0].strip().lower() in lists[status].lower()
                    if (s == status) != in_list:
                        n_mismatch += 1
                        print(f"MISMATCH {key} {c}: glyph={s} "
                              f"overview_list_contains={in_list} ({status})")
        print(f"validation: {n_mismatch} mismatches / {n_checks} checks")


if __name__ == "__main__":
    main()
