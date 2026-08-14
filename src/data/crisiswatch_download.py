#!/usr/bin/env python
"""
Download the CrisisWatch bulletin PDF archive from ICG's public asset host.

Source: the PDF links on ICG's own archive page (crisisgroup.org/es/node/21470),
enumerated via an Internet Archive capture of that page. The /sites/default/files/
asset paths serve directly. Polite: sequential, delayed, retried with backoff.
Resumable: existing non-empty files are skipped.

Usage:  python src/data/crisiswatch_download.py --urls external/cw_pdf_urls.txt
Output: external/crisiswatch/pdfs/<sanitized filename>
"""
from __future__ import annotations
import argparse, os, re, subprocess, time, urllib.request

# curl, not urllib: the host's CDN accepts curl's TLS fingerprint with a browser
# UA but 403s Python's ssl stack regardless of headers.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
WAYBACK_PREFIX = re.compile(r"https://web\.archive\.org/web/\d+/")


def fetch(url, path, tries=4):
    for a in range(tries):
        r = subprocess.run(
            ["curl", "-sL", "--fail", "-A", UA, "--max-time", "120",
             "-o", path, url], capture_output=True)
        if r.returncode == 0 and os.path.exists(path):
            with open(path, "rb") as fh:
                if fh.read(4) == b"%PDF":
                    return True
        if r.returncode == 22 and b"404" in r.stderr:
            return False
        wait = 5 * (2 ** a)
        print(f"retry {a+1} (rc={r.returncode}) sleeping {wait}s", flush=True)
        time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", default="external/crisiswatch/pdfs")
    ap.add_argument("--delay", type=float, default=1.5)
    args = ap.parse_args()

    urls = [WAYBACK_PREFIX.sub("", u.strip()) for u in open(args.urls) if u.strip()]
    os.makedirs(args.out, exist_ok=True)
    ok = skip = fail = 0
    for i, url in enumerate(urls):
        name = urllib.request.unquote(url.rsplit("/", 1)[-1])
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        path = os.path.join(args.out, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            skip += 1
            continue
        if fetch(url, path):
            ok += 1
        else:
            if os.path.exists(path):
                os.remove(path)
            fail += 1
            print(f"FAIL {name}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(urls)} ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(args.delay)
    print(f"CRISISWATCH_DOWNLOAD_DONE ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
