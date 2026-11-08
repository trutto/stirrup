#!/usr/bin/env python3
"""Agentic test through the real Stirrup agent loop (/api/agent).

Six realistic tasks (system info, file ops, systemd ops, data processing, debugging, web
research). Fixtures live in /tmp/stirrup-agent; results land in results/agentic-stirrup-*.json.

Usage: python3 tests/harness_agentic.py [base-url]
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ARGS = [a for a in sys.argv[1:]]
BASE = next((a for a in ARGS if a.startswith("http")), "http://127.0.0.1:8090").rstrip("/")
MODEL = "groot"
if "--model" in ARGS:
    try:
        MODEL = ARGS[ARGS.index("--model") + 1]
    except IndexError:
        MODEL = "groot"
ONLY = ""
if "--only" in ARGS:
    try:
        ONLY = ARGS[ARGS.index("--only") + 1]
    except IndexError:
        ONLY = ""
FORCE_THINK = "--think" in ARGS
LLM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = "/tmp/stirrup-agent"
FIX = os.path.join(WORK, "fixtures")
MODELS = os.path.join(LLM, "models")
KEY = ""
try:
    KEY = open(os.path.join(LLM, "run", "local-llm.key")).read().strip()
except OSError:
    pass


def setup():
    os.makedirs(FIX, exist_ok=True)
    open(os.path.join(FIX, "fizz.py"), "w").write(
        'def fizz(n):\n    """Sum of integers 1..n (inclusive)."""\n    total = 0\n'
        '    for i in range(1, n):\n        total += i\n    return total\n')
    open(os.path.join(FIX, "check_fizz.py"), "w").write(
        'import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
        'from fizz import fizz\nok = all(fizz(n) == w for n, w in [(1, 1), (5, 15), (10, 55)])\n'
        'print("TESTS PASS" if ok else "TESTS FAIL")\n')
    open(os.path.join(FIX, "sales.csv"), "w").write(
        "region,product,qty,price\n"
        "EU,widget,10,50.00\nEU,widget,5,50.00\nEU,gadget,12,25.00\n"
        "EU,doohickey,1,597.50\nUS,widget,99,10.00\nUS,gadget,50,5.00\n"
        "APAC,doohickey,7,1000.00\n")
    for f in ("hello.txt", "unit.txt", "answer.txt", "report.md"):
        try:
            os.remove(os.path.join(WORK, f))
        except OSError:
            pass


def sse_agent(prompt, max_steps=24, think=False, timeout=300):
    body = {"id": MODEL, "max_steps": max_steps, "think": think,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/api/agent", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"tools": [], "texts": [], "done": None, "steps": 0, "errors": [], "wall": 0.0,
           "compacted": 0}
    t0 = time.time()
    ev = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event: "):
                    ev = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                try:
                    d = json.loads(line[6:])
                except ValueError:
                    continue
                if ev == "step":
                    out["steps"] = max(out["steps"], d.get("n") or 0)
                elif ev == "tool_call":
                    out["tools"].append({"name": d.get("name"), "args": d.get("args")})
                elif ev == "tool_result":
                    if out["tools"]:
                        out["tools"][-1]["output"] = (d.get("output") or "")[:4000]
                elif ev == "text":
                    out["texts"].append(d.get("content") or "")
                elif ev == "done":
                    out["done"] = d
                elif ev == "error":
                    out["errors"].append(d.get("message") or "")
                elif ev == "compacted":
                    out["compacted"] += 1
    except Exception as e:
        out["errors"].append(repr(e))
    out["wall"] = round(time.time() - t0, 1)
    out["answer"] = ((out.get("done") or {}).get("text") or "\n".join(out["texts"])).strip()
    return out


def check(tasks):
    results = []
    for t in tasks:
        print("== %s: %s" % (t["id"], t["title"]), flush=True)
        r = sse_agent(t["prompt"], max_steps=t.get("max_steps", 24),
                      think=(True if FORCE_THINK else t.get("think", False)),
                      timeout=t.get("timeout", 300))
        try:
            ok, detail = t["check"](r)
        except Exception as e:
            ok, detail = False, "check error: %r" % e
        results.append({
            "id": t["id"], "title": t["title"], "ok": bool(ok), "detail": str(detail)[:300],
            "steps": r["steps"], "tools": [x.get("name") for x in r["tools"]],
            "wall_s": r["wall"], "answer": r["answer"][:600],
            "tool_trace": [{"name": x.get("name"), "args": x.get("args"),
                            "output": (x.get("output") or "")[:600]} for x in r["tools"]],
        })
        print(("  PASS" if ok else "  FAIL") + "  " + str(detail)[:140] +
              "  (%d steps, %ss, tools=%d)" % (r["steps"], r["wall"], len(r["tools"])), flush=True)
    return results


def fetch_github_latest(repo="ggml-org/llama.cpp"):
    req = urllib.request.Request(
        "https://api.github.com/repos/%s/releases/latest" % repo,
        headers={"User-Agent": "stirrup-test/1.0", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    return d.get("tag_name") or "", (d.get("published_at") or "")[:10]


def fetch_hn_title():
    def api(url):
        req = urllib.request.Request(url, headers={"User-Agent": "stirrup-test/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    top = api("https://hacker-news.firebaseio.com/v0/topstories.json")[:1]
    item = api("https://hacker-news.firebaseio.com/v0/item/%d.json" % top[0])
    return item.get("title") or ""


def report_lines():
    path = os.path.join(WORK, "report.md")
    if not os.path.exists(path):
        return []
    lines = [x.strip() for x in open(path).read().splitlines() if x.strip()]
    lines = [x for x in lines if not x.lower().startswith("sources:")]
    return [x.strip("`-* ").strip() for x in lines]


def check_a6(r):
    try:
        tag, published = fetch_github_latest()
        hn_title = fetch_hn_title()
    except Exception as e:
        return False, "live fact check failed: %r" % e
    L = report_lines()
    if len(L) != 3:
        return False, "expected 3 lines, got %d: %r" % (len(L), L[:4])
    ok = (L[0] == tag and L[1] == published and
          (L[2] == hn_title or hn_title.lower() in L[2].lower()))
    return ok, "want %s / %s / %s, got %s / %s / %s" % (
        tag, published, hn_title[:40], L[0], L[1], L[2][:40])


def main():
    setup()
    pid = ""
    try:
        pid = subprocess.run(["systemctl", "--user", "show", "local-llm-harness.service",
                              "-p", "MainPID", "--value"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    tasks = [
        {"id": "A1-sysinfo", "title": "Real machine state: CPU load + battery",
         "prompt": "Report the current 1-minute CPU load average and the battery charge percentage "
                   "of this laptop using the available tools (sysinfo is the fast, exact one). "
                   "Answer with exactly one line: load=<number> battery=<number>%.",
         "check": lambda r: (bool(re.search(r"load\s*=?\s*(\d+\.\d+)", r["answer"], re.I) and
                                 re.search(r"battery\s*=?\s*(\d+)\s*%", r["answer"], re.I)) and
                             "sysinfo" in [x.get("name") for x in r["tools"]],
                             r["answer"][:100] + " | tools=" +
                             ",".join(str(x.get("name")) for x in r["tools"]))},
        {"id": "A2-file", "title": "Create a file and verify it",
         "prompt": "Create the file %s/hello.txt containing exactly the text stirrup agentic ok, "
                   "then print its contents and answer done." % WORK,
         "check": lambda r: (os.path.exists(os.path.join(WORK, "hello.txt")) and
                             open(os.path.join(WORK, "hello.txt")).read().strip() == "stirrup agentic ok",
                             "hello.txt=" + repr(open(os.path.join(WORK, "hello.txt")).read()[:60])
                             if os.path.exists(os.path.join(WORK, "hello.txt")) else "missing")},
        {"id": "A3-systemd", "title": "Find the running unit / port / workdir",
         "prompt": "Find the systemd USER unit whose process is listening on 127.0.0.1:8090 using "
                   "port_owner(8090) and systemctl --user show. Write one line to %s/unit.txt: "
                   "<unit name>, <MainPID>, <WorkingDirectory> separated by the pipe character. "
                   "Do not guess." % WORK,
         "check": lambda r: (
             (lambda o: ("local-llm-harness.service" in o and pid and pid in o and
                         "default-workspace/llm" in o, o[:150]))(
                 open(os.path.join(WORK, "unit.txt")).read().strip()
                 if os.path.exists(os.path.join(WORK, "unit.txt")) else ""))},
        {"id": "A4-data", "title": "CSV analysis: EU revenue + best product",
         "prompt": "Read %s/sales.csv. Compute (a) total EU revenue = sum(qty*price) for rows where "
                   "region is EU rounded to 2 decimals and (b) the EU product with the highest total "
                   "qty. Write exactly two lines to %s/answer.txt: line 1 the number, line 2 the "
                   "product name." % (FIX, WORK),
         "check": lambda r: (
             (lambda A: (len(A) >= 2 and "1647.50" in A[0] and A[1].lower() == "widget",
                         " | ".join(A[:2])))(
                 [x.strip() for x in (open(os.path.join(WORK, "answer.txt")).read().splitlines()
                                      if os.path.exists(os.path.join(WORK, "answer.txt")) else []) if x.strip()]))},
        {"id": "A5-debug", "title": "Fix a bug and prove the test passes",
         "prompt": "The file %s/fizz.py has a bug. Its test is %s/check_fizz.py. Fix the bug in "
                   "fizz.py (only that file) so the test passes. Run the test to confirm, then say "
                   "which function you changed and whether it passed." % (FIX, FIX),
         "check": lambda r: (
             (lambda p: ("TESTS PASS" in p, p.strip()[-40:]))(
                 subprocess.run([sys.executable, os.path.join(FIX, "check_fizz.py")],
                                capture_output=True, text=True).stdout))},
        {"id": "A6-research", "title": "Live web research into a strict 3-line file",
         "prompt": "Use github_latest_release(ggml-org/llama.cpp) for the first two lines and "
                   "hn_top(1) for the third. Write exactly three lines to %s/report.md and nothing "
                   "else: line 1 the exact latest llama.cpp release tag, line 2 its publish date "
                   "(YYYY-MM-DD), line 3 the exact title of the top Hacker News story right now. "
                   "No prefixes, no markdown. Then say done." % WORK,
         "check": check_a6},
    ]
    if ONLY:
        ids = [x.strip() for x in ONLY.split(",") if x.strip()]
        tasks = [t for t in tasks if t["id"] in ids]
    results = check(tasks)
    passed = sum(1 for x in results if x["ok"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(LLM, "results"), exist_ok=True)
    with open(os.path.join(LLM, "results", "agentic-stirrup-%s.json" % stamp), "w") as f:
        json.dump({"base": BASE, "at": stamp, "passed": passed, "total": len(results),
                   "results": results}, f, indent=2)
    print("\nStirrup agentic test: %d/%d passed" % (passed, len(results)))
    try:
        for f in os.listdir(WORK):
            p = os.path.join(WORK, f)
            if os.path.isfile(p):
                os.remove(p)
    except OSError:
        pass
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
