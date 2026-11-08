#!/usr/bin/env python3
"""Stirrup harness self-test — API/model checks against a running harness.

Usage: python3 tests/harness_selftest.py [http://host:port]
Exit code 0 = all checks passed. Creates and deletes one throwaway session.
"""
import json
import os
import re
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090").rstrip("/")
LLM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []


def rec(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": str(detail)[:160]})
    print(("PASS  " if ok else "FAIL  ") + name + "  " + str(detail)[:120])


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post(path, body, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def sse(path, body, timeout=300):
    """POST and collect events: {'meta': {...}, 'choices text', 'usage': {...}, 'done': {...}}."""
    out = {"reasoning": "", "content": "", "meta": None, "usage": None, "done": None,
           "tool_calls": [], "tool_results": [], "events": []}
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    ev = None
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
            out["events"].append(ev or "")
            if ev == "compacted":
                out["compacted"] = d
            elif ev == "tool_call":
                out["tool_calls"].append(d)
            elif ev == "tool_result":
                out["tool_results"].append(d)
            elif ev == "usage":
                out["usage"] = d
            elif ev == "done":
                out["done"] = d
            elif ev in ("step", "token", "thinking", "text"):
                ch = d if ev == "thinking" else {}
                if ev == "thinking" and d.get("content"):
                    out["reasoning"] += d["content"]
                if ev == "text" and d.get("content"):
                    out["content"] += d["content"]
            elif d.get("model") and d.get("choices"):
                # plain OpenAI-style chat chunk (the chat endpoint sends them after "event: meta")
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    out["reasoning"] += delta["reasoning_content"]
                if delta.get("content"):
                    out["content"] += delta["content"]
                t = d.get("timings") or {}
                if t.get("predicted_per_second"):
                    out.setdefault("timings", t)
            elif d.get("choices"):
                delta = (d.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    out["reasoning"] += delta["reasoning_content"]
                if delta.get("content"):
                    out["content"] += delta["content"]
            elif "promptTokens" in d or "prompt_tokens" in d or "model" in d:
                if not out["meta"]:
                    out["meta"] = d
            if d.get("timings") and not out.get("timings"):
                out["timings"] = d.get("timings")
            if d.get("usage") and not out.get("usage"):
                out["usage"] = d.get("usage")
    return out


def main():
    # 1-5: service + branding
    try:
        h = get("/healthz")
        st = get("/api/state")
        rec("service health", h.get("ok") is True and st.get("phase") == "ready",
            "%s / %s" % (st.get("phase"), st.get("active")))
    except Exception as e:
        rec("service health", False, repr(e))
        print("harness unreachable; aborting")
        return 1
    mode = st.get("active") or "groot"
    ver = get("/api/version")
    man = get("/manifest.webmanifest")
    rec("branding (api/manifest)", ver.get("brand") == "Stirrup" and man.get("name") == "Stirrup",
        "%s / %s" % (ver.get("brand"), man.get("name")))
    feats = ver.get("features") or []
    rec("feature flags", all(x in feats for x in ("session-sync", "mcp", "compaction", "hf-download")),
        ",".join(feats[-5:]))
    html = urllib.request.urlopen(BASE + "/", timeout=20).read().decode("utf-8", "replace")
    rec("page title", "<title>Stirrup</title>" in html, "Stirrup")

    # 6-7: chat + thinking + usage
    chat = sse("/api/chat", {"id": mode, "think": True,
                             "messages": [{"role": "user", "content": "In one sentence: why is the sky blue?"}],
                             "max_tokens": 700}, timeout=180)
    rec("chat answer", bool(chat["content"].strip()) and re.search(r"blue|scatter|Rayleigh", chat["content"], re.I),
        chat["content"][:90].replace("\n", " "))
    rec("live thinking", bool(chat["reasoning"].strip()), "%d reasoning chars" % len(chat["reasoning"]))
    us = chat.get("usage") or {}
    rec("usage tokens", bool(us.get("promptTokens") or us.get("prompt_tokens") or
                             (chat.get("meta") or {}).get("promptTokens")),
        "prompt=%s completion=%s" % (us.get("promptTokens") or (chat.get("meta") or {}).get("promptTokens"),
                                     us.get("completionTokens") or us.get("completion_tokens")))

    # 8: agent round-trip (count gguf files)
    models = os.path.join(LLM, "models")
    expect = len([f for f in os.listdir(models) if f.endswith(".gguf")]) if os.path.isdir(models) else 0
    ag = sse("/api/agent", {"id": mode, "max_steps": 12, "think": False,
                            "messages": [{"role": "user", "content":
                                          "Count the .gguf files in %s and answer with just the number. Use bash." % models}]},
             timeout=240)
    final = (ag.get("done") or {}).get("text") or ag["content"]
    rec("agent round-trip", bool(ag["tool_calls"]) and str(expect) in final,
        "answer contains %d, tools=%d" % (expect, len(ag["tool_calls"])))

    # 9-10: MCP add -> approve -> agent call -> remove
    server = {"name": "selftest_echo",
              "command": "python3 " + os.path.join(LLM, "tests", "mcp_echo_server.py")}
    mcp_out = {}
    try:
        post("/api/mcp/add", server, timeout=30)
        ready = False
        for _ in range(40):
            time.sleep(0.5)
            mcp = get("/api/mcp")
            sv = next((s for s in (mcp.get("servers") or []) if s["name"] == "selftest_echo"), None)
            if sv and sv.get("status") == "ready" and sv.get("tools"):
                ready = True
                break
        post("/api/mcp/toggle", {"server": "selftest_echo", "tool": "echo", "approved": True}, timeout=30)
        mcp_out = sse("/api/agent", {"id": mode, "max_steps": 6, "think": False,
                                     "messages": [{"role": "user", "content":
                                                   "Use the MCP tool mcp__selftest_echo__echo with text HARNESS-SELFTEST "
                                                   "and answer with exactly what it returns. Use only that tool."}]},
                      timeout=180) if ready else {}
        ok = ready and any(c.get("name") == "mcp__selftest_echo__echo"
                           for c in mcp_out.get("tool_calls", [])) and "MCP ECHO: HARNESS-SELFTEST" in \
             ((mcp_out.get("done") or {}).get("text") or "") + mcp_out.get("content", "")
        rec("MCP approve + agent call", ok, "ready=%s tools=%d" % (ready, len(mcp_out.get("tool_calls", []))))
    finally:
        try:
            after = json.loads(post("/api/mcp/remove", {"server": "selftest_echo"}, timeout=30))
            rec("MCP cleanup", not (after.get("servers") or []), "servers=%d" % len(after.get("servers") or []))
        except Exception as e:
            rec("MCP cleanup", False, repr(e))

    # 11-12: Hugging Face APIs
    hf = get("/api/hf/search?q=Qwen2.5-0.5B-Instruct-GGUF", timeout=60)
    rec("HF search", bool(hf.get("results")), "%d repos" % len(hf.get("results") or []))
    try:
        files = get("/api/hf/files?repo=Qwen%2FQwen2.5-0.5B-Instruct-GGUF", timeout=90)
        fl = files.get("files") or []
        rec("HF file list", bool(fl) and bool(fl[0].get("sha256")) and fl[0].get("vram_gb"),
            "%d GGUF e.g. %s" % (len(fl), fl[0].get("quant") if fl else "?"))
    except Exception as e:
        rec("HF file list", False, repr(e))

    # 13: session sync API (create -> visible -> delete -> tombstone)
    sid = "selftest-" + str(int(time.time()))
    try:
        post("/api/sessions", {"sessions": [{"id": sid, "title": "selftest sync",
                                             "created": int(time.time() * 1000),
                                             "updated": int(time.time() * 1000),
                                             "msgs": [{"role": "user", "content": "sync test"}]}]}, timeout=30)
        seen = any(s.get("id") == sid for s in (get("/api/sessions").get("sessions") or []))
        post("/api/sessions", {"deletes": [sid]}, timeout=30)
        data = get("/api/sessions")
        gone = not any(s.get("id") == sid for s in (data.get("sessions") or []))
        rec("session sync API", seen and gone, "created=%s deleted=%s" % (seen, gone))
    except Exception as e:
        rec("session sync API", False, repr(e))
        try:
            post("/api/sessions", {"deletes": [sid]}, timeout=10)
        except Exception:
            pass

    # 14: compaction helper (no live model needed)
    try:
        sys.path.insert(0, os.path.join(LLM, "scripts"))
        import harness
        harness._summarise = lambda mm, text: "Summary: disk 646 GiB free; 27 gguf files."
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(40):
            msgs.append({"role": "user", "content": ("question %d " % i) * 400})
            msgs.append({"role": "assistant", "content": ("answer %d " % i) * 400})
        out, info = harness.compact_context(mode if mode in harness.MODELS else "groot", msgs, "selftest")
        rec("compaction helper", bool(info) and len(out) == 7 and "Summary:" in out[0]["content"],
            "81 -> %d msgs, %s tok summary" % (len(out), (info or {}).get("summaryTokens")))
    except Exception as e:
        rec("compaction helper", False, repr(e))

    # 15-21: new agent tools, sandbox, policy API, meta wiring
    try:
        sys.path.insert(0, os.path.join(LLM, "scripts"))
        import harness
        si = harness.tool_sysinfo()
        rec("sysinfo tool", "load_1m=" in si and "memory_total=" in si, si.splitlines()[1][:80])
        po = harness.tool_port_owner(8090)
        rec("port_owner tool", "local-llm-harness.service" in po and "pid=" in po,
            po.splitlines()[2][:80] if len(po.splitlines()) > 2 else po[:80])
        gh = harness.tool_github_latest_release("ggml-org/llama.cpp")
        rec("github release tool", "tag=" in gh and "published=" in gh, gh.splitlines()[2][:80])
        hn = harness.tool_hn_top(1)
        rec("hn top tool", hn.startswith("hn_top:") and "|" in hn,
            hn.splitlines()[1][:80] if len(hn.splitlines()) > 1 else hn[:80])
        blk = harness.run_tool("write_file", {"path": "/etc/stirrup-selftest", "content": "x"})
        rec("write sandbox", blk.startswith("blocked:"), blk[:80])
        pol = get("/api/agent/policy")
        rec("agent policy API", pol.get("approval") in ("allow", "ask") and
            pol.get("auto_think") in ("off", "on-failure", "auto") and
            pol.get("time_budget_s") and pol.get("write_roots"), json.dumps(pol)[:90])
    except Exception as e:
        rec("agent tools/policy", False, repr(e))
    try:
        policy_ok = True
        post("/api/agent/policy", {"approval": "ask"}, timeout=20)
        policy_ok = get("/api/agent/policy").get("approval") == "ask"
        post("/api/agent/policy", {"approval": "ask", "time_budget_s": 240,
                                   "auto_think": "auto"}, timeout=20)
        p2 = get("/api/agent/policy")
        policy_ok = policy_ok and p2.get("time_budget_s") == 240 and p2.get("auto_think") == "auto"
        rec("agent policy switch", policy_ok, "ask + 240s + auto-think")
    except Exception as e:
        rec("agent policy switch", False, repr(e))
    finally:
        try:
            post("/api/agent/policy", {"approval": "allow", "time_budget_s": 300,
                                       "auto_think": "on-failure"}, timeout=20)
        except Exception:
            pass
    try:
        am = sse("/api/agent", {"id": mode, "max_steps": 2, "think": False,
                                "messages": [{"role": "user", "content": "Reply with just: ok"}]},
                 timeout=120)
        meta = am.get("meta") or {}
        rec("agent meta policy/budget", meta.get("approval") == "allow" and meta.get("timeBudget") == 300
            and meta.get("autoThink") == "on-failure" and meta.get("thinking") is False,
            "approval=%s budget=%s autoThink=%s" % (meta.get("approval"), meta.get("timeBudget"),
                                                    meta.get("autoThink")))
    except Exception as e:
        rec("agent meta policy/budget", False, repr(e))

    # 22: auto-think modes (AUTO plan prompt with think=false)
    try:
        plan = ("Explain step by step how you would check disk usage on this machine, then run "
                "exactly one command to check it.")
        def agent_think_state():
            r = sse("/api/agent", {"id": mode, "max_steps": 2, "think": False,
                                   "messages": [{"role": "user", "content": plan}]}, timeout=120)
            m = r.get("meta") or {}
            return m.get("thinking"), m.get("autoThink"), bool(r.get("reasoning"))
        post("/api/agent/policy", {"auto_think": "off"}, timeout=20)
        off_thinking = agent_think_state()
        post("/api/agent/policy", {"auto_think": "auto"}, timeout=20)
        auto_thinking = agent_think_state()
        ok = (off_thinking[0] is False and off_thinking[1] == "off" and
              auto_thinking[0] is True and auto_thinking[1] == "auto")
        rec("auto-think modes", ok, "off=%s auto=%s" % (off_thinking[:2], auto_thinking[:2]))
    except Exception as e:
        rec("auto-think modes", False, repr(e))
    finally:
        try:
            post("/api/agent/policy", {"auto_think": "on-failure", "approval": "allow",
                                       "time_budget_s": 300}, timeout=20)
        except Exception:
            pass

    # 23: VoiceStudio tool wiring
    try:
        sys.path.insert(0, os.path.join(LLM, "scripts"))
        import harness
        vl = harness.tool_voice_list()
        vs = harness.tool_voice_speak({"text": "VoiceStudio self test.", "format": "wav"})
        path = vs.split("speech saved: ", 1)[-1].split(" (", 1)[0] if vs.startswith("speech saved: ") else ""
        ok = "VoiceStudio:" in vl and path and os.path.isfile(path) and os.path.getsize(path) > 1000
        rec("VoiceStudio TTS tools", bool(ok), vs[:110] if vs else vl[:110])
    except Exception as e:
        rec("VoiceStudio TTS tools", False, repr(e))

    passed = sum(1 for x in results if x["ok"])
    print("\nStirrup self-test: %d/%d passed" % (passed, len(results)))
    out_path = os.path.join(LLM, "results", "selftest-last.json")
    try:
        with open(out_path, "w") as f:
            json.dump({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "base": BASE, "results": results}, f, indent=2)
    except OSError:
        pass
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
