"""
LLM-backed Intent and Validation agents (Together API, OpenAI-compatible).

This module supplies the *agentic* Intent Agent used by Experiment 5. It is the
only component of the pipeline that plans, calls tools and can be sent back to
re-do its work; every other stage (candidate screening, Selection Agent,
Explanation Agent, feasibility contract) is byte-identical to E1. E5 is
therefore a single-component substitution in exactly the same sense as E2 and
E3, which is what keeps the nested ablation chain interpretable.

Six properties a reviewer will look for, and where each lives:

    goal-directed reasoning  INTENT_SYS -- the LLM decides *how* to extract
    tool use                 TOOL_SPECS + _dispatch -- four typed tools
    planning / iteration     intent_agent() -- multi-turn until the frame closes
    inter-agent messaging    run_agentic() -- typed append-only blackboard
    negotiation / critique   validation_agent() -- may return RETRY
    autonomy with deferral   validation_agent() -- may return ESCALATE

Determinism and reproducibility
-------------------------------
Every call is made at ``temperature=0`` with a fixed ``seed`` and is cached on
disk keyed by a hash of (model, messages, tools). Committing
``results/llm_cache.json`` lets anyone reproduce E5 bit-exactly with **no API
key**, which pre-empts the standard "LLM results are not reproducible"
objection. Set ``XAGENTIC_LLM_OFFLINE=1`` to assert that no network call is
made (a cache miss then raises instead of silently costing money).

Usage
-----
    from agents import run_agentic, agent_stats, flush_cache
    bb = run_agentic(text, select_fn, explain_fn)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xagentic_common import (  # noqa: E402
    CATALOGUE, LEXICON, SLICES, OUT, SEED, feasible_set,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
MODEL = os.environ.get("XAGENTIC_AGENT_MODEL",
                       "meta-llama/Llama-3.3-70B-Instruct-Turbo")
MAX_TOKENS = 512

# Together list price for Llama-3.3-70B-Instruct-Turbo (USD per 1M tokens).
# Recorded here so the paper can quote a cost without a second lookup.
PRICE_IN_PER_M = float(os.environ.get("XAGENTIC_PRICE_IN", "0.88"))
PRICE_OUT_PER_M = float(os.environ.get("XAGENTIC_PRICE_OUT", "0.88"))

OFFLINE = os.environ.get("XAGENTIC_LLM_OFFLINE", "").strip().lower() \
    not in ("", "0", "false", "no", "off")

CACHE_PATH = Path(OUT) / "llm_cache.json"
_cache: dict = {}
if CACHE_PATH.exists():
    try:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _cache = {}

_session = None

STATS = {
    "calls": 0, "cache_hits": 0, "tok_in": 0, "tok_out": 0,
    "latency_s": 0.0, "api_errors": 0, "parse_failures": 0,
    "model": MODEL, "seed": SEED,
}


def _sess():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
    return _session


def _slim(msg: dict) -> dict:
    """Keep only the fields the chat API will accept back as history."""
    out = {"role": msg.get("role", "assistant"),
           "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
        # A tool-calling turn must carry null content, not "".
        out["content"] = msg.get("content") or None
    return out


def _chat(messages, tools=None, max_tokens=MAX_TOKENS):
    """Deterministic, disk-cached chat completion. Returns the message dict."""
    key = hashlib.sha256(json.dumps(
        {"m": MODEL, "msg": messages, "t": tools, "mt": max_tokens},
        sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if key in _cache:
        STATS["cache_hits"] += 1
        return _cache[key]

    if OFFLINE:
        raise RuntimeError(
            "XAGENTIC_LLM_OFFLINE is set but the LLM cache missed. The cache "
            "does not cover this run (different corpus, seed or prompt).")

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY is not set and the call is not cached")

    body = {"model": MODEL, "messages": messages, "temperature": 0.0,
            "seed": SEED, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    out = None
    ok = False
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            # Qwen3.7-Max is streaming-only on Together.
            use_stream = MODEL == "Qwen/Qwen3.7-Max"
            req_body = dict(body)
            if use_stream:
                req_body["stream"] = True
                req_body["stream_options"] = {"include_usage": True}

            r = _sess().post(
                TOGETHER_URL,
                json=req_body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
                stream=use_stream,
            )
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()

            if not use_stream:
                j = r.json()
                u = j.get("usage") or {}
                out = j["choices"][0]["message"]
            else:
                content_parts = []
                tool_calls_by_index = {}
                u = {}

                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue

                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue

                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break

                    chunk = json.loads(payload)

                    if chunk.get("usage"):
                        u = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}

                    if delta.get("content"):
                        content_parts.append(delta["content"])

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)

                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": "",
                                },
                            }

                        dst = tool_calls_by_index[idx]

                        if tc.get("id"):
                            dst["id"] += tc["id"]

                        if tc.get("type"):
                            dst["type"] = tc["type"]

                        fn = tc.get("function") or {}

                        if fn.get("name"):
                            dst["function"]["name"] += fn["name"]

                        if fn.get("arguments"):
                            dst["function"]["arguments"] += fn["arguments"]

                out = {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                }

                if tool_calls_by_index:
                    out["tool_calls"] = [
                        tool_calls_by_index[i]
                        for i in sorted(tool_calls_by_index)
                    ]

            STATS["latency_s"] += time.perf_counter() - t0
            STATS["calls"] += 1
            STATS["tok_in"] += int(u.get("prompt_tokens", 0))
            STATS["tok_out"] += int(u.get("completion_tokens", 0))
            ok = True
            break
        except Exception as e:  # pragma: no cover - network dependent
            STATS["api_errors"] += 1
            if attempt == 4:
                print(f"     [warn] LLM call failed permanently: "
                      f"{type(e).__name__}: {e}")
                out = {"role": "assistant", "content": ""}
                break
            time.sleep(1.5 * (attempt + 1))

    if out is None:
        # Five consecutive 429s is not an exception, so this path is reachable
        # without ever entering the handler above.
        STATS["api_errors"] += 1
        print("     [warn] LLM call abandoned after repeated rate limiting")
        out = {"role": "assistant", "content": ""}

    # Cache on provenance, not on content: a genuine API success is cached even
    # if the completion was empty, while a synthesised failure fallback never
    # is. Caching a failure would bake an outage into llm_cache.json, and
    # refusing to cache a legitimately empty answer would break the
    # XAGENTIC_LLM_OFFLINE replay guarantee.
    if ok:
        _cache[key] = out
        if (STATS["calls"] % 10) == 0:
            flush_cache()
    return out


def flush_cache():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def agent_stats():
    """Snapshot of call/token/cost accounting for e5_summary.json."""
    s = dict(STATS)
    s["cost_usd"] = (s["tok_in"] / 1e6 * PRICE_IN_PER_M
                     + s["tok_out"] / 1e6 * PRICE_OUT_PER_M)
    s["cache_size"] = len(_cache)
    total = s["calls"] + s["cache_hits"]
    s["cache_hit_rate"] = s["cache_hits"] / max(total, 1)
    s["mean_api_latency_s"] = s["latency_s"] / max(s["calls"], 1)
    s["offline"] = OFFLINE
    return s


def reset_stats():
    for k in ("calls", "cache_hits", "tok_in", "tok_out", "api_errors",
              "parse_failures"):
        STATS[k] = 0
    STATS["latency_s"] = 0.0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_kpi_envelope",
        "description": (
            "Return the 3GPP-informed KPI envelope for a standardised slice "
            "type: (min,max) for latency ms, throughput Mbps, reliability %, "
            "device density per km2 and mobility km/h."),
        "parameters": {"type": "object", "properties": {
            "slice_type": {"type": "string", "enum": SLICES}},
            "required": ["slice_type"]}}},
    {"type": "function", "function": {
        "name": "convert_units",
        "description": (
            "Normalise a KPI value to the canonical unit: latency->ms, "
            "throughput->Mbps, density->devices/km2, mobility->km/h."),
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"},
            "unit": {"type": "string",
                     "description": "e.g. s, ms, us, Gbps, Mbps, kbps, "
                                    "per_m2, per_km2, kmh, mph, m/s"},
            "kpi": {"type": "string",
                    "enum": ["latency", "throughput", "density", "mobility"]}},
            "required": ["value", "unit", "kpi"]}}},
    {"type": "function", "function": {
        "name": "query_lexicon",
        "description": (
            "Test a use-case phrase against the design-time vertical lexicon. "
            "Returns the matched slice or null. Use this to TEST a hypothesis, "
            "not to decide: a null result means the phrasing is novel, not "
            "that the use case is unclassifiable."),
        "parameters": {"type": "object", "properties": {
            "phrase": {"type": "string"}}, "required": ["phrase"]}}},
    {"type": "function", "function": {
        "name": "check_feasibility",
        "description": (
            "Given the KPI slots extracted so far, return the slices that are "
            "not provably excluded by the operator safety contract. Omit any "
            "KPI the request did not state."),
        "parameters": {"type": "object", "properties": {
            "latency_ms": {"type": "number"},
            "throughput_mbps": {"type": "number"},
            "reliability_pct": {"type": "number"},
            "density_per_km2": {"type": "number"},
            "mobility_kmh": {"type": "number"}},
            "required": []}}},
]

_FACTOR = {
    ("latency", "s"): 1000.0, ("latency", "sec"): 1000.0,
    ("latency", "ms"): 1.0, ("latency", "us"): 0.001, ("latency", "µs"): 0.001,
    ("throughput", "gbps"): 1000.0, ("throughput", "mbps"): 1.0,
    ("throughput", "kbps"): 0.001, ("throughput", "bps"): 1e-6,
    ("density", "per_km2"): 1.0, ("density", "per_m2"): 1e6,
    ("density", "per_km^2"): 1.0, ("density", "km2"): 1.0,
    ("mobility", "kmh"): 1.0, ("mobility", "km_h"): 1.0,
    ("mobility", "mph"): 1.609, ("mobility", "m_s"): 3.6, ("mobility", "ms"): 3.6,
}


def _envelope(s):
    c = CATALOGUE[s]
    return {"latency_ms": list(c["lat"]), "throughput_mbps": list(c["thr"]),
            "reliability_pct": list(c["rel"]), "density_per_km2": list(c["den"]),
            "mobility_kmh": list(c["mob"]), "sst": c["sst"]}


def _dispatch(name, args):
    try:
        if name == "get_kpi_envelope":
            s = str(args.get("slice_type", ""))
            if s not in CATALOGUE:
                return {"error": f"unknown slice {s!r}; expected one of {SLICES}"}
            return _envelope(s)

        if name == "convert_units":
            kpi = str(args.get("kpi", "")).lower()
            unit = str(args.get("unit", "")).lower().replace("/", "_").strip()
            f = _FACTOR.get((kpi, unit))
            if f is None:
                return {"value": float(args.get("value", 0.0)),
                        "note": f"unit {unit!r} not recognised for {kpi!r}; "
                                f"value returned unchanged"}
            return {"value": float(args.get("value", 0.0)) * f}

        if name == "query_lexicon":
            p = str(args.get("phrase", "")).lower()
            hits = [s for s in SLICES
                    if any(t.lower() in p for t in LEXICON[s])]
            return {"match": hits[0] if len(hits) == 1 else None, "all": hits}

        if name == "check_feasibility":
            import numpy as np
            row = {k: (float(args[k]) if args.get(k) is not None else np.nan)
                   for k in ("latency_ms", "throughput_mbps", "reliability_pct",
                             "density_per_km2", "mobility_kmh")}
            return {"feasible": feasible_set(row)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": f"unknown tool {name}"}


def _json_block(s):
    """Extract the last balanced JSON object from a model response."""
    if not s:
        raise ValueError("empty content")
    i, j = s.index("{"), s.rindex("}")
    return json.loads(s[i:j + 1])


# ---------------------------------------------------------------------------
# Intent Agent
# ---------------------------------------------------------------------------

INTENT_SYS = """You are the Intent Agent in a 5G network-slice selection system.

GOAL: turn a free-form service request into a complete, unit-normalised slot frame.

You MAY call tools. Suggested strategy:
 1. Extract every KPI the request actually states. Normalise with convert_units.
 2. If a KPI is NOT stated, leave it null. Never invent a value.
 3. Identify the use-case phrase describing WHAT the service is.
 4. Call query_lexicon on that phrase. A null match means the phrasing is novel,
    NOT that it is unclassifiable: reason from the SEMANTICS of the use case and
    compare the stated KPIs against get_kpi_envelope for each slice.
 5. Optionally call check_feasibility to see which slices remain admissible.
 6. Emit slice_hint and an honest confidence in [0,1].

Slice semantics:
 eMBB  = enhanced mobile broadband: high throughput, video, media, human users.
 URLLC = ultra-reliable low-latency: control, safety, automation, tight deadlines.
 mMTC  = massive machine-type: very many low-rate sensors/meters, delay tolerant.

Finish by returning ONLY this JSON object and no prose:
{"latency_ms":float|null,"throughput_mbps":float|null,"reliability_pct":float|null,
 "density_per_km2":float|null,"mobility_kmh":float|null,
 "use_case_phrase":string,"slice_hint":"eMBB"|"URLLC"|"mMTC"|null,
 "confidence":float,"reasoning":string}"""

_FRAME_KEYS = ("latency_ms", "throughput_mbps", "reliability_pct",
               "density_per_km2", "mobility_kmh")


def _empty_frame(reason=""):
    f = {k: None for k in _FRAME_KEYS}
    f.update(use_case_phrase="", slice_hint=None, confidence=0.0,
             reasoning=reason, ok=False)
    return f


def _coerce_frame(d):
    out = {}
    for k in _FRAME_KEYS:
        v = d.get(k)
        try:
            out[k] = None if v is None else float(v)
        except (TypeError, ValueError):
            out[k] = None
    hint = d.get("slice_hint")
    out["slice_hint"] = hint if hint in SLICES else None
    out["use_case_phrase"] = str(d.get("use_case_phrase", ""))[:200]
    try:
        out["confidence"] = min(max(float(d.get("confidence", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out["reasoning"] = str(d.get("reasoning", ""))[:400]
    out["ok"] = True
    return out


def intent_agent(text, feedback=None, max_turns=4):
    """Multi-turn, tool-using extraction. Returns (frame, tool_trace)."""
    msgs = [{"role": "system", "content": INTENT_SYS},
            {"role": "user", "content": f"Service request:\n{text}"}]
    if feedback:
        msgs.append({"role": "user", "content":
                     "The Validation Agent REJECTED your previous frame with "
                     f"this reason: {feedback}\nRe-extract, fixing that issue."})

    trace = []
    for _ in range(max_turns):
        m = _chat(msgs, tools=TOOL_SPECS)
        # Drop tool calls carrying no id. An OpenAI-compatible API rejects both
        # a tool message with an empty tool_call_id AND an assistant turn whose
        # tool_call is left unanswered, so the advertised list and the answered
        # list must be filtered together, before the turn enters the history.
        tcs = [tc for tc in (m.get("tool_calls") or []) if tc.get("id")]
        m = dict(m)
        if tcs:
            m["tool_calls"] = tcs
        else:
            m.pop("tool_calls", None)
        msgs.append(_slim(m))

        if not tcs:
            try:
                return _coerce_frame(_json_block(m.get("content", ""))), trace
            except Exception:
                STATS["parse_failures"] += 1
                msgs.append({"role": "user",
                             "content": "Return ONLY the JSON object."})
                continue

        for tc in tcs:
            fn = tc.get("function", {}).get("name", "")
            raw = tc.get("function", {}).get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                args = {}
            res = _dispatch(fn, args)
            trace.append({"tool": fn, "args": args, "result": res})
            msgs.append({"role": "tool",
                         "tool_call_id": tc["id"],
                         "name": fn,
                         "content": json.dumps(res, default=str)})

    return _empty_frame("max turns exceeded"), trace


# ---------------------------------------------------------------------------
# Validation Agent
# ---------------------------------------------------------------------------

VALID_SYS = """You are the Validation Agent. You audit a slice decision before execution.

You receive the original request, the extracted slot frame, the slice the
Selection Agent chose, the feasibility-contract result, and the top-3 feature
attributions produced by the Explanation Agent.

Return ONE verdict:
 ACCEPT   - the frame is faithful to the request AND the attribution supports
            the chosen slice.
 RETRY    - the slot frame misreads the request: a KPI is wrong, missing, or
            mis-scaled, or the use-case phrase is wrong. Say precisely what.
 ESCALATE - the frame is right but the evidence is genuinely weak or
            self-contradictory: the contract is invalid, or the dominant
            attribution contradicts the stated use case.

Be strict about RETRY: only use it when re-extraction could plausibly fix the
problem. Return ONLY:
{"verdict":"ACCEPT"|"RETRY"|"ESCALATE","reason":string,"confidence":float}"""


def validation_agent(text, frame, slice_sel, contract_ok, top_attr):
    payload = {"request": text,
               "slot_frame": {k: frame.get(k) for k in
                              list(_FRAME_KEYS) + ["use_case_phrase",
                                                   "slice_hint", "confidence"]},
               "selected_slice": slice_sel,
               "contract_valid": bool(contract_ok),
               "top_attributions": top_attr}
    m = _chat([{"role": "system", "content": VALID_SYS},
               {"role": "user", "content": json.dumps(payload, default=str)}],
              max_tokens=256)
    try:
        d = _json_block(m.get("content", ""))
        v = str(d.get("verdict", "")).strip().upper()
        if v not in ("ACCEPT", "RETRY", "ESCALATE"):
            raise ValueError(f"bad verdict {v!r}")
        try:
            c = min(max(float(d.get("confidence", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError):
            c = 0.0
        return {"verdict": v, "reason": str(d.get("reason", ""))[:300],
                "confidence": c}
    except Exception:
        STATS["parse_failures"] += 1
        return {"verdict": "ESCALATE", "reason": "unparseable verdict",
                "confidence": 0.0}


# ---------------------------------------------------------------------------
# Orchestrated pipeline
# ---------------------------------------------------------------------------

def run_agentic(text, select_fn, explain_fn, max_retry=1):
    """Full agentic loop over one intent.

    `select_fn(frame) -> (slice_label, proba_vector)`
    `explain_fn(frame) -> list of (feature_name, signed_attribution)`

    Returns the blackboard: a typed, append-only record of everything the
    agents wrote, which is also the artefact an operator would audit.
    """
    bb = {"intent_text": text, "trace": [], "retries": 0}
    feedback = None
    frame = slice_sel = proba = top_attr = None
    ok = False
    verdict = {"verdict": "ESCALATE", "reason": "not run", "confidence": 0.0}

    for attempt in range(max_retry + 1):
        frame, ttrace = intent_agent(text, feedback)
        bb["slot_frame"] = frame
        bb["trace"] += ttrace

        slice_sel, proba = select_fn(frame)
        top_attr = explain_fn(frame)
        ok = _contract(slice_sel, frame)
        verdict = validation_agent(text, frame, slice_sel, ok, top_attr)
        bb["trace"].append({"attempt": attempt, "verdict": verdict})
        bb["retries"] = attempt

        if verdict["verdict"] != "RETRY":
            break
        feedback = verdict["reason"]

    bb.update(decision=slice_sel, proba=proba, attribution=top_attr,
              contract_valid=ok, verdict=verdict,
              escalate=(verdict["verdict"] == "ESCALATE"))
    return bb


def _contract(slice_sel, frame):
    import numpy as np
    row = {k: (frame.get(k) if frame.get(k) is not None else np.nan)
           for k in _FRAME_KEYS}
    return slice_sel in feasible_set(row)
