"""LLM client: caching, mock mode, usage tracking.

- Model from config.LLM_MODEL (gpt-4.1-mini); key read from OPENAI_API_KEY.
- Every call is cached on disk by a hash of (model, system, user, schema), so
  re-running never repeats a paid call.
- mode="mock" makes no network call and returns deterministic fake output, for
  testing the pipeline.
- Each call (cached or not) is logged with input/output tokens, latency and
  estimated cost. Cached calls log cost 0 (cached=True) so overhead counts
  only real API usage.

Prices are USD per 1M tokens and are an ASSUMPTION for gpt-4.1-mini; override
with env LLM_PRICE_IN / LLM_PRICE_OUT if the price differs.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .config import LLM_MODEL, ROOT

CACHE_DIR = ROOT / "data" / "llm_cache"
PRICE_IN = float(os.getenv("LLM_PRICE_IN", "0.40"))
PRICE_OUT = float(os.getenv("LLM_PRICE_OUT", "1.60"))


def _key(model: str, system: str, user: str, schema: dict) -> str:
    blob = json.dumps([model, system, user, schema], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class LLM:
    def __init__(self, mode: str = "mock", model: str = LLM_MODEL,
                 cache_name: str = "cache", temperature: float = 0.0):
        if mode not in ("live", "mock"):
            raise ValueError("mode must be 'live' or 'mock'")
        self.mode, self.model, self.temperature = mode, model, temperature
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # mock and live caches are separate so fake outputs never leak
        self.cache_path = CACHE_DIR / f"{cache_name}_{mode}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r
        self.client = None
        if mode == "live":
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("OPENAI_API_KEY is not set")
            from openai import OpenAI
            self.client = OpenAI()
        self.log: list[dict] = []
        self._lock = threading.Lock()

    def call(self, stage: str, system: str, user: str, schema: dict,
             schema_name: str = "out") -> dict:
        k = _key(self.model, system, user, schema)
        if k in self.cache:
            r = self.cache[k]
            rec = {"stage": stage, "cached": True, "in_tok": r["in_tok"],
                   "out_tok": r["out_tok"], "latency_s": 0.0, "cost_usd": 0.0}
            self.log.append(rec)
            return {**r["parsed"], "_usage": rec}

        t0 = time.time()
        if self.mode == "mock":
            parsed = self._mock(schema, user)
            in_tok = (len(system) + len(user)) // 4
            out_tok = len(json.dumps(parsed)) // 4
        else:
            resp = self.client.responses.create(
                model=self.model, temperature=self.temperature,
                input=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
                text={"format": {"type": "json_schema", "name": schema_name,
                                 "schema": schema, "strict": True}})
            parsed = json.loads(resp.output_text)
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
        lat = time.time() - t0
        cost = (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6
        with self._lock:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": k, "parsed": parsed,
                                    "in_tok": in_tok,
                                    "out_tok": out_tok}) + "\n")
            self.cache[k] = {"key": k, "parsed": parsed, "in_tok": in_tok,
                             "out_tok": out_tok}
        rec = {"stage": stage, "cached": False, "in_tok": in_tok,
               "out_tok": out_tok, "latency_s": round(lat, 3),
               "cost_usd": cost}
        with self._lock:
            self.log.append(rec)
        return {**parsed, "_usage": rec}

    @staticmethod
    def _mock(schema: dict, user: str) -> dict:
        h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
        out = {}
        for name, spec in schema["properties"].items():
            if "enum" in spec:
                out[name] = spec["enum"][h % len(spec["enum"])]
            elif spec.get("type") == "boolean":
                out[name] = bool(h % 2)
            else:
                out[name] = "[mock output]"
        return out

    def summary(self) -> dict:
        real = [r for r in self.log if not r["cached"]]
        return {"calls_total": len(self.log), "calls_real": len(real),
                "calls_cached": len(self.log) - len(real),
                "in_tokens": sum(r["in_tok"] for r in real),
                "out_tokens": sum(r["out_tok"] for r in real),
                "latency_s": round(sum(r["latency_s"] for r in real), 2),
                "cost_usd": round(sum(r["cost_usd"] for r in real), 6),
                "mode": self.mode, "model": self.model,
                "price_per_1M_in_out": [PRICE_IN, PRICE_OUT]}
