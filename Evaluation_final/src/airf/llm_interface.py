# src/airf/llm_interface.py
import os
import json
import time
from typing import Dict, Any, Optional
import requests

class GeminiAdapter:
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None, model: str = "gemini-pro"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.api_url = api_url or os.environ.get("GEMINI_API_URL", "https://api.gemini.example/v1/generate")
        self.model = model
        if not self.api_key:
            raise RuntimeError("GeminiAdapter requires API key in GEMINI_API_KEY environment variable")

    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        for attempt in range(3):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                time.sleep(1 + attempt * 2)
        raise RuntimeError(f"Gemini API call failed after retries: {last_exc}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "format": "json"
        }
        out = self._call_api(payload)
        # Expecting model to return {"text": "..."} or structured JSON
        if isinstance(out, dict) and "text" in out:
            return out["text"]
        # fallback: raw json string in "result"
        if isinstance(out, dict) and "result" in out:
            return out["result"]
        return json.dumps(out)

class DummyAdapter:
    def __init__(self):
        pass
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        # Minimal deterministic fallback for offline testing
        return '{"message":"dummy"}'
