"""
gemini_compat.py — Compatibility layer for the deprecated google.generativeai package.

The `google.generativeai` package is deprecated; new code must use
`google.genai`. This module provides drop-in helpers that keep the old
call-site style (model objects with `.generate_content()` and
`response.text`) working on top of the new `google.genai` Client API.

Usage (mimics the old API):
    from gemini_compat import genai_model
    model = genai_model("gemini-2.5-flash", system_instruction="...")
    text = model.generate_content("Hello").text
"""

import json
from pathlib import Path

import google.genai as _genai


def _get_api_key() -> str:
    config_path = Path(__file__).resolve().parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


class _CompatModel:
    """Wraps a google.genai model as a google.generativeai-style model."""

    def __init__(self, client, model_name: str, system_instruction: str = None):
        self._client = client
        self._model_name = model_name
        self._system_instruction = system_instruction

    def generate_content(self, prompt, config=None):
        from google.genai import types

        cfg = config or {}
        if self._system_instruction:
            cfg = dict(cfg)
            cfg["system_instruction"] = self._system_instruction
        kwargs = {}
        if cfg:
            kwargs["config"] = types.GenerateContentConfig(**cfg)
        response = self._client.models.generate_content(
            model=self._model_name, contents=prompt, **kwargs
        )
        return _CompatResponse(response)


class _CompatResponse:
    """Exposes `.text` on a google.genai GenerateContentResponse."""

    def __init__(self, response):
        self._response = response

    @property
    def text(self) -> str:
        parts = []
        for c in getattr(self._response, "candidates", None) or []:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        return "".join(parts)


def genai_model(model_name: str, system_instruction: str = None,
                 api_key: str = None, client=None) -> _CompatModel:
    """Return a google.generativeai-style model backed by google.genai."""
    if client is None:
        client = _genai.Client(api_key=api_key or _get_api_key())
    return _CompatModel(client, model_name, system_instruction)
