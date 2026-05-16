# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM-as-judge reward scorer.

Uses a chat-completions API compatible with OpenAI to grade model responses
against the provided ground-truth answer. Responses are graded strictly as
correct (1.0) or incorrect (0.0). The API endpoint and model can be configured
with environment variables.

Required env vars:
- LLM_JUDGE_API_KEY: API key for the judge model.
Optional env vars:
- LLM_JUDGE_BASE_URL (default: https://pro.xiaoai.plus/v1)
- LLM_JUDGE_MODEL (default: gpt-4o-2024-08-06)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

LLM_JUDGE_API_KEY="sk-sFgE7YlELkvEFDzX4tunLbFG4PTX51j8hDBS4sr3MVwN0BIN"
LLM_JUDGE_BASE_URL="https://pro.xiaoai.plus/v1"
LLM_JUDGE_MODEL="gpt-4o-2024-08-06"

_DEFAULT_BASE_URL = "https://pro.xiaoai.plus/v1"
_DEFAULT_MODEL = "gpt-4o-2024-08-06"


def _post_chat(messages: list[dict[str, str]], model: str, base_url: str, api_key: str, timeout: int = 120) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_score(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except Exception:
        return {"score": 0.0, "reason": "parse_error", "extracted_answer": "", "raw": content}

    score_val = parsed.get("score", 0)
    try:
        score = 1.0 if float(score_val) >= 0.5 else 0.0
    except Exception:
        score = 0.0

    # Ensure extracted_answer is always a string, never None
    extracted_answer = parsed.get("extracted_answer", "") or ""
    
    return {
        "score": score,
        "reason": parsed.get("reason", "") or "",
        "extracted_answer": extracted_answer,
        "raw": content,
    }


def compute_score(solution_str: str, ground_truth: str, extra_info: Dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    extra_info = extra_info or {}
    question = extra_info.get("prompt", "")

    api_key = os.getenv("LLM_JUDGE_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_JUDGE_API_KEY is not set")

    base_url = os.getenv("LLM_JUDGE_BASE_URL", _DEFAULT_BASE_URL)
    model = os.getenv("LLM_JUDGE_MODEL", _DEFAULT_MODEL)

    system_prompt = (
        "You are a strict math grader. Compare the model answer to the ground truth final answer. "
        "Only return JSON with keys: score (0 or 1), extracted_answer (short), reason. "
        "Score 1 only when the model answer matches the ground truth.")

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth final answer:\n"
        f"{ground_truth}\n\n"
        "Model answer to grade:\n"
        f"{solution_str}\n\n"
        "Respond with a JSON object now.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        resp = _post_chat(messages, model=model, base_url=base_url, api_key=api_key)
        content = resp["choices"][0]["message"]["content"]
        return _parse_score(content)
    except urllib.error.HTTPError as e:
        return {"score": 0.0, "reason": f"http_error:{e.code}", "extracted_answer": "", "raw": e.read().decode("utf-8", errors="ignore")}
    except Exception as e:  # noqa: BLE001
        return {"score": 0.0, "reason": f"exception:{e}", "extracted_answer": "", "raw": ""}


__all__ = ["compute_score"]
