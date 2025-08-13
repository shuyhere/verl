#!/usr/bin/env python3
# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from verl.utils.reward_score import math as verl_math
from verl.utils.reward_score import gsm8k as verl_gsm8k


_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_tokenizer(tokenizer_name: str):
    if tokenizer_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    return _TOKENIZER_CACHE[tokenizer_name]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _extract_answer(text: str) -> Optional[str]:
    # Prefer last boxed answer; fall back to None
    boxed = verl_math.last_boxed_only_string(text or "")
    if boxed is None:
        # Fallback to GSM8K flexible numeric extraction
        try:
            return verl_gsm8k.extract_solution(text or "", method="flexible")
        except Exception:
            return None
    try:
        return verl_math.remove_boxed(boxed)
    except Exception:
        # Fallback to flexible extraction if boxed removal fails
        try:
            return verl_gsm8k.extract_solution(text or "", method="flexible")
        except Exception:
            return None


def _normalize_ground_truth(gt: Any) -> Optional[str]:
    # Accept string or list/tuple of strings
    if gt is None:
        return None
    if isinstance(gt, (list, tuple)):
        candidates = [str(x) for x in gt if x is not None and str(x).strip() != ""]
        return candidates[-1] if len(candidates) > 0 else None
    return str(gt)


def compute_score_batch(
    *,
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Optional[str]],
    extra_infos: Optional[List[Optional[dict]]] = None,
    # kwargs below are injected via reward_manager's reward_kwargs
    tokenizer_name: str = "Qwen/Qwen3-4B",
    alpha: float = 0.1,
    check_eos: bool = False,
):
    """Length-aware, batch reward with correctness gating.

    For each question, compute the mean/std length among correct responses in the batch,
    then apply a sigmoid-normalized penalty to each correct response length.

    Returns a list with one entry per sample. Each entry is a dict with:
    - score: final scalar reward
    - accuracy: 0/1 correctness
    - response_length: token length used in penalty
    - question: identifier grouped by (extra_info['question'] if provided, else solution_str index)
    """
    tokenizer = _get_tokenizer(tokenizer_name)

    n = len(solution_strs)
    if extra_infos is None:
        extra_infos = [None] * n

    # First pass: compute accuracy and lengths; collect per-question lengths for correct responses
    per_question_correct_lengths: Dict[str, List[int]] = {}
    acc: List[float] = [0.0] * n
    lengths: List[int] = [0] * n
    questions: List[str] = [""] * n

    for i in range(n):
        resp = solution_strs[i] or ""
        gt = _normalize_ground_truth(ground_truths[i])
        info = extra_infos[i] or {}
        # Question identifier for grouping
        qid = info.get("question") if isinstance(info, dict) else None
        if not qid:
            qid = f"q_{i}"
        questions[i] = str(qid)

        # Token length similar to external server (strip specials by decode/encode round-trip)
        ids = tokenizer.encode(resp, add_special_tokens=False)
        clean = tokenizer.decode(ids, skip_special_tokens=True)
        length = len(tokenizer.encode(clean, add_special_tokens=False))
        lengths[i] = length

        # EOS constraint (optional)
        if check_eos and tokenizer.eos_token_id is not None:
            has_eos = tokenizer.eos_token_id in ids
            if not has_eos:
                acc[i] = 0.0
                continue

        # Correctness via Verl math utils (boxed answer equivalence)
        pred = _extract_answer(resp)
        print("*"*100)
        print(f"gt: {gt}")
        print(f"pred: {pred}")
        print("*"*100)
        if pred is None or gt is None:
            acc[i] = 0.0
        else:
            acc[i] = 1.0 if verl_math.is_equiv(pred, gt) else 0.0

        if acc[i] > 0.0:
            per_question_correct_lengths.setdefault(questions[i], []).append(length)
            
        print(f"acc: {acc[i]}")

    # Second pass: compute rewards with length penalty for correct responses
    outputs: List[Dict[str, Any]] = []
    for i in range(n):
        question = questions[i]
        base_acc = acc[i]
        length = lengths[i]

        if base_acc <= 0.0:
            reward = 0.0
        else:
            lens = per_question_correct_lengths.get(question, [])
            if len(lens) == 0:
                reward = 0.0
            else:
                mean_len = sum(lens) / float(len(lens))
                # population std with epsilon
                var = sum((l - mean_len) ** 2 for l in lens) / float(len(lens))
                std = math.sqrt(var + 1e-7)
                rel = (length - mean_len) / (std + 1e-7)
                reward = float(base_acc * (1.0 - alpha * _sigmoid(rel)))
                print(f"reward: {reward}")

        outputs.append(
            {
                "score": reward,
                "accuracy": base_acc,
                "response_length": length,
                "question": question,
            }
        )

    return outputs


