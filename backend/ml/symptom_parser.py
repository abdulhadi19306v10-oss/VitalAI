"""Free-text multi-symptom parser with deterministic local NLP and optional LLM translation."""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.ml.symptom_aliases import (
    CANONICAL_SYNONYMS,
    SYMPTOM_ALIASES,
    normalize_token,
    resolve_symptom_key,
)

_NEGATION_RE = re.compile(
    r"^(?:no|not|without|denies|denied|absent|negative\s+for|rules?\s+out)\s+(.+)$",
    re.IGNORECASE,
)


@dataclass
class ParseResult:
    """Result of parsing free-text symptom notes into canonical dataset keys."""

    matched_keys: list[str]
    unmapped_terms: list[str] = field(default_factory=list)
    negated_terms: list[str] = field(default_factory=list)
    mode: str = "local_parser"
    note: str = ""


def _build_phrase_index(valid_features: set[str]) -> list[tuple[str, str]]:
    """Return sorted (normalized_space_phrase, canonical_key) longest-first."""
    phrases: dict[str, str] = {}
    for key in valid_features:
        phrase = key.replace("_", " ").replace("(", "").replace(")", "").strip()
        if phrase:
            phrases[phrase] = key
        for alias in CANONICAL_SYNONYMS.get(key, ()):
            alias_phrase = alias.replace("_", " ").strip()
            if alias_phrase and alias_phrase not in phrases:
                phrases[alias_phrase] = key

    for alias_key, target in SYMPTOM_ALIASES.items():
        if target in valid_features:
            alias_phrase = alias_key.replace("_", " ").strip()
            if alias_phrase and alias_phrase not in phrases:
                phrases[alias_phrase] = target

    ordered = sorted(phrases.items(), key=lambda kv: len(kv[0]), reverse=True)
    return ordered


def _split_clauses(text: str) -> list[str]:
    """Split free text into candidate clinical clauses on punctuation and conjunctions."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"[,;\n•|]+|\b(?:and|with|plus|also)\b", cleaned, flags=re.IGNORECASE)
    return [part.strip(" .:-") for part in parts if part.strip(" .:-")]


def parse_symptoms_locally(text: str, valid_features: list[str]) -> ParseResult:
    """Deterministically extract canonical symptom keys from free text without external APIs."""
    valid_set = set(valid_features)
    phrase_index = _build_phrase_index(valid_set)
    all_phrases = [phrase for phrase, _ in phrase_index]
    phrase_to_key = dict(phrase_index)

    matched: list[str] = []
    seen_matched: set[str] = set()
    negated: list[str] = []
    unmapped: list[str] = []

    for clause in _split_clauses(text):
        neg_match = _NEGATION_RE.match(clause)
        if neg_match:
            negated.append(neg_match.group(1).strip())
            continue

        # Direct token/alias lookup first
        direct = resolve_symptom_key(clause)
        if direct in valid_set:
            if direct not in seen_matched:
                seen_matched.add(direct)
                matched.append(direct)
            continue

        # Longest-phrase substring scan within the clause (handles sentences like "patient has severe chest pain")
        lowered = re.sub(r"[^a-z0-9\s]", " ", clause.lower())
        lowered = re.sub(r"\s+", " ", lowered).strip()
        working = f" {lowered} "
        clause_hits = 0

        for phrase, target_key in phrase_index:
            token = f" {phrase} "
            if token in working:
                if target_key not in seen_matched:
                    seen_matched.add(target_key)
                    matched.append(target_key)
                working = working.replace(token, " ")
                clause_hits += 1

        if clause_hits > 0:
            continue

        # Strict fuzzy typo tolerance (0.87+ avoids unsafe cross-matches like earache -> headache)
        if 4 <= len(lowered) <= 45:
            close = difflib.get_close_matches(lowered, all_phrases, n=1, cutoff=0.87)
            if close:
                target_key = phrase_to_key[close[0]]
                if target_key not in seen_matched:
                    seen_matched.add(target_key)
                    matched.append(target_key)
                continue

        # Ignore filler-only words while reporting genuine unrecognized symptoms
        stripped_words = [
            w
            for w in lowered.split()
            if w
            not in {
                "patient",
                "has",
                "have",
                "having",
                "feels",
                "feeling",
                "experiencing",
                "reports",
                "complains",
                "of",
                "severe",
                "mild",
                "moderate",
                "acute",
                "since",
                "days",
                "day",
                "today",
                "yesterday",
                "for",
                "the",
                "a",
                "an",
                "in",
                "on",
            }
        ]
        if stripped_words:
            unmapped.append(clause)

    return ParseResult(
        matched_keys=matched,
        unmapped_terms=unmapped,
        negated_terms=negated,
        mode="local_parser",
        note="Parsed locally using clinical synonym & typo matcher.",
    )


def resolve_llm_config(
    provider_override: str | None = None,
    api_key_override: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Detect active LLM provider, key, endpoint, and model without leaking secrets."""
    req_provider = (provider_override or "auto").strip().lower()
    req_key = (api_key_override or "").strip()

    if req_provider == "local":
        return None, None, None, None

    gemini_key = req_key if req_provider == "gemini" and req_key else os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = req_key if req_provider == "groq" and req_key else os.getenv("GROQ_API_KEY", "").strip()
    openai_key = (
        req_key
        if req_provider in ("openai", "custom") and req_key
        else os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
    )
    custom_model = os.getenv("LLM_MODEL", "").strip()
    custom_base = os.getenv("LLM_BASE_URL", "").strip()

    if req_provider == "auto" and req_key:
        if req_key.startswith("AIza"):
            gemini_key = req_key
        elif req_key.startswith("gsk_"):
            groq_key = req_key
        else:
            openai_key = req_key

    if (req_provider in ("auto", "gemini")) and gemini_key:
        return "gemini", gemini_key, None, custom_model or "gemini-2.5-flash"

    if (req_provider in ("auto", "groq")) and groq_key:
        return (
            "groq",
            groq_key,
            custom_base or "https://api.groq.com/openai/v1/chat/completions",
            custom_model or "llama-3.1-8b-instant",
        )

    if (req_provider in ("auto", "openai", "custom")) and openai_key:
        return (
            "openai",
            openai_key,
            custom_base or "https://api.openai.com/v1/chat/completions",
            custom_model or "gpt-4o-mini",
        )

    return None, None, None, None


def _build_llm_prompt(text: str, valid_features: list[str]) -> str:
    feature_list = ", ".join(valid_features)
    return (
        "You are a strict medical symptom normalizer.\n"
        "Map ONLY symptoms that the patient currently HAS from the input text onto "
        "exact keys from the ALLOWED_KEYS list below. Ignore negated symptoms (e.g. 'no fever') "
        "and list them under negated_terms. Any symptom not covered by ALLOWED_KEYS must go under "
        "unmapped_terms — NEVER invent keys.\n\n"
        f"ALLOWED_KEYS: [{feature_list}]\n\n"
        f"PATIENT_TEXT: {text}\n\n"
        'Return strict JSON only: {"matched_symptoms": [...], "negated_terms": [...], "unmapped_terms": [...]}'
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    return json.loads(cleaned)


def parse_symptoms_smart(
    text: str,
    valid_features: list[str],
    *,
    provider: str | None = None,
    api_key: str | None = None,
    http_client: httpx.Client | None = None,
) -> ParseResult:
    """Use optional cheap LLM API when configured, with strict whitelist & local fallback."""
    resolved_provider, key, base_url, model_name = resolve_llm_config(provider, api_key)
    if not resolved_provider or not key:
        return parse_symptoms_locally(text, valid_features)

    valid_set = set(valid_features)
    prompt = _build_llm_prompt(text, valid_features)

    try:
        owns_client = http_client is None
        client = http_client or httpx.Client(timeout=12.0)
        try:
            if resolved_provider == "gemini":
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model_name}:generateContent?key={key}"
                )
                resp = client.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.0,
                            "responseMimeType": "application/json",
                        },
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                raw_text = body["candidates"][0]["content"]["parts"][0]["text"]
            else:
                assert base_url is not None
                resp = client.post(
                    base_url,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {
                                "role": "system",
                                "content": "Output valid JSON only mapping symptoms to allowed keys.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                raw_text = body["choices"][0]["message"]["content"]
        finally:
            if owns_client:
                client.close()

        parsed = _extract_json_object(raw_text)
        raw_matched = parsed.get("matched_symptoms", [])
        raw_unmapped = [str(x) for x in parsed.get("unmapped_terms", [])]
        raw_negated = [str(x) for x in parsed.get("negated_terms", [])]

        verified: list[str] = []
        seen: set[str] = set()
        for item in raw_matched:
            canon = resolve_symptom_key(str(item))
            if canon in valid_set:
                if canon not in seen:
                    seen.add(canon)
                    verified.append(canon)
            else:
                norm_item = normalize_token(str(item)).replace("_", " ")
                if norm_item:
                    raw_unmapped.append(norm_item)

        return ParseResult(
            matched_keys=verified,
            unmapped_terms=list(dict.fromkeys(raw_unmapped)),
            negated_terms=list(dict.fromkeys(raw_negated)),
            mode=f"ai_{resolved_provider}",
            note=f"Translated via {resolved_provider} ({model_name}) and verified against 131 dataset keys.",
        )
    except Exception as exc:
        fallback = parse_symptoms_locally(text, valid_features)
        fallback.mode = "local_fallback"
        fallback.note = (
            f"AI translation unavailable ({type(exc).__name__}); used local synonym matcher."
        )
        return fallback
