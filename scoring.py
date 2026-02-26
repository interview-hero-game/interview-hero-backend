from __future__ import annotations

import re

from ml_model import analyze_sentiment


HESITATION_WORDS = {
    "um",
    "uh",
    "like",
    "maybe",
    "i guess",
    "sort of",
    "kind of",
    "not sure",
    "probably",
    "எனக்கு தெரியாது",
    "நான் நினைக்கிறேன்",
    "பரவாயில்லை",
}

DECISIVE_WORDS = {
    "definitely",
    "certainly",
    "absolutely",
    "i will",
    "i can",
    "i am",
    "confident",
    "clear",
    "committed",
    "focused",
    "நிச்சயம்",
    "முடியும்",
    "உறுதி",
}

EMOTIONAL_STABILITY_WORDS = {
    "calm",
    "support",
    "plan",
    "listen",
    "adapt",
    "balance",
    "reflect",
    "empathy",
    "responsible",
    "consistent",
}


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.lower(), flags=re.UNICODE)


def calculate_confidence_score(text: str) -> float:
    words = _normalized_words(text)
    if not words:
        return 0.0

    raw_text = text.lower()
    hesitation_hits = sum(raw_text.count(term) for term in HESITATION_WORDS)
    decisive_hits = sum(raw_text.count(term) for term in DECISIVE_WORDS)

    base = 0.55
    length_bonus = min(len(words) / 180, 0.2)
    decisive_bonus = min(decisive_hits * 0.08, 0.3)
    hesitation_penalty = min(hesitation_hits * 0.1, 0.5)

    confidence = base + length_bonus + decisive_bonus - hesitation_penalty
    return max(0.0, min(1.0, round(confidence, 3)))


def detect_confidence(text: str) -> float:
    return calculate_confidence_score(text)


def evaluate_answer(text: str) -> dict[str, float | str]:
    sentiment_label, sentiment_score = analyze_sentiment(text)
    confidence_score = detect_confidence(text)
    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence_score": confidence_score,
    }


def is_clarity_good(text: str) -> bool:
    words = _normalized_words(text)
    if len(words) < 6:
        return False

    unique_ratio = len(set(words)) / max(len(words), 1)
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    avg_sentence_len = len(words) / sentence_count
    return unique_ratio >= 0.45 and 4 <= avg_sentence_len <= 35


def _is_boss_answer_strong(text: str, sentiment_label: str, sentiment_score: float) -> bool:
    words = _normalized_words(text)
    if len(words) < 18:
        return False

    raw_text = text.lower()
    normalized_label = sentiment_label.lower()
    stability_hits = sum(raw_text.count(term) for term in EMOTIONAL_STABILITY_WORDS)
    structure_words = ["first", "then", "finally", "because", "therefore"]
    structure_hits = sum(raw_text.count(term) for term in structure_words)

    emotionally_stable = (
        normalized_label != "negative" or sentiment_score < 0.55
    ) and stability_hits >= 2
    realistic = structure_hits >= 1 and any(
        token in raw_text for token in ["team", "deadline", "plan", "risk", "feedback"]
    )
    return emotionally_stable and realistic


def calculate_stars(
    level: int,
    question_number: int,
    text: str,
    sentiment_label: str,
    sentiment_score: float,
    confidence_score: float,
) -> int:
    normalized_label = sentiment_label.lower()
    clarity = is_clarity_good(text)

    if level == 1:
        if question_number == 1:
            if confidence_score < 0.3:
                return 0
            stars = 1
            if confidence_score > 0.7:
                stars += 1
            return max(0, min(2, stars))

        if normalized_label == "positive" and sentiment_score > 0.7 and clarity:
            return 1
        return 0

    if level == 2:
        if normalized_label == "positive" and sentiment_score > 0.7 and clarity:
            return 1
        return 0

    if level == 3:
        return 1 if _is_boss_answer_strong(text, normalized_label, sentiment_score) else 0

    return 0
