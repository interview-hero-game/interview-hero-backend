from __future__ import annotations

from transformers import pipeline


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
)


def _extract_stars(label_raw: str) -> int:
    label_clean = label_raw.strip().lower()
    first_token = label_clean.split()[0] if label_clean else ""

    if first_token.isdigit():
        stars = int(first_token)
        return max(1, min(5, stars))

    if "neg" in label_clean:
        return 1
    if "pos" in label_clean:
        return 5
    return 3


def analyze_sentiment(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return "NEUTRAL", 0.5

    result = sentiment_analyzer(text[:2000])[0]
    stars = _extract_stars(str(result.get("label", "3 stars")))

    if stars <= 2:
        label = "NEGATIVE"
    elif stars == 3:
        label = "NEUTRAL"
    else:
        label = "POSITIVE"

    sentiment_score = round(stars / 5, 3)
    return label, sentiment_score
