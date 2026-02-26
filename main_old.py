from __future__ import annotations

import asyncio
import base64
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

from database import DatabaseManager
from levels import (
    build_boss_question,
    get_fixed_question,
    get_level_config,
    get_next_position,
    is_fixed_question,
)
from ml_model import analyze_sentiment
from scoring import calculate_confidence_score, calculate_stars


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "interview_hero")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required. Add it to backend/.env")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
db_manager = DatabaseManager(mongo_uri=MONGODB_URI, db_name=MONGODB_DB)


class StartGameResponse(BaseModel):
    session_id: str
    level: int
    level_name: str
    question_number: int
    next_question: str
    total_stars: int


async def generate_dynamic_question(
    level: int,
    question_number: int,
    hero_name: Optional[str],
    life_lesson: Optional[str],
) -> str:
    level_cfg = get_level_config(level)

    if level == 3:
        return build_boss_question(life_lesson)

    system_prompt = (
        "You generate one short, cinematic interview question for a game. "
        "Return only the question text. Keep it supportive and engaging."
    )

    user_prompt = (
        f"Level: {level_cfg.name}\n"
        f"Theme: {level_cfg.theme}\n"
        f"Question number: {question_number}\n"
        f"Hero name context: {hero_name or 'Unknown'}\n"
        f"Life lesson context: {life_lesson or 'Unknown'}\n"
        "Make the question practical and easy to answer in 20-40 seconds."
    )

    try:
        completion = await client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=80,
        )
        question = (completion.choices[0].message.content or "").strip()
        if question:
            return question
    except Exception:
        pass

    fallback = {
        (1, 1): "What is one strength you are proud of, and why?",
        (1, 2): "What challenge helps you grow the most right now?",
        (1, 3): "When do you feel most confident while helping others?",
        (2, 1): "How has your hero influenced your daily decisions?",
        (2, 2): "Which habit from your hero would you like to build this month?",
        (2, 3): "Tell me about a moment you acted bravely like your hero.",
    }
    return fallback.get((level, question_number), "What inspires you to keep improving every day?")


async def generate_ai_reply(
    transcript: str,
    level_name: str,
    sentiment_label: str,
    confidence_score: float,
) -> str:
    system_prompt = (
        "You are an anime-style mentor in a confidence training arena. "
        "Give concise feedback in 2-4 sentences. Be supportive, analytical, and motivating."
    )
    user_prompt = (
        f"Level: {level_name}\n"
        f"Transcript: {transcript}\n"
        f"Sentiment: {sentiment_label}\n"
        f"Confidence score: {confidence_score}\n"
        "Respond with encouragement, one specific strength, and one actionable improvement."
    )

    completion = await client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=220,
    )
    return (completion.choices[0].message.content or "Keep going, hero!").strip()


async def transcribe_audio(audio_file: UploadFile) -> str:
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    transcript = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio_file.filename or "recording.webm", audio_bytes, audio_file.content_type or "audio/webm"),
    )

    text = getattr(transcript, "text", "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Unable to transcribe audio")
    return text


async def synthesize_tts(text: str) -> str:
    speech = await client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice="alloy",
        input=text,
        format="mp3",
    )

    if hasattr(speech, "aread"):
        audio_bytes = await speech.aread()
    else:
        audio_bytes = speech.read()

    return base64.b64encode(audio_bytes).decode("utf-8")


def extract_hero_name(text: str) -> Optional[str]:
    match = re.search(r"\b(?:is|was|like)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})", text)
    if match:
        return match.group(1).strip()

    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return None

    candidate = " ".join(words[:3]).strip().title()
    return candidate if candidate else None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db_manager.connect()
    yield
    await db_manager.disconnect()


app = FastAPI(title="Interview Hero API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/api/start-game", response_model=StartGameResponse)
async def start_game():
    first_question = await generate_dynamic_question(level=1, question_number=1, hero_name=None, life_lesson=None)
    session_id = await db_manager.create_session(first_question=first_question)
    return StartGameResponse(
        session_id=session_id,
        level=1,
        level_name=get_level_config(1).name,
        question_number=1,
        next_question=first_question,
        total_stars=0,
    )


@app.post("/api/process-audio")
async def process_audio(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") == "completed":
        raise HTTPException(status_code=400, detail="Game already completed")

    current_level = int(session["level"])
    question_number = int(session["question_number"])
    level_name = get_level_config(current_level).name

    transcript = await transcribe_audio(audio)
    sentiment_result = await asyncio.to_thread(analyze_sentiment, transcript)
    sentiment_label = str(sentiment_result["sentiment_label"])
    sentiment_score = float(sentiment_result["sentiment_score"])
    confidence_score = calculate_confidence_score(transcript)

    stars_awarded = calculate_stars(
        level=current_level,
        question_number=question_number,
        text=transcript,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        confidence_score=confidence_score,
    )

    total_stars = max(0, min(9, int(session.get("total_stars", 0)) + stars_awarded))

    hero_name = session.get("hero_name")
    life_lesson = session.get("life_lesson")

    if current_level == 1 and question_number == 4 and not hero_name:
        hero_name = extract_hero_name(transcript)

    if current_level == 2 and question_number == 4:
        life_lesson = transcript.strip()

    ai_text = await generate_ai_reply(
        transcript=transcript,
        level_name=level_name,
        sentiment_label=sentiment_label,
        confidence_score=confidence_score,
    )
    audio_base64 = await synthesize_tts(ai_text)

    next_level, next_question_number, boss_unlocked = get_next_position(current_level, question_number)
    completed = next_level is None

    next_question = None
    if not completed and next_level is not None and next_question_number is not None:
        if is_fixed_question(next_level, next_question_number):
            next_question = get_fixed_question(next_level, next_question_number, hero_name=hero_name)
        elif next_level == 3:
            next_question = build_boss_question(life_lesson)
        else:
            next_question = await generate_dynamic_question(
                level=next_level,
                question_number=next_question_number,
                hero_name=hero_name,
                life_lesson=life_lesson,
            )

    answer_payload = {
        "answered_at": datetime.now(timezone.utc),
        "level": current_level,
        "question_number": question_number,
        "question": session.get("current_question"),
        "transcript": transcript,
        "confidence_score": confidence_score,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "stars_awarded": stars_awarded,
        "ai_text": ai_text,
    }

    await db_manager.update_session_after_answer(
        session_id=session_id,
        answer_payload=answer_payload,
        next_level=next_level,
        next_question_number=next_question_number,
        next_question=next_question,
        total_stars=total_stars,
        hero_name=hero_name,
        life_lesson=life_lesson,
        completed=completed,
    )

    return {
        "transcript": transcript,
        "ai_text": ai_text,
        "stars_awarded": stars_awarded,
        "total_stars": total_stars,
        "confidence_score": confidence_score,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "next_question": next_question,
        "level": next_level if next_level is not None else current_level,
        "level_name": get_level_config(next_level).name if next_level else level_name,
        "question_number": next_question_number,
        "audio_base64": audio_base64,
        "boss_unlocked": boss_unlocked,
        "game_completed": completed,
        "hero_name": hero_name,
        "life_lesson": life_lesson,
    }
