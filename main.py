from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
import uvicorn

from database import DatabaseManager
from levels import (
    build_boss_question,
    get_fixed_question,
    get_level_config,
    get_next_position,
    is_fixed_question,
)
from scoring import calculate_stars, evaluate_answer


load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "interview_hero")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required. Add it to backend/.env")

genai_client = genai.Client(api_key=GEMINI_API_KEY)
db_manager = DatabaseManager(mongo_uri=MONGODB_URI, db_name=MONGODB_DB)


async def generate_gemini_text(contents: list[str]) -> str:
    models_to_try = [
        GEMINI_MODEL,
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.0-pro",
    ]
    seen = set()
    
    for model_name in models_to_try:
        if not model_name or model_name in seen:
            continue
        seen.add(model_name)
        try:
            logger.debug(f"Trying Gemini model: {model_name}")
            response = await asyncio.to_thread(
                genai_client.models.generate_content,
                model=model_name,
                contents=contents,
            )
            text = getattr(response, "text", None)
            if text:
                logger.debug(f"Success with {model_name}: {text[:100]}...")
                return text.strip()
            else:
                logger.warning(f"No text in response from {model_name}")
        except Exception as e:
            logger.error(f"Error with {model_name}: {type(e).__name__}: {str(e)}")
            continue

    logger.error("All Gemini models failed to generate text")
    return ""


class StartGameResponse(BaseModel):
    session_id: str
    level: int
    level_name: str
    question_number: int
    next_question: str
    total_stars: int


class ProcessAnswerRequest(BaseModel):
    session_id: str
    answer: str


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
        "You generate one short, cinematic interview question for a confidence-building game. "
        "Return only the question text. Keep it supportive, engaging, and UNIQUE each time. "
        "Make each question different from previous ones - use creative variation."
    )

    user_prompt = (
        f"Level: {level_cfg.name}\n"
        f"Theme: {level_cfg.theme}\n"
        f"Question number: {question_number}\n"
        f"Hero name context: {hero_name or 'Unknown'}\n"
        f"Life lesson context: {life_lesson or 'Unknown'}\n"
        "Create a UNIQUE question each time. Make it practical, personal, and answerable in 20-40 seconds. "
        "Focus on self-reflection, strengths, challenges, or values. Ensure variety."
    )

    question = await generate_gemini_text([system_prompt, user_prompt])
    if question:
        return question

    # Expanded fallback with more variety
    import random
    fallback_pool = {
        (1, 1): [
            "What is one strength you are proud of, and why?",
            "Describe a moment when you felt truly confident.",
            "What personal quality helps you overcome challenges?",
            "Tell me about a time you surprised yourself with your abilities.",
        ],
        (1, 2): [
            "What challenge helps you grow the most right now?",
            "What skill are you currently working to improve?",
            "Describe an obstacle you recently faced and how you handled it.",
            "What's something difficult you're learning to accept about yourself?",
        ],
        (1, 3): [
            "When do you feel most confident while helping others?",
            "How do you stay motivated when things get tough?",
            "What values guide your important decisions?",
            "Describe a situation where you had to stand up for what you believe in.",
        ],
        (2, 1): [
            "How has your hero influenced your daily decisions?",
            "What trait from your hero do you admire most?",
            "When did you first connect with your hero's story?",
        ],
        (2, 2): [
            "Which habit from your hero would you like to build this month?",
            "How does your hero handle failure or setbacks?",
            "What lesson from your hero applies to your life right now?",
        ],
        (2, 3): [
            "Tell me about a moment you acted bravely like your hero.",
            "How would your hero approach a current challenge you're facing?",
            "What would your hero say to you right now?",
        ],
    }
    
    pool = fallback_pool.get((level, question_number))
    if pool:
        return random.choice(pool)
    
    return "What inspires you to keep improving every day?"


async def generate_ai_reply(
    transcript: str,
    level_name: str,
    sentiment_label: str,
    confidence_score: float,
    level: int = 1,
    question_number: int = 1,
    hero_name: Optional[str] = None,
    life_lesson: Optional[str] = None,
) -> str:
    import random
    
    # Boss Level (Level 3) - Generate responsible life advice
    if level == 3:
        system_prompt = (
            "You are a wise and hilariously sarcastic mentor delivering life-changing advice. "
            "The user has completed their final boss challenge. Your job is to: "
            "1. Acknowledge their answer with humor and authenticity "
            "2. Extract the KEY WISDOM from their response "
            "3. Give funny but DEEPLY RESPONSIBLE life advice based on their answer and the challenge scenario "
            "4. Make it memorable and impactful - this is their final wisdom. "
            "Keep it to 3-5 sentences. Balance roast humor with genuine, life-changing advice."
        )
        user_prompt = (
            f"Life Lesson Context: {life_lesson or 'their personal wisdom'}\n"
            f"User's Boss Level Answer: '{transcript}'\n"
            f"Sentiment: {sentiment_label}\n"
            f"Confidence: {confidence_score:.1f}%\n\n"
            "This is their FINAL WISDOM - the culmination of their journey. "
            "Give them funny but responsible life advice that will stay with them forever. "
            "Reference their answer and the lesson they've learned."
        )
    else:
        # Regular levels - Confidence-based feedback
        if confidence_score < 40:
            tone = "Low confidence - HEAVY ENCOURAGEMENT + roasting"
            instructions = (
                "The user is struggling with confidence. Roast their answer for being too timid, "
                "but MASSIVELY encourage them. Make them feel like they can do this. "
                "Include hype energy like 'You got this!', 'Don't give up!', 'You're stronger than you think!' "
                "Mix savage humor WITH strong motivation."
            )
        elif confidence_score < 70:
            tone = "Medium confidence - BALANCED roasting and motivation"
            instructions = (
                "Good effort but room for growth. Roast them playfully about what could be better, "
                "but acknowledge their effort. Be funny but fair. Throw in some motivational energy."
            )
        else:
            tone = "High confidence - CELEBRATION + light roasting"
            instructions = (
                "They crushed it! Celebrate their answer with enthusiasm and appreciation. "
                "You can still make a light roasting joke, but the PRIMARY tone should be celebrating their greatness. "
                "Use words like 'Fire!', 'Absolutely did that!', 'Now THAT'S what I'm talking about!', 'You're a beast!' "
                "They deserve major hype. End on a note of appreciation for their confidence."
            )
        
        system_prompt = (
            "You are a hilariously sarcastic anime-style mentor in a confidence training arena. "
            "Your job is to give contextual feedback based on the user's confidence level. "
            f"Current tone: {tone}\n"
            "Keep it to 2-4 sentences max. Mix humor with genuine feedback. Be authentic and dynamic."
        )

        if level == 1 and question_number == 4:
            hero_ref = hero_name or transcript or "their hero"
            user_prompt = (
                f"Level: {level_name}\n"
                f"User answer (favorite comic hero): '{transcript}'\n"
                f"Extracted hero name: {hero_ref}\n"
                f"Sentiment: {sentiment_label}\n"
                f"Confidence score: {confidence_score:.1f}%\n"
                f"\nInstructions: {instructions}\n"
                "Your response MUST reference the hero they named and react to that choice. "
                "Say something about that hero specifically while keeping the confidence-based tone."
            )
        else:
            user_prompt = (
                f"Level: {level_name}\n"
                f"User answer: '{transcript}'\n"
                f"Sentiment: {sentiment_label}\n"
                f"Confidence score: {confidence_score:.1f}%\n"
                f"\nInstructions: {instructions}\n"
                "Give a response that fits the confidence level above."
            )

    reply = await generate_gemini_text([system_prompt, user_prompt])
    
    # Fallback replies
    if level == 3:
        # Boss level fallbacks
        fallback_replies = [
            "That's the energy right there! You just discovered your true potential. Now go change the world with that wisdom!",
            "Wow, now THAT'S a life lesson! The fact that you can articulate that means you're ready to inspire others!",
            "Honest, powerful, and real. You've got the tools to handle anything life throws at you. Go be the hero!",
            "That's not just an answer, that's your life philosophy in action. Use it wisely, legend!",
        ]
        return reply or random.choice(fallback_replies)
    else:
        # Regular level fallbacks based on confidence
        fallback_replies = {
            "low": [
                "Okay, that was rough... but hey, you SHOWED UP! That's already a win. You've got this!",
                "Yikes... but I see potential in you. Don't back down now! Come back stronger!",
                "That answer was weak... but your courage to answer? THAT'S strong. Keep going!",
                "Rough around the edges... but everyone starts somewhere. You're gonna make it!",
            ],
            "medium": [
                "Not bad, not bad! You're getting there. One more push and you'll nail it!",
                "Solid effort! There's room to shine brighter, but you're on the right track!",
                "Getting warmer! I can feel the growth. Keep that momentum going!",
                "Good attempt! You're finding your voice. Let's make it even stronger next time!",
            ],
            "high": [
                "NOW THAT'S WHAT I'M TALKING ABOUT! You absolutely crushed that!",
                "YOOO! That was FIRE! Your confidence is showing and it's BEAUTIFUL!",
                "Okay okay okay... I see you being a BEAST right now! Absolutely phenomenal!",
                "THAT'S MY HERO! You just went full legend mode! Perfection!",
            ],
        }
        
        if reply:
            return reply.strip()
        
        if level == 1 and question_number == 4:
            hero_ref = hero_name or "your hero"
            return f"Nice pick with {hero_ref}. That's a bold choice. Keep going and bring more of that hero energy next time!"

        if confidence_score < 40:
            return random.choice(fallback_replies["low"])
        elif confidence_score < 70:
            return random.choice(fallback_replies["medium"])
        else:
            return random.choice(fallback_replies["high"])


async def synthesize_tts(text: str) -> str:
    # Gemini doesn't provide TTS natively.
    # Frontend uses browser Speech Synthesis API for audio narration.
    # Return empty placeholder for compatibility with frontend audio player.
    empty_audio = bytes(4)  # Minimal audio placeholder
    return base64.b64encode(empty_audio).decode("utf-8")


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


@app.post("/api/process-answer")
async def process_answer(request: ProcessAnswerRequest):
    session = await db_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.get("status") == "completed":
        raise HTTPException(status_code=400, detail="Game already completed")

    transcript = request.answer.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Answer cannot be empty")

    current_level = int(session["level"])
    question_number = int(session["question_number"])
    level_name = get_level_config(current_level).name

    # Run ML analysis
    analysis = await asyncio.to_thread(evaluate_answer, transcript)
    sentiment_label = str(analysis["sentiment_label"])
    sentiment_score = float(analysis["sentiment_score"])
    confidence_score = float(analysis["confidence_score"])

    # Calculate stars
    stars_awarded = calculate_stars(
        level=current_level,
        question_number=question_number,
        text=transcript,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        confidence_score=confidence_score,
    )

    total_stars = max(0, min(9, int(session.get("total_stars", 0)) + stars_awarded))

    # Extract hero name if on level 1 Q4
    hero_name = session.get("hero_name")
    life_lesson = session.get("life_lesson")

    if current_level == 1 and question_number == 4 and not hero_name:
        hero_name = extract_hero_name(transcript)

    # Extract life lesson if on level 2 Q4
    if current_level == 2 and question_number == 4:
        life_lesson = transcript.strip()

    # Generate AI response and TTS
    ai_text = await generate_ai_reply(
        transcript=transcript,
        level_name=level_name,
        sentiment_label=sentiment_label,
        confidence_score=confidence_score,
        level=current_level,
        question_number=question_number,
        hero_name=hero_name,
        life_lesson=life_lesson,
    )
    audio_base64 = await synthesize_tts(ai_text)

    # Determine next position
    next_level, next_question_number, boss_unlocked = get_next_position(current_level, question_number)
    completed = next_level is None

    # Generate next question if game continues
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

    # Update MongoDB session
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
        session_id=request.session_id,
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
