from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LevelConfig:
    id: int
    name: str
    total_questions: int
    max_stars: int
    theme: str


LEVELS: dict[int, LevelConfig] = {
    1: LevelConfig(
        id=1,
        name="Know Yourself",
        total_questions=4,
        max_stars=4,
        theme="Self-evaluation with fun",
    ),
    2: LevelConfig(
        id=2,
        name="Your Hero",
        total_questions=4,
        max_stars=4,
        theme="Based on hero from level 1",
    ),
    3: LevelConfig(
        id=3,
        name="Boss Level - You are The Hero!!",
        total_questions=1,
        max_stars=1,
        theme="Analyze life lesson and apply it in a scenario",
    ),
}


FIXED_QUESTIONS: dict[tuple[int, int], str] = {
    (1, 4): "Who is your favorite comic hero?",
    (2, 4): "What life lesson did you learn from {hero_name}?",
}


def get_level_config(level: int) -> LevelConfig:
    return LEVELS[level]


def get_level_name(level: int) -> str:
    return LEVELS[level].name


def is_fixed_question(level: int, question_number: int) -> bool:
    return (level, question_number) in FIXED_QUESTIONS


def get_fixed_question(level: int, question_number: int, hero_name: Optional[str] = None) -> str:
    question = FIXED_QUESTIONS[(level, question_number)]
    if "{hero_name}" in question:
        return question.format(hero_name=hero_name or "your hero")
    return question


def build_boss_question(life_lesson: Optional[str]) -> str:
    import random
    
    lesson = life_lesson or "your life lesson"
    
    # Randomized scenario templates based on life lesson context
    scenarios = [
        f"Scenario: You're leading a small team during a stressful project deadline. "
        f"How would you apply '{lesson}' to keep everyone motivated and deliver successfully?",
        
        f"Scenario: A close friend is struggling with self-doubt and comes to you for advice. "
        f"Using '{lesson}' as your guide, how would you help them regain confidence?",
        
        f"Scenario: You're facing a major career decision and feel uncertain about the right path. "
        f"How would '{lesson}' help you make a wise choice and move forward?",
        
        f"Scenario: Your team is divided on an important issue with different viewpoints. "
        f"How would you use '{lesson}' to bring everyone together and find a solution?",
        
        f"Scenario: You've made a significant mistake at work. How would you apply '{lesson}' "
        f"to take responsibility and rebuild trust with your team?",
        
        f"Scenario: You're mentoring a new team member who's overwhelmed and losing confidence. "
        f"Using '{lesson}', how would you guide them to succeed and believe in themselves?",
    ]
    
    return random.choice(scenarios)


def get_next_position(level: int, question_number: int) -> tuple[Optional[int], Optional[int], bool]:
    level_config = get_level_config(level)
    if question_number < level_config.total_questions:
        return level, question_number + 1, False

    if level == 1:
        return 2, 1, False
    if level == 2:
        return 3, 1, True
    return None, None, False
