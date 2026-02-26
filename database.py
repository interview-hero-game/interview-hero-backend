from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


class DatabaseManager:
    def __init__(self, mongo_uri: str, db_name: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        self.client = AsyncIOMotorClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        await self.db.sessions.create_index("created_at")

    async def disconnect(self) -> None:
        if self.client:
            self.client.close()

    async def create_session(self, first_question: str) -> str:
        now = datetime.now(timezone.utc)
        session_doc = {
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "level": 1,
            "question_number": 1,
            "current_question": first_question,
            "total_stars": 0,
            "hero_name": None,
            "life_lesson": None,
            "answers": [],
        }
        result = await self.db.sessions.insert_one(session_doc)
        return str(result.inserted_id)

    async def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        if not ObjectId.is_valid(session_id):
            return None
        doc = await self.db.sessions.find_one({"_id": ObjectId(session_id)})
        if not doc:
            return None
        doc["_id"] = str(doc["_id"])
        return doc

    async def update_session_after_answer(
        self,
        session_id: str,
        answer_payload: dict[str, Any],
        next_level: Optional[int],
        next_question_number: Optional[int],
        next_question: Optional[str],
        total_stars: int,
        hero_name: Optional[str],
        life_lesson: Optional[str],
        completed: bool,
    ) -> None:
        now = datetime.now(timezone.utc)
        update_payload: dict[str, Any] = {
            "updated_at": now,
            "total_stars": total_stars,
            "hero_name": hero_name,
            "life_lesson": life_lesson,
            "$push": {"answers": answer_payload},
        }

        if completed:
            update_payload.update(
                {
                    "status": "completed",
                    "level": 3,
                    "question_number": 1,
                    "current_question": None,
                }
            )
        else:
            update_payload.update(
                {
                    "status": "active",
                    "level": next_level,
                    "question_number": next_question_number,
                    "current_question": next_question,
                }
            )

        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {k: v for k, v in update_payload.items() if k != "$push"},
                "$push": update_payload["$push"],
            },
        )
