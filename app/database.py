from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
import logging

logger = logging.getLogger(__name__)

client = AsyncIOMotorClient(settings.DATABASE_URL)
db = client[settings.MONGO_INITDB_DATABASE]

# Collections
Results = db.workflow_results
ChatSessions = db.chat_sessions
ResolvedClaims = db.resolved_claims


async def init_db():
    await Results.create_index([("created_at", -1)])
    await Results.create_index([("session_id", 1), ("created_at", -1)])
    await ChatSessions.create_index([("session_id", 1)], unique=True)
    await ResolvedClaims.create_index([("claim_id", 1)], unique=True)
    await ResolvedClaims.create_index([("resolved_at", -1)])
    logger.info("Connected to MongoDB and indexes created.")
