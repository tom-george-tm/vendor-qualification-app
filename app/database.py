from pymongo import MongoClient
from app.config import settings

client = MongoClient(settings.DATABASE_URL)
db = client[settings.MONGO_INITDB_DATABASE]

# Collection for storing workflow results
Results = db.workflow_results

# Create index for quick lookup if needed
Results.create_index([("workflow_id", 1)])
print('Connected to MongoDB...')
