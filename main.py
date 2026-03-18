from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import init_db
from app.routers import workflow, tracker, dashboard, chat, dashboard_chat, approvals_tracker, claims


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Vendor Qualification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(claims.router, tags=["claims"], prefix="/api/claims")
app.include_router(workflow.router, tags=["Workflow"], prefix="/api/workflow")
app.include_router(tracker.router, tags=["Tracker"], prefix="/api/tracker")
app.include_router(dashboard.router, tags=["Dashboard"], prefix="/api/dashboard")
app.include_router(chat.router, tags=["Chat"], prefix="/api/chat")
app.include_router(dashboard_chat.router, tags=["Dashboard Chat"], prefix="/api/dashboard/chat")
app.include_router(approvals_tracker.router, tags=["approvals-tracker"], prefix="/approvals-tracker")


@app.get("/api/healthchecker")
def root():
    return {"message": "Welcome to the refactored Vendor Qualification API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
