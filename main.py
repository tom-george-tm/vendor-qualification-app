import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import init_db
from app.email_listener import email_listener_loop
from app.routers import workflow, tracker, dashboard, chat, dashboard_chat, approvals_tracker, claims


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Start the background email listener if enabled
    listener_task = None
    print(f"[EMAIL LISTENER] Enabled={settings.EMAIL_LISTENER_ENABLED}, "
          f"IMAP_USER={settings.IMAP_USERNAME}, "
          f"IMAP_PASS={'***' if settings.IMAP_PASSWORD else 'NOT SET'}")
    if settings.EMAIL_LISTENER_ENABLED and settings.IMAP_USERNAME and settings.IMAP_PASSWORD:
        listener_task = asyncio.create_task(email_listener_loop())
        print("📬 Background email listener task created.")
    else:
        print("⚠️ Email listener disabled or IMAP credentials not set — skipping.")

    yield

    # Gracefully cancel the listener on shutdown
    if listener_task and not listener_task.done():
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass


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
