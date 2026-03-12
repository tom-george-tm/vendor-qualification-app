from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import workflow, chat

app = FastAPI(title="Vendor Qualification API")

origins = [
    settings.CLIENT_ORIGIN,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allowing all for easier testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow.router, tags=['Workflow'], prefix='/api/workflow')
app.include_router(chat.router, tags=['Chat'], prefix='/api/chat')

@app.get("/api/healthchecker")
def root():
    return {"message": "Welcome to the refactored Vendor Qualification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
