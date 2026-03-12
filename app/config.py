from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import EmailStr
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str
    MONGO_INITDB_DATABASE: str
    CLIENT_ORIGIN: str

    # SMTP (Deprecated but kept for now)
    EMAIL_HOST: Optional[str] = None
    EMAIL_PORT: Optional[int] = None
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    EMAIL_FROM: Optional[EmailStr] = None

    # Gmail API Settings
    GMAIL_CLIENT_ID: str
    GMAIL_CLIENT_SECRET: str
    GMAIL_ACCESS_TOKEN: Optional[str] = None
    GMAIL_REFRESH_TOKEN: str
    GMAIL_EXPIRE_TIME: int = 0
    GMAIL_TO: EmailStr
    
    # New settings for the workflow
    WORKFLOW_API_URL: str = "http://localhost:8000/workflow/163f55aa-4a92-4164-94cb-a4210e1d7509"
    NOTIFICATION_EMAIL: EmailStr = "admin@example.com"

    # OpenAI
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(env_file='./.env', extra='ignore')


settings = Settings()
