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

    # Gmail API Settings (all optional — email notifications skipped if not set)
    GMAIL_CLIENT_ID: Optional[str] = None
    GMAIL_CLIENT_SECRET: Optional[str] = None
    GMAIL_ACCESS_TOKEN: Optional[str] = None
    GMAIL_REFRESH_TOKEN: Optional[str] = None
    GMAIL_EXPIRE_TIME: int = 0
    GMAIL_TO: Optional[EmailStr] = None

    # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: str = "your-azure-openai-api-key"
    AZURE_OPENAI_ENDPOINT: str = "https://your-resource.openai.azure.com/"
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"

    # Deployment names (set these to match your Azure deployment names)
    AZURE_OPENAI_DEPLOYMENT_MINI: str = "gpt-4o-mini"   # GPT-4o mini deployment
    AZURE_OPENAI_DEPLOYMENT_GPT4O: str = "gpt-4o"        # GPT-4o deployment

    # Azure Blob Storage
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_NAME: str = ""
    AZURE_STORAGE_CONTAINER_URL: Optional[str] = None

    model_config = SettingsConfigDict(env_file='./.env', extra='ignore')


settings = Settings()
