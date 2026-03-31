from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "local"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    forecast_service_host: str = "localhost"
    forecast_service_port: int = 8001

    control_service_host: str = "localhost"
    control_service_port: int = 8002

    simulation_service_host: str = "localhost"
    simulation_service_port: int = 8003

    dashboard_port: int = 8501

    kma_api_key: str = ""
    data_dir: str = "./data"
    model_dir: str = "./data/models"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()