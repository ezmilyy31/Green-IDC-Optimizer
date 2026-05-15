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

    # /control/rl 엔드포인트가 로드할 best 모델 경로 (.zip). _vecnorm.pkl 페어가 같은 위치에 있어야 함.
    # 효율 우선 정책 — 평상시·heat_wave·chiller_derate에서 PUE 최우수.
    rl_model_path: str = "./data/models/sac-wetbulb-1m.zip"

    # /control/rl-hybrid 엔드포인트가 사용할 안전 우선 정책 경로 (.zip).
    # 부하/온도 신호로 위기 감지 시 best 대신 사용 → 모든 시나리오 위반 0% 달성.
    rl_safety_model_path: str = "./data/models/sac-dr-fresh-1m.zip"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()