"""
app/utils/config_loader.py

Loads all four config files and exposes a single typed Config object.

    config = load_config("./config")

    config.services.litellm.base_url   # "http://192.168.3.51:4000"
    config.models.chat                 # "loyal-macaroni-maid-7b"
    config.personality.system_prompt   # full persona string
    config.params["chat_llm"]          # raw dict — flexible per model
    config.simulation.enabled          # True/False

All four files are validated on load. Missing required fields raise
ConfigError listing all problems at once.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when config files are missing, malformed, or invalid."""
    pass


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LiteLLMConfig:
    base_url: str
    api_key:  str = "sk-local"


@dataclass
class STTConfig:
    base_url:    str
    type:        str = "websocket"
    chunk_ms:    int = 250
    sample_rate: int = 16000
    channels:    int = 1


@dataclass
class TTSConfig:
    base_url: str
    voice:    str = "af_heart"
    type:     str = "http_stream"


@dataclass
class VisionServiceConfig:
    base_url: str
    type:     str = "openai_compatible"


@dataclass
class ServicesConfig:
    litellm: LiteLLMConfig
    stt:     STTConfig
    tts:     TTSConfig
    vision:  VisionServiceConfig


@dataclass
class ModelsConfig:
    chat:       str
    classifier: str
    vision:     str
    embedding:  str


@dataclass
class MemoryConfig:
    db_path:               str = "./data/memory.db"
    retrieval_top_k:       int = 5
    context_budget_tokens: int = 800


@dataclass
class VisionPipelineConfig:
    frame_interval_ms:          int = 2000
    frame_interval_active_ms:   int = 800
    frame_interval_speaking_ms: int = 4000
    max_pixels:                 int = 1280
    jpeg_quality:               int = 75


@dataclass
class LoggingConfig:
    level: str = "INFO"
    path:  str = "./logs/eyes_and_ears.log"


@dataclass
class AppConfig:
    services:        ServicesConfig
    models:          ModelsConfig
    memory:          MemoryConfig         = field(default_factory=MemoryConfig)
    vision_pipeline: VisionPipelineConfig = field(default_factory=VisionPipelineConfig)
    logging:         LoggingConfig        = field(default_factory=LoggingConfig)


@dataclass
class PersonalityConfig:
    name:          str = "Aria"
    persona:       str = ""
    system_prompt: str = ""
    tone_notes:    str = ""


@dataclass
class CannedResponse:
    trigger:  str
    response: str


@dataclass
class SimulationConfig:
    enabled:          bool                = False
    stt_delay_ms:     int                 = 300
    vision_delay_ms:  int                 = 800
    llm_delay_ms:     int                 = 1200
    canned_responses: list[CannedResponse] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Single source of truth for all runtime configuration.
    Convenience properties proxy the most-used nested fields.
    """
    app:         AppConfig
    params:      dict              # raw params.yaml — stays a dict, flexible per model
    personality: PersonalityConfig
    simulation:  SimulationConfig
    config_dir:  str = ""

    # ---- convenience properties ----------------------------------------

    @property
    def services(self) -> ServicesConfig:
        return self.app.services

    @property
    def models(self) -> ModelsConfig:
        return self.app.models

    @property
    def memory(self) -> MemoryConfig:
        return self.app.memory

    @property
    def vision_pipeline(self) -> VisionPipelineConfig:
        return self.app.vision_pipeline

    @property
    def log_cfg(self) -> LoggingConfig:
        return self.app.logging

    @property
    def sim_mode(self) -> bool:
        """True when simulation.yaml has enabled: true."""
        return self.simulation.enabled

    def summary(self) -> dict:
        """Safe loggable snapshot — no secrets."""
        return {
            "config_dir":       self.config_dir,
            "sim_mode":         self.sim_mode,
            "chat_model":       self.models.chat,
            "classifier_model": self.models.classifier,
            "vision_model":     self.models.vision,
            "litellm_url":      self.services.litellm.base_url,
            "stt_url":          self.services.stt.base_url,
            "tts_url":          self.services.tts.base_url,
            "vision_url":       self.services.vision.base_url,
            "memory_db":        self.memory.db_path,
            "personality_name": self.personality.name,
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_config(config_dir: str = "./config") -> Config:
    """
    Load and validate all four config files from config_dir.

    Raises ConfigError if:
      - config_dir does not exist
      - any required file is missing
      - any YAML is malformed
      - any required field is absent or empty

    All validation errors are collected and reported together.
    """
    path = Path(config_dir)
    if not path.exists():
        raise ConfigError(f"Config directory not found: {config_dir!r}")

    app_raw         = _load_yaml(path / "config.yaml",      required=True)
    params_raw      = _load_yaml(path / "params.yaml",      required=True)
    personality_raw = _load_yaml(path / "personality.yaml", required=True)
    simulation_raw  = _load_yaml(path / "simulation.yaml",  required=False) or {}

    errors: list[str] = []

    services    = _parse_services(app_raw.get("services", {}),   errors)
    models      = _parse_models(app_raw.get("models",    {}),    errors)
    memory      = _parse_memory(app_raw.get("memory",    {}))
    vis_pip     = _parse_vision_pipeline(app_raw.get("vision_pipeline", {}))
    log_cfg     = _parse_logging(app_raw.get("logging",  {}))
    personality = _parse_personality(personality_raw,             errors)
    simulation  = _parse_simulation(simulation_raw)
    params      = params_raw if isinstance(params_raw, dict) else {}

    if errors:
        raise ConfigError(
            "Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    cfg = Config(
        app = AppConfig(
            services        = services,
            models          = models,
            memory          = memory,
            vision_pipeline = vis_pip,
            logging         = log_cfg,
        ),
        params      = params,
        personality = personality,
        simulation  = simulation,
        config_dir  = str(path.resolve()),
    )

    logger.info(
        "Config loaded | dir=%s sim_mode=%s model=%s",
        config_dir, cfg.sim_mode, cfg.models.chat
    )
    return cfg


# ---------------------------------------------------------------------------
# Internal — YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path, required: bool = True) -> Optional[dict]:
    try:
        import yaml
    except ImportError:
        raise ConfigError("PyYAML not installed. Run: pip install pyyaml")

    if not path.exists():
        if required:
            raise ConfigError(f"Required config file not found: {path.name}")
        return None

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigError(f"Failed to parse {path.name}: {exc}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path.name} must contain a YAML mapping, got {type(data).__name__}"
        )
    return data


# ---------------------------------------------------------------------------
# Internal — section parsers
# ---------------------------------------------------------------------------

def _parse_services(raw: dict, errors: list[str]) -> ServicesConfig:
    llm = raw.get("litellm", {})
    stt = raw.get("stt",     {})
    tts = raw.get("tts",     {})
    vis = raw.get("vision",  {})

    _require(llm, "base_url", "services.litellm.base_url", errors)
    _require(stt, "base_url", "services.stt.base_url",     errors)
    _require(tts, "base_url", "services.tts.base_url",     errors)
    _require(vis, "base_url", "services.vision.base_url",  errors)

    return ServicesConfig(
        litellm = LiteLLMConfig(
            base_url = llm.get("base_url", ""),
            api_key  = llm.get("api_key",  "sk-local"),
        ),
        stt = STTConfig(
            base_url    = stt.get("base_url", ""),
            type        = stt.get("type",        "websocket"),
            chunk_ms    = int(stt.get("chunk_ms",    250)),
            sample_rate = int(stt.get("sample_rate", 16000)),
            channels    = int(stt.get("channels",    1)),
        ),
        tts = TTSConfig(
            base_url = tts.get("base_url", ""),
            voice    = tts.get("voice",    "af_heart"),
            type     = tts.get("type",     "http_stream"),
        ),
        vision = VisionServiceConfig(
            base_url = vis.get("base_url", ""),
            type     = vis.get("type",     "openai_compatible"),
        ),
    )


def _parse_models(raw: dict, errors: list[str]) -> ModelsConfig:
    _require(raw, "chat",       "models.chat",       errors)
    _require(raw, "classifier", "models.classifier", errors)
    _require(raw, "vision",     "models.vision",     errors)
    _require(raw, "embedding",  "models.embedding",  errors)

    return ModelsConfig(
        chat       = raw.get("chat",       ""),
        classifier = raw.get("classifier", ""),
        vision     = raw.get("vision",     ""),
        embedding  = raw.get("embedding",  ""),
    )


def _parse_memory(raw: dict) -> MemoryConfig:
    return MemoryConfig(
        db_path               = raw.get("db_path",               "./data/memory.db"),
        retrieval_top_k       = int(raw.get("retrieval_top_k",   5)),
        context_budget_tokens = int(raw.get("context_budget_tokens", 800)),
    )


def _parse_vision_pipeline(raw: dict) -> VisionPipelineConfig:
    return VisionPipelineConfig(
        frame_interval_ms          = int(raw.get("frame_interval_ms",          2000)),
        frame_interval_active_ms   = int(raw.get("frame_interval_active_ms",   800)),
        frame_interval_speaking_ms = int(raw.get("frame_interval_speaking_ms", 4000)),
        max_pixels                 = int(raw.get("max_pixels",                 1280)),
        jpeg_quality               = int(raw.get("jpeg_quality",               75)),
    )


def _parse_logging(raw: dict) -> LoggingConfig:
    level = str(raw.get("level", "INFO")).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    return LoggingConfig(
        level = level,
        path  = raw.get("path", "./logs/eyes_and_ears.log"),
    )


def _parse_personality(raw: dict, errors: list[str]) -> PersonalityConfig:
    if not raw.get("system_prompt", "").strip():
        errors.append("personality.yaml: system_prompt is required and cannot be empty")
    return PersonalityConfig(
        name          = raw.get("name",          "Aria"),
        persona       = raw.get("persona",       ""),
        system_prompt = raw.get("system_prompt", ""),
        tone_notes    = raw.get("tone_notes",    ""),
    )


def _parse_simulation(raw: dict) -> SimulationConfig:
    canned = [
        CannedResponse(trigger=str(r.get("trigger", "")), response=str(r.get("response", "")))
        for r in raw.get("canned_responses", [])
        if isinstance(r, dict)
    ]
    return SimulationConfig(
        enabled          = bool(raw.get("enabled",        False)),
        stt_delay_ms     = int(raw.get("stt_delay_ms",    300)),
        vision_delay_ms  = int(raw.get("vision_delay_ms", 800)),
        llm_delay_ms     = int(raw.get("llm_delay_ms",    1200)),
        canned_responses = canned,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require(raw: dict, key: str, label: str, errors: list[str]) -> None:
    val = raw.get(key, "")
    if not val or (isinstance(val, str) and not val.strip()):
        errors.append(f"{label} is required but missing or empty")
