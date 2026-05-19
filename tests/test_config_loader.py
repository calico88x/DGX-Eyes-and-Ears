"""
tests/test_config_loader.py

Fully offline tests for the config loader and logger setup.
Uses pytest's tmp_path to write real YAML files — no mocking needed.

Run with:  pytest tests/test_config_loader.py -v
"""

import logging
import pytest
from pathlib import Path

from app.utils.config_loader import (
    ConfigError,
    Config,
    LiteLLMConfig,
    SimulationConfig,
    load_config,
    _parse_simulation,
    _parse_memory,
    _parse_vision_pipeline,
    _parse_logging,
)


# ---------------------------------------------------------------------------
# YAML fixtures
# ---------------------------------------------------------------------------

VALID_CONFIG = """
services:
  litellm:
    base_url: "http://192.168.3.51:4000"
    api_key: "sk-local"
  stt:
    base_url: "ws://192.168.3.51:9090/asr"
    chunk_ms: 250
    sample_rate: 16000
    channels: 1
  tts:
    base_url: "http://192.168.3.51:8880"
    voice: "af_heart"
  vision:
    base_url: "http://192.168.3.51:8000/v1"

models:
  chat: "loyal-macaroni-maid-7b"
  classifier: "Qwen/Qwen2.5-1.5B-Instruct"
  vision: "Qwen/Qwen2.5-VL-7B-Instruct"
  embedding: "Qwen/Qwen2.5-1.5B-Instruct"

memory:
  db_path: "./data/memory.db"
  retrieval_top_k: 5
  context_budget_tokens: 800

vision_pipeline:
  frame_interval_ms: 2000
  frame_interval_active_ms: 800
  frame_interval_speaking_ms: 4000
  max_pixels: 1280
  jpeg_quality: 75

logging:
  level: "INFO"
  path: "./logs/eyes_and_ears.log"
"""

VALID_PARAMS = """
chat_llm:
  temperature: 0.85
  max_tokens: 512
  top_p: 0.95
  stream: true

classifier_llm:
  temperature: 0.1
  max_tokens: 150
  stream: false

vision_model:
  temperature: 0.4
  max_tokens: 300
  stream: false

embedding_model:
  dimensions: 1536

classifier_thresholds:
  speech_respond_threshold: 0.60
  stream_react_threshold: 0.70
  stream_override_threshold: 0.85
  stream_interrupt_threshold: 0.90
"""

VALID_PERSONALITY = """
name: "Aria"
persona: "curious, warm, attentive"
system_prompt: |
  You are Aria, a conversational AI companion.
  You can see and hear what the user is sharing.
tone_notes: "casual, reactive, genuine"
"""

VALID_SIMULATION = """
enabled: true
stt_delay_ms: 300
vision_delay_ms: 800
llm_delay_ms: 1200
canned_responses:
  - trigger: "hello"
    response: "Hey! Good to see you."
  - trigger: "what is that"
    response: "Looks interesting!"
"""


def write_config_dir(tmp_path: Path, overrides: dict = {}) -> str:
    """Write all four YAML files to tmp_path. overrides replaces named files."""
    files = {
        "config.yaml":      VALID_CONFIG,
        "params.yaml":      VALID_PARAMS,
        "personality.yaml": VALID_PERSONALITY,
        "simulation.yaml":  VALID_SIMULATION,
    }
    files.update(overrides)
    for name, content in files.items():
        if content is not None:
            (tmp_path / name).write_text(content, encoding="utf-8")
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Happy path — valid config
# ---------------------------------------------------------------------------

class TestLoadConfigValid:
    def test_returns_config_object(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert isinstance(cfg, Config)

    def test_config_dir_is_set(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert str(tmp_path.resolve()) in cfg.config_dir

    def test_litellm_base_url(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services.litellm.base_url == "http://192.168.3.51:4000"

    def test_litellm_api_key(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services.litellm.api_key == "sk-local"

    def test_stt_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services.stt.base_url    == "ws://192.168.3.51:9090/asr"
        assert cfg.services.stt.chunk_ms    == 250
        assert cfg.services.stt.sample_rate == 16000
        assert cfg.services.stt.channels    == 1

    def test_tts_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services.tts.base_url == "http://192.168.3.51:8880"
        assert cfg.services.tts.voice    == "af_heart"

    def test_vision_service_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services.vision.base_url == "http://192.168.3.51:8000/v1"
        assert cfg.services.vision.type     == "openai_compatible"

    def test_models_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.models.chat       == "loyal-macaroni-maid-7b"
        assert cfg.models.classifier == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.models.vision     == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert cfg.models.embedding  == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_memory_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.memory.db_path               == "./data/memory.db"
        assert cfg.memory.retrieval_top_k       == 5
        assert cfg.memory.context_budget_tokens == 800

    def test_vision_pipeline_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.vision_pipeline.frame_interval_ms          == 2000
        assert cfg.vision_pipeline.frame_interval_active_ms   == 800
        assert cfg.vision_pipeline.frame_interval_speaking_ms == 4000
        assert cfg.vision_pipeline.max_pixels                 == 1280
        assert cfg.vision_pipeline.jpeg_quality               == 75

    def test_personality_config(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.personality.name == "Aria"
        assert "Aria" in cfg.personality.system_prompt
        assert cfg.personality.persona == "curious, warm, attentive"

    def test_simulation_enabled(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.simulation.enabled    is True
        assert cfg.simulation.stt_delay_ms    == 300
        assert cfg.simulation.vision_delay_ms == 800
        assert cfg.simulation.llm_delay_ms    == 1200

    def test_simulation_canned_responses(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert len(cfg.simulation.canned_responses) == 2
        assert cfg.simulation.canned_responses[0].trigger  == "hello"
        assert cfg.simulation.canned_responses[0].response == "Hey! Good to see you."

    def test_params_is_dict(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert isinstance(cfg.params, dict)
        assert "chat_llm" in cfg.params
        assert "classifier_llm" in cfg.params
        assert "classifier_thresholds" in cfg.params

    def test_params_chat_llm_values(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        chat = cfg.params["chat_llm"]
        assert chat["temperature"] == 0.85
        assert chat["max_tokens"]  == 512

    def test_classifier_thresholds_accessible(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        t = cfg.params["classifier_thresholds"]
        assert t["speech_respond_threshold"]   == 0.60
        assert t["stream_react_threshold"]     == 0.70
        assert t["stream_override_threshold"]  == 0.85
        assert t["stream_interrupt_threshold"] == 0.90


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------

class TestConvenienceProperties:
    def test_services_shortcut(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.services is cfg.app.services

    def test_models_shortcut(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.models is cfg.app.models

    def test_memory_shortcut(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.memory is cfg.app.memory

    def test_vision_pipeline_shortcut(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.vision_pipeline is cfg.app.vision_pipeline

    def test_sim_mode_property(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        assert cfg.sim_mode is True

    def test_sim_mode_false_when_disabled(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path, {
            "simulation.yaml": "enabled: false\n"
        }))
        assert cfg.sim_mode is False

    def test_summary_returns_dict(self, tmp_path):
        cfg = load_config(write_config_dir(tmp_path))
        s = cfg.summary()
        assert isinstance(s, dict)
        assert "chat_model"   in s
        assert "litellm_url"  in s
        assert "sim_mode"     in s
        assert "memory_db"    in s

    def test_summary_no_api_key(self, tmp_path):
        """API key must not appear in the summary."""
        cfg = load_config(write_config_dir(tmp_path))
        s   = cfg.summary()
        assert "sk-local" not in str(s)


# ---------------------------------------------------------------------------
# Error cases — missing files
# ---------------------------------------------------------------------------

class TestMissingFiles:
    def test_missing_config_dir_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path / "nonexistent"))

    def test_missing_config_yaml_raises(self, tmp_path):
        write_config_dir(tmp_path, {"config.yaml": None})
        (tmp_path / "config.yaml").unlink(missing_ok=True)
        with pytest.raises(ConfigError, match="config.yaml"):
            load_config(str(tmp_path))

    def test_missing_params_yaml_raises(self, tmp_path):
        write_config_dir(tmp_path)
        (tmp_path / "params.yaml").unlink()
        with pytest.raises(ConfigError, match="params.yaml"):
            load_config(str(tmp_path))

    def test_missing_personality_yaml_raises(self, tmp_path):
        write_config_dir(tmp_path)
        (tmp_path / "personality.yaml").unlink()
        with pytest.raises(ConfigError, match="personality.yaml"):
            load_config(str(tmp_path))

    def test_missing_simulation_yaml_is_ok(self, tmp_path):
        """simulation.yaml is optional."""
        write_config_dir(tmp_path)
        (tmp_path / "simulation.yaml").unlink()
        cfg = load_config(str(tmp_path))
        assert cfg.simulation.enabled is False


# ---------------------------------------------------------------------------
# Error cases — malformed YAML
# ---------------------------------------------------------------------------

class TestMalformedYAML:
    def test_malformed_config_yaml_raises(self, tmp_path):
        write_config_dir(tmp_path, {"config.yaml": "key: [unclosed"})
        with pytest.raises(ConfigError, match="parse"):
            load_config(str(tmp_path))

    def test_non_mapping_config_raises(self, tmp_path):
        write_config_dir(tmp_path, {"config.yaml": "- just a list\n- not a mapping\n"})
        with pytest.raises(ConfigError, match="mapping"):
            load_config(str(tmp_path))

    def test_empty_config_yaml_raises(self, tmp_path):
        """Empty file has no required keys — should list them all."""
        write_config_dir(tmp_path, {"config.yaml": "\n"})
        with pytest.raises(ConfigError):
            load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Error cases — missing required fields
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    def test_missing_litellm_base_url(self, tmp_path):
        bad = VALID_CONFIG.replace('base_url: "http://192.168.3.51:4000"', "", 1)
        write_config_dir(tmp_path, {"config.yaml": bad})
        with pytest.raises(ConfigError, match="litellm.base_url"):
            load_config(str(tmp_path))

    def test_missing_stt_base_url(self, tmp_path):
        bad = VALID_CONFIG.replace('base_url: "ws://192.168.3.51:9090/asr"', "", 1)
        write_config_dir(tmp_path, {"config.yaml": bad})
        with pytest.raises(ConfigError, match="stt.base_url"):
            load_config(str(tmp_path))

    def test_missing_chat_model(self, tmp_path):
        bad = VALID_CONFIG.replace('chat: "loyal-macaroni-maid-7b"', "", 1)
        write_config_dir(tmp_path, {"config.yaml": bad})
        with pytest.raises(ConfigError, match="models.chat"):
            load_config(str(tmp_path))

    def test_multiple_errors_reported_together(self, tmp_path):
        """All missing fields should appear in one ConfigError, not just the first."""
        bad = "\nservices:\n  litellm:\n    api_key: sk-local\nmodels:\n  chat: test\n"
        write_config_dir(tmp_path, {"config.yaml": bad})
        with pytest.raises(ConfigError) as exc_info:
            load_config(str(tmp_path))
        msg = str(exc_info.value)
        # Multiple bullets should be present
        assert msg.count("•") > 1

    def test_empty_system_prompt_raises(self, tmp_path):
        bad = "name: Aria\npersona: cool\nsystem_prompt: ''\n"
        write_config_dir(tmp_path, {"personality.yaml": bad})
        with pytest.raises(ConfigError, match="system_prompt"):
            load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_memory_defaults_when_section_absent(self, tmp_path):
        no_mem = "\n".join(
            l for l in VALID_CONFIG.splitlines()
            if not l.startswith("memory")
        )
        write_config_dir(tmp_path, {"config.yaml": no_mem})
        cfg = load_config(str(tmp_path))
        assert cfg.memory.db_path               == "./data/memory.db"
        assert cfg.memory.retrieval_top_k       == 5
        assert cfg.memory.context_budget_tokens == 800

    def test_vision_pipeline_defaults(self, tmp_path):
        no_vis = "\n".join(
            l for l in VALID_CONFIG.splitlines()
            if not l.startswith("vision_pipeline")
        )
        write_config_dir(tmp_path, {"config.yaml": no_vis})
        cfg = load_config(str(tmp_path))
        assert cfg.vision_pipeline.frame_interval_ms == 2000
        assert cfg.vision_pipeline.jpeg_quality      == 75

    def test_logging_defaults(self, tmp_path):
        no_log = "\n".join(
            l for l in VALID_CONFIG.splitlines()
            if not l.startswith("logging")
        )
        write_config_dir(tmp_path, {"config.yaml": no_log})
        cfg = load_config(str(tmp_path))
        assert cfg.log_cfg.level == "INFO"

    def test_invalid_log_level_coerces_to_info(self):
        cfg = _parse_logging({"level": "BANANA"})
        assert cfg.level == "INFO"

    def test_simulation_defaults_when_absent(self, tmp_path):
        write_config_dir(tmp_path)
        (tmp_path / "simulation.yaml").unlink()
        cfg = load_config(str(tmp_path))
        assert cfg.simulation.enabled          is False
        assert cfg.simulation.stt_delay_ms     == 300
        assert cfg.simulation.canned_responses == []

    def test_stt_defaults(self, tmp_path):
        minimal = VALID_CONFIG.replace(
            "    chunk_ms: 250\n    sample_rate: 16000\n    channels: 1\n", ""
        )
        write_config_dir(tmp_path, {"config.yaml": minimal})
        cfg = load_config(str(tmp_path))
        assert cfg.services.stt.chunk_ms    == 250
        assert cfg.services.stt.sample_rate == 16000


# ---------------------------------------------------------------------------
# Unit tests for individual parsers
# ---------------------------------------------------------------------------

class TestParsers:
    def test_parse_memory(self):
        m = _parse_memory({"db_path": "./custom.db", "retrieval_top_k": 10})
        assert m.db_path         == "./custom.db"
        assert m.retrieval_top_k == 10

    def test_parse_memory_empty(self):
        m = _parse_memory({})
        assert m.db_path == "./data/memory.db"

    def test_parse_vision_pipeline(self):
        v = _parse_vision_pipeline({"frame_interval_ms": 500, "jpeg_quality": 60})
        assert v.frame_interval_ms == 500
        assert v.jpeg_quality      == 60

    def test_parse_simulation_canned_responses(self):
        raw = {
            "enabled": True,
            "canned_responses": [
                {"trigger": "hi",    "response": "Hello!"},
                {"trigger": "bye",   "response": "Goodbye!"},
            ]
        }
        s = _parse_simulation(raw)
        assert s.enabled is True
        assert len(s.canned_responses) == 2
        assert s.canned_responses[1].trigger == "bye"

    def test_parse_simulation_ignores_non_dict_responses(self):
        raw = {"canned_responses": ["not a dict", 42, {"trigger": "ok", "response": "yes"}]}
        s = _parse_simulation(raw)
        assert len(s.canned_responses) == 1

    def test_parse_logging_valid_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            cfg = _parse_logging({"level": level})
            assert cfg.level == level

    def test_parse_logging_case_insensitive(self):
        cfg = _parse_logging({"level": "debug"})
        assert cfg.level == "DEBUG"


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

class TestLoggerSetup:
    def test_setup_logging_does_not_raise(self, tmp_path):
        from app.utils.logger import setup_logging
        cfg = load_config(write_config_dir(tmp_path))
        # Override log path to tmp so we don't write to project logs/
        cfg.app.logging.path = str(tmp_path / "test.log")
        setup_logging(cfg)   # should not raise

    def test_log_file_is_created(self, tmp_path):
        from app.utils.logger import setup_logging
        cfg = load_config(write_config_dir(tmp_path))
        cfg.app.logging.path = str(tmp_path / "test.log")
        setup_logging(cfg)
        logging.getLogger("test").info("test message")
        assert (tmp_path / "test.log").exists()

    def test_get_logger_returns_logger(self):
        from app.utils.logger import get_logger
        lg = get_logger("test.module")
        assert isinstance(lg, logging.Logger)
        assert lg.name == "test.module"
