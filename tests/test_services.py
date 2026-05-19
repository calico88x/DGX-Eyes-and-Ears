"""
tests/test_services.py

Offline tests for all service adapters and their simulation stubs.
Real services are tested with mocked HTTP (unittest.mock).
Sim services are tested directly — no mocking needed.

Run with:  pytest tests/test_services.py -v
"""

import json
import struct
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.services.llm_service       import LLMService
from app.services.tts_service       import TTSService, _SILENCE_WAV
from app.services.stt_service       import STTService
from app.services.vision_service    import VisionService
from app.services.embedding_service import EmbeddingService
from app.simulation.sim_services    import (
    SimLLMService,
    SimTTSService,
    SimSTTService,
    SimVisionService,
    _make_silence_wav,
)
from app.utils.config_loader import (
    Config, AppConfig, ServicesConfig, ModelsConfig,
    LiteLLMConfig, STTConfig, TTSConfig, VisionServiceConfig,
    MemoryConfig, VisionPipelineConfig, LoggingConfig,
    PersonalityConfig, SimulationConfig, CannedResponse,
)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

def make_config(sim_enabled: bool = False) -> Config:
    return Config(
        app = AppConfig(
            services = ServicesConfig(
                litellm = LiteLLMConfig(
                    base_url = "http://localhost:4000",
                    api_key  = "sk-test",
                ),
                stt = STTConfig(
                    base_url    = "ws://localhost:9090/asr",
                    chunk_ms    = 250,
                    sample_rate = 16000,
                    channels    = 1,
                ),
                tts = TTSConfig(
                    base_url = "http://localhost:8880",
                    voice    = "af_heart",
                ),
                vision = VisionServiceConfig(
                    base_url = "http://localhost:8000/v1",
                ),
            ),
            models = ModelsConfig(
                chat       = "test-chat-model",
                classifier = "test-classifier-model",
                vision     = "test-vision-model",
                embedding  = "test-embedding-model",
            ),
            memory          = MemoryConfig(),
            vision_pipeline = VisionPipelineConfig(),
            logging         = LoggingConfig(),
        ),
        params = {
            "chat_llm":    {"temperature": 0.85, "max_tokens": 512},
            "vision_model": {"temperature": 0.4,  "max_tokens": 300},
            "classifier_llm": {"temperature": 0.1, "max_tokens": 150},
        },
        personality = PersonalityConfig(
            name="Aria", system_prompt="You are Aria."
        ),
        simulation = SimulationConfig(
            enabled=sim_enabled,
            stt_delay_ms=50,
            vision_delay_ms=50,
            llm_delay_ms=50,
            canned_responses=[
                CannedResponse(trigger="hello", response="Hey there!"),
                CannedResponse(trigger="game",  response="Looks exciting!"),
            ],
        ),
    )


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def sim_config():
    return make_config(sim_enabled=True)


# ---------------------------------------------------------------------------
# SSE helpers for LLM mocking
# ---------------------------------------------------------------------------

def make_sse_response(*chunks: str, include_done: bool = True) -> list[bytes]:
    """Build a list of SSE byte lines as requests would return."""
    lines = []
    for chunk in chunks:
        data = json.dumps({"choices": [{"delta": {"content": chunk}}]})
        lines.append(f"data: {data}".encode())
        lines.append(b"")   # blank line between events
    if include_done:
        lines.append(b"data: [DONE]")
    return lines


def mock_streaming_response(chunks: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = iter(make_sse_response(*chunks))
    return resp


# ---------------------------------------------------------------------------
# LLMService
# ---------------------------------------------------------------------------

class TestLLMService:
    def test_stream_chat_yields_chunks(self, config):
        llm  = LLMService(config)
        mock = mock_streaming_response(["Hello", " world", "!"])
        with patch("requests.post", return_value=mock):
            result = list(llm.stream_chat([{"role": "user", "content": "hi"}]))
        assert result == ["Hello", " world", "!"]

    def test_chat_returns_full_string(self, config):
        llm  = LLMService(config)
        mock = mock_streaming_response(["Hello", " world"])
        with patch("requests.post", return_value=mock):
            result = llm.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_system_prompt_prepended(self, config):
        llm  = LLMService(config)
        mock = mock_streaming_response(["ok"])
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["messages"] = json["messages"]
            return mock_streaming_response(["ok"])

        with patch("requests.post", side_effect=capture):
            list(llm.stream_chat(
                [{"role": "user", "content": "hi"}],
                system_prompt="Be helpful."
            ))

        assert captured["messages"][0]["role"]    == "system"
        assert captured["messages"][0]["content"] == "Be helpful."

    def test_no_duplicate_system_message(self, config):
        llm  = LLMService(config)
        msgs = [
            {"role": "system", "content": "existing"},
            {"role": "user",   "content": "hi"},
        ]
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["messages"] = json["messages"]
            return mock_streaming_response(["ok"])

        with patch("requests.post", side_effect=capture):
            list(llm.stream_chat(msgs, system_prompt="new"))

        system_msgs = [m for m in captured["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "existing"

    def test_uses_correct_model(self, config):
        llm  = LLMService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["model"] = json["model"]
            return mock_streaming_response(["ok"])

        with patch("requests.post", side_effect=capture):
            list(llm.stream_chat([{"role": "user", "content": "hi"}]))

        assert captured["model"] == "test-chat-model"

    def test_stream_is_true_in_payload(self, config):
        llm  = LLMService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["stream"] = json.get("stream")
            return mock_streaming_response(["ok"])

        with patch("requests.post", side_effect=capture):
            list(llm.stream_chat([{"role": "user", "content": "hi"}]))

        assert captured["stream"] is True

    def test_connection_error_yields_nothing(self, config):
        import requests as req
        llm = LLMService(config)
        with patch("requests.post", side_effect=req.RequestException("refused")):
            result = list(llm.stream_chat([{"role": "user", "content": "hi"}]))
        assert result == []

    def test_skips_blank_delta(self, config):
        llm  = LLMService(config)
        lines = [
            b'data: {"choices": [{"delta": {"content": ""}}]}',
            b'data: {"choices": [{"delta": {"content": "hi"}}]}',
            b"data: [DONE]",
        ]
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.iter_lines.return_value = iter(lines)
        with patch("requests.post", return_value=mock):
            result = list(llm.stream_chat([{"role": "user", "content": "x"}]))
        assert result == ["hi"]

    def test_malformed_sse_line_skipped(self, config):
        llm  = LLMService(config)
        lines = [
            b"data: not-json-at-all",
            b'data: {"choices": [{"delta": {"content": "ok"}}]}',
            b"data: [DONE]",
        ]
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.iter_lines.return_value = iter(lines)
        with patch("requests.post", return_value=mock):
            result = list(llm.stream_chat([{"role": "user", "content": "x"}]))
        assert result == ["ok"]


# ---------------------------------------------------------------------------
# TTSService
# ---------------------------------------------------------------------------

class TestTTSService:
    def test_synthesize_returns_bytes(self, config):
        tts  = TTSService(config)
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.content = b"fake-mp3-bytes"
        with patch("requests.post", return_value=mock):
            result = tts.synthesize("Hello world")
        assert isinstance(result, bytes)
        assert result == b"fake-mp3-bytes"

    def test_sends_correct_payload(self, config):
        tts      = TTSService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured.update(json)
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.content = b"audio"
            return m

        with patch("requests.post", side_effect=capture):
            tts.synthesize("Say this")

        assert captured["input"]           == "Say this"
        assert captured["voice"]           == "af_heart"
        assert captured["response_format"] == "wav"

    def test_voice_override(self, config):
        tts      = TTSService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["voice"] = json["voice"]
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.content = b"audio"
            return m

        with patch("requests.post", side_effect=capture):
            tts.synthesize("hello", voice="custom_voice")

        assert captured["voice"] == "custom_voice"

    def test_empty_text_returns_silence(self, config):
        tts = TTSService(config)
        with patch("requests.post") as mock_post:
            result = tts.synthesize("")
        mock_post.assert_not_called()
        assert result == _SILENCE_WAV

    def test_connection_error_returns_silence(self, config):
        import requests as req
        tts = TTSService(config)
        with patch("requests.post", side_effect=req.RequestException("down")):
            result = tts.synthesize("Hello")
        assert result == _SILENCE_WAV


# ---------------------------------------------------------------------------
# VisionService
# ---------------------------------------------------------------------------

class TestVisionService:
    def _mock_vision_response(self, text: str) -> MagicMock:
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {
            "choices": [{"message": {"content": text}}]
        }
        return mock

    def test_describe_frame_returns_string(self, config):
        vis  = VisionService(config)
        mock = self._mock_vision_response("User is coding in Python.")
        with patch("requests.post", return_value=mock):
            result = vis.describe_frame(b"fake-image-bytes")
        assert result == "User is coding in Python."

    def test_sends_base64_image(self, config):
        import base64
        vis      = VisionService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["content"] = json["messages"][0]["content"]
            return self._mock_vision_response("ok")

        with patch("requests.post", side_effect=capture):
            vis.describe_frame(b"\xff\xd8\xff")   # JPEG magic bytes

        image_part = captured["content"][0]
        assert image_part["type"] == "image_url"
        assert "base64" in image_part["image_url"]["url"]

    def test_empty_bytes_returns_empty_string(self, config):
        vis = VisionService(config)
        with patch("requests.post") as mock_post:
            result = vis.describe_frame(b"")
        mock_post.assert_not_called()
        assert result == ""

    def test_connection_error_returns_empty_string(self, config):
        import requests as req
        vis = VisionService(config)
        with patch("requests.post", side_effect=req.RequestException("down")):
            result = vis.describe_frame(b"image")
        assert result == ""

    def test_custom_prompt_sent(self, config):
        vis      = VisionService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["content"] = json["messages"][0]["content"]
            return self._mock_vision_response("ok")

        with patch("requests.post", side_effect=capture):
            vis.describe_frame(b"image", prompt="What game is this?")

        text_part = captured["content"][1]
        assert text_part["text"] == "What game is this?"

    def test_uses_vision_model(self, config):
        vis      = VisionService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["model"] = json["model"]
            return self._mock_vision_response("ok")

        with patch("requests.post", side_effect=capture):
            vis.describe_frame(b"image")

        assert captured["model"] == "test-vision-model"


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    def _mock_embed_response(self, vector: list[float]) -> MagicMock:
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"data": [{"embedding": vector}]}
        return mock

    def test_embed_returns_float_list(self, config):
        emb    = EmbeddingService(config)
        vector = [0.1] * 8
        with patch("requests.post", return_value=self._mock_embed_response(vector)):
            result = emb.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == 8

    def test_embed_empty_string_returns_none(self, config):
        emb = EmbeddingService(config)
        with patch("requests.post") as mock_post:
            result = emb.embed("")
        mock_post.assert_not_called()
        assert result is None

    def test_embed_whitespace_returns_none(self, config):
        emb = EmbeddingService(config)
        with patch("requests.post") as mock_post:
            result = emb.embed("   ")
        mock_post.assert_not_called()
        assert result is None

    def test_connection_error_returns_none(self, config):
        import requests as req
        emb = EmbeddingService(config)
        with patch("requests.post", side_effect=req.RequestException("down")):
            result = emb.embed("test")
        assert result is None

    def test_uses_embedding_model(self, config):
        emb      = EmbeddingService(config)
        captured = {}

        def capture(url, json=None, **kwargs):
            captured["model"] = json["model"]
            return self._mock_embed_response([0.1])

        with patch("requests.post", side_effect=capture):
            emb.embed("test text")

        assert captured["model"] == "test-embedding-model"

    def test_embed_plugs_into_memory_writer(self, config):
        """Verify embed() signature matches EmbedFn expected by MemoryWriter."""
        from app.memory.db     import MemoryDB
        from app.memory.writer import MemoryWriter

        emb  = EmbeddingService(config)
        mock = self._mock_embed_response([0.1, 0.2, 0.3])

        with MemoryDB(":memory:") as db:
            with patch("requests.post", return_value=mock):
                writer = MemoryWriter(db, embed_fn=emb.embed)
                sid    = writer.create_session()
                eid    = writer.write_episode(sid, "user", "test content")
                assert eid is not None


# ---------------------------------------------------------------------------
# STTService (interface only — WebSocket is integration-tested live)
# ---------------------------------------------------------------------------

class TestSTTServiceInterface:
    def test_on_transcript_registers_callback(self, config):
        stt = STTService(config)
        cb  = lambda t, f, c: None
        stt.on_transcript(cb)
        assert cb in stt._callbacks

    def test_multiple_callbacks_registered(self, config):
        stt = STTService(config)
        stt.on_transcript(lambda t, f, c: None)
        stt.on_transcript(lambda t, f, c: None)
        assert len(stt._callbacks) == 2

    def test_send_audio_drops_when_queue_full(self, config):
        stt = STTService(config)
        for _ in range(100):
            stt.send_audio(b"\x00" * 512)   # should not raise

    def test_not_connected_initially(self, config):
        stt = STTService(config)
        assert stt.connected is False

    def test_handle_message_fires_callback(self, config):
        stt    = STTService(config)
        fired  = []
        stt.on_transcript(lambda t, f, c: fired.append((t, f, c)))
        stt._handle_message(json.dumps({
            "status": "active_transcription",
            "lines": [],
            "buffer_transcription": "hello there",
            "buffer_diarization": "",
        }))
        assert len(fired) == 1
        assert fired[0][0] == "hello there"
        assert fired[0][1] is True

    def test_handle_message_ignores_empty_text(self, config):
        stt   = STTService(config)
        fired = []
        stt.on_transcript(lambda t, f, c: fired.append(t))
        stt._handle_message(json.dumps({"text": "", "is_final": True}))
        assert fired == []

    def test_handle_message_ignores_malformed_json(self, config):
        stt = STTService(config)
        stt._handle_message("not json at all")   # should not raise

    def test_callback_error_does_not_crash_handler(self, config):
        stt = STTService(config)
        stt.on_transcript(lambda t, f, c: (_ for _ in ()).throw(RuntimeError("boom")))
        stt._handle_message(json.dumps({"text": "hello", "is_final": True}))


# ---------------------------------------------------------------------------
# SimLLMService
# ---------------------------------------------------------------------------

class TestSimLLMService:
    def test_stream_chat_yields_strings(self, sim_config):
        sim    = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        result = list(sim.stream_chat([{"role": "user", "content": "hi"}]))
        assert all(isinstance(c, str) for c in result)
        assert len(result) > 0

    def test_chat_returns_string(self, sim_config):
        sim    = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        result = sim.chat([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_canned_trigger_matched(self, sim_config):
        sim = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        result = sim.chat([{"role": "user", "content": "hello there"}])
        assert "Hey there!" in result

    def test_game_trigger_matched(self, sim_config):
        sim = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        result = sim.chat([{"role": "user", "content": "what about this game"}])
        assert "Looks exciting!" in result

    def test_cycles_default_responses(self, sim_config):
        sim = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        results = [
            sim.chat([{"role": "user", "content": "xyz-no-match"}])
            for _ in range(6)
        ]
        # Should cycle — not all identical
        assert len(set(results)) > 1

    def test_no_user_messages_returns_default(self, sim_config):
        sim = SimLLMService(sim_config, word_delay_s=0, first_token_delay_s=0)
        result = sim.chat([{"role": "system", "content": "be helpful"}])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SimTTSService
# ---------------------------------------------------------------------------

class TestSimTTSService:
    def test_synthesize_returns_bytes(self, sim_config):
        sim    = SimTTSService(sim_config)
        result = sim.synthesize("Hello world")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_synthesize_empty_returns_silence(self, sim_config):
        sim    = SimTTSService(sim_config)
        result = sim.synthesize("")
        assert result[:4] == b"RIFF"   # valid WAV header

    def test_wav_header_is_valid(self, sim_config):
        sim    = SimTTSService(sim_config)
        result = sim.synthesize("Test audio output please speak")
        assert result[:4]  == b"RIFF"
        assert result[8:12] == b"WAVE"

    def test_make_silence_wav_structure(self):
        wav = _make_silence_wav(0.5)
        assert wav[:4]  == b"RIFF"
        assert wav[8:12] == b"WAVE"
        # Validate data chunk length
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 16000   # 0.5s * 16000 samples * 2 bytes

    def test_longer_text_makes_longer_audio(self, sim_config):
        sim   = SimTTSService(sim_config)
        short = sim.synthesize("Hi")
        long  = sim.synthesize("Hello this is a longer sentence with more words in it")
        assert len(long) > len(short)


# ---------------------------------------------------------------------------
# SimSTTService
# ---------------------------------------------------------------------------

class TestSimSTTService:
    def test_inject_phrase_fires_callback(self, sim_config):
        sim   = SimSTTService(sim_config)
        fired = []
        sim.on_transcript(lambda t, f, c: fired.append((t, f, c)))
        sim.inject_phrase("hello world", is_final=True, confidence=0.95)
        assert len(fired) == 1
        assert fired[0] == ("hello world", True, 0.95)

    def test_multiple_callbacks_all_fired(self, sim_config):
        sim    = SimSTTService(sim_config)
        fired1 = []
        fired2 = []
        sim.on_transcript(lambda t, f, c: fired1.append(t))
        sim.on_transcript(lambda t, f, c: fired2.append(t))
        sim.inject_phrase("test")
        assert fired1 == ["test"]
        assert fired2 == ["test"]

    def test_callback_error_does_not_crash(self, sim_config):
        sim = SimSTTService(sim_config)
        sim.on_transcript(lambda t, f, c: (_ for _ in ()).throw(RuntimeError("boom")))
        sim.inject_phrase("test")   # should not raise

    def test_send_audio_is_noop(self, sim_config):
        sim = SimSTTService(sim_config)
        sim.send_audio(b"\x00" * 512)   # should not raise

    def test_not_connected_before_start(self, sim_config):
        sim = SimSTTService(sim_config)
        assert sim.connected is False

    def test_connected_after_start(self, sim_config):
        sim = SimSTTService(sim_config, phrase_interval_s=999)
        sim.start()
        assert sim.connected is True
        sim.stop()

    def test_not_connected_after_stop(self, sim_config):
        sim = SimSTTService(sim_config, phrase_interval_s=999)
        sim.start()
        sim.stop()
        assert sim.connected is False

    def test_background_thread_emits_phrases(self, sim_config):
        """Integration: thread runs and calls back within a few seconds."""
        sim_config.simulation.stt_delay_ms = 10
        sim   = SimSTTService(sim_config, phrase_interval_s=0.3)
        fired = []
        sim.on_transcript(lambda t, f, c: fired.append(t) if f else None)
        sim.start()
        time.sleep(1.2)
        sim.stop()
        assert len(fired) >= 1


# ---------------------------------------------------------------------------
# SimVisionService
# ---------------------------------------------------------------------------

class TestSimVisionService:
    def test_describe_frame_returns_string(self, sim_config):
        vis    = SimVisionService(sim_config)
        result = vis.describe_frame(b"fake-image")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cycles_through_descriptions(self, sim_config):
        vis = SimVisionService(sim_config)
        results = [vis.describe_frame(b"img") for _ in range(10)]
        # Should cycle — not all identical
        assert len(set(results)) > 1

    def test_set_description_overrides_next(self, sim_config):
        vis = SimVisionService(sim_config)
        vis.set_description("Custom description for testing.")
        result = vis.describe_frame(b"img")
        assert result == "Custom description for testing."

    def test_empty_bytes_still_returns_string(self, sim_config):
        vis    = SimVisionService(sim_config)
        result = vis.describe_frame(b"")
        assert isinstance(result, str)

    def test_custom_prompt_accepted(self, sim_config):
        vis    = SimVisionService(sim_config)
        result = vis.describe_frame(b"img", prompt="What is happening?")
        assert isinstance(result, str)   # prompt doesn't affect sim output
