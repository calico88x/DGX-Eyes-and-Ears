"""
tests/test_loop.py

Integration tests for PipelineLoop in simulation mode.
No real services, no network, no audio hardware required.

These tests start the loop in a thread, inject events, and verify
the pipeline progresses through correct state transitions.

Run with:  pytest tests/test_loop.py -v
"""

import threading
import time
import pytest

from app.core.loop          import PipelineLoop
from app.core.state_machine import State, Event
from app.utils.config_loader import (
    Config, AppConfig, ServicesConfig, ModelsConfig,
    LiteLLMConfig, STTConfig, TTSConfig, VisionServiceConfig,
    MemoryConfig, VisionPipelineConfig, LoggingConfig,
    PersonalityConfig, SimulationConfig, CannedResponse,
)


# ---------------------------------------------------------------------------
# Config fixture — sim mode, in-memory DB, fast timings
# ---------------------------------------------------------------------------

def make_sim_config(tmp_path=None) -> Config:
    db_path = str(tmp_path / "test.db") if tmp_path else ":memory:"
    return Config(
        app=AppConfig(
            services=ServicesConfig(
                litellm = LiteLLMConfig(base_url="http://localhost:4000"),
                stt     = STTConfig(base_url="ws://localhost:9090"),
                tts     = TTSConfig(base_url="http://localhost:8880"),
                vision  = VisionServiceConfig(base_url="http://localhost:8000"),
            ),
            models=ModelsConfig(
                chat="test", classifier="test", vision="test", embedding="test"
            ),
            memory = MemoryConfig(
                db_path               = db_path,
                context_budget_tokens = 800,
            ),
            vision_pipeline = VisionPipelineConfig(
                frame_interval_ms          = 500,
                frame_interval_active_ms   = 200,
                frame_interval_speaking_ms = 1000,
            ),
            logging = LoggingConfig(level="WARNING", path="./logs/test.log"),
        ),
        params={
            "chat_llm":    {"temperature": 0.8, "max_tokens": 50},
            "classifier_thresholds": {
                "speech_respond_threshold":   0.60,
                "stream_react_threshold":     0.70,
                "stream_override_threshold":  0.85,
                "stream_interrupt_threshold": 0.90,
            },
        },
        personality=PersonalityConfig(
            name="Aria",
            system_prompt="You are Aria.",
        ),
        simulation=SimulationConfig(
            enabled         = True,
            stt_delay_ms    = 10,
            vision_delay_ms = 50,
            llm_delay_ms    = 50,
            canned_responses=[
                CannedResponse(trigger="hello", response="Hey there!"),
                CannedResponse(trigger="game",  response="Looks fun!"),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_state(loop: PipelineLoop, state: State, timeout_s: float = 5.0) -> bool:
    """Poll until the loop reaches the target state or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if loop.state_machine and loop.state_machine.state == state:
            return True
        time.sleep(0.05)
    return False


def wait_for_state_any(
    loop: PipelineLoop, states: set, timeout_s: float = 5.0
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if loop.state_machine and loop.state_machine.state in states:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Initialisation tests (no loop running)
# ---------------------------------------------------------------------------

class TestLoopInitialisation:
    def test_creates_without_error(self, tmp_path):
        config = make_sim_config(tmp_path)
        loop   = PipelineLoop(config)
        assert loop is not None

    def test_starts_not_running(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        assert loop._running is False

    def test_no_session_before_start(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        assert loop._session_id is None


# ---------------------------------------------------------------------------
# Setup methods (called without full start)
# ---------------------------------------------------------------------------

class TestLoopSetup:
    def test_setup_core_initialises_state_machine(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        loop._setup_core()
        assert loop.state_machine is not None
        assert loop.state_machine.state == State.IDLE

    def test_setup_services_sim_mode(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        loop._setup_core()
        loop._setup_services()
        assert loop.llm is not None
        assert loop.tts is not None
        assert loop.stt is not None
        assert loop.vision_svc is not None
        assert loop.classifier is not None

    def test_setup_memory_opens_db(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        loop._setup_memory()
        assert loop.db is not None
        assert loop.writer is not None
        assert loop.reader is not None
        loop.db.close()

    def test_vision_interval_by_state(self, tmp_path):
        loop = PipelineLoop(make_sim_config(tmp_path))
        loop._setup_core()

        loop.state_machine.force_state(State.IDLE)
        assert loop._get_vision_interval() == 500

        loop.state_machine.force_state(State.SPEAKING)
        assert loop._get_vision_interval() == 1000


# ---------------------------------------------------------------------------
# STT callback — unit tests (no loop running)
# ---------------------------------------------------------------------------

class TestSTTCallback:
    def setup_method(self):
        self.loop = PipelineLoop(make_sim_config())
        self.loop._setup_core()
        self.loop._setup_services()
        self.loop._setup_memory()
        self.loop._session_id = self.loop.writer.create_session()

    def teardown_method(self):
        if self.loop.db:
            self.loop.db.close()

    def test_transcript_updates_shared_state(self):
        self.loop._on_transcript("hello there", True, 0.9)
        assert self.loop._latest_transcript == "hello there"
        assert self.loop._transcript_confidence == 0.9

    def test_speech_detected_from_idle(self):
        assert self.loop.state_machine.state == State.IDLE
        self.loop._on_transcript("test", True, 0.9)
        assert self.loop.state_machine.state == State.LISTENING

    def test_speech_detected_from_watching(self):
        self.loop.state_machine.force_state(State.WATCHING)
        self.loop._on_transcript("test", True, 0.9)
        assert self.loop.state_machine.state == State.LISTENING

    def test_interrupt_from_speaking(self):
        self.loop.state_machine.force_state(State.SPEAKING)
        self.loop._on_transcript("wait", True, 0.9)
        assert self.loop.state_machine.state == State.USER_INTERRUPTED

    def test_interrupt_from_thinking(self):
        self.loop.state_machine.force_state(State.THINKING)
        self.loop._on_transcript("stop", True, 0.9)
        assert self.loop.state_machine.state == State.USER_INTERRUPTED

    def test_empty_transcript_ignored(self):
        self.loop._on_transcript("", True, 0.9)
        assert self.loop.state_machine.state == State.IDLE   # unchanged

    def test_whitespace_transcript_ignored(self):
        self.loop._on_transcript("   ", True, 0.9)
        assert self.loop._latest_transcript == ""


# ---------------------------------------------------------------------------
# Tick — unit tests (no loop running)
# ---------------------------------------------------------------------------

class TestTick:
    def setup_method(self):
        self.loop = PipelineLoop(make_sim_config())
        self.loop._setup_core()
        self.loop._setup_services()
        self.loop._setup_memory()
        self.loop._session_id = self.loop.writer.create_session()

    def teardown_method(self):
        if self.loop.db:
            self.loop.db.close()

    def test_tick_does_not_crash_on_idle(self):
        self.loop._tick()   # should not raise

    def test_tick_skipped_during_generation(self):
        self.loop._generation_active = True
        # Should return early without touching classifier
        self.loop._tick()   # no crash, no state change
        assert self.loop.state_machine.state == State.IDLE

    def test_tick_with_salient_speech_transitions_to_thinking(self):
        """Inject high-salience speech and verify pipeline starts."""
        # Override classifier to always return RESPOND with high salience
        from app.simulation.sim_classifier import SimClassifier, Scenario
        self.loop.classifier = SimClassifier(
            mode     = "scenario",
            scenario = Scenario.USER_SPEAKS,
            simulated_latency_ms = 0,
        )
        # Set up as if user just spoke
        self.loop.state_machine.process_event(Event.SPEECH_DETECTED)
        self.loop._latest_transcript    = "what is on screen"
        self.loop._transcript_confidence = 0.9

        self.loop._tick()

        # Should have entered THINKING and started generation
        assert self.loop.state_machine.state in (
            State.THINKING, State.SPEAKING, State.IDLE
        )


# ---------------------------------------------------------------------------
# Generation pipeline — unit test (direct call)
# ---------------------------------------------------------------------------

class TestGenerationPipeline:
    def setup_method(self):
        self.loop = PipelineLoop(make_sim_config())
        self.loop._setup_core()
        self.loop._setup_services()
        self.loop._setup_memory()
        self.loop._session_id = self.loop.writer.create_session()
        self.loop.state_machine.force_state(State.THINKING)
        self.loop._generation_active = True

    def teardown_method(self):
        if self.loop.db:
            self.loop.db.close()

    def test_generation_pipeline_runs_without_error(self):
        self.loop._generation_pipeline("hello there", "desktop visible")
        # Should have ended cleanly
        assert self.loop._generation_active is False

    def test_generation_transitions_to_speaking_then_back(self):
        states_seen = []
        self.loop.state_machine.on_state_change(
            lambda f, t, e: states_seen.append(t)
        )
        self.loop._generation_pipeline("hello", "screen visible")
        assert State.SPEAKING in states_seen

    def test_generation_writes_episodes_to_memory(self):
        self.loop._generation_pipeline("tell me about this", "coding editor")

        episodes = self.loop.reader.get_recent_episodes(
            self.loop._session_id, limit=10
        )
        assert len(episodes) >= 1

    def test_generation_updates_context_builder_history(self):
        self.loop._generation_pipeline("hello", "")
        assert self.loop.context_builder.turn_count >= 1

    def test_generation_resets_active_flag_on_error(self):
        # Patch LLM to raise
        original = self.loop.llm.stream_chat
        def bad_stream(*a, **k):
            raise RuntimeError("LLM is down")
        self.loop.llm.stream_chat = bad_stream

        self.loop._generation_pipeline("test", "")
        assert self.loop._generation_active is False

        self.loop.llm.stream_chat = original

    def test_generation_resets_state_on_error(self):
        def bad_stream(*a, **k):
            raise RuntimeError("LLM is down")
        self.loop.llm.stream_chat = bad_stream

        self.loop._generation_pipeline("test", "")
        assert self.loop.state_machine.state == State.IDLE


# ---------------------------------------------------------------------------
# Full integration test — loop runs in background thread
# ---------------------------------------------------------------------------

class TestLoopIntegration:
    """
    Runs the full PipelineLoop in a background thread and verifies
    that canned phrases drive real state transitions.
    """

    def test_sim_loop_starts_and_stops(self, tmp_path):
        config = make_sim_config(tmp_path)
        loop   = PipelineLoop(config)
        errors = []

        def run():
            try:
                loop.start()
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        time.sleep(1.0)
        loop.stop()
        t.join(timeout=5)

        assert errors == [], f"Loop raised: {errors}"

    def test_sim_stt_drives_state_transitions(self, tmp_path):
        """SimSTT fires a phrase → classifier → state machine cycles through."""
        config = make_sim_config(tmp_path)
        loop   = PipelineLoop(config)
        states_seen = []

        def run():
            loop.start()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Wait for state machine to exist
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if loop.state_machine:
                loop.state_machine.on_state_change(
                    lambda f, to, e: states_seen.append(to)
                )
                break
            time.sleep(0.05)

        # Let it run for a few seconds — SimSTT fires after ~1s
        time.sleep(4.0)
        loop.stop()
        t.join(timeout=5)

        # Should have seen at least LISTENING at some point
        assert len(states_seen) > 0

    def test_inject_phrase_triggers_generation(self, tmp_path):
        """Manually inject a phrase and confirm generation runs."""
        config = make_sim_config(tmp_path)
        loop   = PipelineLoop(config)
        generated = []

        original_pipeline = loop._generation_pipeline

        def tracked_pipeline(transcript, vision, frame=None):
            generated.append(transcript)
            original_pipeline(transcript, vision)

        def run():
            # Patch after services are set up
            loop._setup_core()
            loop._setup_services()
            loop._setup_memory()
            loop._session_id = loop.writer.create_session()
            loop.stt.on_transcript(loop._on_transcript)
            loop.stt.start()
            loop._generation_pipeline = tracked_pipeline
            loop._running = True
            try:
                loop._main_loop()
            except Exception:
                pass

        t = threading.Thread(target=run, daemon=True)
        t.start()

        # Wait for STT to be ready
        time.sleep(0.5)

        # Inject a phrase directly
        loop.stt.inject_phrase("hello there friend", is_final=True, confidence=0.95)

        # Wait for generation to be triggered
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not generated:
            time.sleep(0.1)

        loop.stop()
        t.join(timeout=5)

        assert len(generated) >= 1
        assert "hello" in generated[0].lower()

    def test_memory_persists_across_turns(self, tmp_path):
        """After a turn, episodes should exist in the DB."""
        config = make_sim_config(tmp_path)
        loop   = PipelineLoop(config)

        loop._setup_core()
        loop._setup_services()
        loop._setup_memory()
        loop._session_id = loop.writer.create_session()

        # Run generation directly
        loop.state_machine.force_state(State.THINKING)
        loop._generation_active = True
        loop._generation_pipeline("what is on my screen", "a coding editor")

        # Check DB
        stats = loop.db.stats()
        assert stats["episodes"] >= 1

        loop.db.close()
