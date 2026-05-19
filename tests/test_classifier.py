"""
tests/test_classifier.py

Fully offline tests for Classifier and SimClassifier.
No LLM calls. Real Classifier._call_llm is patched with unittest.mock.

Run with:  pytest tests/test_classifier.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.classifier import (
    Classifier,
    ClassifierAction,
    ClassifierInput,
    ClassifierOutput,
    ClassifierThresholds,
)
from app.core.state_machine import Event, State
from app.simulation.sim_classifier import (
    ClassifierRule,
    DEFAULT_RULES,
    Scenario,
    SimClassifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    return {
        "services": {
            "litellm": {
                "base_url": "http://localhost:4000",
                "api_key":  "sk-test",
            }
        },
        "models": {
            "classifier": "Qwen/Qwen2.5-1.5B-Instruct"
        },
    }


@pytest.fixture
def minimal_params():
    return {
        "classifier_llm": {
            "temperature": 0.1,
            "max_tokens":  150,
            "top_p":       0.9,
        },
        "classifier_thresholds": {
            "speech_respond_threshold":   0.60,
            "stream_react_threshold":     0.70,
            "stream_override_threshold":  0.85,
            "stream_interrupt_threshold": 0.90,
        },
    }


@pytest.fixture
def classifier(minimal_config, minimal_params):
    return Classifier(minimal_config, minimal_params)


def make_llm_response(**kwargs) -> str:
    """Build a realistic JSON string as the LLM would return."""
    defaults = {
        "speech_salience": 0.0,
        "stream_salience": 0.0,
        "primary_source":  "none",
        "action":          "HOLD",
        "reason":          "test",
        "confidence":      0.8,
    }
    defaults.update(kwargs)
    return json.dumps(defaults)


def make_input(**kwargs) -> ClassifierInput:
    defaults = dict(
        current_state      = State.IDLE,
        speech_transcript  = "",
        speech_confidence  = 0.0,
        vision_description = "",
        audio_event_label  = "",
        active_since_ms    = 0,
    )
    defaults.update(kwargs)
    return ClassifierInput(**defaults)


# ---------------------------------------------------------------------------
# ClassifierInput
# ---------------------------------------------------------------------------

class TestClassifierInput:
    def test_default_fields(self):
        inp = ClassifierInput(current_state=State.IDLE)
        assert inp.speech_transcript   == ""
        assert inp.speech_confidence   == 0.0
        assert inp.vision_description  == ""
        assert inp.audio_event_label   == ""
        assert inp.active_since_ms     == 0

    def test_all_fields(self):
        inp = make_input(
            current_state      = State.LISTENING,
            speech_transcript  = "what is that",
            speech_confidence  = 0.92,
            vision_description = "racing game, car crash",
            audio_event_label  = "impact",
            active_since_ms    = 1500,
        )
        assert inp.current_state     == State.LISTENING
        assert inp.speech_transcript == "what is that"
        assert inp.speech_confidence == 0.92


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    def test_prompt_includes_state(self, classifier):
        inp = make_input(current_state=State.LISTENING)
        prompt = classifier._build_prompt(inp)
        assert "LISTENING" in prompt

    def test_prompt_includes_speech_when_present(self, classifier):
        inp = make_input(speech_transcript="hello there", speech_confidence=0.9)
        prompt = classifier._build_prompt(inp)
        assert "hello there" in prompt
        assert "0.90" in prompt

    def test_prompt_shows_none_for_absent_speech(self, classifier):
        inp = make_input()
        prompt = classifier._build_prompt(inp)
        assert "[none]" in prompt

    def test_prompt_includes_vision_when_present(self, classifier):
        inp = make_input(vision_description="user is coding in VSCode")
        prompt = classifier._build_prompt(inp)
        assert "coding in VSCode" in prompt

    def test_prompt_includes_audio_event(self, classifier):
        inp = make_input(audio_event_label="explosion")
        prompt = classifier._build_prompt(inp)
        assert "explosion" in prompt

    def test_prompt_includes_active_duration(self, classifier):
        inp = make_input(active_since_ms=2500)
        prompt = classifier._build_prompt(inp)
        assert "2500" in prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_parses_valid_json(self, classifier):
        raw = make_llm_response(action="RESPOND", speech_salience=0.85)
        out = classifier._parse_response(raw)
        assert out.action          == ClassifierAction.RESPOND
        assert out.speech_salience == 0.85

    def test_parses_all_fields(self, classifier):
        raw = make_llm_response(
            action="REACT", speech_salience=0.1, stream_salience=0.9,
            primary_source="vision", reason="exciting event", confidence=0.88
        )
        out = classifier._parse_response(raw)
        assert out.action          == ClassifierAction.REACT
        assert out.stream_salience == 0.9
        assert out.primary_source  == "vision"
        assert out.reason          == "exciting event"
        assert out.confidence      == 0.88

    def test_strips_markdown_fences(self, classifier):
        raw = "```json\n" + make_llm_response(action="RESPOND") + "\n```"
        out = classifier._parse_response(raw)
        assert out.action == ClassifierAction.RESPOND

    def test_strips_plain_fences(self, classifier):
        raw = "```\n" + make_llm_response(action="WATCH") + "\n```"
        out = classifier._parse_response(raw)
        assert out.action == ClassifierAction.WATCH

    def test_unknown_action_defaults_to_hold(self, classifier):
        raw = make_llm_response(action="BANANA")
        out = classifier._parse_response(raw)
        assert out.action == ClassifierAction.HOLD

    def test_missing_salience_defaults_to_zero(self, classifier):
        raw = json.dumps({"action": "HOLD", "reason": "x", "confidence": 0.5})
        out = classifier._parse_response(raw)
        assert out.speech_salience == 0.0
        assert out.stream_salience == 0.0

    def test_raises_on_malformed_json(self, classifier):
        import json
        with pytest.raises(json.JSONDecodeError):
            classifier._parse_response("not json at all")


# ---------------------------------------------------------------------------
# Classify — patching _call_llm
# ---------------------------------------------------------------------------

class TestClassify:
    def test_classify_returns_output(self, classifier):
        with patch.object(
            classifier, "_call_llm",
            return_value=make_llm_response(action="RESPOND", speech_salience=0.8)
        ):
            out = classifier.classify(make_input(current_state=State.LISTENING))
        assert isinstance(out, ClassifierOutput)
        assert out.action == ClassifierAction.RESPOND

    def test_classify_records_latency(self, classifier):
        with patch.object(classifier, "_call_llm", return_value=make_llm_response()):
            out = classifier.classify(make_input(current_state=State.IDLE))
        assert out.latency_ms >= 0

    def test_classify_records_raw_response(self, classifier):
        raw = make_llm_response(action="WATCH")
        with patch.object(classifier, "_call_llm", return_value=raw):
            out = classifier.classify(make_input(current_state=State.IDLE))
        assert out.raw_response == raw

    def test_classify_fallback_on_llm_failure(self, classifier):
        with patch.object(classifier, "_call_llm", side_effect=Exception("timeout")):
            out = classifier.classify(make_input(current_state=State.IDLE))
        assert out.action          == ClassifierAction.HOLD
        assert out.speech_salience == 0.0

    def test_classify_fallback_on_bad_json(self, classifier):
        with patch.object(classifier, "_call_llm", return_value="not json"):
            out = classifier.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.HOLD

    def test_classify_fallback_on_network_error(self, classifier):
        import requests
        with patch.object(
            classifier, "_call_llm",
            side_effect=requests.RequestException("connection refused")
        ):
            out = classifier.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.HOLD


# ---------------------------------------------------------------------------
# to_event — threshold gating
# ---------------------------------------------------------------------------

class TestToEvent:

    # --- RESPOND ---

    def test_respond_above_threshold_fires_speech_ended(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.RESPOND,
            speech_salience=0.85, stream_salience=0.0,
            primary_source="speech", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.LISTENING)
        assert event == Event.SPEECH_ENDED

    def test_respond_below_threshold_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.RESPOND,
            speech_salience=0.40, stream_salience=0.0,
            primary_source="speech", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.LISTENING)
        assert event is None

    def test_respond_at_exact_threshold_fires(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.RESPOND,
            speech_salience=0.60, stream_salience=0.0,
            primary_source="speech", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.LISTENING)
        assert event == Event.SPEECH_ENDED

    # --- REACT ---

    def test_react_above_threshold_fires_salience_high(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.REACT,
            speech_salience=0.0, stream_salience=0.80,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.WATCHING)
        assert event == Event.SALIENCE_HIGH

    def test_react_below_threshold_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.REACT,
            speech_salience=0.0, stream_salience=0.50,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.WATCHING)
        assert event is None

    # --- INTERRUPT from SPEAKING ---

    def test_interrupt_speaking_above_threshold_fires_stream_interrupt(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.INTERRUPT,
            speech_salience=0.0, stream_salience=0.95,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.SPEAKING)
        assert event == Event.STREAM_INTERRUPT

    def test_interrupt_speaking_below_threshold_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.INTERRUPT,
            speech_salience=0.0, stream_salience=0.80,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.SPEAKING)
        assert event is None

    # --- INTERRUPT from LISTENING ---

    def test_interrupt_listening_above_threshold_fires_salience_override(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.INTERRUPT,
            speech_salience=0.3, stream_salience=0.90,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.LISTENING)
        assert event == Event.SALIENCE_OVERRIDE

    def test_interrupt_listening_below_threshold_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.INTERRUPT,
            speech_salience=0.3, stream_salience=0.70,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.LISTENING)
        assert event is None

    # --- INTERRUPT from irrelevant state ---

    def test_interrupt_from_idle_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.INTERRUPT,
            speech_salience=0.0, stream_salience=0.99,
            primary_source="vision", reason="", confidence=0.9
        )
        event = classifier.to_event(out, State.IDLE)
        assert event is None

    # --- No-transition actions ---

    def test_listen_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.LISTEN,
            speech_salience=0.7, stream_salience=0.0,
            primary_source="speech", reason="", confidence=0.9
        )
        assert classifier.to_event(out, State.LISTENING) is None

    def test_watch_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.WATCH,
            speech_salience=0.0, stream_salience=0.4,
            primary_source="vision", reason="", confidence=0.9
        )
        assert classifier.to_event(out, State.WATCHING) is None

    def test_hold_returns_none(self, classifier):
        out = ClassifierOutput(
            action=ClassifierAction.HOLD,
            speech_salience=0.0, stream_salience=0.0,
            primary_source="none", reason="", confidence=0.0
        )
        assert classifier.to_event(out, State.IDLE) is None


# ---------------------------------------------------------------------------
# ClassifierThresholds
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_defaults(self):
        t = ClassifierThresholds({})
        assert t.speech_respond   == 0.60
        assert t.stream_react     == 0.70
        assert t.stream_override  == 0.85
        assert t.stream_interrupt == 0.90

    def test_custom_values(self):
        t = ClassifierThresholds({
            "speech_respond_threshold":   0.50,
            "stream_react_threshold":     0.65,
            "stream_override_threshold":  0.80,
            "stream_interrupt_threshold": 0.95,
        })
        assert t.speech_respond   == 0.50
        assert t.stream_react     == 0.65
        assert t.stream_override  == 0.80
        assert t.stream_interrupt == 0.95


# ---------------------------------------------------------------------------
# SimClassifier — scenario mode
# ---------------------------------------------------------------------------

class TestSimClassifierScenario:
    def test_returns_classifier_output(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.USER_SPEAKS)
        out = sim.classify(make_input(current_state=State.LISTENING))
        assert isinstance(out, ClassifierOutput)

    def test_user_speaks_scenario_returns_respond(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.USER_SPEAKS)
        out = sim.classify(make_input(current_state=State.LISTENING))
        assert out.action == ClassifierAction.RESPOND
        assert out.speech_salience > 0.6

    def test_exciting_stream_returns_react(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.EXCITING_STREAM_EVENT)
        out = sim.classify(make_input(current_state=State.WATCHING))
        assert out.action == ClassifierAction.REACT
        assert out.stream_salience > 0.7

    def test_silence_returns_hold(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.SILENCE)
        out = sim.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.HOLD

    def test_cycles_through_scenarios_when_none_set(self):
        sim = SimClassifier(mode="scenario", scenario=None, simulated_latency_ms=0)
        results = [
            sim.classify(make_input(current_state=State.IDLE)).action
            for _ in range(len(Scenario))
        ]
        # Should have visited multiple different actions
        assert len(set(results)) > 1

    def test_set_scenario_changes_output(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.SILENCE, simulated_latency_ms=0)
        out1 = sim.classify(make_input(current_state=State.IDLE))
        sim.set_scenario(Scenario.USER_SPEAKS)
        out2 = sim.classify(make_input(current_state=State.LISTENING))
        assert out1.action != out2.action

    def test_raw_response_contains_scenario_name(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.BORING_STREAM, simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.WATCHING))
        assert "boring_stream" in out.raw_response

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SimClassifier(mode="magic")


# ---------------------------------------------------------------------------
# SimClassifier — rule mode
# ---------------------------------------------------------------------------

class TestSimClassifierRules:
    def test_matches_speech_rule(self):
        sim = SimClassifier(mode="rule", simulated_latency_ms=0)
        inp = make_input(
            current_state     = State.LISTENING,
            speech_transcript = "what is on the screen right now",
            speech_confidence = 0.92,
        )
        out = sim.classify(inp)
        assert out.action == ClassifierAction.RESPOND

    def test_matches_vision_keyword_rule(self):
        sim = SimClassifier(mode="rule", simulated_latency_ms=0)
        inp = make_input(
            current_state      = State.WATCHING,
            vision_description = "there was a massive explosion on screen",
        )
        out = sim.classify(inp)
        assert out.action == ClassifierAction.REACT

    def test_matches_silence_rule(self):
        sim = SimClassifier(mode="rule", simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.HOLD

    def test_no_match_returns_hold(self):
        sim = SimClassifier(mode="rule", rules=[], simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.HOLD

    def test_custom_rule_takes_priority(self):
        custom_rule = ClassifierRule(
            name     = "always_watch",
            match_fn = lambda i: True,   # matches everything
            action   = ClassifierAction.WATCH,
            stream_salience = 0.3,
            primary_source  = "vision",
            reason   = "custom rule",
        )
        sim = SimClassifier(mode="rule", rules=[custom_rule], simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.IDLE))
        assert out.action == ClassifierAction.WATCH

    def test_rule_match_fn_exception_skips_rule(self):
        bad_rule = ClassifierRule(
            name     = "bad_rule",
            match_fn = lambda i: (_ for _ in ()).throw(ValueError("oops")),
            action   = ClassifierAction.RESPOND,
        )
        sim = SimClassifier(
            mode  = "rule",
            rules = [bad_rule] + DEFAULT_RULES,
            simulated_latency_ms = 0
        )
        # Should not crash — bad rule is skipped, falls through to DEFAULT_RULES
        out = sim.classify(make_input(current_state=State.IDLE))
        assert isinstance(out, ClassifierOutput)


# ---------------------------------------------------------------------------
# SimClassifier — to_event (same threshold logic as real Classifier)
# ---------------------------------------------------------------------------

class TestSimToEvent:
    def test_respond_above_threshold_fires(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.USER_SPEAKS, simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.LISTENING))
        event = sim.to_event(out, State.LISTENING)
        assert event == Event.SPEECH_ENDED

    def test_react_above_threshold_fires(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.EXCITING_STREAM_EVENT, simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.WATCHING))
        event = sim.to_event(out, State.WATCHING)
        assert event == Event.SALIENCE_HIGH

    def test_hold_never_fires(self):
        sim = SimClassifier(mode="scenario", scenario=Scenario.SILENCE, simulated_latency_ms=0)
        out = sim.classify(make_input(current_state=State.IDLE))
        event = sim.to_event(out, State.IDLE)
        assert event is None


# ---------------------------------------------------------------------------
# Integration: Classifier + StateMachine (fully offline)
# ---------------------------------------------------------------------------

class TestClassifierStateMachineIntegration:
    """
    Verify that SimClassifier output correctly drives StateMachine transitions.
    No services required.
    """

    def test_user_speaks_drives_full_listen_think_arc(self):
        from app.core.state_machine import StateMachine

        sm  = StateMachine()
        sim = SimClassifier(mode="scenario", scenario=Scenario.USER_SPEAKS, simulated_latency_ms=0)

        # User starts speaking
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING

        # Classifier scores the speech
        inp = make_input(
            current_state     = sm.state,
            speech_transcript = "hey what is that on screen",
            speech_confidence = 0.88,
        )
        out   = sim.classify(inp)
        event = sim.to_event(out, sm.state)

        assert event == Event.SPEECH_ENDED
        sm.process_event(event)
        assert sm.state == State.THINKING

    def test_exciting_stream_drives_watch_to_thinking(self):
        from app.core.state_machine import StateMachine

        sm  = StateMachine()
        sim = SimClassifier(mode="scenario", scenario=Scenario.EXCITING_STREAM_EVENT, simulated_latency_ms=0)

        sm.notify_stream_active(True)
        assert sm.state == State.WATCHING

        inp   = make_input(current_state=sm.state, vision_description="car crash on screen")
        out   = sim.classify(inp)
        event = sim.to_event(out, sm.state)

        assert event == Event.SALIENCE_HIGH
        sm.process_event(event)
        assert sm.state == State.THINKING

    def test_stream_overrides_active_speech(self):
        from app.core.state_machine import StateMachine

        sm  = StateMachine()
        sim = SimClassifier(
            mode="scenario",
            scenario=Scenario.STREAM_OVERRIDES_SPEECH,
            simulated_latency_ms=0
        )

        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING

        inp   = make_input(current_state=sm.state, vision_description="critical alert")
        out   = sim.classify(inp)
        event = sim.to_event(out, sm.state)

        assert event == Event.SALIENCE_OVERRIDE
        sm.process_event(event)
        assert sm.state == State.STREAM_INTERRUPTED

    def test_boring_stream_does_not_trigger_response(self):
        from app.core.state_machine import StateMachine

        sm  = StateMachine()
        sim = SimClassifier(mode="scenario", scenario=Scenario.BORING_STREAM, simulated_latency_ms=0)

        sm.notify_stream_active(True)
        inp   = make_input(current_state=sm.state, vision_description="desktop wallpaper")
        out   = sim.classify(inp)
        event = sim.to_event(out, sm.state)

        assert event is None
        assert sm.state == State.WATCHING  # unchanged
