"""
tests/test_state_machine.py

Comprehensive offline tests for the StateMachine.
No external services or models required.

Run with:  pytest tests/test_state_machine.py -v
"""

import time
import pytest
from app.core.state_machine import StateMachine, State, Event, StateTransition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sm(**kwargs) -> StateMachine:
    """Convenience factory — keeps tests concise."""
    return StateMachine(**kwargs)


def walk(sm: StateMachine, *events: Event) -> None:
    """Feed a sequence of events into the machine."""
    for e in events:
        sm.process_event(e)


def full_listen_and_speak(sm: StateMachine) -> None:
    """Drive the machine through a complete listen → speak arc."""
    walk(sm,
         Event.SPEECH_DETECTED,
         Event.SPEECH_ENDED,
         Event.FIRST_TOKEN,
         Event.SPEAK_COMPLETE)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_starts_in_idle(self):
        sm = make_sm()
        assert sm.state == State.IDLE

    def test_history_empty_on_init(self):
        sm = make_sm()
        assert sm.history == []

    def test_time_in_state_is_non_negative(self):
        sm = make_sm()
        assert sm.time_in_state >= 0

    def test_reasoning_disabled_by_default(self):
        sm = make_sm()
        assert sm.reasoning_enabled is False

    def test_reasoning_enabled_via_flag(self):
        sm = make_sm(reasoning_enabled=True)
        assert sm.reasoning_enabled is True

    def test_summary_returns_expected_keys(self):
        sm = make_sm()
        s = sm.summary()
        assert "state" in s
        assert "time_in_state_s" in s
        assert "stream_active" in s
        assert "transition_count" in s


# ---------------------------------------------------------------------------
# Core happy-path flows
# ---------------------------------------------------------------------------

class TestListenAndRespond:
    def test_idle_to_listening_on_speech(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING

    def test_listening_to_thinking_when_speech_ends(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        assert sm.state == State.THINKING

    def test_thinking_to_speaking_on_first_token(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED, Event.FIRST_TOKEN)
        assert sm.state == State.SPEAKING

    def test_speaking_to_idle_when_no_stream(self):
        sm = make_sm()
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.FIRST_TOKEN,
             Event.SPEAK_COMPLETE)
        assert sm.state == State.IDLE

    def test_speaking_to_watching_when_stream_active(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.FIRST_TOKEN,
             Event.SPEAK_COMPLETE)
        assert sm.state == State.WATCHING

    def test_process_event_returns_true_on_valid_transition(self):
        sm = make_sm()
        result = sm.process_event(Event.SPEECH_DETECTED)
        assert result is True

    def test_process_event_returns_false_on_ignored_event(self):
        sm = make_sm()
        result = sm.process_event(Event.SPEAK_COMPLETE)
        assert result is False


# ---------------------------------------------------------------------------
# Speech priority (USER SPEECH > STREAM > IDLE)
# ---------------------------------------------------------------------------

class TestSpeechPriority:
    def test_speech_wins_over_watching(self):
        """Fundamental priority rule: user speech always pulls from WATCHING."""
        sm = make_sm()
        sm.process_event(Event.STREAM_ACTIVE)
        assert sm.state == State.WATCHING
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING

    def test_stream_triggers_thinking_from_watching(self):
        sm = make_sm()
        walk(sm, Event.STREAM_ACTIVE, Event.SALIENCE_HIGH)
        assert sm.state == State.THINKING

    def test_stream_salience_overrides_speech_only_above_threshold(self):
        """SALIENCE_OVERRIDE is how the classifier signals stream > speech."""
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING
        sm.process_event(Event.SALIENCE_OVERRIDE)
        assert sm.state == State.STREAM_INTERRUPTED


# ---------------------------------------------------------------------------
# Interruption flows
# ---------------------------------------------------------------------------

class TestInterruptions:
    def test_user_interrupts_speaking(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED, Event.FIRST_TOKEN)
        sm.process_event(Event.USER_INTERRUPT)
        assert sm.state == State.USER_INTERRUPTED

    def test_user_interrupts_thinking(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        sm.process_event(Event.USER_INTERRUPT)
        assert sm.state == State.USER_INTERRUPTED

    def test_after_user_interrupt_speech_resumes_listening(self):
        sm = make_sm()
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.FIRST_TOKEN,
             Event.USER_INTERRUPT,
             Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING

    def test_after_user_interrupt_speech_end_resumes_thinking(self):
        """Short burst interrupt: user finishes quickly, AI re-thinks."""
        sm = make_sm()
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.FIRST_TOKEN,
             Event.USER_INTERRUPT,
             Event.SPEECH_ENDED)
        assert sm.state == State.THINKING

    def test_stream_interrupts_speaking(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED, Event.FIRST_TOKEN)
        sm.process_event(Event.STREAM_INTERRUPT)
        assert sm.state == State.INTERRUPT

    def test_ai_resumes_speaking_after_self_interrupt(self):
        sm = make_sm()
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.FIRST_TOKEN,
             Event.STREAM_INTERRUPT,
             Event.FIRST_TOKEN)
        assert sm.state == State.SPEAKING

    def test_stream_interrupted_recovers_to_watching(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        sm.process_event(Event.SALIENCE_OVERRIDE)
        assert sm.state == State.STREAM_INTERRUPTED
        sm.process_event(Event.STREAM_ACTIVE)
        assert sm.state == State.WATCHING

    def test_stream_interrupted_recovers_to_listening_if_speech(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        sm.process_event(Event.SALIENCE_OVERRIDE)
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING


# ---------------------------------------------------------------------------
# Stream presence
# ---------------------------------------------------------------------------

class TestStreamPresence:
    def test_stream_active_from_idle(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        assert sm.state == State.WATCHING

    def test_stream_inactive_from_watching(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        sm.notify_stream_active(False)
        assert sm.state == State.IDLE

    def test_duplicate_stream_active_no_double_transition(self):
        """notify_stream_active is idempotent — no spurious transitions."""
        sm = make_sm()
        transitions = []
        sm.on_state_change(lambda f, t, e: transitions.append((f, t)))
        sm.notify_stream_active(True)
        sm.notify_stream_active(True)   # second call — already active
        assert len(transitions) == 1

    def test_stream_flag_affects_speak_complete_destination(self):
        sm = make_sm()
        sm.notify_stream_active(True)

        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED, Event.FIRST_TOKEN)
        assert sm.state == State.SPEAKING

        sm.notify_stream_active(False)
        sm.process_event(Event.SPEAK_COMPLETE)
        assert sm.state == State.IDLE

    def test_stream_inactive_from_listening_goes_idle(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        sm.process_event(Event.SPEECH_DETECTED)
        sm.process_event(Event.STREAM_INACTIVE)
        assert sm.state == State.IDLE


# ---------------------------------------------------------------------------
# Reasoning mode (optional, flag-gated)
# ---------------------------------------------------------------------------

class TestReasoningMode:
    def test_reasoning_start_ignored_when_disabled(self):
        sm = make_sm(reasoning_enabled=False)
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        result = sm.process_event(Event.REASONING_START)
        assert result is False
        assert sm.state == State.THINKING  # stayed in THINKING

    def test_reasoning_start_accepted_when_enabled(self):
        sm = make_sm(reasoning_enabled=True)
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        sm.process_event(Event.REASONING_START)
        assert sm.state == State.REASONING

    def test_reasoning_complete_returns_to_thinking(self):
        sm = make_sm(reasoning_enabled=True)
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.REASONING_START,
             Event.REASONING_COMPLETE)
        assert sm.state == State.THINKING

    def test_user_interrupt_during_reasoning(self):
        sm = make_sm(reasoning_enabled=True)
        walk(sm,
             Event.SPEECH_DETECTED,
             Event.SPEECH_ENDED,
             Event.REASONING_START)
        sm.process_event(Event.USER_INTERRUPT)
        assert sm.state == State.USER_INTERRUPTED


# ---------------------------------------------------------------------------
# Ignored / invalid events
# ---------------------------------------------------------------------------

class TestIgnoredEvents:
    def test_speak_complete_ignored_in_idle(self):
        sm = make_sm()
        result = sm.process_event(Event.SPEAK_COMPLETE)
        assert result is False
        assert sm.state == State.IDLE

    def test_generation_complete_ignored_in_watching(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        result = sm.process_event(Event.GENERATION_COMPLETE)
        assert result is False
        assert sm.state == State.WATCHING

    def test_stream_interrupt_ignored_when_not_speaking(self):
        sm = make_sm()
        result = sm.process_event(Event.STREAM_INTERRUPT)
        assert result is False

    def test_salience_high_ignored_in_listening(self):
        """SALIENCE_HIGH only triggers from WATCHING; LISTENING needs SALIENCE_OVERRIDE."""
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        result = sm.process_event(Event.SALIENCE_HIGH)
        assert result is False
        assert sm.state == State.LISTENING

    def test_state_unchanged_on_ignored_event(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        sm.process_event(Event.FIRST_TOKEN)  # invalid from WATCHING
        assert sm.state == State.WATCHING


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_fires_on_transition(self):
        sm = make_sm()
        fired = []
        sm.on_state_change(lambda f, t, e: fired.append((f, t, e)))
        sm.process_event(Event.SPEECH_DETECTED)
        assert len(fired) == 1
        assert fired[0] == (State.IDLE, State.LISTENING, Event.SPEECH_DETECTED)

    def test_callback_not_fired_on_ignored_event(self):
        sm = make_sm()
        fired = []
        sm.on_state_change(lambda f, t, e: fired.append((f, t, e)))
        sm.process_event(Event.SPEAK_COMPLETE)
        assert fired == []

    def test_multiple_callbacks_all_fire(self):
        sm = make_sm()
        log_a, log_b = [], []
        sm.on_state_change(lambda f, t, e: log_a.append(t))
        sm.on_state_change(lambda f, t, e: log_b.append(t))
        sm.process_event(Event.SPEECH_DETECTED)
        assert log_a == [State.LISTENING]
        assert log_b == [State.LISTENING]

    def test_callback_error_does_not_crash_machine(self):
        """A bad callback should not break the state machine."""
        sm = make_sm()
        sm.on_state_change(lambda f, t, e: (_ for _ in ()).throw(RuntimeError("boom")))
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.state == State.LISTENING  # still transitioned

    def test_callback_receives_correct_states(self):
        sm = make_sm()
        received = []
        sm.on_state_change(lambda f, t, e: received.append((f.name, t.name, e.name)))
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        assert received[0] == ("IDLE", "LISTENING", "SPEECH_DETECTED")
        assert received[1] == ("LISTENING", "THINKING", "SPEECH_ENDED")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_records_each_transition(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        assert len(sm.history) == 2

    def test_history_entries_are_correct_type(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        assert isinstance(sm.history[0], StateTransition)

    def test_history_records_correct_states_and_event(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        h = sm.history[0]
        assert h.from_state == State.IDLE
        assert h.to_state == State.LISTENING
        assert h.event == Event.SPEECH_DETECTED

    def test_history_has_timestamp(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        assert sm.history[0].timestamp > 0

    def test_history_returns_copy(self):
        """Mutating the returned list should not affect internal history."""
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        h = sm.history
        h.clear()
        assert len(sm.history) == 1

    def test_ignored_events_not_in_history(self):
        sm = make_sm()
        sm.process_event(Event.SPEAK_COMPLETE)  # ignored
        assert len(sm.history) == 0


# ---------------------------------------------------------------------------
# Reset and force_state
# ---------------------------------------------------------------------------

class TestResetAndForce:
    def test_reset_returns_to_idle(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED, Event.FIRST_TOKEN)
        sm.reset()
        assert sm.state == State.IDLE

    def test_history_preserved_after_reset(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED)
        count_before = len(sm.history)
        sm.reset()
        assert len(sm.history) > count_before

    def test_force_state_sets_state(self):
        sm = make_sm()
        sm.force_state(State.SPEAKING, reason="test")
        assert sm.state == State.SPEAKING

    def test_force_state_fires_callbacks(self):
        sm = make_sm()
        fired = []
        sm.on_state_change(lambda f, t, e: fired.append(t))
        sm.force_state(State.SPEAKING, reason="test")
        assert State.SPEAKING in fired

    def test_machine_usable_after_reset(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        sm.reset()
        result = sm.process_event(Event.SPEECH_DETECTED)
        assert result is True
        assert sm.state == State.LISTENING


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_reflects_current_state(self):
        sm = make_sm()
        sm.process_event(Event.SPEECH_DETECTED)
        s = sm.summary()
        assert s["state"] == "LISTENING"

    def test_summary_transition_count(self):
        sm = make_sm()
        walk(sm, Event.SPEECH_DETECTED, Event.SPEECH_ENDED)
        assert sm.summary()["transition_count"] == 2

    def test_summary_stream_active_reflects_reality(self):
        sm = make_sm()
        sm.notify_stream_active(True)
        assert sm.summary()["stream_active"] is True


# ---------------------------------------------------------------------------
# Thread safety (basic smoke test)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_events_do_not_crash(self):
        """
        Fire events from multiple threads. We don't assert a specific final
        state (threading order is non-deterministic) but the machine must
        not raise or corrupt internal state.
        """
        import threading
        sm = make_sm()
        errors = []

        def spam(event):
            for _ in range(50):
                try:
                    sm.process_event(event)
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=spam, args=(Event.SPEECH_DETECTED,)),
            threading.Thread(target=spam, args=(Event.SPEECH_ENDED,)),
            threading.Thread(target=spam, args=(Event.STREAM_ACTIVE,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        # State must be a valid State enum member
        assert isinstance(sm.state, State)
