"""
app/core/state_machine.py

Central state controller for the Eyes and Ears pipeline.

Controls when the AI listens, watches, thinks, speaks, and interrupts.
All pipeline components subscribe to state changes via callbacks and
drive transitions by feeding events into process_event().

Priority (hardcoded):
    USER SPEECH > STREAM SALIENCE > IDLE

Thread-safe. No external dependencies.
"""

import threading
import time
import logging
from enum import Enum, auto
from typing import Callable, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & Event Enums
# ---------------------------------------------------------------------------

class State(Enum):
    """All possible states of the pipeline loop."""
    IDLE             = auto()   # No inputs. Loop is running but quiet.
    LISTENING        = auto()   # User is speaking. Mic input active.
    WATCHING         = auto()   # Stream share is active. No speech detected.
    THINKING         = auto()   # LLM is generating but no tokens output yet.
    REASONING        = auto()   # Optional: reasoning model pass before response.
    SPEAKING         = auto()   # TTS is outputting audio.
    USER_INTERRUPTED = auto()   # User spoke while AI was thinking or speaking.
    STREAM_INTERRUPTED = auto() # Stream salience overrode active user speech.
    INTERRUPT        = auto()   # AI interrupts itself to react to stream event.


class Event(Enum):
    """Events that drive state transitions. Emitted by pipeline services."""
    SPEECH_DETECTED    = auto()  # VAD: voice activity started
    SPEECH_ENDED       = auto()  # VAD: voice activity stopped (user done speaking)
    STREAM_ACTIVE      = auto()  # Screen share has content
    STREAM_INACTIVE    = auto()  # Screen share went quiet or disconnected
    SALIENCE_HIGH      = auto()  # Classifier: stream salience crossed LLM trigger threshold
    SALIENCE_OVERRIDE  = auto()  # Classifier: stream salience exceeded active speech salience
    USER_INTERRUPT     = auto()  # User spoke while AI was in THINKING or SPEAKING
    STREAM_INTERRUPT   = auto()  # Stream salience exceeded interrupt threshold during SPEAKING
    FIRST_TOKEN        = auto()  # LLM emitted its first output token
    GENERATION_COMPLETE = auto() # LLM finished generating (no TTS yet)
    SPEAK_COMPLETE     = auto()  # TTS finished speaking
    REASONING_START    = auto()  # Reasoning pass started (flag-gated)
    REASONING_COMPLETE = auto()  # Reasoning pass finished; proceed to generation


# ---------------------------------------------------------------------------
# Transition Table
# ---------------------------------------------------------------------------

# (current_state, event) -> next_state
# Events not in this table are silently ignored for that state.
TRANSITIONS: dict[tuple[State, Event], State] = {

    # --- IDLE ---
    (State.IDLE, Event.SPEECH_DETECTED):         State.LISTENING,
    (State.IDLE, Event.STREAM_ACTIVE):           State.WATCHING,

    # --- LISTENING ---
    # Speech ends naturally → send context to LLM
    (State.LISTENING, Event.SPEECH_ENDED):       State.THINKING,
    # Stream becomes more salient than the user's speech → defer to stream
    (State.LISTENING, Event.SALIENCE_OVERRIDE):  State.STREAM_INTERRUPTED,
    # Stream dropped while listening (rare edge case)
    (State.LISTENING, Event.STREAM_INACTIVE):    State.IDLE,

    # --- WATCHING ---
    # Speech always wins over watching
    (State.WATCHING, Event.SPEECH_DETECTED):     State.LISTENING,
    # Stream content is salient enough to trigger a response
    (State.WATCHING, Event.SALIENCE_HIGH):       State.THINKING,
    # Stream ended
    (State.WATCHING, Event.STREAM_INACTIVE):     State.IDLE,

    # --- THINKING ---
    # LLM starts producing tokens
    (State.THINKING, Event.FIRST_TOKEN):         State.SPEAKING,
    # User spoke while we were generating → stop and listen
    (State.THINKING, Event.USER_INTERRUPT):      State.USER_INTERRUPTED,
    # Optional reasoning pass (flag-gated in _resolve_transition)
    (State.THINKING, Event.REASONING_START):     State.REASONING,

    # --- REASONING (optional) ---
    # Reasoning complete → back to generating
    (State.REASONING, Event.REASONING_COMPLETE): State.THINKING,
    # User interrupted during reasoning
    (State.REASONING, Event.USER_INTERRUPT):     State.USER_INTERRUPTED,

    # --- SPEAKING ---
    # User spoke while AI was speaking → stop and listen
    (State.SPEAKING, Event.USER_INTERRUPT):      State.USER_INTERRUPTED,
    # Stream event so salient the AI should react mid-sentence
    (State.SPEAKING, Event.STREAM_INTERRUPT):    State.INTERRUPT,
    # SPEAK_COMPLETE destination is dynamic (see _resolve_transition)
    # → WATCHING if stream active, IDLE if not

    # --- INTERRUPT ---
    # AI interrupted itself; resumes speaking after re-generating
    (State.INTERRUPT, Event.FIRST_TOKEN):        State.SPEAKING,

    # --- USER_INTERRUPTED ---
    # User is still / continues talking
    (State.USER_INTERRUPTED, Event.SPEECH_DETECTED): State.LISTENING,
    # User stopped (short burst); return to pre-interrupt context
    (State.USER_INTERRUPTED, Event.SPEECH_ENDED):    State.THINKING,

    # --- STREAM_INTERRUPTED ---
    # Stream still active; switch to watching mode
    (State.STREAM_INTERRUPTED, Event.STREAM_ACTIVE):    State.WATCHING,
    # User starts speaking again after stream override
    (State.STREAM_INTERRUPTED, Event.SPEECH_DETECTED):  State.LISTENING,
    # Stream also dropped
    (State.STREAM_INTERRUPTED, Event.STREAM_INACTIVE):  State.IDLE,
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StateTransition:
    """Record of a single state change."""
    from_state: State
    to_state:   State
    event:      Event
    timestamp:  float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"[{self.timestamp:.3f}] "
            f"{self.from_state.name} → {self.to_state.name} "
            f"via {self.event.name}"
        )


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class StateMachine:
    """
    Central state controller for the Eyes and Ears pipeline.

    Usage
    -----
    sm = StateMachine()
    sm.on_state_change(my_callback)     # subscribe to transitions
    sm.process_event(Event.SPEECH_DETECTED)  # drive transitions

    Callbacks receive: (from_state: State, to_state: State, event: Event)
    Callbacks should be non-blocking. Use queues for heavy work.

    Thread Safety
    -------------
    All public methods are thread-safe. The internal lock is not reentrant,
    so callbacks must not call process_event() synchronously.
    """

    def __init__(self, reasoning_enabled: bool = False):
        """
        Parameters
        ----------
        reasoning_enabled : bool
            If True, REASONING_START events are processed and the machine
            can enter the REASONING state. Default False.
        """
        self._state: State = State.IDLE
        self._lock = threading.Lock()
        self._reasoning_enabled = reasoning_enabled
        self._state_entered_at: float = time.time()
        self._history: list[StateTransition] = []
        self._callbacks: list[Callable[[State, State, Event], None]] = []
        self._stream_active: bool = False

        logger.info("StateMachine initialized | state=IDLE | reasoning=%s", reasoning_enabled)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Current state. Thread-safe read."""
        with self._lock:
            return self._state

    @property
    def time_in_state(self) -> float:
        """Seconds elapsed since entering the current state."""
        return time.time() - self._state_entered_at

    @property
    def history(self) -> list[StateTransition]:
        """Full transition history. Returns a copy."""
        with self._lock:
            return list(self._history)

    @property
    def reasoning_enabled(self) -> bool:
        return self._reasoning_enabled

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def on_state_change(
        self,
        callback: Callable[[State, State, Event], None]
    ) -> None:
        """
        Register a callback invoked on every state transition.

        Signature: callback(from_state, to_state, event)
        Callbacks are called synchronously under the lock — keep them fast.
        Use asyncio queues or threading.Queue for heavier downstream work.
        """
        self._callbacks.append(callback)
        logger.debug("Callback registered: %s", callback.__name__)

    def process_event(self, event: Event) -> bool:
        """
        Feed an event into the machine. Transitions if valid for current state.

        Returns
        -------
        bool
            True if a transition occurred. False if event was ignored.
        """
        with self._lock:
            next_state = self._resolve_transition(self._state, event)

            if next_state is None:
                logger.debug(
                    "Event ignored | state=%s event=%s",
                    self._state.name, event.name
                )
                return False

            self._do_transition(event, next_state)
            return True

    def notify_stream_active(self, active: bool) -> None:
        """
        Notify the machine of screen share presence changes.
        Automatically emits STREAM_ACTIVE or STREAM_INACTIVE as needed.

        Call this from the vision service whenever capture starts/stops.
        """
        with self._lock:
            changed = self._stream_active != active
            self._stream_active = active

        if changed:
            event = Event.STREAM_ACTIVE if active else Event.STREAM_INACTIVE
            logger.debug("Stream presence changed | active=%s", active)
            self.process_event(event)

    def reset(self) -> None:
        """
        Force machine to IDLE. Use for pipeline restart or error recovery.
        Preserves history.
        """
        logger.warning("StateMachine reset to IDLE")
        with self._lock:
            self._force_state_unsafe(State.IDLE, "reset")

    def force_state(self, state: State, reason: str = "manual") -> None:
        """
        Force machine into any state directly. Avoid in normal operation.
        Intended for testing and error recovery only.
        """
        with self._lock:
            self._force_state_unsafe(state, reason)

    def summary(self) -> dict:
        """Return a snapshot of current machine status for logging/UI."""
        with self._lock:
            return {
                "state":            self._state.name,
                "time_in_state_s":  round(time.time() - self._state_entered_at, 3),
                "stream_active":    self._stream_active,
                "reasoning_enabled": self._reasoning_enabled,
                "transition_count": len(self._history),
            }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _resolve_transition(self, state: State, event: Event) -> Optional[State]:
        """
        Determine the next state for (state, event).
        Applies feature flags and dynamic destination logic.
        Must be called under self._lock.
        """
        # Reasoning is opt-in — gate its events behind the flag
        if event == Event.REASONING_START and not self._reasoning_enabled:
            logger.debug("REASONING_START ignored: reasoning_enabled=False")
            return None

        # Look up in transition table
        next_state = TRANSITIONS.get((state, event))

        # Dynamic destination: SPEAK_COMPLETE goes to WATCHING or IDLE
        # depending on whether the stream is currently active
        if state == State.SPEAKING and event == Event.SPEAK_COMPLETE:
            next_state = State.WATCHING if self._stream_active else State.IDLE

        return next_state

    def _do_transition(self, event: Event, next_state: State) -> None:
        """
        Execute a validated transition and fire callbacks.
        Must be called under self._lock.
        """
        prev_state = self._state
        duration   = round(time.time() - self._state_entered_at, 3)

        self._state = next_state
        self._state_entered_at = time.time()

        transition = StateTransition(
            from_state=prev_state,
            to_state=next_state,
            event=event,
        )
        self._history.append(transition)

        logger.info(
            "Transition | %s → %s | event=%s | was_in_state_for=%.3fs",
            prev_state.name, next_state.name, event.name, duration
        )

        for callback in self._callbacks:
            try:
                callback(prev_state, next_state, event)
            except Exception as exc:
                logger.error(
                    "Callback error during transition %s→%s: %s",
                    prev_state.name, next_state.name, exc
                )

    def _force_state_unsafe(self, state: State, reason: str) -> None:
        """Force state without acquiring lock. Caller must hold self._lock."""
        prev = self._state
        self._state = state
        self._state_entered_at = time.time()
        self._history.append(
            StateTransition(from_state=prev, to_state=state, event=Event.SPEECH_DETECTED)
        )
        logger.warning(
            "Force state | %s → %s | reason=%s",
            prev.name, state.name, reason
        )
        for callback in self._callbacks:
            try:
                callback(prev, state, Event.SPEECH_DETECTED)
            except Exception as exc:
                logger.error("Callback error during force_state: %s", exc)
