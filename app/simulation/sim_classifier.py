"""
app/simulation/sim_classifier.py

Offline drop-in replacement for Classifier.
Identical classify() and to_event() interface — no LLM calls.

Two modes:

1. SCENARIO — named, pre-built test situations.
   sim = SimClassifier(mode="scenario", scenario="user_speaks")

2. RULE — pattern-matching on input fields.
   sim = SimClassifier(mode="rule", rules=[...])

Both produce realistic ClassifierOutput and ClassifierInput objects
so the rest of the pipeline runs exactly as it would with live services.

Usage in simulation loop
------------------------
from app.simulation.sim_classifier import SimClassifier, Scenario

sim = SimClassifier(mode="scenario", scenario=Scenario.EXCITING_STREAM_EVENT)
output = sim.classify(inp)
event  = sim.to_event(output, current_state)
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

from app.core.classifier import (
    Classifier,
    ClassifierAction,
    ClassifierInput,
    ClassifierOutput,
    ClassifierThresholds,
)
from app.core.state_machine import Event, State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

class Scenario(str, Enum):
    """
    Named test situations covering the main classifier decision paths.
    Use these to drive the full pipeline in simulation without any services.
    """
    USER_SPEAKS            = "user_speaks"           # Normal speech, AI should respond
    USER_SPEAKS_LOW_SAL    = "user_speaks_low_sal"   # Short/quiet speech, AI holds
    EXCITING_STREAM_EVENT  = "exciting_stream"       # Highly salient screen content
    BORING_STREAM          = "boring_stream"         # Screen active but low salience
    STREAM_OVERRIDES_SPEECH = "stream_overrides"     # Stream beats active speech
    STREAM_INTERRUPTS_AI   = "stream_interrupts_ai"  # Stream interrupts AI speaking
    SILENCE                = "silence"               # Nothing happening
    HOLD_AMBIGUOUS         = "hold_ambiguous"        # Classifier uncertain

# Maps Scenario → (ClassifierAction, speech_salience, stream_salience, source, reason)
_SCENARIO_TABLE: dict[Scenario, tuple] = {
    Scenario.USER_SPEAKS: (
        ClassifierAction.RESPOND, 0.85, 0.10, "speech",
        "User asked a clear question."
    ),
    Scenario.USER_SPEAKS_LOW_SAL: (
        ClassifierAction.HOLD, 0.35, 0.05, "speech",
        "Speech detected but content not salient enough to respond."
    ),
    Scenario.EXCITING_STREAM_EVENT: (
        ClassifierAction.REACT, 0.05, 0.82, "vision",
        "Exciting event visible on screen — car crash in game."
    ),
    Scenario.BORING_STREAM: (
        ClassifierAction.WATCH, 0.00, 0.30, "vision",
        "Screen active but content is routine."
    ),
    Scenario.STREAM_OVERRIDES_SPEECH: (
        ClassifierAction.INTERRUPT, 0.40, 0.88, "vision",
        "Stream event more salient than user speech."
    ),
    Scenario.STREAM_INTERRUPTS_AI: (
        ClassifierAction.INTERRUPT, 0.05, 0.93, "vision",
        "Critical event on screen while AI is speaking."
    ),
    Scenario.SILENCE: (
        ClassifierAction.HOLD, 0.00, 0.00, "none",
        "No signal detected."
    ),
    Scenario.HOLD_AMBIGUOUS: (
        ClassifierAction.HOLD, 0.45, 0.45, "none",
        "Ambiguous input — holding for more context."
    ),
}


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class ClassifierRule:
    """
    A pattern-matching rule for SimClassifier in RULE mode.

    match_fn receives a ClassifierInput and returns True if this rule applies.
    Rules are evaluated in order; first match wins.
    """
    name:     str
    match_fn: Callable[[ClassifierInput], bool]
    action:   ClassifierAction
    speech_salience: float = 0.0
    stream_salience: float = 0.0
    primary_source:  str   = "none"
    reason:          str   = ""
    confidence:      float = 0.8


# ---------------------------------------------------------------------------
# Default rules (reasonable baseline for full-pipeline sim)
# ---------------------------------------------------------------------------

DEFAULT_RULES: list[ClassifierRule] = [
    ClassifierRule(
        name    = "user_speaking_salient",
        match_fn = lambda i: len(i.speech_transcript) > 10 and i.speech_confidence > 0.7,
        action  = ClassifierAction.RESPOND,
        speech_salience = 0.85, stream_salience = 0.05,
        primary_source  = "speech",
        reason  = "User speech detected with high confidence.",
        confidence = 0.9,
    ),
    ClassifierRule(
        name    = "user_speaking_low_confidence",
        match_fn = lambda i: len(i.speech_transcript) > 0 and i.speech_confidence <= 0.7,
        action  = ClassifierAction.LISTEN,
        speech_salience = 0.50, stream_salience = 0.0,
        primary_source  = "speech",
        reason  = "Speech detected but confidence low — still listening.",
        confidence = 0.7,
    ),
    ClassifierRule(
        name    = "exciting_vision_keyword",
        match_fn = lambda i: any(
            re.search(r"\b" + kw + r"\b", i.vision_description.lower())
            for kw in ["explosion", "crash", "fire", "goal", "error", "warning", "alert"]
        ),
        action  = ClassifierAction.REACT,
        speech_salience = 0.0, stream_salience = 0.85,
        primary_source  = "vision",
        reason  = "High-salience keyword detected in screen description.",
        confidence = 0.85,
    ),
    ClassifierRule(
        name    = "vision_active_boring",
        match_fn = lambda i: bool(i.vision_description) and not i.speech_transcript,
        action  = ClassifierAction.WATCH,
        speech_salience = 0.0, stream_salience = 0.25,
        primary_source  = "vision",
        reason  = "Screen content active but not salient.",
        confidence = 0.7,
    ),
    ClassifierRule(
        name    = "silence",
        match_fn = lambda i: (
            not i.speech_transcript
            and not i.vision_description
            and not i.audio_event_label
        ),
        action  = ClassifierAction.HOLD,
        speech_salience = 0.0, stream_salience = 0.0,
        primary_source  = "none",
        reason  = "No signal.",
        confidence = 1.0,
    ),
]


# ---------------------------------------------------------------------------
# SimClassifier
# ---------------------------------------------------------------------------

class SimClassifier:
    """
    Offline classifier for simulation and testing.

    Identical interface to Classifier. No LLM calls.
    Adds artificial latency to simulate realistic timing.

    Parameters
    ----------
    mode : str
        "scenario" — use a named Scenario (cycles through if scenario=None).
        "rule"     — pattern-match on input fields using ClassifierRule list.

    scenario : Scenario | None
        Used in scenario mode. If None, cycles through all scenarios in order.

    rules : list[ClassifierRule] | None
        Used in rule mode. Defaults to DEFAULT_RULES if None.

    simulated_latency_ms : float
        Added sleep to simulate real LLM latency. Default 80ms.

    thresholds_cfg : dict
        Optional threshold overrides (same keys as params.yaml
        classifier_thresholds section).
    """

    def __init__(
        self,
        mode: str = "rule",
        scenario: Optional[Scenario] = None,
        rules: Optional[list[ClassifierRule]] = None,
        simulated_latency_ms: float = 80.0,
        thresholds_cfg: Optional[dict] = None,
    ):
        if mode not in ("scenario", "rule"):
            raise ValueError(f"mode must be 'scenario' or 'rule', got {mode!r}")

        self._mode              = mode
        self._scenario          = scenario
        self._rules             = rules if rules is not None else DEFAULT_RULES
        self._latency_s         = simulated_latency_ms / 1000.0
        self._thresholds        = ClassifierThresholds(thresholds_cfg or {})
        self._scenario_cycle    = list(Scenario)
        self._cycle_index       = 0

        logger.info(
            "SimClassifier initialized | mode=%s scenario=%s rules=%d latency=%.0fms",
            mode,
            scenario.value if scenario else "cycle",
            len(self._rules),
            simulated_latency_ms,
        )

    # -----------------------------------------------------------------------
    # Public — identical interface to Classifier
    # -----------------------------------------------------------------------

    def classify(self, inp: ClassifierInput) -> ClassifierOutput:
        """
        Return a simulated ClassifierOutput. No LLM call.
        Applies artificial latency to simulate realistic timing.
        """
        t0 = time.monotonic()

        if self._mode == "scenario":
            output = self._classify_scenario(inp)
        else:
            output = self._classify_rules(inp)

        # Simulate LLM latency
        elapsed = time.monotonic() - t0
        remaining = self._latency_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

        output.latency_ms = round((time.monotonic() - t0) * 1000, 1)

        logger.debug(
            "SimClassifier | action=%-10s speech_sal=%.2f stream_sal=%.2f latency=%.1fms",
            output.action.value,
            output.speech_salience,
            output.stream_salience,
            output.latency_ms,
        )

        return output

    def to_event(
        self,
        output: ClassifierOutput,
        current_state: State,
    ) -> Optional[Event]:
        """
        Identical threshold-gated mapping as the real Classifier.
        Reuses the same logic so simulation thresholds are consistent.
        """
        action = output.action
        t      = self._thresholds

        if action == ClassifierAction.RESPOND:
            return Event.SPEECH_ENDED if output.speech_salience >= t.speech_respond else None

        if action == ClassifierAction.REACT:
            return Event.SALIENCE_HIGH if output.stream_salience >= t.stream_react else None

        if action == ClassifierAction.INTERRUPT:
            if current_state == State.SPEAKING:
                return (
                    Event.STREAM_INTERRUPT
                    if output.stream_salience >= t.stream_interrupt
                    else None
                )
            if current_state == State.LISTENING:
                return (
                    Event.SALIENCE_OVERRIDE
                    if output.stream_salience >= t.stream_override
                    else None
                )
            return None

        return None   # LISTEN / WATCH / HOLD

    # -----------------------------------------------------------------------
    # Scenario helpers
    # -----------------------------------------------------------------------

    def set_scenario(self, scenario: Scenario) -> None:
        """Switch to a different scenario mid-run."""
        self._scenario = scenario
        logger.debug("SimClassifier scenario changed to %s", scenario.value)

    def next_scenario(self) -> Scenario:
        """Advance the cycle and return the scenario that will be used next."""
        s = self._scenario_cycle[self._cycle_index % len(self._scenario_cycle)]
        self._cycle_index += 1
        return s

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _classify_scenario(self, inp: ClassifierInput) -> ClassifierOutput:
        scenario = self._scenario or self.next_scenario()
        action, speech_sal, stream_sal, source, reason = _SCENARIO_TABLE[scenario]
        return ClassifierOutput(
            action          = action,
            speech_salience = speech_sal,
            stream_salience = stream_sal,
            primary_source  = source,
            reason          = f"[sim:{scenario.value}] {reason}",
            confidence      = 0.9,
            raw_response    = f"<simulated scenario={scenario.value}>",
        )

    def _classify_rules(self, inp: ClassifierInput) -> ClassifierOutput:
        for rule in self._rules:
            try:
                if rule.match_fn(inp):
                    logger.debug("SimClassifier matched rule: %s", rule.name)
                    return ClassifierOutput(
                        action          = rule.action,
                        speech_salience = rule.speech_salience,
                        stream_salience = rule.stream_salience,
                        primary_source  = rule.primary_source,
                        reason          = f"[sim:{rule.name}] {rule.reason}",
                        confidence      = rule.confidence,
                        raw_response    = f"<simulated rule={rule.name}>",
                    )
            except Exception as exc:
                logger.warning("Rule %r match_fn raised: %s", rule.name, exc)

        # No rule matched — safe fallback
        return ClassifierOutput(
            action          = ClassifierAction.HOLD,
            speech_salience = 0.0,
            stream_salience = 0.0,
            primary_source  = "none",
            reason          = "[sim:no_match] No rule matched input.",
            confidence      = 0.5,
            raw_response    = "<simulated no_match>",
        )
