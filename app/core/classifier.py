"""
app/core/classifier.py

Lightweight LLM-based classifier for pipeline flow control.

Receives a snapshot of all live inputs (speech transcript, vision description,
audio event) plus the current state machine state, then scores salience and
returns a routing action. That action is translated to a State Machine Event
via to_event().

Synchronous — no streaming. Must complete before the next pipeline tick.
Model: Qwen2.5-1.5B-Instruct via LiteLLM.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests

from app.core.state_machine import Event, State

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier vocabulary
# ---------------------------------------------------------------------------

class ClassifierAction(Enum):
    """
    Routing decisions the classifier can emit.
    Maps to State Machine Events via Classifier.to_event().
    """
    LISTEN    = "LISTEN"    # User is actively speaking — keep collecting
    RESPOND   = "RESPOND"   # User finished; content is worth responding to
    WATCH     = "WATCH"     # Stream active but below reaction threshold
    REACT     = "REACT"     # Stream content salient enough to trigger a response
    INTERRUPT = "INTERRUPT" # Input so salient it should override the current flow
    HOLD      = "HOLD"      # Insufficient signal — wait for more input


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClassifierInput:
    """
    Snapshot of all live pipeline signals for one classifier tick.
    Fields are optional — absent signals default to empty/zero.
    """
    current_state:       State
    speech_transcript:   str   = ""
    speech_confidence:   float = 0.0
    vision_description:  str   = ""
    audio_event_label:   str   = ""
    active_since_ms:     int   = 0   # ms the current state has been active


@dataclass
class ClassifierOutput:
    """
    Classifier decision for one pipeline tick.
    All salience scores are 0.0–1.0.
    """
    action:          ClassifierAction
    speech_salience: float
    stream_salience: float
    primary_source:  str           # "speech" | "vision" | "audio_event" | "none"
    reason:          str           # one-sentence explanation (for logs / UI)
    confidence:      float
    raw_response:    str   = ""    # raw LLM text, for debugging
    latency_ms:      float = 0.0


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class ClassifierThresholds:
    """
    Salience thresholds that gate which actions become State Machine Events.
    All sourced from params.yaml → classifier_thresholds.
    Defaults are conservative — tune upward for a less reactive agent.
    """

    def __init__(self, cfg: dict):
        # Minimum speech salience for RESPOND to fire SPEECH_ENDED
        self.speech_respond   = cfg.get("speech_respond_threshold",   0.60)
        # Minimum stream salience for REACT to fire SALIENCE_HIGH
        self.stream_react     = cfg.get("stream_react_threshold",     0.70)
        # Minimum stream salience to override active user speech
        self.stream_override  = cfg.get("stream_override_threshold",  0.85)
        # Minimum stream salience to interrupt AI while SPEAKING
        self.stream_interrupt = cfg.get("stream_interrupt_threshold", 0.90)

    def __repr__(self) -> str:
        return (
            f"Thresholds("
            f"speech_respond={self.speech_respond}, "
            f"stream_react={self.stream_react}, "
            f"stream_override={self.stream_override}, "
            f"stream_interrupt={self.stream_interrupt})"
        )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class Classifier:
    """
    Scores salience and routes pipeline inputs to State Machine Events.

    Usage
    -----
    classifier = Classifier(config, params)
    output = classifier.classify(input)
    event  = classifier.to_event(output, current_state)
    if event:
        state_machine.process_event(event)

    Simulation
    ----------
    Use SimClassifier (app/simulation/sim_classifier.py) for offline testing.
    Both expose identical classify() and to_event() interfaces.
    """

    # System prompt is kept tight — 1.5B model needs clear, minimal instructions.
    SYSTEM_PROMPT = """\
You are a real-time conversation flow controller for an AI agent.
Analyze the inputs and return a JSON routing decision. No preamble. JSON only.

ACTIONS:
LISTEN    — user is actively speaking, keep collecting audio
RESPOND   — user finished speaking and said something worth responding to
WATCH     — screen content is active but not salient enough to react
REACT     — screen content is salient enough for the AI to comment on
INTERRUPT — input is so salient it should override current speech or response
HOLD      — not enough signal to act, wait

SALIENCE: 0.0 = not interesting, 1.0 = extremely salient / urgent

OUTPUT FORMAT (JSON only, no markdown):
{
  "speech_salience": 0.0,
  "stream_salience": 0.0,
  "primary_source": "speech|vision|audio_event|none",
  "action": "LISTEN|RESPOND|WATCH|REACT|INTERRUPT|HOLD",
  "reason": "one sentence",
  "confidence": 0.0
}"""

    def __init__(self, config, params: dict):
        """
        Parameters
        ----------
        config : Config | dict
            Accepts the typed Config object (production) or a raw dict (tests).
        params : dict
            Raw params.yaml contents.
        """
        # Support typed Config object (production) and raw dict (tests)
        if hasattr(config, "services"):
            svc            = config.services.litellm
            self._base_url = svc.base_url
            self._api_key  = svc.api_key
            self._model    = config.models.classifier
        else:
            svc            = config["services"]["litellm"]
            self._base_url = svc["base_url"]
            self._api_key  = svc.get("api_key", "sk-local")
            self._model    = config["models"]["classifier"]

        self._llm_params  = params.get("classifier_llm", {})
        self._thresholds  = ClassifierThresholds(
            params.get("classifier_thresholds", {})
        )

        logger.info(
            "Classifier ready | model=%s thresholds=%s",
            self._model, self._thresholds
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def classify(self, inp: ClassifierInput) -> ClassifierOutput:
        """
        Score and route one pipeline tick. Synchronous.

        On LLM failure returns a safe HOLD output — pipeline keeps running.
        """
        t0 = time.monotonic()
        prompt = self._build_prompt(inp)
        raw = ""

        try:
            raw = self._call_llm(prompt)
            output = self._parse_response(raw)
        except json.JSONDecodeError as exc:
            logger.error("Classifier JSON parse failed: %s | raw=%r", exc, raw)
            output = self._fallback_output("JSON parse error")
        except requests.RequestException as exc:
            logger.error("Classifier LLM request failed: %s", exc)
            output = self._fallback_output("LLM request failed")
        except Exception as exc:
            logger.error("Classifier unexpected error: %s", exc)
            output = self._fallback_output(str(exc))

        output.raw_response = raw
        output.latency_ms   = round((time.monotonic() - t0) * 1000, 1)

        logger.debug(
            "Classifier | action=%-10s speech_sal=%.2f stream_sal=%.2f "
            "source=%-12s latency=%.1fms reason=%r",
            output.action.value,
            output.speech_salience,
            output.stream_salience,
            output.primary_source,
            output.latency_ms,
            output.reason,
        )

        return output

    def to_event(
        self,
        output: ClassifierOutput,
        current_state: State,
    ) -> Optional[Event]:
        """
        Translate a ClassifierOutput into a State Machine Event.

        Applies threshold gates — an action below its threshold fires nothing.
        Returns None for LISTEN / WATCH / HOLD (no transition needed).

        Parameters
        ----------
        output : ClassifierOutput
            Result from classify().
        current_state : State
            The machine's state at the moment of this decision.
            Required to select the correct interrupt event.
        """
        action = output.action
        t      = self._thresholds

        # User finished speaking — fire SPEECH_ENDED if salient enough
        if action == ClassifierAction.RESPOND:
            if output.speech_salience >= t.speech_respond:
                return Event.SPEECH_ENDED
            logger.debug(
                "RESPOND below threshold (%.2f < %.2f) — holding",
                output.speech_salience, t.speech_respond
            )
            return None

        # Stream content warrants a reaction from WATCHING
        if action == ClassifierAction.REACT:
            if output.stream_salience >= t.stream_react:
                return Event.SALIENCE_HIGH
            logger.debug(
                "REACT below threshold (%.2f < %.2f) — watching",
                output.stream_salience, t.stream_react
            )
            return None

        # Override: stream beats active speech or mid-sentence AI output
        if action == ClassifierAction.INTERRUPT:
            if current_state == State.SPEAKING:
                if output.stream_salience >= t.stream_interrupt:
                    return Event.STREAM_INTERRUPT
                logger.debug(
                    "INTERRUPT (SPEAKING) below threshold (%.2f < %.2f)",
                    output.stream_salience, t.stream_interrupt
                )
            elif current_state == State.LISTENING:
                if output.stream_salience >= t.stream_override:
                    return Event.SALIENCE_OVERRIDE
                logger.debug(
                    "INTERRUPT (LISTENING) below threshold (%.2f < %.2f)",
                    output.stream_salience, t.stream_override
                )
            return None

        # LISTEN / WATCH / HOLD — no state transition
        return None

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _build_prompt(self, inp: ClassifierInput) -> str:
        """Build the user-turn content from the input snapshot."""
        lines = [f"State: {inp.current_state.name}"]

        if inp.speech_transcript:
            lines.append(
                f'Speech: "{inp.speech_transcript}" '
                f'(conf={inp.speech_confidence:.2f})'
            )
        else:
            lines.append("Speech: [none]")

        if inp.vision_description:
            lines.append(f'Screen: "{inp.vision_description}"')
        else:
            lines.append("Screen: [none]")

        if inp.audio_event_label:
            lines.append(f"Audio event: {inp.audio_event_label}")

        lines.append(f"Active for: {inp.active_since_ms}ms")

        return "\n".join(lines)

    def _call_llm(self, user_prompt: str) -> str:
        """POST to LiteLLM and return raw response text."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model":       self._model,
            "stream":      False,
            "temperature": self._llm_params.get("temperature", 0.1),
            "max_tokens":  self._llm_params.get("max_tokens",  150),
            "top_p":       self._llm_params.get("top_p",       0.9),
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_response(self, text: str) -> ClassifierOutput:
        """
        Parse LLM response into ClassifierOutput.
        Handles markdown fences the model may add despite instructions.
        """
        clean = text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        data = json.loads(clean)   # raises JSONDecodeError if still malformed

        action_str = str(data.get("action", "HOLD")).upper().strip()
        try:
            action = ClassifierAction[action_str]
        except KeyError:
            logger.warning("Unknown action %r — defaulting HOLD", action_str)
            action = ClassifierAction.HOLD

        return ClassifierOutput(
            action          = action,
            speech_salience = float(data.get("speech_salience", 0.0)),
            stream_salience = float(data.get("stream_salience", 0.0)),
            primary_source  = str(data.get("primary_source",   "none")),
            reason          = str(data.get("reason",           "")),
            confidence      = float(data.get("confidence",     0.0)),
        )

    def _fallback_output(self, reason: str = "error") -> ClassifierOutput:
        """Safe default emitted when the LLM call or parse fails."""
        return ClassifierOutput(
            action          = ClassifierAction.HOLD,
            speech_salience = 0.0,
            stream_salience = 0.0,
            primary_source  = "none",
            reason          = f"classifier fallback: {reason}",
            confidence      = 0.0,
        )
