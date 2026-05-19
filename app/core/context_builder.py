"""
app/core/context_builder.py

Assembles everything the LLM needs to see each turn into a clean
messages array, ready to pass to LLMService.stream_chat().

Each turn the builder receives:
  - The current user transcript (from STT)
  - The current vision description (from VisionService)
  - Retrieved memory context (from MemoryReader)
  - The pipeline's current state
  - The running conversation history

And produces:
  - system_prompt : str         Full system prompt with persona + memory injected
  - messages      : list[dict]  Conversation history trimmed to context budget

Sliding window
--------------
The conversation history is trimmed from the oldest end whenever the
estimated token count exceeds context_budget_tokens. The system prompt
and the current turn are always preserved. This prevents the context
from filling and breaking the pipeline mid-conversation.

Separation of concerns
-----------------------
ContextBuilder knows nothing about services or state transitions.
It only reads config and produces prompt data. The loop calls it.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.core.state_machine import State
from app.memory.reader import MemoryContext
from app.utils.config_loader import Config

logger = logging.getLogger(__name__)

# Approximate chars per token — used for budget estimation
_CHARS_PER_TOKEN = 4


def _estimate_tokens(system_prompt: str, messages: list[dict]) -> int:
    """Rough token count: total characters / 4. No tokenizer dependency."""
    total = len(system_prompt)
    for msg in messages:
        total += len(msg.get("content", "")) + 10  # 10 chars overhead per message
    return total // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurnInput:
    """
    All inputs available for one pipeline turn.
    Fields are optional — absent inputs are handled gracefully.
    """
    transcript:         str   = ""    # STT output (final transcript)
    vision_description: str   = ""    # VisionService output
    current_state:      State = State.IDLE
    source:             str   = "speech"   # "speech" | "vision" | "audio_event"


@dataclass
class BuiltContext:
    """
    Output of ContextBuilder.build() — ready for the LLM.
    """
    system_prompt:   str
    messages:        list[dict]
    token_estimate:  int
    history_trimmed: bool   # True if old messages were dropped this turn


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """
    Assembles the LLM prompt for each pipeline turn.

    Parameters
    ----------
    config : Config
        Loaded app config. Uses personality and memory budget settings.
    """

    def __init__(self, config: Config):
        self._personality     = config.personality
        self._budget          = config.memory.context_budget_tokens
        self._max_history     = 40     # hard cap on conversation turns kept
        self._history: list[dict] = []

        logger.info(
            "ContextBuilder ready | budget=%d tokens persona=%s",
            self._budget, self._personality.name
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def build(
        self,
        turn:           TurnInput,
        memory_context: Optional[MemoryContext] = None,
    ) -> BuiltContext:
        """
        Build the full LLM context for one pipeline turn.

        Parameters
        ----------
        turn           : TurnInput       Current inputs from the pipeline.
        memory_context : MemoryContext   Retrieved memories (may be None in sim mode).

        Returns
        -------
        BuiltContext   System prompt + trimmed messages list.
        """
        system_prompt = self._build_system_prompt(turn, memory_context)
        user_message  = self._build_user_message(turn)

        # Add to history
        if user_message:
            self._history.append({"role": "user", "content": user_message})

        # Trim history to fit budget
        messages, trimmed = self._trim_history(system_prompt)

        token_estimate = _estimate_tokens(system_prompt, messages)

        logger.debug(
            "Context built | turns=%d tokens~=%d trimmed=%s source=%s",
            len(messages), token_estimate, trimmed, turn.source
        )

        return BuiltContext(
            system_prompt   = system_prompt,
            messages        = messages,
            token_estimate  = token_estimate,
            history_trimmed = trimmed,
        )

    def add_assistant_response(self, text: str) -> None:
        """
        Record the assistant's response in the conversation history.
        Call this after the LLM finishes speaking.
        """
        if text.strip():
            self._history.append({"role": "assistant", "content": text.strip()})
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def clear_history(self) -> None:
        """Reset the conversation history. Called at session start/end."""
        self._history.clear()
        logger.info("Conversation history cleared")

    @property
    def history(self) -> list[dict]:
        """Read-only snapshot of current conversation history."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return len(self._history)

    # -----------------------------------------------------------------------
    # System prompt assembly
    # -----------------------------------------------------------------------

    def _build_system_prompt(
        self,
        turn:    TurnInput,
        memory:  Optional[MemoryContext],
    ) -> str:
        parts: list[str] = []

        # 1. Core persona
        parts.append(self._personality.system_prompt.strip())

        # 2. Tone notes (if present)
        if self._personality.tone_notes.strip():
            parts.append(f"Tone: {self._personality.tone_notes.strip()}")

        # 3. Current state context
        parts.append(self._format_state_context(turn))

        # 4. Memory context (if available)
        if memory and memory.prompt_text:
            parts.append(memory.prompt_text)

        return "\n\n".join(p for p in parts if p.strip())

    def _format_state_context(self, turn: TurnInput) -> str:
        """Describe what the AI can currently perceive."""
        lines = [f"[PERCEPTION] Current state: {turn.current_state.name}"]

        if turn.vision_description:
            lines.append(f"Screen: {turn.vision_description}")

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # User message assembly
    # -----------------------------------------------------------------------

    def _build_user_message(self, turn: TurnInput) -> str:
        """
        Build the user-turn content from available inputs.

        Speech is the primary input. Vision is included as context
        when it triggered the turn (source == "vision").
        """
        if not turn.transcript and not turn.vision_description:
            return ""

        if turn.source == "vision" and not turn.transcript:
            # AI reacting to screen content without user speech
            return f"[reacting to screen] {turn.vision_description}"

        if turn.transcript:
            return turn.transcript.strip()

        return ""

    # -----------------------------------------------------------------------
    # Sliding window
    # -----------------------------------------------------------------------

    def _trim_history(
        self,
        system_prompt: str,
    ) -> tuple[list[dict], bool]:
        """
        Return a copy of history trimmed to fit the token budget.

        Always preserves the most recent message.
        Drops oldest messages first.
        """
        if not self._history:
            return [], False

        # Enforce hard cap first
        if len(self._history) > self._max_history:
            dropped = len(self._history) - self._max_history
            self._history = self._history[dropped:]
            logger.debug("History hard-capped: dropped %d turns", dropped)

        messages = list(self._history)
        trimmed  = False

        # Trim until estimate fits budget
        while len(messages) > 1:
            estimate = _estimate_tokens(system_prompt, messages)
            if estimate <= self._budget:
                break
            messages = messages[1:]   # drop oldest
            trimmed  = True

        if trimmed:
            logger.info(
                "History trimmed to %d messages to fit %d token budget",
                len(messages), self._budget
            )

        return messages, trimmed
