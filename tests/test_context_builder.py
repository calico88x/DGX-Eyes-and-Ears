"""
tests/test_context_builder.py

Fully offline tests for ContextBuilder.
No services, no LLM calls, no memory DB required.

Run with:  pytest tests/test_context_builder.py -v
"""

import pytest
from unittest.mock import MagicMock

from app.core.context_builder import ContextBuilder, TurnInput, BuiltContext
from app.core.state_machine import State
from app.memory.reader import MemoryContext, EpisodeResult, FactResult
from app.utils.config_loader import (
    Config, AppConfig, ServicesConfig, ModelsConfig,
    LiteLLMConfig, STTConfig, TTSConfig, VisionServiceConfig,
    MemoryConfig, VisionPipelineConfig, LoggingConfig,
    PersonalityConfig, SimulationConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(budget: int = 800) -> Config:
    return Config(
        app=AppConfig(
            services=ServicesConfig(
                litellm=LiteLLMConfig(base_url="http://localhost:4000"),
                stt=STTConfig(base_url="ws://localhost:9090"),
                tts=TTSConfig(base_url="http://localhost:8880"),
                vision=VisionServiceConfig(base_url="http://localhost:8000"),
            ),
            models=ModelsConfig(
                chat="test-model", classifier="test-cls",
                vision="test-vis", embedding="test-emb",
            ),
            memory=MemoryConfig(context_budget_tokens=budget),
            vision_pipeline=VisionPipelineConfig(),
            logging=LoggingConfig(),
        ),
        params={},
        personality=PersonalityConfig(
            name="Aria",
            persona="curious and warm",
            system_prompt="You are Aria, a conversational AI companion.",
            tone_notes="casual and genuine",
        ),
        simulation=SimulationConfig(),
    )


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def builder(config):
    return ContextBuilder(config)


def make_turn(**kwargs) -> TurnInput:
    defaults = dict(
        transcript         = "",
        vision_description = "",
        current_state      = State.IDLE,
        source             = "speech",
    )
    defaults.update(kwargs)
    return TurnInput(**defaults)


def make_memory_context(
    episodes: list = None,
    facts:    list = None,
    prompt_text: str = "",
) -> MemoryContext:
    return MemoryContext(
        episodes        = episodes or [],
        facts           = facts    or [],
        entities        = [],
        prompt_text     = prompt_text,
        token_estimate  = len(prompt_text) // 4,
        retrieved_count = 0,
    )


# ---------------------------------------------------------------------------
# TurnInput
# ---------------------------------------------------------------------------

class TestTurnInput:
    def test_defaults(self):
        t = TurnInput()
        assert t.transcript         == ""
        assert t.vision_description == ""
        assert t.current_state      == State.IDLE
        assert t.source             == "speech"

    def test_all_fields(self):
        t = TurnInput(
            transcript         = "hello",
            vision_description = "game on screen",
            current_state      = State.LISTENING,
            source             = "vision",
        )
        assert t.transcript  == "hello"
        assert t.source      == "vision"


# ---------------------------------------------------------------------------
# ContextBuilder initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_history_empty_on_init(self, builder):
        assert builder.history == []

    def test_turn_count_zero_on_init(self, builder):
        assert builder.turn_count == 0

    def test_history_returns_copy(self, builder):
        h = builder.history
        h.append({"role": "user", "content": "test"})
        assert builder.turn_count == 0   # original unchanged


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

class TestSystemPromptAssembly:
    def test_contains_persona(self, builder):
        turn = make_turn()
        ctx  = builder.build(turn)
        assert "Aria" in ctx.system_prompt

    def test_contains_system_prompt_text(self, builder):
        turn = make_turn()
        ctx  = builder.build(turn)
        assert "You are Aria" in ctx.system_prompt

    def test_contains_tone_notes(self, builder):
        turn = make_turn()
        ctx  = builder.build(turn)
        assert "casual and genuine" in ctx.system_prompt

    def test_contains_state(self, builder):
        turn = make_turn(current_state=State.LISTENING)
        ctx  = builder.build(turn)
        assert "LISTENING" in ctx.system_prompt

    def test_contains_vision_description_when_present(self, builder):
        turn = make_turn(vision_description="user is playing a game")
        ctx  = builder.build(turn)
        assert "user is playing a game" in ctx.system_prompt

    def test_no_vision_in_system_when_absent(self, builder):
        turn = make_turn()
        ctx  = builder.build(turn)
        assert "Screen:" not in ctx.system_prompt

    def test_memory_context_injected(self, builder):
        mem  = make_memory_context(prompt_text="[MEMORY]\nFact: user likes jazz.\n[/MEMORY]")
        turn = make_turn(transcript="hi")
        ctx  = builder.build(turn, memory_context=mem)
        assert "user likes jazz" in ctx.system_prompt

    def test_no_memory_when_context_none(self, builder):
        turn = make_turn(transcript="hi")
        ctx  = builder.build(turn, memory_context=None)
        assert "[MEMORY]" not in ctx.system_prompt

    def test_no_memory_when_prompt_text_empty(self, builder):
        mem  = make_memory_context(prompt_text="")
        turn = make_turn(transcript="hi")
        ctx  = builder.build(turn, memory_context=mem)
        assert "[MEMORY]" not in ctx.system_prompt

    def test_empty_tone_notes_not_added(self):
        config = make_config()
        config.personality.tone_notes = ""
        builder = ContextBuilder(config)
        turn = make_turn()
        ctx  = builder.build(turn)
        assert "Tone:" not in ctx.system_prompt


# ---------------------------------------------------------------------------
# User message assembly
# ---------------------------------------------------------------------------

class TestUserMessageAssembly:
    def test_speech_transcript_becomes_user_message(self, builder):
        turn = make_turn(transcript="what is that on screen")
        ctx  = builder.build(turn)
        last = ctx.messages[-1]
        assert last["role"]    == "user"
        assert last["content"] == "what is that on screen"

    def test_vision_source_without_speech(self, builder):
        turn = make_turn(
            vision_description = "explosion on screen",
            source             = "vision",
        )
        ctx = builder.build(turn)
        assert len(ctx.messages) > 0
        assert "reacting to screen" in ctx.messages[-1]["content"]
        assert "explosion" in ctx.messages[-1]["content"]

    def test_vision_source_with_speech_uses_speech(self, builder):
        turn = make_turn(
            transcript         = "whoa did you see that",
            vision_description = "explosion on screen",
            source             = "vision",
        )
        ctx = builder.build(turn)
        assert ctx.messages[-1]["content"] == "whoa did you see that"

    def test_no_transcript_no_vision_no_user_message(self, builder):
        turn = make_turn()
        ctx  = builder.build(turn)
        assert ctx.messages == []

    def test_transcript_whitespace_stripped(self, builder):
        turn = make_turn(transcript="  hello there  ")
        ctx  = builder.build(turn)
        assert ctx.messages[-1]["content"] == "hello there"


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_grows_with_turns(self, builder):
        for i in range(3):
            builder.build(make_turn(transcript=f"message {i}"))
        assert builder.turn_count == 3

    def test_add_assistant_response_recorded(self, builder):
        builder.build(make_turn(transcript="hi"))
        builder.add_assistant_response("Hello!")
        assert builder.history[-1] == {"role": "assistant", "content": "Hello!"}

    def test_blank_assistant_response_ignored(self, builder):
        builder.build(make_turn(transcript="hi"))
        count_before = builder.turn_count
        builder.add_assistant_response("   ")
        assert builder.turn_count == count_before

    def test_full_exchange_in_history(self, builder):
        builder.build(make_turn(transcript="hello"))
        builder.add_assistant_response("Hey there!")
        builder.build(make_turn(transcript="how are you"))
        builder.add_assistant_response("I'm doing great!")

        history = builder.history
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_clear_history_resets(self, builder):
        builder.build(make_turn(transcript="hi"))
        builder.add_assistant_response("Hello!")
        builder.clear_history()
        assert builder.history == []
        assert builder.turn_count == 0

    def test_messages_include_full_history(self, builder):
        builder.build(make_turn(transcript="first"))
        builder.add_assistant_response("response one")
        ctx = builder.build(make_turn(transcript="second"))
        assert len(ctx.messages) == 3   # first user, assistant, second user

    def test_no_message_added_for_empty_turn(self, builder):
        builder.build(make_turn())   # empty turn
        assert builder.turn_count == 0


# ---------------------------------------------------------------------------
# Sliding window / token budget
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_history_trimmed_false_when_within_budget(self, builder):
        ctx = builder.build(make_turn(transcript="short message"))
        assert ctx.history_trimmed is False

    def test_history_trimmed_true_when_over_budget(self):
        config  = make_config(budget=50)   # tiny budget
        builder = ContextBuilder(config)

        # Fill with many long messages
        for i in range(20):
            builder.build(make_turn(transcript="x" * 100))
            builder.add_assistant_response("y" * 100)

        ctx = builder.build(make_turn(transcript="final"))
        assert ctx.history_trimmed is True

    def test_token_estimate_fits_budget_after_trim(self):
        config  = make_config(budget=100)
        builder = ContextBuilder(config)

        for i in range(20):
            builder.build(make_turn(transcript="x" * 100))
            builder.add_assistant_response("y" * 100)

        ctx = builder.build(make_turn(transcript="z" * 10))
        assert ctx.token_estimate <= 200   # some headroom for system prompt

    def test_current_message_always_preserved(self):
        config  = make_config(budget=50)
        builder = ContextBuilder(config)

        for i in range(20):
            builder.build(make_turn(transcript="x" * 100))
            builder.add_assistant_response("y" * 100)

        ctx = builder.build(make_turn(transcript="KEEP THIS"))
        last = ctx.messages[-1]
        assert last["content"] == "KEEP THIS"

    def test_hard_cap_enforced(self):
        config  = make_config(budget=99999)   # huge budget — cap enforced differently
        builder = ContextBuilder(config)
        builder._max_history = 5   # force small cap for test

        for i in range(10):
            builder.build(make_turn(transcript=f"msg {i}"))
            builder.add_assistant_response(f"resp {i}")

        assert len(builder.history) <= 5

    def test_token_estimate_is_positive(self, builder):
        ctx = builder.build(make_turn(transcript="hello there"))
        assert ctx.token_estimate > 0

    def test_token_estimate_grows_with_content(self, builder):
        ctx_short = builder.build(make_turn(transcript="hi"))
        builder.clear_history()
        ctx_long = builder.build(make_turn(transcript="x" * 500))
        assert ctx_long.token_estimate > ctx_short.token_estimate


# ---------------------------------------------------------------------------
# BuiltContext output shape
# ---------------------------------------------------------------------------

class TestBuiltContext:
    def test_returns_built_context(self, builder):
        ctx = builder.build(make_turn(transcript="hi"))
        assert isinstance(ctx, BuiltContext)

    def test_system_prompt_is_string(self, builder):
        ctx = builder.build(make_turn(transcript="hi"))
        assert isinstance(ctx.system_prompt, str)
        assert len(ctx.system_prompt) > 0

    def test_messages_is_list(self, builder):
        ctx = builder.build(make_turn(transcript="hi"))
        assert isinstance(ctx.messages, list)

    def test_each_message_has_role_and_content(self, builder):
        builder.build(make_turn(transcript="hi"))
        builder.add_assistant_response("hello")
        ctx = builder.build(make_turn(transcript="how are you"))
        for msg in ctx.messages:
            assert "role"    in msg
            assert "content" in msg

    def test_messages_roles_are_valid(self, builder):
        builder.build(make_turn(transcript="hi"))
        builder.add_assistant_response("hello")
        ctx = builder.build(make_turn(transcript="follow up"))
        valid_roles = {"user", "assistant", "system"}
        for msg in ctx.messages:
            assert msg["role"] in valid_roles


# ---------------------------------------------------------------------------
# Integration: full turn cycle
# ---------------------------------------------------------------------------

class TestFullTurnCycle:
    def test_complete_conversation_arc(self):
        """Simulate a multi-turn conversation and verify context is coherent."""
        config  = make_config()
        builder = ContextBuilder(config)

        # Turn 1: user speaks
        mem  = make_memory_context(prompt_text="[MEMORY]\nFact: user likes racing games.\n[/MEMORY]")
        ctx1 = builder.build(
            make_turn(
                transcript         = "hey what do you think of this game",
                vision_description = "racing game on screen",
                current_state      = State.LISTENING,
            ),
            memory_context = mem,
        )
        builder.add_assistant_response("Oh that looks like a racing game! Exciting!")

        # Turn 2: vision triggers reaction
        ctx2 = builder.build(
            make_turn(
                vision_description = "car crash in the game",
                current_state      = State.WATCHING,
                source             = "vision",
            )
        )
        builder.add_assistant_response("Whoa! That crash was intense!")

        # Turn 3: user speaks again
        ctx3 = builder.build(
            make_turn(
                transcript    = "did you see that",
                current_state = State.LISTENING,
            )
        )

        # Verify progressive history
        assert ctx1.token_estimate > 0
        assert "racing game" in ctx1.system_prompt
        assert "user likes racing games" in ctx1.system_prompt

        assert "reacting to screen" in ctx2.messages[-1]["content"]

        # ctx3 should have full history
        assert len(ctx3.messages) >= 4   # 2 user + 2 assistant from previous turns

    def test_vision_and_speech_both_in_system(self, builder):
        mem  = make_memory_context(prompt_text="[MEMORY]\nUser enjoys coding.\n[/MEMORY]")
        turn = make_turn(
            transcript         = "what should I refactor here",
            vision_description = "VS Code with Python code open",
            current_state      = State.LISTENING,
        )
        ctx = builder.build(turn, memory_context=mem)

        assert "VS Code"      in ctx.system_prompt
        assert "User enjoys coding" in ctx.system_prompt
        assert ctx.messages[-1]["content"] == "what should I refactor here"
