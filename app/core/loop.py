"""
app/core/loop.py

The main pipeline loop — ties every component together.

Startup sequence
----------------
1. Load config → setup logging
2. Open memory DB, create session
3. Initialise services (real or sim based on config.sim_mode)
4. Register STT transcript callback
5. Start STT background thread
6. Start vision background thread
7. Run main loop (blocking) until stop() is called

Main loop (one tick, ~100ms)
-----------------------------
• Assemble ClassifierInput from latest STT + vision
• Run classifier → get action + salience scores
• Translate action → State Machine Event
• Feed event → state machine
• If state machine entered THINKING → run generation pipeline

Generation pipeline (runs in generation thread)
-----------------------------------------------
1. Snapshot transcript + vision description
2. Fetch memory context from DB
3. Build LLM prompt via ContextBuilder
4. Stream LLM → collect chunks
5. Fire FIRST_TOKEN on first chunk → state: SPEAKING
6. Check for USER_INTERRUPT each chunk — abort if interrupted
7. Synthesise full response via TTS
8. Play audio
9. Write user + assistant episodes to memory
10. Fire SPEAK_COMPLETE → state: IDLE or WATCHING

Sim mode
--------
Set simulation.enabled = true in simulation.yaml.
All services are replaced with offline stubs. No network calls.
The SimSTT fires canned phrases on a timer — the full pipeline
runs end-to-end without any real models or hardware.
"""

import logging
import threading
import time
from typing import Optional

from app.core.classifier    import Classifier, ClassifierInput
from app.core.context_builder import ContextBuilder, TurnInput
from app.core.state_machine  import Event, State, StateMachine
from app.memory.db           import MemoryDB
from app.memory.reader       import MemoryReader
from app.memory.writer       import MemoryWriter
from app.services.audio_player import play_audio, configure_audio
from app.services.mic_service  import MicService
from app.utils.config_loader import Config
from app.utils.logger        import setup_logging

logger = logging.getLogger(__name__)

# UI event bus — imported lazily so UI deps are optional
def _try_publish(event: dict) -> None:
    """Publish to the UI event bus. No-op if UI is not installed."""
    try:
        from app.ui.event_bus import event_bus
        event_bus.publish(event)
    except Exception:
        pass


class PipelineLoop:
    """
    The Eyes and Ears main pipeline loop.

    Usage
    -----
    config = load_config("./config")
    loop   = PipelineLoop(config)
    loop.start()      # blocks until KeyboardInterrupt or loop.stop()
    """

    TICK_INTERVAL_S = 0.1    # 10 ticks/second

    def __init__(self, config: Config):
        self._config  = config
        self._running = False

        # Shared state — updated by background threads, read on each tick
        self._latest_transcript    = ""
        self._transcript_confidence = 0.0
        self._latest_vision        = ""
        self._latest_frame: Optional[bytes] = None   # current JPEG, replaced each capture
        self._last_source          = "speech"
        self._transcript_lock      = threading.Lock()
        self._vision_lock          = threading.Lock()

        # Generation pipeline — runs in its own thread while main loop continues
        self._generation_thread: Optional[threading.Thread] = None
        self._generation_active = False
        self._session_id:       Optional[str] = None
        self._last_generation_ended_at: float = 0.0
        self._last_vision_reaction_at:  float = 0.0
        self._cooldown_s:       float = 8.0    # pause after speaking before reacting to stream

        self._embed_fn:        Optional[object]        = None
        self.state_machine:    Optional[StateMachine]   = None
        self.context_builder:  Optional[ContextBuilder] = None
        self.mic:              Optional[MicService]     = None
        self.classifier        = None
        self.llm               = None
        self.tts               = None
        self.stt               = None
        self.vision_svc        = None
        self.db:               Optional[MemoryDB]       = None
        self.writer:           Optional[MemoryWriter]   = None
        self.reader:           Optional[MemoryReader]   = None

        logger.info(
            "PipelineLoop created | sim_mode=%s", config.sim_mode
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the pipeline. Blocks until stop() is called or KeyboardInterrupt.
        Call from the main thread.
        """
        setup_logging(self._config)
        logger.info("=" * 60)
        logger.info("Eyes and Ears — starting up")
        logger.info("sim_mode=%s  model=%s", self._config.sim_mode, self._config.models.chat)
        logger.info("=" * 60)

        self._setup_core()
        self._setup_services()
        self._setup_memory()

        # Configure audio device from params.yaml → audio section
        audio_cfg = self._config.params.get("audio", {})
        configure_audio(
            output_device = audio_cfg.get("output_device"),
            input_device  = audio_cfg.get("input_device"),
        )

        self._session_id = self.writer.create_session(
            tags=["sim" if self._config.sim_mode else "live"]
        )
        logger.info("Session started | id=%s", self._session_id)

        _try_publish({
            "type":       "startup",
            "sim_mode":   self._config.sim_mode,
            "model":      self._config.models.chat,
            "session_id": self._session_id,
        })

        logger.info("Pipeline running. Press Ctrl+C to stop.")
        self._running = True

        # Start UI server if configured
        ui_cfg = self._config.params.get("ui", {})
        if ui_cfg.get("enabled", True):
            from app.ui.server import start_ui_server
            start_ui_server(
                host = ui_cfg.get("host", "0.0.0.0"),
                port = int(ui_cfg.get("port", 7860)),
            )

        # Register STT callback and start STT
        self.stt.on_transcript(self._on_transcript)
        self.stt.start()

        # Start microphone capture → feeds audio to STT WebSocket
        # Only enqueues audio when STT is actually connected
        self.mic = MicService(
            input_device = audio_cfg.get("input_device"),
            sample_rate  = self._config.services.stt.sample_rate,
            chunk_ms     = self._config.services.stt.chunk_ms,
            on_audio     = lambda chunk: self.stt.send_audio(chunk) if self.stt.connected else None,
        )
        self.mic.start()

        # Start vision thread
        vision_thread = threading.Thread(
            target = self._vision_loop,
            name   = "vision",
            daemon = True,
        )
        vision_thread.start()

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — shutting down")
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the loop to stop cleanly."""
        self._running = False

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _setup_core(self) -> None:
        self.state_machine   = StateMachine(
            reasoning_enabled = False
        )
        self.context_builder = ContextBuilder(self._config)

        # Log state transitions
        self.state_machine.on_state_change(self._on_state_change)

    def _setup_services(self) -> None:
        cfg = self._config

        if cfg.sim_mode:
            from app.simulation.sim_classifier import SimClassifier, Scenario
            from app.simulation.sim_services   import (
                SimLLMService, SimTTSService,
                SimSTTService, SimVisionService,
            )
            self.classifier = SimClassifier(mode="rule", simulated_latency_ms=20)
            self.llm        = SimLLMService(cfg, word_delay_s=0.03, first_token_delay_s=0.2)
            self.tts        = SimTTSService(cfg)
            self.stt        = SimSTTService(cfg, phrase_interval_s=5.0)
            self.vision_svc = SimVisionService(cfg)
            self._embed_fn  = None   # sim: no embeddings
            logger.info("Simulation services loaded")

        else:
            from app.core.classifier            import Classifier
            from app.services.llm_service       import LLMService
            from app.services.tts_service       import TTSService
            from app.services.stt_service       import STTService
            from app.services.vision_service    import VisionService
            from app.services.embedding_service import EmbeddingService
            from app.simulation.sim_classifier  import SimClassifier, ClassifierRule
            from app.core.classifier            import ClassifierAction

            # Rule-based classifier: instant (CPU only), no GPU competition.
            # Vision is ambient context — speech includes it automatically.
            # Unprompted vision reactions gated to every 30s so speech stays primary.
            loop_ref = self

            live_rules = [
                ClassifierRule(
                    name     = "user_speaking",
                    match_fn = lambda i: len(i.speech_transcript) > 5 and i.speech_confidence > 0.65,
                    action   = ClassifierAction.RESPOND,
                    speech_salience = 0.92, stream_salience = 0.0,
                    primary_source  = "speech",
                    reason   = "User is speaking.",
                    confidence = 0.9,
                ),
                ClassifierRule(
                    name     = "visual_content",
                    match_fn = lambda i: (
                        bool(i.vision_description)
                        and len(i.vision_description) > 20
                        and not i.speech_transcript
                        and (time.monotonic() - loop_ref._last_vision_reaction_at) > 30.0
                    ),
                    action   = ClassifierAction.REACT,
                    speech_salience = 0.0, stream_salience = 0.75,
                    primary_source  = "vision",
                    reason   = "Screen has new content worth noting.",
                    confidence = 0.8,
                ),
                ClassifierRule(
                    name     = "silence",
                    match_fn = lambda i: not i.speech_transcript and not i.vision_description,
                    action   = ClassifierAction.HOLD,
                    speech_salience = 0.0, stream_salience = 0.0,
                    primary_source  = "none",
                    reason   = "No signal.",
                    confidence = 1.0,
                ),
            ]

            self.classifier = SimClassifier(
                mode            = "rule",
                rules           = live_rules,
                simulated_latency_ms = 0,
                thresholds_cfg  = cfg.params.get("classifier_thresholds", {}),
            )
            self.llm        = LLMService(cfg)
            self.tts        = TTSService(cfg)
            self.stt        = STTService(cfg)
            self.vision_svc = VisionService(cfg)
            self._embed_fn  = EmbeddingService(cfg).embed
            logger.info("Live services loaded")

    def _setup_memory(self) -> None:
        self.db     = MemoryDB(self._config.memory.db_path)
        self.db.open()
        self.writer = MemoryWriter(self.db, embed_fn=self._embed_fn)
        self.reader = MemoryReader(
            self.db,
            embed_fn          = self._embed_fn,
            recency_half_life_hours = 24,
        )
        logger.info("Memory DB opened | path=%s", self._config.memory.db_path)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def _main_loop(self) -> None:
        while self._running:
            tick_start = time.monotonic()

            try:
                self._tick()
            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)

            # Sleep remainder of tick interval
            elapsed = time.monotonic() - tick_start
            sleep   = max(0.0, self.TICK_INTERVAL_S - elapsed)
            time.sleep(sleep)

    def _tick(self) -> None:
        """One pipeline iteration."""
        current_state = self.state_machine.state

        # Don't run classifier while generation is active — state machine
        # handles the THINKING/SPEAKING arc internally
        if self._generation_active:
            return

        # Cooldown after speaking — suppress stream-triggered generation briefly
        # so the AI doesn't immediately react to its own output
        since_last = time.monotonic() - self._last_generation_ended_at
        if since_last < self._cooldown_s:
            return

        # Snapshot inputs
        with self._transcript_lock:
            transcript  = self._latest_transcript
            confidence  = self._transcript_confidence

        with self._vision_lock:
            vision_desc  = self._latest_vision
            # Take the current frame for speech responses — cleared after use
            # so we don't hold onto stale bytes. Vision reactions use the
            # text description path instead (classifier needs text to score).
            current_frame: Optional[bytes] = self._latest_frame if transcript else None
            if current_frame:
                self._latest_frame = None

        # Classifier
        inp = ClassifierInput(
            current_state      = current_state,
            speech_transcript  = transcript,
            speech_confidence  = confidence,
            vision_description = vision_desc,
            active_since_ms    = int(self.state_machine.time_in_state * 1000),
        )
        output = self.classifier.classify(inp)
        event  = self.classifier.to_event(output, current_state)

        log = logger.info if event else logger.debug
        log(
            "Classifier: action=%-10s speech=%.2f stream=%.2f → event=%s",
            output.action.value,
            output.speech_salience,
            output.stream_salience,
            event.name if event else "none",
        )

        _try_publish({
            "type":            "classifier",
            "action":          output.action.value,
            "speech_salience": round(output.speech_salience, 3),
            "stream_salience": round(output.stream_salience, 3),
            "reason":          output.reason,
        })

        if vision_desc:
            _try_publish({"type": "vision", "description": vision_desc})

        if event:
            transitioned = self.state_machine.process_event(event)

            if transitioned and self.state_machine.state == State.THINKING:
                self._last_source = output.primary_source
                self._start_generation(transcript, vision_desc, current_frame)

    def _start_generation(
        self,
        transcript:  str,
        vision_desc: str,
        frame:       Optional[bytes] = None,
    ) -> None:
        """Kick off the generation pipeline in a background thread."""
        if self._generation_active:
            return

        self._generation_active = True
        if not transcript:
            self._last_vision_reaction_at = time.monotonic()
        self._generation_thread = threading.Thread(
            target = self._generation_pipeline,
            args   = (transcript, vision_desc, frame),
            name   = "generation",
            daemon = True,
        )
        self._generation_thread.start()

    # -----------------------------------------------------------------------
    # Generation pipeline
    # -----------------------------------------------------------------------

    def _generation_pipeline(
        self,
        transcript:  str,
        vision_desc: str,
        frame:       Optional[bytes] = None,
    ) -> None:
        """
        Full generate → speak → remember arc.
        Runs in a background thread so the main loop stays responsive.

        When frame is provided (speech-triggered), the raw JPEG is injected
        directly into the LLM request so the VL model sees actual pixels.
        Vision reactions use the text description path instead.
        Frame bytes are passed in and not stored — GC'd after this call.
        """
        try:
            # Clear consumed transcript
            with self._transcript_lock:
                self._latest_transcript    = ""
                self._transcript_confidence = 0.0

            # Fetch memory
            memory_ctx = None
            if transcript and self._session_id:
                try:
                    memory_ctx = self.reader.get_context_for_prompt(
                        query          = transcript,
                        session_id     = self._session_id,
                        budget_tokens  = self._config.memory.context_budget_tokens,
                    )
                except Exception as exc:
                    logger.warning("Memory fetch failed: %s", exc)

            # Build context — skip text description when injecting actual frame
            turn = TurnInput(
                transcript         = transcript,
                vision_description = vision_desc if not frame else "",
                current_state      = self.state_machine.state,
                source             = self._last_source,
            )
            built = self.context_builder.build(turn, memory_ctx)

            # Stream LLM — attach frame for speech responses
            response_chunks: list[str] = []
            first_token_fired = False

            for chunk in self.llm.stream_chat(
                built.messages,
                system_prompt = built.system_prompt,
                image_bytes   = frame,
            ):
                if not first_token_fired:
                    self.state_machine.process_event(Event.FIRST_TOKEN)
                    first_token_fired = True

                # Honour interruptions
                if self.state_machine.state == State.USER_INTERRUPTED:
                    logger.info("Generation interrupted by user")
                    break

                response_chunks.append(chunk)

            response = "".join(response_chunks).strip()

            # TTS + playback
            if response and self.state_machine.state == State.SPEAKING:
                logger.info("Speaking | chars=%d", len(response))
                audio = self.tts.synthesize(response)
                if not self._config.sim_mode:
                    play_audio(audio, blocking=True)
                else:
                    logger.debug("Sim mode — skipping audio playback")

            # Write to memory
            if self._session_id:
                if transcript:
                    self.writer.write_episode(
                        session_id      = self._session_id,
                        role            = "user",
                        content         = transcript,
                        source          = "speech",
                        salience_weight = 0.7,
                    )
                    _try_publish({"type": "user_turn", "text": transcript})
                if response:
                    self.writer.write_episode(
                        session_id      = self._session_id,
                        role            = "assistant",
                        content         = response,
                        source          = "speech",
                        salience_weight = 0.6,
                    )
                    _try_publish({
                        "type":    "assistant_turn",
                        "text":    response,
                        "persona": self._config.personality.name,
                    })
                # Publish memory stats after each turn
                try:
                    _try_publish({"type": "memory_stats", "stats": self.db.stats()})
                except Exception:
                    pass

            # Update conversation history
            if response:
                self.context_builder.add_assistant_response(response)

            # Complete turn
            self.state_machine.process_event(Event.SPEAK_COMPLETE)
            self._last_generation_ended_at = time.monotonic()

            # Vision description is NOT cleared here — it persists as ambient
            # context so the LLM has screen awareness even when speech triggers.
            # Unprompted vision reactions are gated by _last_vision_reaction_at.

            logger.debug("Turn complete | response_chars=%d", len(response))

        except Exception as exc:
            logger.error("Generation pipeline error: %s", exc, exc_info=True)
            self.state_machine.reset()
        finally:
            self._generation_active = False

    # -----------------------------------------------------------------------
    # STT callback
    # -----------------------------------------------------------------------

    def _on_transcript(self, text: str, is_final: bool, confidence: float) -> None:
        """Called from the STT background thread on every transcript segment."""
        if not text.strip():
            return

        state = self.state_machine.state

        # User speaks while AI is generating/speaking → interrupt
        if state in (State.THINKING, State.SPEAKING, State.REASONING):
            self.state_machine.process_event(Event.USER_INTERRUPT)
            logger.info("User interrupt | text=%r state=%s", text[:40], state.name)

        with self._transcript_lock:
            self._latest_transcript     = text
            self._transcript_confidence = confidence

        # Fire SPEECH_DETECTED to move IDLE/WATCHING → LISTENING
        if state in (State.IDLE, State.WATCHING):
            self.state_machine.process_event(Event.SPEECH_DETECTED)

        logger.debug(
            "Transcript | final=%s conf=%.2f text=%r",
            is_final, confidence, text[:60]
        )
        _try_publish({
            "type":       "transcript",
            "text":       text,
            "is_final":   is_final,
            "confidence": confidence,
        })

    # -----------------------------------------------------------------------
    # Vision loop
    # -----------------------------------------------------------------------

    def _vision_loop(self) -> None:
        """
        Background thread: captures frames periodically and describes them.

        Skips vision inference entirely while generation is active so the
        GPU is free for the LLM. Vision and LLM are mutually exclusive —
        they share the same GPU and cannot run efficiently in parallel.
        """
        logger.info("Vision loop started")

        while self._running:
            interval_ms = self._get_vision_interval()

            # Give the GPU fully to the LLM during generation.
            # Vision will resume on the next tick after generation completes.
            if self._generation_active:
                time.sleep(interval_ms / 1000.0)
                continue

            try:
                frame_bytes = self._capture_frame()
                if frame_bytes:
                    # Store the raw frame — used for direct image injection
                    # on speech responses. Replaced each cycle, never accumulates.
                    with self._vision_lock:
                        self._latest_frame = frame_bytes

                    description = self.vision_svc.describe_frame(frame_bytes)
                    if description:
                        with self._vision_lock:
                            self._latest_vision = description
                        self.state_machine.notify_stream_active(True)
                        logger.info("Vision: %s", description[:120])
                else:
                    self.state_machine.notify_stream_active(False)

            except Exception as exc:
                logger.warning("Vision loop error: %s", exc)

            time.sleep(interval_ms / 1000.0)

        logger.info("Vision loop stopped")

    def _capture_frame(self) -> Optional[bytes]:
        """
        Capture a screen frame and resize it before returning.
        In sim mode returns a 1-pixel placeholder so SimVisionService fires.
        In live mode uses mss for fast cross-platform screenshots, then
        resizes to max_pixels width (config: vision_pipeline.max_pixels)
        and compresses as JPEG to keep visual token count within model context.
        """
        if self._config.sim_mode:
            return b"\xff\xd8\xff"   # minimal JPEG magic — triggers sim vision

        try:
            import mss
            from PIL import Image
            import io

            with mss.mss() as sct:
                monitor = sct.monitors[1]   # primary monitor
                img     = sct.grab(monitor)

                # Convert raw pixels → PIL Image
                pil_img = Image.frombytes("RGB", img.size, img.rgb)

                # Resize to max_pixels width maintaining aspect ratio.
                # This keeps visual token count within the model's context window.
                # A 4K screenshot at full resolution can generate 10k+ tokens.
                max_w = self._config.vision_pipeline.max_pixels
                if pil_img.width > max_w:
                    ratio   = max_w / pil_img.width
                    new_h   = int(pil_img.height * ratio)
                    pil_img = pil_img.resize((max_w, new_h), Image.LANCZOS)

                # Encode as JPEG (much smaller than PNG, acceptable quality for description)
                buf = io.BytesIO()
                pil_img.save(
                    buf,
                    format  = "JPEG",
                    quality = self._config.vision_pipeline.jpeg_quality,
                )
                return buf.getvalue()

        except ImportError:
            logger.warning(
                "mss or Pillow not installed — screen capture unavailable. "
                "pip install mss Pillow"
            )
            return None
        except Exception as exc:
            logger.warning("Screen capture failed: %s", exc)
            return None

    def _get_vision_interval(self) -> int:
        """Return frame interval in ms based on current state."""
        state = self.state_machine.state if self.state_machine else State.IDLE
        vp    = self._config.vision_pipeline

        if state == State.SPEAKING:
            return vp.frame_interval_speaking_ms
        if self._generation_active:
            return vp.frame_interval_active_ms
        return vp.frame_interval_ms

    # -----------------------------------------------------------------------
    # State change observer
    # -----------------------------------------------------------------------

    def _on_state_change(
        self, prev: State, new: State, event: Event
    ) -> None:
        logger.info(
            "State: %s → %s  [%s]  (%.2fs in prev)",
            prev.name, new.name, event.name,
            self.state_machine.time_in_state,
        )
        _try_publish({
            "type":       "state_change",
            "state":      new.name,
            "prev_state": prev.name,
            "event":      event.name,
        })

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Shutting down...")
        self._running = False

        self.stt.stop()

        if self.mic:
            self.mic.stop()

        if self._generation_thread and self._generation_thread.is_alive():
            self._generation_thread.join(timeout=5)

        if self._session_id and self.writer:
            try:
                summary = f"Session ended. {self.context_builder.turn_count} turns."
                self.writer.close_session(self._session_id, summary)
            except Exception as exc:
                logger.warning("Failed to close session: %s", exc)

        if self.db:
            self.db.close()

        _try_publish({"type": "shutdown"})
        stats = self.state_machine.summary() if self.state_machine else {}
        logger.info("Shutdown complete | transitions=%d", stats.get("transition_count", 0))
