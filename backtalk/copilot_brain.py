# backtalk: GitHub Copilot SDK brain.
# Copyright (C) 2026 Jared Rhodenizer (@jaredrhod)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Copilot SDK adapter for backtalk's streaming voice loop."""
import asyncio
import os
import re
from pathlib import Path

from copilot import CopilotClient
from copilot.session_events import (
    AssistantMessageData,
    AssistantMessageDeltaData,
    SessionErrorData,
    SessionIdleData,
    ToolExecutionStartData,
)

from backtalk.config import CFG, DISCIPLINE
from backtalk.vlog import log

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
SESSION_FILE = os.path.join(
    CFG["signals_dir"] or str(Path(__file__).resolve().parent.parent),
    ".backtalk_copilot_session",
)


class CopilotBrain:
    """Keep one Copilot session alive and stream complete sentences."""

    def __init__(self, model: str | None = None, can_use_tool=None,
                 resume_id: str | None = None):
        self.model = model or CFG["model"]
        self._permission_handler = can_use_tool
        self._resume_id = resume_id
        self._client: CopilotClient | None = None
        self._session = None
        self._active = False
        self._idle = asyncio.Event()
        self._idle.set()
        self.session = {
            "turns": 0,
            "out_tokens": 0,
            "in_tokens": 0,
            "cost": 0.0,
        }

    def _session_options(self) -> dict:
        identity_path = Path(CFG["agent_dir"]) / "CLAUDE.md"
        identity = identity_path.read_text(encoding="utf-8") \
            if identity_path.is_file() else ""
        content = "\n\n".join(part for part in (identity, DISCIPLINE) if part)
        opts = {
            "model": self.model,
            "working_directory": CFG["agent_dir"],
            "system_message": {"mode": "append", "content": content},
            "events": {"streaming": True},
            "permissions": {"on_request": self._permission_handler},
        }
        extra_dirs = CFG.get("extra_dirs") or []
        if extra_dirs:
            opts["filesystem"] = {"additional_directories": extra_dirs}
        effort = str(CFG.get("effort") or "").strip().lower()
        if effort in ("low", "medium", "high", "xhigh", "max"):
            opts["inference"] = {"reasoning_effort": effort}
        return opts

    async def start(self):
        self._client = CopilotClient(mode="coding-agent")
        await self._client.start()
        options = self._session_options()
        if self._resume_id:
            try:
                self._session = await self._client.resume_session(
                    self._resume_id, **options)
                log(f"[brain] resumed Copilot session "
                    f"{self._resume_id[:8]}")
            except Exception as exc:
                log(f"[brain] Copilot resume failed ({str(exc)[:100]}), "
                    "starting fresh")
                self._session = await self._client.create_session(**options)
        else:
            self._session = await self._client.create_session(**options)
        self._resume_id = None

    async def set_permission_mode(self, backtalk_mode: str):
        # The callback reads backtalk.main's live auto-approve flag, so no
        # Copilot session recreation is needed when the user changes modes.
        return None

    async def context_usage(self):
        return None

    def _remember_session(self):
        if not CFG.get("resume_last_session") or not self._session:
            return
        try:
            Path(SESSION_FILE).write_text(
                self._session.session_id, encoding="utf-8")
        except OSError as exc:
            log(f"[brain] couldn't save Copilot session id: {exc}")

    async def _new_session(self):
        if self._session:
            await self._session.disconnect()
        self._session = await self._client.create_session(
            **self._session_options())
        self._active = False
        self._idle.set()

    async def command(self, cmd: str) -> str:
        """Handle the subset of voice-console commands supported by the SDK."""
        parts = cmd.strip().split(maxsplit=1)
        if not parts:
            return ""
        command = parts[0].lower()
        value = parts[1] if len(parts) == 2 else ""
        if command == "/clear":
            await self._new_session()
            return "Conversation cleared."
        if command == "/model":
            if not value:
                return "error: specify a Copilot model"
            try:
                await self._session.set_model(value)
            except Exception as exc:
                return f"error: {exc}"
            self.model = value
            return f"Model set to {value}."
        if command == "/effort":
            if value not in ("low", "medium", "high", "xhigh", "max"):
                return f"error: unsupported reasoning effort {value!r}"
            try:
                await self._session.set_model(
                    self.model, reasoning_effort=value)
            except Exception as exc:
                return f"error: {exc}"
            return f"Reasoning effort set to {value}."
        if command == "/compact":
            return ("error: manual compaction is not available in Copilot "
                    "mode; Copilot manages context automatically")
        return f"error: {command} is not available in Copilot mode"

    async def interrupt(self):
        if self._session and self._active:
            await self._session.abort()

    async def reset_turn(self, timeout: float = 8.0):
        if not self._session or not self._active:
            return
        try:
            await self.interrupt()
            await asyncio.wait_for(self._idle.wait(), timeout)
        except asyncio.TimeoutError:
            log("[brain] Copilot turn did not stop cleanly; rebuilding "
                "the session")
            await self._new_session()

    async def _drain_after_abort(self, events):
        if self._session and self._active:
            try:
                await self._session.abort()
            except Exception as exc:
                log(f"[brain] Copilot abort after interruption: {exc}")

            async def drain():
                while True:
                    event = await events.get()
                    if isinstance(event.data, SessionIdleData):
                        return

            try:
                await asyncio.wait_for(drain(), 8)
                self._active = False
                self._idle.set()
            except asyncio.TimeoutError:
                log("[brain] Copilot turn failed to settle; rebuilding "
                    "the session")
                await self._new_session()

    async def ask_stream(self, utterance: str):
        """Yield sentences as Copilot streams them."""
        await self.reset_turn()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()

        def receive(event):
            loop.call_soon_threadsafe(events.put_nowait, event)

        unsubscribe = self._session.on(receive)
        self._active = True
        self._idle.clear()
        buf = ""
        saw_delta = False
        seen_messages: set[str] = set()
        try:
            await self._session.send(utterance)
            while True:
                event = await events.get()
                if getattr(event, "agent_id", None):
                    continue
                data = event.data
                if isinstance(data, AssistantMessageDeltaData):
                    saw_delta = True
                    buf += data.delta_content
                    while match := _SENTENCE_END.search(buf):
                        sentence, buf = (buf[:match.end()].strip(),
                                         buf[match.end():])
                        if sentence:
                            yield sentence
                elif isinstance(data, AssistantMessageData):
                    if data.message_id not in seen_messages:
                        seen_messages.add(data.message_id)
                        if data.output_tokens:
                            self.session["out_tokens"] += data.output_tokens
                        if not saw_delta and data.content:
                            buf += data.content
                elif isinstance(data, ToolExecutionStartData):
                    tail = buf.strip()
                    buf = ""
                    if tail:
                        yield tail
                elif isinstance(data, SessionErrorData):
                    raise RuntimeError(data.message)
                elif isinstance(data, SessionIdleData):
                    self.session["turns"] += 1
                    self._active = False
                    self._idle.set()
                    break
            tail = buf.strip()
            if tail:
                yield tail
            self._remember_session()
        finally:
            if self._active:
                await self._drain_after_abort(events)
            unsubscribe()

    async def stop(self):
        if self._session:
            await self._session.disconnect()
            self._session = None
        if self._client:
            await self._client.stop()
            self._client = None


if __name__ == "__main__":
    async def demo():
        brain = CopilotBrain()
        await brain.start()
        try:
            async for sentence in brain.ask_stream(
                    "Say hello to me in one short sentence."):
                print(sentence, flush=True)
        finally:
            await brain.stop()

    asyncio.run(demo())
