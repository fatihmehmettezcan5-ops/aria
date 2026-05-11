"""The agent loop.

Streams model output token by token. While streaming, watches for
<TOOL_CALL>...</TOOL_CALL> spans. When one is detected:
  1. stop generation,
  2. parse the call,
  3. execute the tool,
  4. format the result back into the prompt,
  5. resume generation.

Yields high-level `AgentEvent` objects that the API layer can serialise to SSE.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent.prompt_builder import build_prompt_ids
from inference.generator import GenerationConfig, Generator
from tokenizer.special_tokens import SpecialTokens
from tools.executor import execute_call, format_tool_result_for_model
from tools.parser import find_tool_calls
from tools.registry import ToolRegistry
from tools.schema import ToolContext


@dataclass
class AgentEvent:
    type: str            # "token" | "tool_start" | "tool_end" | "done" | "error"
    data: dict[str, Any] = field(default_factory=dict)


MAX_STEPS = 4


class AgentOrchestrator:
    def __init__(
        self,
        generator: Generator,
        registry: ToolRegistry,
        *,
        gen_cfg: GenerationConfig | None = None,
    ) -> None:
        self.generator = generator
        self.registry = registry
        self.gen_cfg = gen_cfg or GenerationConfig()

    async def run(
        self,
        *,
        language: str,
        history: list[tuple[str, str]],
        user_message: str,
        ctx: ToolContext,
    ) -> AsyncIterator[AgentEvent]:
        S = SpecialTokens
        tok = self.generator.tokenizer
        tool_specs = self.registry.schemas()

        # Build the initial prompt.
        prompt_ids = build_prompt_ids(
            tok,
            language=language,
            tool_specs=tool_specs,
            history=history,
            user_message=user_message,
        )

        # Track tool spans across steps so we can persist them.
        tool_calls_done: list[dict[str, Any]] = []
        final_text = ""

        for step in range(MAX_STEPS):
            stop_strings = [S.TOOL_CALL_END, S.END]
            cfg = GenerationConfig(
                max_new_tokens=self.gen_cfg.max_new_tokens,
                temperature=self.gen_cfg.temperature,
                top_k=self.gen_cfg.top_k,
                top_p=self.gen_cfg.top_p,
                repetition_penalty=self.gen_cfg.repetition_penalty,
                stop_strings=stop_strings,
            )

            collected_ids: list[int] = []
            try:
                for tid in self.generator.stream(prompt_ids, cfg):
                    collected_ids.append(tid)
                    # Stream visible (non-special) characters.
                    piece = tok.decode([tid], skip_special=True)
                    if piece:
                        yield AgentEvent("token", {"content": piece})
            except Exception as e:  # noqa: BLE001
                yield AgentEvent("error", {"message": f"generation_failed: {e}"})
                return

            assistant_text = tok.decode(collected_ids, skip_special=False)
            # If a tool call was emitted, execute it then loop.
            calls = find_tool_calls(assistant_text)
            if calls:
                # Take the first complete call.
                call = calls[0]
                yield AgentEvent("tool_start", {
                    "name": call.name, "arguments": call.arguments,
                })
                execution = await execute_call(call, self.registry, ctx)
                tool_calls_done.append({
                    "name": execution.name,
                    "arguments": execution.arguments,
                    "result": execution.result,
                    "summary": execution.summary,
                })
                yield AgentEvent("tool_end", {
                    "name": execution.name,
                    "summary": execution.summary,
                    "result": execution.result,
                })

                # Re-prime the prompt with the tool call + result spans, then
                # continue generation in the same assistant turn.
                call_json_str = json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                )
                result_str = format_tool_result_for_model(execution)
                prompt_ids = build_prompt_ids(
                    tok,
                    language=language,
                    tool_specs=tool_specs,
                    history=history,
                    user_message=user_message,
                    prior_tool_call_text=call_json_str,
                    prior_tool_result_text=result_str,
                )
                continue

            # No tool call → final answer.
            final_text = tok.decode(collected_ids, skip_special=True).strip()
            break
        else:
            yield AgentEvent("error", {"message": f"max_steps_reached ({MAX_STEPS})"})

        yield AgentEvent("done", {
            "final_text": final_text,
            "tool_calls": tool_calls_done,
        })
