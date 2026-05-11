"""Build the model prompt: system → tool docs → conversation.

Tokens are emitted as a flat ID list using the tokenizer's chat-format helpers.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from tokenizer.special_tokens import SpecialTokens
from tokenizer.tokenizer import Tokenizer


SYSTEM_PROMPT_EN = """You are Aria, a small but helpful AI assistant.
Today's UTC date is {date}.

Tools available:
{tools}

When you need a tool, output exactly:
<TOOL_CALL>{{"name": "<tool>", "arguments": {{...}}}}</TOOL_CALL>
Then stop. The system will execute the tool and return the result wrapped in
<TOOL_RESULT>...</TOOL_RESULT>. After receiving the result, write the final answer.

If a tool returned hits with `filename` (knowledge base) or `url` (web), cite
them inline as [1], [2] in numerical order of appearance. Never invent sources.
Be brief. Use Markdown."""


SYSTEM_PROMPT_TR = """Sen Aria adlı küçük ama yardımcı bir yapay zekâ asistanısın.
Bugünün UTC tarihi: {date}.

Mevcut araçlar:
{tools}

Bir araç gerektiğinde tam olarak şu formatta çıktı ver:
<TOOL_CALL>{{"name": "<tool>", "arguments": {{...}}}}</TOOL_CALL>
Sonra dur. Sistem aracı çalıştırıp sonucu <TOOL_RESULT>...</TOOL_RESULT>
içinde döner. Sonuç geldikten sonra son cevabını yaz.

Bir araç `filename` (bilgi tabanı) veya `url` (web) içeriyorsa, ortaya çıkış
sırasına göre [1], [2] gibi satır içi alıntı yap. Asla uydurma kaynak verme.
Kısa ol. Markdown kullan."""


def system_prompt(language: str, tool_specs: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tools_text = "\n".join(
        f"- {t['name']}: {t['description']}" for t in tool_specs
    ) or "(none)"
    template = SYSTEM_PROMPT_TR if (language or "en").lower().startswith("tr") else SYSTEM_PROMPT_EN
    return template.format(date=today, tools=tools_text)


def build_prompt_ids(
    tokenizer: Tokenizer,
    *,
    language: str,
    tool_specs: list[dict],
    history: list[tuple[str, str]],   # [(role, content), ...]
    user_message: str,
    prior_tool_call_text: str | None = None,
    prior_tool_result_text: str | None = None,
) -> list[int]:
    """Construct the full prompt as token IDs ending with the assistant role tag.

    `prior_tool_call_text` / `prior_tool_result_text` allow the agent loop to
    feed back tool results in the same turn.
    """
    S = SpecialTokens
    sys_text = system_prompt(language, tool_specs)
    turns: list[tuple[str, str]] = [("system", sys_text)] + history + [("user", user_message)]
    ids = tokenizer.encode_chat(turns, add_bos=True)
    # Open the assistant role.
    ids.append(tokenizer.token_to_id[S.ASSISTANT])
    # If we already produced a tool call + got a result, append both spans
    # then keep going as the assistant.
    if prior_tool_call_text is not None and prior_tool_result_text is not None:
        ids.append(tokenizer.token_to_id[S.TOOL_CALL])
        ids.extend(tokenizer.encode(prior_tool_call_text))
        ids.append(tokenizer.token_to_id[S.TOOL_CALL_END])
        ids.append(tokenizer.token_to_id[S.TOOL_RESULT])
        ids.extend(tokenizer.encode(prior_tool_result_text))
        ids.append(tokenizer.token_to_id[S.TOOL_RESULT_END])
    return ids
