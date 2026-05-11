"""Reserved special tokens used across the system."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    PAD: str = "<PAD>"
    UNK: str = "<UNK>"
    BOS: str = "<BOS>"
    EOS: str = "<EOS>"
    SEP: str = "<SEP>"
    MASK: str = "<MASK>"
    TOOL_CALL: str = "<TOOL_CALL>"
    TOOL_CALL_END: str = "</TOOL_CALL>"
    TOOL_RESULT: str = "<TOOL_RESULT>"
    TOOL_RESULT_END: str = "</TOOL_RESULT>"
    USER: str = "<USER>"
    ASSISTANT: str = "<ASSISTANT>"
    SYSTEM: str = "<SYSTEM>"
    END: str = "<END>"  # end of a role-turn

    @classmethod
    def all(cls) -> list[str]:
        return [
            cls.PAD, cls.UNK, cls.BOS, cls.EOS, cls.SEP, cls.MASK,
            cls.TOOL_CALL, cls.TOOL_CALL_END,
            cls.TOOL_RESULT, cls.TOOL_RESULT_END,
            cls.USER, cls.ASSISTANT, cls.SYSTEM, cls.END,
        ]
