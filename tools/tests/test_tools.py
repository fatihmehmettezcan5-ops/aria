import asyncio

import pytest

from tokenizer.special_tokens import SpecialTokens
from tools.builtin.calculator import safe_eval
from tools.parser import find_tool_calls, has_open_tool_call
from tools.registry import get_default_registry
from tools.schema import ToolContext


def test_calculator_safe_eval():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(15 + 9) * 3") == 72
    assert safe_eval("sqrt(16)") == 4


def test_calculator_rejects_bad_input():
    with pytest.raises(Exception):
        safe_eval("__import__('os').system('echo hi')")
    with pytest.raises(Exception):
        safe_eval("open('x')")


@pytest.mark.asyncio
async def test_registry_calls_calculator():
    reg = get_default_registry()
    res = await reg.call("calculator", {"expression": "10*10"}, ToolContext())
    assert res["result"] == 100


@pytest.mark.asyncio
async def test_registry_unknown():
    reg = get_default_registry()
    res = await reg.call("nope", {}, ToolContext())
    assert "error" in res


def test_parse_tool_calls():
    S = SpecialTokens
    txt = (
        "Sure, "
        f'{S.TOOL_CALL}{{"name": "calculator", "arguments": {{"expression": "1+1"}}}}{S.TOOL_CALL_END}'
        " let's see."
    )
    calls = find_tool_calls(txt)
    assert len(calls) == 1
    assert calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "1+1"}


def test_has_open_tool_call():
    S = SpecialTokens
    assert has_open_tool_call(f"abc {S.TOOL_CALL} {{") is True
    assert has_open_tool_call(f"{S.TOOL_CALL}x{S.TOOL_CALL_END}") is False
