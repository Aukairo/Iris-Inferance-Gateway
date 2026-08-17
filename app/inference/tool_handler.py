import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from app.models.schemas import ChatCompletionToolCall, FunctionCall
from app.core.logging import logger


TOOL_SYSTEM_PROMPT_TEMPLATE = """

# Tool Calling Instructions
You have access to the following tools:
<tools>
{tools_json}
</tools>

To call a tool, respond with a JSON object enclosed in <tool_call>...</tool_call> tags.
Format:
<tool_call>
{{"name": "function_name", "arguments": {{"arg_name": "arg_value"}}}}
</tool_call>

CRITICAL INSTRUCTIONS:
1. You MUST ONLY call tools that are explicitly listed in the <tools> block above. Do NOT invent or call any tools not listed.
2. If a tool call fails, is unavailable, or returns an error, do NOT repeat the exact same failed tool call. Choose an alternative available tool or proceed with the information available.
3. If you call a tool, do not add extraneous conversational text before or after the tag.
"""


def format_tool_system_prompt(
    tools: List[Dict[str, Any]],
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
) -> str:
    """
    Generates system instructions describing the available tools and expected output format.
    Supports tool_choice: "none", "auto", "required", or {"type": "function", "function": {"name": ...}}.
    """
    if not tools or tool_choice == "none":
        return ""

    clean_tools = []
    for tool in tools:
        if isinstance(tool, dict):
            # Support standard OpenAI {"type": "function", "function": {...}} format
            if tool.get("type") == "function" and "function" in tool:
                clean_tools.append(tool["function"])
            else:
                clean_tools.append(tool)
        else:
            clean_tools.append(tool)

    tools_json = json.dumps(clean_tools, indent=2)
    prompt = TOOL_SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)

    if tool_choice == "required":
        prompt += "\nIMPORTANT: You MUST call at least one of the tools listed above in your response."
    elif isinstance(tool_choice, dict):
        target_name = None
        if tool_choice.get("type") == "function" and "function" in tool_choice:
            target_name = tool_choice["function"].get("name")
        elif "name" in tool_choice:
            target_name = tool_choice["name"]
        if target_name:
            prompt += f"\nIMPORTANT: You MUST call the tool '{target_name}' in your response."

    return prompt


def extract_reasoning_content(text: str) -> Tuple[Optional[str], str]:
    """
    Extracts reasoning/thinking blocks enclosed in <think>...</think> or <thought>...</thought> tags.
    Returns (reasoning_content, clean_text).
    """
    if not text:
        return None, text

    reasoning_parts = []
    clean_text = text

    matches = list(re.finditer(r"<(think|thought)>\s*(.*?)\s*(?:</\1>|$)", text, re.DOTALL | re.IGNORECASE))
    if matches:
        for match in matches:
            reasoning_parts.append(match.group(2).strip())
            clean_text = clean_text.replace(match.group(0), "")

        reasoning = "\n\n".join(reasoning_parts).strip() if reasoning_parts else None
        clean_text = clean_text.strip()
        return reasoning, clean_text

    return None, text


def parse_tool_calls(text: str) -> Tuple[Optional[str], Optional[List[ChatCompletionToolCall]], str]:
    """
    Parses generation text to extract tool calls and return (clean_content, tool_calls, finish_reason).

    Supports tool calling formats across multiple model families:
    1. <tool_call>...</tool_call> tags (Qwen 2.5, Hermes, DeepSeek)
    2. [TOOL_CALLS] [...] or [TOOL_CALLS] {...} (Mistral v0.3 / Mixtral)
    3. <call:function_name>{...}</call:function_name> or <call:func_name>(...)</call:func_name> (Llama 3.1 / 3.2)
    4. ```json ... ``` code blocks containing tool call objects
    5. [Call Tool: name(args)] legacy text format
    6. Direct JSON objects containing tool calls
    """
    if not text:
        return None, None, "stop"

    raw_text = text.strip()
    tool_calls: List[ChatCompletionToolCall] = []
    clean_text = text

    # Strategy 1: Look for <tool_call>...</tool_call> blocks (Qwen 2.5, Hermes, DeepSeek)
    tool_call_matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", raw_text, re.DOTALL | re.IGNORECASE))
    if tool_call_matches:
        for match in tool_call_matches:
            block_content = match.group(1).strip()
            clean_text = clean_text.replace(match.group(0), "")
            try:
                parsed = json.loads(block_content)
                extracted = _convert_to_tool_calls(parsed)
                tool_calls.extend(extracted)
            except Exception:
                json_match = re.search(r"\{.*\}", block_content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        extracted = _convert_to_tool_calls(parsed)
                        tool_calls.extend(extracted)
                    except Exception as e:
                        logger.warning("Failed to parse JSON inside <tool_call>: %s", str(e))

    # Strategy 2: Look for [TOOL_CALLS] [...] (Mistral v0.3 / Mixtral format)
    if not tool_calls:
        mistral_matches = list(re.finditer(r"\[TOOL_CALLS\]\s*(\[.*?\]|\{.*?\})\s*(?:\[/TOOL_CALLS\]|$)", raw_text, re.DOTALL | re.IGNORECASE))
        for match in mistral_matches:
            block_content = match.group(1).strip()
            clean_text = clean_text.replace(match.group(0), "")
            try:
                parsed = json.loads(block_content)
                extracted = _convert_to_tool_calls(parsed)
                if extracted:
                    tool_calls.extend(extracted)
            except Exception as e:
                logger.warning("Failed to parse Mistral [TOOL_CALLS] payload: %s", str(e))

    # Strategy 3: Look for <call:function_name>{...}</call:function_name> or <call:function_name>(...)</call:function_name> (Llama 3.1 / 3.2 format)
    if not tool_calls:
        llama_matches = list(re.finditer(r"<call:([a-zA-Z0-9_-]+)>\s*(\{.*?\}|\(.*?\))\s*(?:</call:\1>|</call>|$)", raw_text, re.DOTALL | re.IGNORECASE))
        for match in llama_matches:
            func_name = match.group(1)
            raw_payload = match.group(2).strip()
            clean_text = clean_text.replace(match.group(0), "")

            if raw_payload.startswith("{") and raw_payload.endswith("}"):
                try:
                    args_obj = json.loads(raw_payload)
                    args_str = json.dumps(args_obj)
                except Exception:
                    args_str = raw_payload
            else:
                raw_inner = raw_payload.strip("()")
                args_str = json.dumps({"args": raw_inner}) if raw_inner else "{}"

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(
                ChatCompletionToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=func_name, arguments=args_str)
                )
            )

    # Strategy 4: Look for ```json ... ``` code blocks containing tool calls if no tool_call tags matched
    if not tool_calls:
        json_code_blocks = list(re.finditer(r"```(?:json)?\s*(\{\s*\"name\".*?\}|\[\s*\{\s*\"name\".*?\}\s*\])\s*```", raw_text, re.DOTALL))
        for match in json_code_blocks:
            block_content = match.group(1).strip()
            try:
                parsed = json.loads(block_content)
                extracted = _convert_to_tool_calls(parsed)
                if extracted:
                    tool_calls.extend(extracted)
                    clean_text = clean_text.replace(match.group(0), "")
            except Exception:
                pass

    # Strategy 5: Look for legacy [Call Tool: name(args)]
    if not tool_calls:
        legacy_matches = list(re.finditer(r"\[Call Tool:\s*([a-zA-Z0-9_-]+)\((.*?)\)\]", raw_text, re.DOTALL))
        for match in legacy_matches:
            func_name = match.group(1)
            raw_args = match.group(2).strip()
            clean_text = clean_text.replace(match.group(0), "")

            if raw_args.startswith("{") and raw_args.endswith("}"):
                args_str = raw_args
            else:
                args_str = json.dumps({"args": raw_args}) if raw_args else "{}"

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            tool_calls.append(
                ChatCompletionToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=func_name, arguments=args_str)
                )
            )

    # Strategy 6: Raw JSON objects containing tool calls (e.g. {"name": "...", "arguments": ...} or {"action": "...", ...})
    if not tool_calls:
        json_obj_matches = list(re.finditer(r"(\{(?:[^{}]|\{[^{}]*\})*\}|\[\s*\{.*\}\s*\])", raw_text, re.DOTALL))
        for match in json_obj_matches:
            block_content = match.group(1).strip()
            try:
                parsed = json.loads(block_content)
                extracted = _convert_to_tool_calls(parsed)
                if extracted:
                    tool_calls.extend(extracted)
                    clean_text = clean_text.replace(match.group(0), "")
            except Exception:
                pass

    clean_content = clean_text.strip() if clean_text.strip() else None
    finish_reason = "tool_calls" if tool_calls else "stop"

    return clean_content, (tool_calls if tool_calls else None), finish_reason


def _convert_to_tool_calls(parsed: Any) -> List[ChatCompletionToolCall]:
    """
    Helper to standardize arbitrary parsed JSON (dict or list) into ChatCompletionToolCall items.
    Supports name, function, action, parameters, arguments, action_input, etc.
    """
    calls = []
    items = parsed if isinstance(parsed, list) else [parsed]

    for item in items:
        if not isinstance(item, dict):
            continue

        func_name = None
        func_args = None

        if "name" in item:
            func_name = item["name"]
            func_args = item.get("arguments") if item.get("arguments") is not None else (item.get("parameters") if item.get("parameters") is not None else item.get("args"))
        elif "action" in item:
            func_name = item["action"]
            func_args = item.get("action_input") if item.get("action_input") is not None else (item.get("arguments") if item.get("arguments") is not None else item.get("parameters"))
        elif "function" in item and isinstance(item["function"], dict):
            func_name = item["function"].get("name")
            func_args = item["function"].get("arguments") if item["function"].get("arguments") is not None else item["function"].get("parameters")
        elif "type" in item and item["type"] == "function" and "function" in item:
            func_name = item["function"].get("name")
            func_args = item["function"].get("arguments") if item["function"].get("arguments") is not None else item["function"].get("parameters")

        if func_name and isinstance(func_name, str):
            if isinstance(func_args, (dict, list)):
                args_str = json.dumps(func_args)
            elif isinstance(func_args, str):
                args_str = func_args
            else:
                args_str = "{}"

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            calls.append(
                ChatCompletionToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=func_name, arguments=args_str)
                )
            )

    return calls
