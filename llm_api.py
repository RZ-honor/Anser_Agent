"""
LLM API 适配层：把 chat.completions 风格调用转换为 Responses API

背景：问答端点（dasuapi grok-4.6 等）使用 OpenAI Responses 接口（/v1/responses），
而现有 pipeline（工具调用循环 / 基线评测）全部按 chat.completions 格式编写。
本模块做双向格式转换，调用方零改动：

- 请求：messages/tools(tool_choice) → input/tools(responses 工具格式)
- 响应：适配回 chat 风格对象（.choices[0].message.content / .tool_calls / .usage）

config.USE_RESPONSES_API=0 时直接透传 chat.completions（兼容其他端点）。
"""
from types import SimpleNamespace

# tool_choice 透传白名单（两端点语义一致：auto/required/none/具体函数名）


def _convert_tools(tools):
    """chat 工具格式 → responses 工具格式

    chat:    {"type": "function", "function": {"name", "description", "parameters"}}
    responses: {"type": "function", "name", "description", "parameters"}
    """
    converted = []
    for t in tools or []:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            converted.append({
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            converted.append(t)
    return converted


def _convert_messages(messages):
    """chat 消息列表 → responses input 列表

    - system/user/assistant(纯文本) → {role, content}
    - assistant 带 tool_calls → 拆成文本项 + 多个 function_call 项
    - role=tool → function_call_output 项
    - 兼容 SDK 响应对象（pydantic/SimpleNamespace）：工具循环会把上一轮的
      assistant 消息对象直接 append 回消息列表，此处先归一化为 dict
    """
    items = []
    for m in messages:
        if not isinstance(m, dict):
            entry = {"role": getattr(m, "role", "assistant"),
                     "content": getattr(m, "content", None)}
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                entry["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tcs]
            m = entry
        role = m.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", ""),
                "output": m.get("content", ""),
            })
            continue
        tool_calls = m.get("tool_calls")
        if role == "assistant" and tool_calls:
            # assistant 文本与每个工具调用分别入 input（文本可为空则省略）
            content = m.get("content")
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in tool_calls:
                fn = tc["function"] if isinstance(tc, dict) else tc.function
                call_id = tc["id"] if isinstance(tc, dict) else tc.id
                items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": fn["name"] if isinstance(fn, dict) else fn.name,
                    "arguments": fn["arguments"] if isinstance(fn, dict) else fn.arguments,
                })
            continue
        items.append({"role": role, "content": m.get("content", "")})
    return items


def _adapt_response(resp):
    """responses 响应 → chat 风格对象（choices[0].message.content/.tool_calls + usage）"""
    text_parts, tool_calls = [], []
    for item in (resp.output or []):
        itype = getattr(item, "type", None)
        if itype == "message":
            for c in (getattr(item, "content", None) or []):
                if getattr(c, "type", None) == "output_text":
                    text_parts.append(c.text)
        elif itype == "function_call":
            tool_calls.append(SimpleNamespace(
                id=getattr(item, "call_id", ""),
                type="function",
                function=SimpleNamespace(
                    name=getattr(item, "name", ""),
                    arguments=getattr(item, "arguments", "{}"),
                )))
    usage = getattr(resp, "usage", None)
    usage_ns = SimpleNamespace(
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
    )
    usage_ns.total_tokens = usage_ns.prompt_tokens + usage_ns.completion_tokens
    message = SimpleNamespace(
        content="".join(text_parts),
        tool_calls=tool_calls if tool_calls else None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage_ns)


def call_llm(client, model, messages, tools=None, tool_choice=None,
             temperature=0.1, max_tokens=1200, extra_body=None,
             use_responses_api=True):
    """统一调用入口：use_responses_api=True 走 Responses API，否则透传 chat.completions

    Returns:
        chat 风格响应对象（.choices[0].message / .usage），调用方无感知
    """
    if not use_responses_api:
        kwargs = {"model": model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if extra_body:
            kwargs["extra_body"] = extra_body
        return client.chat.completions.create(**kwargs)

    kwargs = {
        "model": model,
        "input": _convert_messages(messages),
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if tools is not None:
        kwargs["tools"] = _convert_tools(tools)
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp = client.responses.create(**kwargs)
    return _adapt_response(resp)
