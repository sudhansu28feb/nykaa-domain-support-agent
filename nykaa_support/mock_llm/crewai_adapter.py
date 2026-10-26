"""
CrewAI BaseLLM Adapter for Deterministic Mock LLM.
Disables telemetry and supports Agent tool execution and final answer synthesis.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
from pydantic import Field

# Ensure telemetry is disabled before crewai imports
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TESTING"] = "true"

from crewai.llms.base_llm import BaseLLM
from nykaa_support.mock_llm.engine import extract_order_id, generate_grounded_response


class CrewAIMockLLM(BaseLLM):
    """Deterministic CrewAI BaseLLM with tool-calling and ReAct execution support."""

    model: str = Field(default="mock-crewai-agent-llm")

    def __init__(self, **data: Any):
        data.setdefault("model", "mock-crewai-agent-llm")
        super().__init__(**data)

    def call(
        self,
        messages: str | List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type] = None,
    ) -> str:
        # Convert messages to a consolidated string representation
        if isinstance(messages, list):
            prompt_str = "\n".join(
                str(m.get("content", "")) if isinstance(m, dict) else str(m)
                for m in messages
            )
        else:
            prompt_str = str(messages)

        # Check if a tool has already been invoked in conversation history
        # (System prompt mentions 'Observation:' as instructions, so strictly check non-system messages)
        has_tool_observation = False
        obs_text = ""
        if isinstance(messages, list) and len(messages) > 2:
            # Check non-system messages for actual tool execution observation
            for m in messages[1:]:
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                if "Observation:" in content or "Tool Output" in content:
                    has_tool_observation = True
                    obs_match = re.search(
                        r'(?:Observation:|Tool Output[:\s]+)(.*?)(?:\nMoving on then|\nTo use a tool|\nTo provide the final|\n[A-Z][a-z]+:|\Z)',
                        content,
                        re.DOTALL
                    )
                    if obs_match:
                        candidate = obs_match.group(1).strip()
                        if candidate:
                            obs_text = candidate
                    break

        # If a tool observation exists, synthesize the final answer
        if has_tool_observation:
            final_text = obs_text if obs_text else "Order inquiry processed successfully with official records."
            return f"Thought: I have received the necessary tool details to resolve the inquiry.\nFinal Answer: {final_text}"

        # Determine available tools from args, from_agent, or prompt text
        available_tools = list(tools) if tools else []
        if not available_tools and from_agent and hasattr(from_agent, "tools"):
            available_tools = list(getattr(from_agent, "tools") or [])

        # If tools are available, find the matching tool to call and dispatch by DECLARED SCHEMA
        if available_tools:
            tool_obj = available_tools[0]
            tool_name = self._select_tool_name(available_tools)
            action_input = self._build_action_input(tool_obj, prompt_str)
            return (
                f"Thought: I should use the available tool to fetch accurate information.\n"
                f"Action: {tool_name}\n"
                f"Action Input: {action_input}"
            )

        # Agent has no tools (e.g. Response Composer Agent or direct response)
        # Check if prompt contains context from previous tasks
        context_match = re.search(
            r'This is the context you(?:\'re|[\’]re| are) working with:\s*(.*?)(?:\nProvide your complete|\nBegin!|\Z)',
            prompt_str,
            re.DOTALL | re.IGNORECASE
        )
        if context_match and context_match.group(1).strip():
            raw_context = context_match.group(1).strip()
            direct_answer = (
                f"Thank you for contacting Nykaa Support. Based on official records: {raw_context} "
                f"Please let us know if you need further assistance with your Nykaa shopping experience."
            )
        else:
            direct_answer = generate_grounded_response(prompt_str)

        return f"Thought: I have sufficient information to answer the inquiry directly.\nFinal Answer: {direct_answer}"

    async def acall(
        self,
        messages: str | List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type] = None,
    ) -> str:
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model
        )

    def supports_stop_words(self) -> bool:
        return False

    def _select_tool_name(self, tools: List[Any]) -> str:
        """Extract tool name from tool dict or object."""
        first = tools[0]
        if isinstance(first, dict):
            return first.get("name", "check_order")
        if hasattr(first, "name"):
            return getattr(first, "name")
        return "check_order"

    def _build_action_input(self, tool_obj: Any, prompt_str: str) -> str:
        """
        Build Action Input JSON strictly complying with Pitfall 2:
        Inspect tool's declared Pydantic argument schema (e.g., args_schema),
        NEVER infer arguments from substring name matching.
        """
        schema_fields = set()
        args_schema = getattr(tool_obj, "args_schema", None)
        if isinstance(tool_obj, dict):
            args_schema = tool_obj.get("args_schema")

        if args_schema:
            if hasattr(args_schema, "model_fields"):
                schema_fields = set(args_schema.model_fields.keys())
            elif hasattr(args_schema, "__fields__"):
                schema_fields = set(args_schema.__fields__.keys())

        # If schema not available directly on the object, inspect JSON schema in system prompt
        if not schema_fields:
            arg_match = re.search(r'Tool Arguments:\s*(\{.*?\})\s*Tool Description:', prompt_str, re.DOTALL)
            if arg_match:
                try:
                    schema_dict = json.loads(arg_match.group(1))
                    schema_fields = set(schema_dict.get("properties", {}).keys())
                except Exception:
                    pass

        # Look specifically in Current Task text first to avoid matching schema description examples
        task_match = re.search(r'Current Task:\s*(.*?)(?:\n\n|\nThis is the expected|\Z)', prompt_str, re.DOTALL)
        task_text = task_match.group(1).strip() if task_match else ""

        # Dispatch based strictly on declared schema fields
        if "record_id" in schema_fields:
            order_id = extract_order_id(task_text) or extract_order_id(prompt_str) or "NYK-10001"
            return json.dumps({"record_id": order_id})
        elif "query" in schema_fields:
            q_text = task_text if task_text else prompt_str[:150].strip()
            # Clean out prompt wrappers if present
            q_clean = re.sub(r'^(?:Customer asks:?|Retrieve official Nykaa policy for:?|Lookup:?)\s*', '', q_text, flags=re.IGNORECASE).strip()
            return json.dumps({"query": q_clean or q_text})
        elif "order_id" in schema_fields:
            order_id = extract_order_id(task_text) or extract_order_id(prompt_str) or "NYK-10001"
            return json.dumps({"order_id": order_id})
        else:
            return json.dumps({"query": task_text or prompt_str[:100].strip()})
