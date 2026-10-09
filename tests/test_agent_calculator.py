import asyncio
import contextlib
import io
import json
import os
import unittest
from collections.abc import Sequence
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from starlette.testclient import TestClient

# The provider is replaced below; constructing the app never needs a real key.
with patch.dict(
    os.environ, {"GOOGLE_API_KEY": "test-dummy-key-0000000000000000000000"}
):
    from app import agent
    from app.main import app
    from smoke_test import run_smoke_test


class ScriptedModel:
    def __init__(self, tools: Sequence[BaseTool] = ()) -> None:
        self.tools = tools

    def bind_tools(self, tools: Sequence[BaseTool]) -> "ScriptedModel":
        return ScriptedModel(tools)

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"Completed: {messages[-1].content}")
        if not self.tools:
            return AIMessage(content="No tools requested.")
        selected = self.tools[0].name
        arguments = (
            {"city": "Barcelona"}
            if selected == "get_weather_forecast"
            else {"expression": str(messages[-1].content)}
        )
        return AIMessage(
            content="",
            tool_calls=[{"name": selected, "args": arguments, "id": "test-call"}],
        )


class AgentCalculatorTests(unittest.TestCase):
    def test_existing_smoke_flow_without_network(self) -> None:
        with (
            patch.object(agent, "llm", ScriptedModel()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_smoke_test()

    def test_health_and_tool_discovery(self) -> None:
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json()["status"], "ok")
            response = client.get("/tools")
            self.assertEqual(response.status_code, 200)
            self.assertIn(
                "calculate_metric", [item["name"] for item in response.json()]
            )

    def test_streamed_calculations_errors_and_recovery(self) -> None:
        cases = (
            ("2 + 3 * 4", "Calculation result: 14"),
            ("().__class__", "Calculation error:"),
            ("__import__('os')", "Calculation error:"),
            ("1 / 0", "Calculation error:"),
            ("6 * 7", "Calculation result: 42"),
        )
        with patch.object(agent, "llm", ScriptedModel()), TestClient(app) as client:
            for index, (expression, expected) in enumerate(cases):
                with self.subTest(expression=expression):
                    response = client.post(
                        "/stream",
                        json={
                            "message": expression,
                            "tools": ["calculate_metric"],
                            "thread_id": f"calculator-test-{index}",
                        },
                    )
                    self.assertEqual(response.status_code, 200)
                    lines = [
                        line[6:]
                        for line in response.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    self.assertEqual(lines[-1], "[DONE]")
                    events = [json.loads(line) for line in lines[:-1]]
                    self.assertEqual(
                        [event["type"] for event in events],
                        ["tool_call", "tool_result", "message"],
                    )
                    self.assertEqual(events[0]["tool"], "calculate_metric")
                    self.assertTrue(events[1]["output"].startswith(expected))
                    self.assertIn(events[1]["output"], events[2]["content"])

    def test_unknown_tools_are_rejected_without_model_calls(self) -> None:
        with (
            TestClient(app) as client,
            patch.object(
                agent, "resolve_model", side_effect=AssertionError("model must not run")
            ),
        ):
            response = client.post(
                "/stream", json={"message": "2 + 3", "tools": ["unknown-tool"]}
            )
            self.assertEqual(response.status_code, 400)

    def test_concurrent_tool_errors_do_not_abort_other_calls(self) -> None:
        calls = [
            {
                "name": "calculate_metric",
                "args": {"expression": expression},
                "id": str(i),
            }
            for i, expression in enumerate(
                ("2 + 3 * 4", "().__class__", "1 / 0", "6 * 7")
            )
        ]
        state: agent.AgentState = {
            "messages": [AIMessage(content="", tool_calls=calls)]
        }
        result = asyncio.run(agent.execute_tools(state))
        messages = result["messages"]
        self.assertEqual(
            [message.tool_call_id for message in messages], ["0", "1", "2", "3"]
        )
        self.assertEqual(messages[0].content, "Calculation result: 14")
        self.assertTrue(str(messages[1].content).startswith("Calculation error:"))
        self.assertTrue(str(messages[2].content).startswith("Calculation error:"))
        self.assertEqual(messages[3].content, "Calculation result: 42")


if __name__ == "__main__":
    unittest.main()
