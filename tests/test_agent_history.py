"""Regression tests for cross-request history and in-request agent execution."""
import os
import tempfile
import unittest
import warnings
from unittest.mock import AsyncMock, patch

warnings.simplefilter("ignore", ResourceWarning)

_TEST_HOME = tempfile.TemporaryDirectory(prefix="agent-prototyping-tests-")
os.environ["AGENT_PRO_HOME"] = _TEST_HOME.name

from fastapi.testclient import TestClient
from app import main as agent


class AgentHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(agent.app)

    def setUp(self):
        with agent.db() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM projects WHERE id <> 'default'")
            conn.execute("UPDATE settings SET value='all' WHERE key='history_limit'")
            conn.execute("INSERT INTO settings(key,value) VALUES('agent_architecture','false') ON CONFLICT(key) DO UPDATE SET value='false'")
            conn.execute(
                "INSERT OR IGNORE INTO projects(id,name,created) VALUES('default','Test',1)"
            )
        self.session_id = "history-test-session"
        with agent.db() as conn:
            conn.execute(
                "INSERT INTO sessions(id,project_id,title,created,updated) VALUES(?,?,?,?,?)",
                (self.session_id, "default", "History test", 1, 1),
            )
            for index in range(1, 7):
                conn.execute(
                    "INSERT INTO messages(id,session_id,role,content,details,created) VALUES(?,?,?,?,?,?)",
                    (f"prior-{index}", self.session_id,
                     "user" if index % 2 else "assistant",
                     f"prior message {index}", None, float(index)),
                )

    def _reset_prior_history(self):
        with agent.db() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (self.session_id,))
            for index in range(1, 7):
                conn.execute(
                    "INSERT INTO messages(id,session_id,role,content,details,created) VALUES(?,?,?,?,?,?)",
                    (f"prior-{index}", self.session_id,
                     "user" if index % 2 else "assistant",
                     f"prior message {index}", None, float(index)),
                )

    def _set_history_limit(self, value):
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value=? WHERE key='history_limit'", (str(value),))

    @staticmethod
    def _tool_call(call_id, tool_name, arguments):
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": arguments},
        }

    def _run_mocked_agent(self, history_limit):
        self._reset_prior_history()
        self._set_history_limit(history_limit)
        tool_definitions = [{
            "type": "function",
            "function": {
                "name": "test-server__resolve",
                "description": "Resolve a test value",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            },
        }]
        responses = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                self._tool_call("call-1", "test-server__resolve", {"value": "first"})
            ]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                self._tool_call("call-2", "test-server__resolve", {"value": "second"})
            ]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "I have both results; the task is complete.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Completed using both tool results.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []

        async def fake_model_chat(messages, tools, cfg):
            # Capture a deep-enough immutable snapshot of every invocation.
            import copy
            snapshots.append(copy.deepcopy(messages))
            self.assertEqual(tools, tool_definitions)
            return responses[len(snapshots) - 1]

        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "first result"}]},
                 {"content": [{"type": "text", "text": "second result"}]},
             ])), \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Run the two-step test"}},
            )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["content"], "Completed using both tool results.")
        self.assertEqual(len(snapshots), 4)
        self.assertTrue(any("Agent completion check" in m.get("content", "") for m in snapshots[3] if m["role"] == "system"))
        return snapshots

    def test_history_window_counts_only_previous_messages(self):
        cases = {
            "all": [f"prior message {i}" for i in range(1, 7)],
            "0": [],
            "1": ["prior message 6"],
            "2": ["prior message 5", "prior message 6"],
            "4": [f"prior message {i}" for i in range(3, 7)],
            "6": [f"prior message {i}" for i in range(1, 7)],
            "8": [f"prior message {i}" for i in range(1, 7)],
            "10": [f"prior message {i}" for i in range(1, 7)],
        }
        for limit, expected_prior in cases.items():
            with self.subTest(history_limit=limit):
                snapshots = self._run_mocked_agent(limit)
                first_call = snapshots[0]
                self.assertEqual(
                    [m["content"] for m in first_call if m["role"] != "system"],
                    expected_prior + ["Run the two-step test"],
                )

    def test_full_current_execution_history_survives_every_iteration(self):
        for limit in ("0", "1", "2", "all"):
            with self.subTest(history_limit=limit):
                snapshots = self._run_mocked_agent(limit)
                self.assertEqual(len(snapshots), 4)
                second = snapshots[1]
                third = snapshots[2]
                # Iteration 2 receives the first assistant tool call and result.
                self.assertTrue(any(m.get("role") == "assistant" and m.get("tool_calls") for m in second))
                self.assertTrue(any(m.get("role") == "tool" and m.get("content") == "first result" for m in second))
                # Iteration 3 retains the entire first cycle and the second cycle.
                self.assertEqual(sum(1 for m in third if m.get("role") == "assistant" and m.get("tool_calls")), 2)
                tool_results = [m.get("content") for m in third if m.get("role") == "tool"]
                self.assertEqual(tool_results, ["first result", "second result"])
                self.assertEqual(
                    [m["content"] for m in third if m["role"] == "user" and m["content"] == "Run the two-step test"],
                    ["Run the two-step test"],
                )

    def test_current_user_message_is_never_dropped_with_zero_history(self):
        snapshots = self._run_mocked_agent("0")
        user_messages = [m["content"] for m in snapshots[0] if m["role"] == "user"]
        self.assertEqual(user_messages, ["Run the two-step test"])

    def test_tool_errors_also_remain_in_execution_history(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        tool_definitions = []
        snapshots = []
        responses = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                self._tool_call("error-call", "test-server__missing", {})
            ]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The tool failed.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The tool failed.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots)-1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=RuntimeError("simulated MCP failure"))), \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Trigger a failing tool"}},
            )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(len(snapshots), 3)
        self.assertTrue(any(m.get("role") == "tool" and "simulated MCP failure" in m.get("content", "") for m in snapshots[1]))
        self.assertTrue(any(m.get("role") == "system" and "Agent completion check" in m.get("content", "") for m in snapshots[2]))

    def test_plain_text_candidate_is_reviewed_and_model_can_request_another_tool(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        tool_definitions = [{
            "type": "function",
            "function": {"name": "test-server__get_profile", "description": "Get a profile", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
        }]
        snapshots = []
        responses = [
            {"message": {"role": "assistant", "content": "The identifier is REC-001. Here is JSON for the next call.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("profile-call", "test-server__get_profile", {"code": "REC-001"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The profile was retrieved.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Here is the verified profile.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]

        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]

        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(return_value={"content": [{"type": "text", "text": "{\"name\":\"North archive\",\"category\":\"Engineering\"}"}]})) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Get the record profile"}},
            )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["content"], "Here is the verified profile.")
        self.assertEqual(run_tool.await_count, 1)
        self.assertEqual(run_tool.await_args.args, ("test-server__get_profile", {"code": "REC-001"}))
        # The second model call is the explicit completion review, and it sees
        # the original plain-text candidate plus the full current transcript.
        self.assertTrue(any(m.get("role") == "assistant" and "JSON for the next call" in m.get("content", "") for m in snapshots[1]))
        self.assertTrue(any(m.get("role") == "system" and "Agent completion check" in m.get("content", "") for m in snapshots[1]))
        self.assertTrue(any(m.get("role") == "tool" and "Engineering" in m.get("content", "") for m in snapshots[2]))
        self.assertTrue(any(m.get("role") == "system" and "Agent completion check" in m.get("content", "") for m in snapshots[3]))
        completion_prompt = next(m["content"] for m in snapshots[3] if m.get("role") == "system" and "Agent completion check" in m.get("content", ""))
        self.assertIn("ALL information and completed ALL actions", completion_prompt)
        self.assertIn("Do not assume that the task is complete merely because a tool executed successfully", completion_prompt)
        self.assertIn("available tool can retrieve the missing information", completion_prompt)
        self.assertIn("native structured tool_calls", completion_prompt)


    def test_all_native_tool_calls_in_a_batch_are_executed_and_returned(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        tool_definitions = [
            {"type": "function", "function": {"name": "directory__get_record_id", "description": "Resolve record identifier", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
            {"type": "function", "function": {"name": "directory__get_record_profile", "description": "Get record profile", "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}}},
        ]
        snapshots = []
        responses = [
            # The controller must honor every native call in the model batch,
            # including calls whose arguments the model chose for this turn.
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                self._tool_call("code-1", "directory__get_record_id", {"name": "North archive"}),
                self._tool_call("profile-premature", "directory__get_record_profile", {"record_id": "INVALID-GUESS"}),
            ]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Both requested operations completed.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Both requested operations completed and their results are available.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]

        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            self.assertEqual(tools, tool_definitions)
            return responses[len(snapshots) - 1]

        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "REC-001"}]},
                 {"content": [{"type": "text", "text": "{\"name\":\"North archive\",\"category\":\"Engineering\"}"}]},
             ])) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Get North archive's record profile"}},
            )

        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["content"], "Both requested operations completed and their results are available.")
        self.assertEqual(run_tool.await_count, 2)
        self.assertEqual(run_tool.await_args_list[0].args, ("directory__get_record_id", {"name": "North archive"}))
        self.assertEqual(run_tool.await_args_list[1].args, ("directory__get_record_profile", {"record_id": "INVALID-GUESS"}))
        # Both calls from the same model response are acknowledged together,
        # executed, and their matching tool results are included on the next turn.
        first_assistant_calls = [m for m in snapshots[1] if m.get("role") == "assistant" and m.get("tool_calls")]
        self.assertEqual(len(first_assistant_calls), 1)
        self.assertEqual([c["id"] for c in first_assistant_calls[0]["tool_calls"]], ["code-1", "profile-premature"])
        self.assertTrue(any(m.get("role") == "tool" and m.get("tool_call_id") == "code-1" and m.get("content") == "REC-001" for m in snapshots[1]))
        self.assertTrue(any(m.get("role") == "tool" and m.get("tool_call_id") == "profile-premature" and "North archive" in m.get("content", "") for m in snapshots[1]))
        self.assertEqual(len(snapshots), 3)


    def test_planner_executor_verifier_architecture_completes_with_evidence(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [{
            "type": "function",
            "function": {
                "name": "records__get_record",
                "description": "Retrieve a record by identifier",
                "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]},
            },
        }]
        plan = {
            "task": "Retrieve and summarize a record",
            "objectives": [{
                "id": "O1",
                "requirement": "Retrieve the requested record",
                "acceptance_criteria": ["The returned record is available in the tool result"],
                "depends_on": [],
            }],
            "clarifications_needed": [],
        }
        verification = {
            "status": "complete",
            "objective_results": [{
                "objective_id": "O1", "status": "satisfied",
                "evidence": ["tool:record-call"],
                "remaining_gap": "",
            }],
            "missing_information": [],
            "recommended_next_steps": [],
            "clarification_question": "",
        }
        snapshots = []
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("record-call", "records__get_record", {"record_id": "R-17"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Record R-17 belongs to Alex in Engineering.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(verification), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]

        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append((copy.deepcopy(messages), copy.deepcopy(tools), dict(cfg)))
            return responses[len(snapshots) - 1]

        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(return_value={"content": [{"type": "text", "text": '{"name":"Alex","category":"Engineering"}'}]})) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Retrieve record R-17 and summarize it"}},
            )
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["completion_status"], "complete")
        self.assertEqual(payload["content"], "Record R-17 belongs to Alex in Engineering.")
        self.assertEqual(payload["task_plan"]["objectives"][0]["id"], "O1")
        self.assertEqual(run_tool.await_count, 1)
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(snapshots[0][1], [])  # Planner cannot execute tools.
        self.assertEqual(snapshots[0][2].get("_response_format"), "json")
        self.assertEqual(snapshots[1][1], tool_definitions)  # Executor receives MCP tools.
        self.assertEqual(snapshots[3][1], [])  # Verifier cannot execute tools.
        self.assertTrue(any("Agent task plan" in m.get("content", "") for m in snapshots[1][0] if m.get("role") == "system"))

    def test_verifier_corrects_one_invalid_assessment_and_then_completes(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        plan = {"task": "Answer a simple arithmetic question", "objectives": [{"id": "O1", "requirement": "Provide the arithmetic result", "acceptance_criteria": ["The result is stated correctly"], "depends_on": []}], "clarifications_needed": []}
        invalid = {"status": "done", "objective_results": []}
        corrected = {"status": "complete", "objective_results": [{"objective_id": "O1", "status": "satisfied", "evidence": ["model_knowledge"], "remaining_gap": ""}], "missing_information": [], "recommended_next_steps": [], "clarification_question": ""}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "4", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(invalid), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(corrected), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=([], []))), \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "What is 2 plus 2?"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["completion_status"], "complete")
        self.assertEqual(result.json()["content"], "4")
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(result.json()["run_budget"]["verifier_attempts"], 2)
        self.assertTrue(any(m.get("role") == "system" and "invalid status" in m.get("content", "") for m in snapshots[3]))

    def test_architecture_recovers_when_verifier_cites_successful_tool_by_name_and_continues(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [
            {"type": "function", "function": {"name": "directory__get_record_id", "description": "Find an record identifier by name", "parameters": {"type": "object", "properties": {"record_label": {"type": "string"}}, "required": ["record_label"]}}},
            {"type": "function", "function": {"name": "directory__get_record_details", "description": "Retrieve record details by code", "parameters": {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}}},
        ]
        plan = {"task": "Retrieve the named record's details", "objectives": [{"id": "O1", "requirement": "Retrieve record details for the named person", "acceptance_criteria": ["A successful record-details result is available"], "depends_on": []}], "clarifications_needed": []}
        incomplete = {"status": "incomplete", "objective_results": [{"objective_id": "O1", "status": "unsatisfied", "evidence": ["tool:directory__get_record_id"], "remaining_gap": "The record identifier was found, but record details have not yet been retrieved"}], "missing_information": ["Record details"], "recommended_next_steps": ["Use the returned record identifier to retrieve record details"], "clarification_question": ""}
        complete = {"status": "complete", "objective_results": [{"objective_id": "O1", "status": "satisfied", "evidence": ["tool:directory__get_record_id", "tool:directory__get_record_details"], "remaining_gap": ""}], "missing_information": [], "recommended_next_steps": [], "clarification_question": ""}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("code-call", "directory__get_record_id", {"record_label": "North archive"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "I found the record identifier REC-001.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(incomplete), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("details-call", "directory__get_record_details", {"record_id": "REC-001"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "North archive's record details were retrieved successfully.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(complete), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append((copy.deepcopy(messages), copy.deepcopy(tools)))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "REC-001"}]},
                 {"content": [{"type": "text", "text": '{"record_label":"North archive","category":"Engineering"}'}]},
             ])) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Get the record identifier first, then the details of record named North archive"}})
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["completion_status"], "complete")
        self.assertEqual(run_tool.await_count, 2)
        self.assertEqual(run_tool.await_args_list[0].args, ("directory__get_record_id", {"record_label": "North archive"}))
        self.assertEqual(run_tool.await_args_list[1].args, ("directory__get_record_details", {"record_id": "REC-001"}))
        self.assertEqual(len(snapshots), 7)
        verifier_messages = snapshots[3][0]
        verifier_instruction = next(m["content"] for m in verifier_messages if m.get("role") == "system" and "completion-verification phase" in m.get("content", ""))
        self.assertIn('"call_id": "code-call"', verifier_instruction)
        self.assertIn('"name": "directory__get_record_id"', verifier_instruction)
        self.assertTrue(any(m.get("role") == "tool" and m.get("tool_call_id") == "code-call" and m.get("content") == "REC-001" for m in snapshots[4][0]))

    def test_architecture_reenters_execution_when_verifier_finds_missing_work(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [{"type": "function", "function": {"name": "records__get_record", "description": "Retrieve a record", "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}}]
        plan = {"task": "Retrieve a record", "objectives": [{"id": "O1", "requirement": "Retrieve the requested record", "acceptance_criteria": ["A successful record result is available"], "depends_on": []}], "clarifications_needed": []}
        incomplete = {"status": "incomplete", "objective_results": [{"objective_id": "O1", "status": "unsatisfied", "evidence": [], "remaining_gap": "No record has been retrieved"}], "missing_information": ["The record details"], "recommended_next_steps": ["Use the available record retrieval capability"], "clarification_question": ""}
        complete = {"status": "complete", "objective_results": [{"objective_id": "O1", "status": "satisfied", "evidence": ["tool:record-after-review"], "remaining_gap": ""}], "missing_information": [], "recommended_next_steps": [], "clarification_question": ""}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "I cannot retrieve the record yet.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(incomplete), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("record-after-review", "records__get_record", {"id": "R-9"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Record R-9 was retrieved successfully.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(complete), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(return_value={"content": [{"type": "text", "text": "Record R-9 retrieved"}]})) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Retrieve record R-9"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["completion_status"], "complete")
        self.assertEqual(result.json()["content"], "Record R-9 was retrieved successfully.")
        self.assertEqual(run_tool.await_count, 1)
        self.assertEqual(len(snapshots), 6)
        self.assertTrue(any("Independent completion verification found" in m.get("content", "") for m in snapshots[3] if m.get("role") == "system"))
        self.assertTrue(any(m.get("role") == "tool" and "Record R-9 retrieved" in m.get("content", "") for m in snapshots[4]))

    def test_architecture_executes_all_native_calls_in_one_model_batch(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [
            {"type": "function", "function": {"name": "source__first", "description": "Get first source", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "source__second", "description": "Get second source", "parameters": {"type": "object", "properties": {}}}},
        ]
        plan = {"task": "Collect two source results", "objectives": [
            {"id": "O1", "requirement": "Collect the first source result", "acceptance_criteria": ["A successful first source result is available"], "depends_on": []},
            {"id": "O2", "requirement": "Collect the second source result", "acceptance_criteria": ["A successful second source result is available"], "depends_on": []},
        ], "clarifications_needed": []}
        # Simulate the common model inconsistency: all requested objectives are
        # satisfied, but the verifier labels the task incomplete due to an
        # optional follow-up suggestion. The controller should normalize this.
        verification = {"status": "incomplete", "objective_results": [
            {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:first-source"], "remaining_gap": ""},
            {"objective_id": "O2", "status": "satisfied", "evidence": ["tool:second-source"], "remaining_gap": ""},
        ], "missing_information": [], "recommended_next_steps": ["Offer additional context if requested"], "clarification_question": ""}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("first-source", "source__first", {}), self._tool_call("second-source", "source__second", {})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Both sources returned the requested information.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(verification), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "first result"}]},
                 {"content": [{"type": "text", "text": "second result"}]},
             ])) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Collect both sources"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["completion_status"], "complete")
        self.assertTrue(result.json()["completion_verification"]["normalized_from_incomplete"])
        self.assertEqual(len(snapshots), 4)  # plan + one batch + candidate + one verifier call
        self.assertEqual(run_tool.await_count, 2)
        self.assertEqual([call.args[0] for call in run_tool.await_args_list], ["source__first", "source__second"])
        self.assertTrue(any(m.get("role") == "tool" and m.get("tool_call_id") == "first-source" for m in snapshots[2]))
        self.assertTrue(any(m.get("role") == "tool" and m.get("tool_call_id") == "second-source" for m in snapshots[2]))

    def test_native_tool_call_without_function_name_is_rejected_without_blank_trace_event(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tools = [{"type": "function", "function": {"name": "records__lookup", "description": "Look up a record", "parameters": {"type": "object", "properties": {}}}}]
        plan = {"task": "Look up a record", "objectives": [{"id": "O1", "requirement": "Look up the requested record", "acceptance_criteria": ["A successful lookup result is available"], "depends_on": []}], "clarifications_needed": []}
        clarification = {"status": "needs_clarification", "objective_results": [{"objective_id": "O1", "status": "uncertain", "evidence": [], "remaining_gap": "The lookup did not execute"}], "missing_information": ["A valid lookup call"], "recommended_next_steps": [], "clarification_question": "Please retry the lookup request."}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "bad-call", "type": "function", "function": {"name": "", "arguments": {}}}]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The lookup could not be executed.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(clarification), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, offered_tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tools, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock()) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Look up this record"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["completion_status"], "needs_clarification")
        run_tool.assert_not_awaited()
        tool_events = [event for event in result.json()["events"] if event.get("type") == "tool"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["name"], "Invalid native tool call")
        self.assertEqual(tool_events[0]["status"], "error")

    def test_repeated_invalid_verifier_stops_without_reexecuting_successful_tools(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [{
            "type": "function",
            "function": {
                "name": "catalog__lookup_record",
                "description": "Look up a record by its label",
                "parameters": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
            },
        }]
        plan = {"task": "Retrieve a requested record", "objectives": [{"id": "O1", "requirement": "Retrieve the matching record", "acceptance_criteria": ["A successful lookup result is available"], "depends_on": []}], "clarifications_needed": []}
        invalid = {"status": "finished", "objective_results": []}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("lookup-1", "catalog__lookup_record", {"label": "North archive"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The North archive record was retrieved: reference R-17, status active.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            *[{"message": {"role": "assistant", "content": __import__('json').dumps(invalid), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}} for _ in range(2)],
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(return_value={"content": [{"type": "text", "text": "Record R-17: status active"}]})) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Find the North archive record"}})
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["completion_status"], "unverified")
        self.assertIn("R-17", payload["content"])
        self.assertIn("could not be completed", payload["content"])
        self.assertEqual(run_tool.await_count, 1)
        self.assertEqual(len(snapshots), 5)
        self.assertEqual(payload["run_budget"]["verifier_attempts"], 2)
        self.assertTrue(any(event.get("type") == "agent_recovery" and event.get("action") == "stop_without_reexecuting_tools" for event in payload["events"]))

    def test_prose_tool_calls_trigger_native_call_recovery_before_verification(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [
            {"type": "function", "function": {"name": "weather__today", "description": "Retrieve today's weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "market__gold_price", "description": "Retrieve current gold price", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ]
        plan = {"task": "Retrieve two current facts", "objectives": [
            {"id": "O1", "requirement": "Retrieve today's weather", "acceptance_criteria": ["A successful weather result is available"], "depends_on": []},
            {"id": "O2", "requirement": "Retrieve the current gold price", "acceptance_criteria": ["A successful gold price result is available"], "depends_on": []},
        ], "clarifications_needed": []}
        verification = {"status": "complete", "objective_results": [
            {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:weather-call"], "remaining_gap": ""},
            {"objective_id": "O2", "status": "satisfied", "evidence": ["tool:gold-call"], "remaining_gap": ""},
        ], "missing_information": [], "recommended_next_steps": [], "clarification_question": ""}
        prose = 'Here are the calls: {"name":"weather__today","parameters":{"city":"Dubai"}}; {"name":"market__gold_price","parameters":{}}'
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": prose, "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("weather-call", "weather__today", {"city": "Dubai"}), self._tool_call("gold-call", "market__gold_price", {})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Today's weather and current gold price were retrieved.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(verification), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "Weather: clear, 25 C"}]},
                 {"content": [{"type": "text", "text": "Gold: AED 250 per gram"}]},
             ])) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Get both weather and gold price today in Dubai"}})
        self.assertEqual(result.status_code, 200, result.text)
        payload = result.json()
        self.assertEqual(payload["completion_status"], "complete")
        self.assertEqual(run_tool.await_count, 2)
        self.assertEqual([call.args[0] for call in run_tool.await_args_list], ["weather__today", "market__gold_price"])
        self.assertEqual(len(snapshots), 5)
        self.assertTrue(any(event.get("type") == "agent_recovery" and event.get("reason") == "unexecuted_tool_proposal_in_assistant_text" for event in payload["events"]))
        self.assertTrue(any(m.get("role") == "system" and "native structured tool_calls" in m.get("content", "") for m in snapshots[2]))
        self.assertFalse(any(event.get("phase") == "verification" for event in payload["events"][:3] if event.get("type") == "llm"))

    def test_invalid_planner_falls_back_and_still_executes_independent_tools(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        with agent.db() as conn:
            conn.execute("UPDATE settings SET value='true' WHERE key='agent_architecture'")
        tool_definitions = [
            {"type": "function", "function": {"name": "source__lookup_primary", "description": "Retrieve a result from the primary source", "parameters": {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}}},
            {"type": "function", "function": {"name": "source__lookup_secondary", "description": "Retrieve a result from the secondary source", "parameters": {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}}},
        ]
        invalid_plan = {"objectives": []}
        verification = {"status": "complete", "objective_results": [{"objective_id": "O1", "status": "satisfied", "evidence": ["tool:primary-call", "tool:secondary-call"], "remaining_gap": ""}], "missing_information": [], "recommended_next_steps": [], "clarification_question": ""}
        responses = [
            {"message": {"role": "assistant", "content": __import__('json').dumps(invalid_plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(invalid_plan), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": [self._tool_call("primary-call", "source__lookup_primary", {"subject": "requested subject"}), self._tool_call("secondary-call", "source__lookup_secondary", {"subject": "requested subject"})]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "Both independent source results were retrieved.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": __import__('json').dumps(verification), "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]
        snapshots = []
        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append((copy.deepcopy(messages), copy.deepcopy(tools)))
            return responses[len(snapshots) - 1]
        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(side_effect=[
                 {"content": [{"type": "text", "text": "Primary source result: value A"}]},
                 {"content": [{"type": "text", "text": "Secondary source result: value B"}]},
             ])) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post("/api/chat", json={"data": {"session_id": self.session_id, "message": "Collect two independent source results for the same subject"}})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["completion_status"], "complete")
        self.assertEqual(run_tool.await_count, 2)
        self.assertEqual(len(snapshots), 5)
        self.assertEqual([call.args[0] for call in run_tool.await_args_list], ["source__lookup_primary", "source__lookup_secondary"])
        self.assertTrue(any(event.get("type") == "agent_fallback" and event.get("phase") == "planning" for event in result.json()["events"]))
        executor_messages, executor_tools = snapshots[2]
        self.assertEqual(executor_tools, tool_definitions)
        self.assertTrue(any("Collect two independent source results for the same subject" in str(m.get("content", "")) for m in executor_messages if m.get("role") == "system"))

    def test_empty_completion_review_preserves_last_nonempty_answer(self):
        self._reset_prior_history()
        self._set_history_limit("0")
        tool_definitions = [{
            "type": "function",
            "function": {
                "name": "catalog__get_current_status",
                "description": "Get the current status",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }]
        snapshots = []
        responses = [
            {"message": {"role": "assistant", "content": "", "tool_calls": [
                self._tool_call("status-call", "catalog__get_current_status", {"city": "North archive"})
            ]}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "The North archive status is active.", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
            {"message": {"role": "assistant", "content": "", "tool_calls": []}, "usage": {}, "timing": {}, "duration_ms": 1, "provider": "test", "model": "mock", "request": {}},
        ]

        async def fake_model_chat(messages, tools, cfg):
            import copy
            snapshots.append(copy.deepcopy(messages))
            return responses[len(snapshots) - 1]

        with patch.object(agent, "session_mcp_tools", new=AsyncMock(return_value=(tool_definitions, []))), \
             patch.object(agent, "run_mcp_tool", new=AsyncMock(return_value={"content": [{"type": "text", "text": "Status: active"}]})) as run_tool, \
             patch.object(agent, "model_chat", new=fake_model_chat):
            result = self.client.post(
                "/api/chat",
                json={"data": {"session_id": self.session_id, "message": "Get the current status for the North archive"}},
            )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["content"], "The North archive status is active.")
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(run_tool.await_count, 1)
        self.assertTrue(any(m.get("role") == "tool" and "Status: active" in m.get("content", "") for m in snapshots[1]))
        self.assertTrue(any(m.get("role") == "system" and "Agent completion check" in m.get("content", "") for m in snapshots[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
