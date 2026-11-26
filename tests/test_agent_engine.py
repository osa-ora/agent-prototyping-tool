"""Structural validation tests for model-authored plans and completion reviews."""
import unittest

from app.agent_engine import AgentRunBudget, fallback_plan, validate_plan, validate_verification, validate_tool_arguments, contains_unexecuted_tool_proposal


class AgentEngineTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "task": "Retrieve and summarize a requested record",
            "objectives": [
                {
                    "id": "O1",
                    "requirement": "Retrieve the requested record",
                    "acceptance_criteria": ["The record fields are present in a successful tool result"],
                    "depends_on": [],
                },
                {
                    "id": "O2",
                    "requirement": "Summarize the record accurately",
                    "acceptance_criteria": ["The answer reflects the retrieved fields"],
                    "depends_on": ["O1"],
                },
            ],
            "clarifications_needed": [],
        }

    def test_global_llm_budget_stops_phase_handoffs(self):
        budget = AgentRunBudget(max_llm_calls=3, max_execution_iterations=10)
        budget.consume("llm_calls")  # planning
        budget.consume("llm_calls")  # execution
        budget.consume("llm_calls")  # verification
        with self.assertRaisesRegex(RuntimeError, "llm calls budget exhausted"):
            budget.consume("llm_calls")

    def test_execution_iteration_budget_is_hard(self):
        budget = AgentRunBudget(max_execution_iterations=2)
        budget.consume("execution_iterations")
        budget.consume("execution_iterations")
        with self.assertRaisesRegex(RuntimeError, "execution iterations budget exhausted"):
            budget.consume("execution_iterations")

    def test_tool_call_budget_is_hard(self):
        budget = AgentRunBudget(max_tool_calls=1)
        budget.consume("tool_calls")
        with self.assertRaisesRegex(RuntimeError, "tool calls budget exhausted"):
            budget.consume("tool_calls")

    def test_consecutive_tool_errors_stop_but_success_resets_counter(self):
        budget = AgentRunBudget(max_consecutive_tool_errors=2)
        budget.record_tool_result(False)
        budget.record_tool_result(True)
        budget.record_tool_result(False)
        with self.assertRaisesRegex(RuntimeError, "consecutive tool failures"):
            budget.record_tool_result(False)

    def test_repeated_execution_verification_handoffs_are_bounded(self):
        budget = AgentRunBudget(max_llm_calls=20, max_verification_cycles=2)
        # Simulate the controller returning from verification to execution.
        for _ in range(2):
            budget.consume("execution_iterations")
            budget.consume("llm_calls")
            budget.consume("verification_cycles")
            budget.consume("llm_calls")
        with self.assertRaisesRegex(RuntimeError, "verification cycles budget exhausted"):
            budget.consume("verification_cycles")

    def test_verification_cycle_budget_is_hard(self):
        budget = AgentRunBudget(max_verification_cycles=1)
        budget.consume("verification_cycles")
        with self.assertRaisesRegex(RuntimeError, "verification cycles budget exhausted"):
            budget.consume("verification_cycles")

    def test_fallback_plan_preserves_original_request_without_domain_assumptions(self):
        request = "Get the current status from two independent sources"
        result = fallback_plan(request, "planner omitted the task")
        self.assertEqual(result["task"], request)
        self.assertEqual(result["objectives"][0]["requirement"], request)
        self.assertEqual(result["objectives"][0]["depends_on"], [])
        self.assertIn("available conversation and execution evidence", result["objectives"][0]["acceptance_criteria"][0])

    def test_plan_is_validated_and_dependencies_are_preserved(self):
        result = validate_plan(self.plan)
        self.assertEqual(result["objectives"][1]["depends_on"], ["O1"])

    def test_plan_rejects_duplicate_objective_ids(self):
        broken = dict(self.plan)
        broken["objectives"] = [self.plan["objectives"][0], dict(self.plan["objectives"][0])]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_plan(broken)

    def test_plan_rejects_dependency_cycles(self):
        broken = dict(self.plan)
        broken["objectives"] = [
            dict(self.plan["objectives"][0], depends_on=["O2"]),
            dict(self.plan["objectives"][1], depends_on=["O1"]),
        ]
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            validate_plan(broken)

    def test_plan_rejects_unknown_dependencies(self):
        broken = dict(self.plan)
        broken["objectives"] = [dict(self.plan["objectives"][0], depends_on=["MISSING"])]
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            validate_plan(broken)

    def test_verifier_must_assess_every_objective(self):
        result = {
            "status": "complete",
            "objective_results": [{"objective_id": "O1", "status": "satisfied", "evidence": ["tool:call-1"], "remaining_gap": ""}],
            "missing_information": [],
            "recommended_next_steps": [],
        }
        with self.assertRaisesRegex(ValueError, "omitted objectives"):
            validate_verification(result, self.plan, {"call-1"})

    def test_verifier_cannot_claim_completion_without_evidence(self):
        result = {
            "status": "complete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:call-1"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "satisfied", "evidence": [], "remaining_gap": ""},
            ],
            "missing_information": [],
            "recommended_next_steps": [],
        }
        with self.assertRaisesRegex(ValueError, "supporting evidence"):
            validate_verification(result, self.plan, {"call-1"})

    def test_verifier_cannot_complete_with_missing_information(self):
        result = {
            "status": "complete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:call-1"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "satisfied", "evidence": ["user_request"], "remaining_gap": ""},
            ],
            "missing_information": ["A requirement was omitted from the plan"],
            "recommended_next_steps": [],
        }
        with self.assertRaisesRegex(ValueError, "missing information"):
            validate_verification(result, self.plan, {"call-1"})

    def test_verifier_cannot_cite_a_tool_call_that_did_not_succeed(self):
        result = {
            "status": "complete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:not-real"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "satisfied", "evidence": ["user_request"], "remaining_gap": ""},
            ],
            "missing_information": [],
            "recommended_next_steps": [],
        }
        with self.assertRaisesRegex(ValueError, "not successfully executed"):
            validate_verification(result, self.plan, {"call-1"})

    def test_verifier_accepts_exact_tool_name_when_it_maps_to_one_successful_call(self):
        result = {
            "status": "incomplete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:directory__get_record_id"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "uncertain", "evidence": [], "remaining_gap": "Record details have not been retrieved"},
            ],
            "missing_information": [],
            "recommended_next_steps": ["Retrieve the record details using the returned code"],
        }
        checked = validate_verification(
            result, self.plan, {"call-1"}, successful_tool_names={"directory__get_record_id": ["call-1"]}
        )
        self.assertEqual(checked["status"], "incomplete")

    def test_verifier_rejects_ambiguous_tool_name_when_tool_was_called_more_than_once(self):
        result = {
            "status": "incomplete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:directory__get_record_id"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "uncertain", "evidence": [], "remaining_gap": "Details missing"},
            ],
            "missing_information": [],
            "recommended_next_steps": ["Retrieve details"],
        }
        with self.assertRaisesRegex(ValueError, "ambiguous tool-name"):
            validate_verification(
                result, self.plan, {"call-1", "call-2"},
                successful_tool_names={"directory__get_record_id": ["call-1", "call-2"]}
            )

    def test_tool_arguments_are_checked_against_advertised_schema(self):
        schema = {
            "type": "object",
            "properties": {"record_id": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["record_id"],
        }
        validate_tool_arguments({"record_id": "R-17", "limit": 3}, schema)
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_tool_arguments({"limit": 3}, schema)
        with self.assertRaisesRegex(ValueError, "type string"):
            validate_tool_arguments({"record_id": 17}, schema)

    def test_optional_follow_up_does_not_block_completion_when_all_objectives_are_satisfied(self):
        result = {
            "status": "incomplete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:call-1"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "satisfied", "evidence": ["tool:call-2"], "remaining_gap": ""},
            ],
            "missing_information": [],
            "recommended_next_steps": ["Offer an optional forecast"],
        }
        checked = validate_verification(result, self.plan, {"call-1", "call-2"})
        self.assertEqual(checked["status"], "complete")
        self.assertTrue(checked["normalized_from_incomplete"])
        self.assertEqual(checked["recommended_next_steps"], [])

    def test_incomplete_verification_is_preserved(self):
        result = {
            "status": "incomplete",
            "objective_results": [
                {"objective_id": "O1", "status": "satisfied", "evidence": ["tool:call-1"], "remaining_gap": ""},
                {"objective_id": "O2", "status": "uncertain", "evidence": [], "remaining_gap": "The summary needs checking"},
            ],
            "missing_information": ["A required field is missing"],
            "recommended_next_steps": ["Retrieve the missing field"],
        }
        checked = validate_verification(result, self.plan, {"call-1"})
        self.assertEqual(checked["status"], "incomplete")
        self.assertEqual(checked["objective_results"][1]["status"], "uncertain")

    def test_plain_text_tool_proposal_is_detected_but_never_interpreted_as_a_call(self):
        tools = [{"type": "function", "function": {"name": "server__lookup", "parameters": {"type": "object"}}}]
        self.assertTrue(contains_unexecuted_tool_proposal(
            '{"name":"server__lookup","parameters":{}}', tools
        ))
        self.assertFalse(contains_unexecuted_tool_proposal(
            'The lookup completed successfully.', tools
        ))
        self.assertFalse(contains_unexecuted_tool_proposal(
            '{"name":"other__lookup","parameters":{}}', tools
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
