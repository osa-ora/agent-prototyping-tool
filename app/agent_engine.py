"""Generic planning and completion-verification primitives for the agent runtime.

This module deliberately contains no MCP names, task-specific workflows, or
argument heuristics. The model owns task interpretation and planning; this
module validates the structure of the plan and verification decision.
"""
from __future__ import annotations

import json
from typing import Any


PLAN_SCHEMA_DESCRIPTION = {
    "task": "A concise restatement of the user's requested outcome.",
    "objectives": [
        {
            "id": "O1",
            "requirement": "A specific requirement that must be satisfied.",
            "acceptance_criteria": ["Observable condition for considering this requirement satisfied."],
            "depends_on": [],
        }
    ],
    "clarifications_needed": [],
}

VERIFICATION_SCHEMA_DESCRIPTION = {
    "status": "complete | incomplete | needs_clarification",
    "objective_results": [
        {
            "objective_id": "O1",
            "status": "satisfied | unsatisfied | uncertain",
            "evidence": ["tool:<successful_tool_call_id>", "user_request", "model_knowledge", "message:<zero_based_transcript_index>"],
            "remaining_gap": "Empty when satisfied; otherwise explain what is missing.",
        }
    ],
    "missing_information": [],
    "recommended_next_steps": [],
    "clarification_question": "A concise question, or an empty string.",
}




class AgentBudgetExceeded(RuntimeError):
    """Raised when a hard per-run safety budget is exhausted."""


class AgentRunBudget:
    """Hard controller-enforced limits shared by every phase of one agent run."""
    def __init__(self, *, max_llm_calls: int = 30, max_execution_iterations: int = 12,
                 max_tool_calls: int = 20, max_verification_cycles: int = 2,
                 max_verifier_attempts: int = 4, max_consecutive_tool_errors: int = 4):
        self.limits = {
            "llm_calls": max_llm_calls,
            "execution_iterations": max_execution_iterations,
            "tool_calls": max_tool_calls,
            "verification_cycles": max_verification_cycles,
            "verifier_attempts": max_verifier_attempts,
            "consecutive_tool_errors": max_consecutive_tool_errors,
        }
        self.used = {key: 0 for key in self.limits}
        self.consecutive_tool_errors = 0

    def consume(self, category: str, amount: int = 1) -> None:
        if category not in self.limits or amount < 0:
            raise ValueError("Unknown agent budget category or invalid amount.")
        if self.used[category] + amount > self.limits[category]:
            raise AgentBudgetExceeded(
                f"Agent run stopped: {category.replace('_', ' ')} budget exhausted "
                f"({self.used[category]}/{self.limits[category]} used)."
            )
        self.used[category] += amount

    def record_tool_result(self, succeeded: bool) -> None:
        if succeeded:
            self.consecutive_tool_errors = 0
        else:
            self.consecutive_tool_errors += 1
            if self.consecutive_tool_errors >= self.limits["consecutive_tool_errors"]:
                raise AgentBudgetExceeded(
                    "Agent run stopped after "
                    f"{self.consecutive_tool_errors} consecutive tool failures."
                )

    def snapshot(self) -> dict[str, Any]:
        return {**self.used, "limits": dict(self.limits),
                "consecutive_tool_errors": self.consecutive_tool_errors}


def _json_object(content: Any, label: str) -> dict[str, Any]:
    if isinstance(content, dict):
        value = content
    else:
        raw = str(content or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw[:4].lower() == "json":
                raw = raw[4:].strip()
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The {label} response was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"The {label} response must be a JSON object.")
    return value



def validate_tool_arguments(arguments: Any, schema: dict[str, Any]) -> None:
    """Validate model-generated arguments against the advertised JSON Schema.

    This is deliberately domain-agnostic: it validates protocol shape only and
    never infers tool prerequisites or rewrites argument values.
    """
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    def check(value: Any, spec: Any, path: str) -> None:
        if not isinstance(spec, dict):
            return
        expected = spec.get("type")
        type_map = {
            "object": dict, "array": list, "string": str, "integer": int,
            "number": (int, float), "boolean": bool, "null": type(None),
        }
        if expected in type_map:
            pytype = type_map[expected]
            if expected == "integer" and isinstance(value, bool):
                valid = False
            elif expected == "number" and isinstance(value, bool):
                valid = False
            else:
                valid = isinstance(value, pytype)
            if not valid:
                raise ValueError(f"{path} must be of type {expected}.")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"{path} must be one of the allowed enum values.")
        if isinstance(value, dict):
            required = spec.get("required", [])
            if isinstance(required, list):
                missing = [key for key in required if key not in value]
                if missing:
                    raise ValueError(f"{path} is missing required argument(s): {', '.join(missing)}.")
            properties = spec.get("properties", {})
            if isinstance(properties, dict):
                for key, child in value.items():
                    if key in properties:
                        check(child, properties[key], f"{path}.{key}")
                    elif spec.get("additionalProperties") is False:
                        raise ValueError(f"{path} contains unsupported argument: {key}.")
            if isinstance(spec.get("minProperties"), int) and len(value) < spec["minProperties"]:
                raise ValueError(f"{path} has too few properties.")
        if isinstance(value, list):
            if isinstance(spec.get("minItems"), int) and len(value) < spec["minItems"]:
                raise ValueError(f"{path} has too few items.")
            if isinstance(spec.get("maxItems"), int) and len(value) > spec["maxItems"]:
                raise ValueError(f"{path} has too many items.")
            item_schema = spec.get("items")
            if isinstance(item_schema, dict):
                for index, child in enumerate(value):
                    check(child, item_schema, f"{path}[{index}]")
        if isinstance(value, str):
            if isinstance(spec.get("minLength"), int) and len(value) < spec["minLength"]:
                raise ValueError(f"{path} is shorter than the minimum length.")
            if isinstance(spec.get("maxLength"), int) and len(value) > spec["maxLength"]:
                raise ValueError(f"{path} exceeds the maximum length.")

    check(arguments, schema or {"type": "object"}, "arguments")


def validate_plan(content: Any) -> dict[str, Any]:
    """Validate a model-authored task plan without inventing or rewriting it."""
    plan = _json_object(content, "task planner")
    task = plan.get("task")
    objectives = plan.get("objectives")
    clarifications = plan.get("clarifications_needed", [])
    if not isinstance(task, str) or not task.strip():
        raise ValueError("The task planner did not provide a task description.")
    if not isinstance(objectives, list) or not objectives:
        raise ValueError("The task planner must identify at least one objective.")
    if not isinstance(clarifications, list) or any(not isinstance(x, str) for x in clarifications):
        raise ValueError("clarifications_needed must be a list of strings.")
    seen: set[str] = set()
    normalized = []
    for item in objectives:
        if not isinstance(item, dict):
            raise ValueError("Every task objective must be an object.")
        oid = item.get("id")
        requirement = item.get("requirement")
        criteria = item.get("acceptance_criteria")
        dependencies = item.get("depends_on", [])
        if not isinstance(oid, str) or not oid.strip() or oid in seen:
            raise ValueError("Objective IDs must be non-empty and unique.")
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError(f"Objective {oid} has no requirement description.")
        if not isinstance(criteria, list) or not criteria or any(not isinstance(x, str) or not x.strip() for x in criteria):
            raise ValueError(f"Objective {oid} must have non-empty acceptance criteria.")
        if not isinstance(dependencies, list) or any(not isinstance(x, str) for x in dependencies):
            raise ValueError(f"Objective {oid} has invalid dependencies.")
        seen.add(oid)
        normalized.append({
            "id": oid,
            "requirement": requirement.strip(),
            "acceptance_criteria": [x.strip() for x in criteria],
            "depends_on": dependencies,
        })
    dependency_map = {item["id"]: set(item["depends_on"]) for item in normalized}
    for item in normalized:
        unknown = dependency_map[item["id"]] - seen
        if unknown:
            raise ValueError(f"Objective {item['id']} references unknown dependencies: {', '.join(sorted(unknown))}")
        if item["id"] in dependency_map[item["id"]]:
            raise ValueError(f"Objective {item['id']} cannot depend on itself.")
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(objective_id: str) -> None:
        if objective_id in visiting:
            raise ValueError("The task plan contains a dependency cycle.")
        if objective_id in visited:
            return
        visiting.add(objective_id)
        for dependency in dependency_map[objective_id]:
            visit(dependency)
        visiting.remove(objective_id)
        visited.add(objective_id)
    for objective_id in dependency_map:
        visit(objective_id)
    return {
        "task": task.strip(),
        "objectives": normalized,
        "clarifications_needed": [x.strip() for x in clarifications if x.strip()],
    }



def fallback_plan(user_request: str, reason: str) -> dict[str, Any]:
    """Build a minimal, domain-neutral task contract from the user's exact request.

    This is a resilience fallback, not a task interpretation engine: it does not
    choose tools, infer prerequisites, or invent additional requirements.
    """
    request = str(user_request or "").strip()
    if not request:
        raise ValueError("Cannot construct a fallback plan without the original user request.")
    return {
        "task": request,
        "objectives": [{
            "id": "O1",
            "requirement": request,
            "acceptance_criteria": [
                "Address the user's requested outcome using the available conversation and execution evidence."
            ],
            "depends_on": [],
        }],
        "clarifications_needed": [],
        "fallback_reason": str(reason or "The model planner did not produce a valid plan."),
    }

def validate_verification(content: Any, plan: dict[str, Any], successful_tool_call_ids: set[str] | None = None, message_count: int | None = None, successful_tool_names: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Validate the verifier's assessment against the exact planned objectives."""
    result = _json_object(content, "completion verifier")
    status = result.get("status")
    if status not in {"complete", "incomplete", "needs_clarification"}:
        raise ValueError("The completion verifier returned an invalid status.")
    rows = result.get("objective_results")
    if not isinstance(rows, list):
        raise ValueError("The completion verifier must assess every objective.")
    expected = {x["id"] for x in plan["objectives"]}
    successful_tool_call_ids = successful_tool_call_ids or set()
    successful_tool_names = successful_tool_names or {}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("objective_id") not in expected:
            raise ValueError("The completion verifier referenced an unknown objective.")
        oid = row["objective_id"]
        if oid in observed:
            raise ValueError(f"The completion verifier assessed objective {oid} more than once.")
        if row.get("status") not in {"satisfied", "unsatisfied", "uncertain"}:
            raise ValueError(f"The completion verifier returned an invalid status for {oid}.")
        evidence = row.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(x, str) for x in evidence):
            raise ValueError(f"The completion verifier returned invalid evidence for {oid}.")
        for reference in evidence:
            if reference in {"user_request", "model_knowledge"}:
                continue
            if reference.startswith("tool:"):
                call_id = reference[5:]
                if call_id in successful_tool_call_ids:
                    continue
                # Some models confuse the requested call ID with the function
                # name. Accept an exact function-name alias only when it maps
                # to one successful invocation; never guess between repeats.
                matching_calls = successful_tool_names.get(call_id, [])
                if len(matching_calls) == 1 and matching_calls[0] in successful_tool_call_ids:
                    continue
                if len(matching_calls) > 1:
                    raise ValueError(f"Objective {oid} uses an ambiguous tool-name evidence reference: {reference}. Cite one exact tool call ID instead.")
                raise ValueError(f"Objective {oid} cites a tool result that was not successfully executed: {reference}.")
            if reference.startswith("message:"):
                try:
                    message_index = int(reference[8:])
                except ValueError as exc:
                    raise ValueError(f"Objective {oid} contains an invalid message evidence reference: {reference}.") from exc
                if message_index < 0 or (message_count is not None and message_index >= message_count):
                    raise ValueError(f"Objective {oid} cites a transcript message that does not exist: {reference}.")
                continue
            raise ValueError(f"Objective {oid} contains an unsupported evidence reference: {reference}.")
        observed[oid] = {
            "objective_id": oid,
            "status": row["status"],
            "evidence": evidence,
            "remaining_gap": str(row.get("remaining_gap") or "").strip(),
        }
    missing = expected - set(observed)
    if missing:
        raise ValueError("The completion verifier omitted objectives: " + ", ".join(sorted(missing)))
    if status == "complete" and any(x["status"] != "satisfied" or not x["evidence"] for x in observed.values()):
        raise ValueError("The verifier cannot declare completion unless every objective is satisfied with supporting evidence.")
    missing_information = result.get("missing_information", [])
    next_steps = result.get("recommended_next_steps", [])
    if not isinstance(missing_information, list) or any(not isinstance(x, str) for x in missing_information):
        raise ValueError("missing_information must be a list of strings.")
    if not isinstance(next_steps, list) or any(not isinstance(x, str) for x in next_steps):
        raise ValueError("recommended_next_steps must be a list of strings.")
    if status == "complete" and (missing_information or next_steps):
        raise ValueError("The verifier cannot declare completion while missing information or recommended next steps remain.")
    clarification_question = str(result.get("clarification_question") or "").strip()
    if status == "needs_clarification" and not clarification_question:
        raise ValueError("The verifier must provide a clarification question when requesting clarification.")
    # Models sometimes label a task incomplete solely because they suggest optional
    # follow-up work, despite marking every required objective satisfied with valid
    # evidence. Optional suggestions do not invalidate completion of the request.
    normalized_from_incomplete = False
    if (status == "incomplete" and not missing_information
            and observed and all(x["status"] == "satisfied" and x["evidence"] for x in observed.values())):
        status = "complete"
        next_steps = []  # discard only optional suggestions after all required work is evidenced
        normalized_from_incomplete = True
    return {
        "status": status,
        "normalized_from_incomplete": normalized_from_incomplete,
        "objective_results": [observed[x["id"]] for x in plan["objectives"]],
        "missing_information": missing_information,
        "recommended_next_steps": next_steps,
        "clarification_question": clarification_question,
    }


def contains_unexecuted_tool_proposal(content: Any, tools: list[dict[str, Any]]) -> bool:
    """Detect, but never execute, tool-call-shaped prose naming an available tool.

    This protects the native tool-call protocol: JSON-looking assistant text is
    only a signal to ask the model to retry with structured tool_calls.
    """
    text = str(content or "")
    if not text or not any(token in text for token in ("{", "parameters", "arguments")):
        return False
    names = [
        str((item.get("function") or {}).get("name") or "")
        for item in tools
    ]
    return any(name and name in text for name in names)


def tool_catalog_for_planner(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Provide capability context without enabling tool execution in the planner phase."""
    catalog = []
    for tool in tools:
        function = tool.get("function") or {}
        catalog.append({
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters", {}),
        })
    return catalog


def planning_prompt(tools: list[dict[str, Any]]) -> str:
    catalog = tool_catalog_for_planner(tools)
    return (
        "You are the task-planning phase of a general-purpose AI agent. Analyze the user's actual request, "
        "including relevant conversation context and all selected instructions. Decompose it into the "
        "smallest useful set of independently assessable objectives. Do not assume any domain-specific "
        "workflow. Use the available capability catalog only to understand what information or actions "
        "may be obtainable; do not execute or simulate tools. Include dependencies only when genuinely "
        "required. Include clarification needs if essential input is missing. Return ONLY one JSON object "
        "matching this shape (no Markdown): " + json.dumps(PLAN_SCHEMA_DESCRIPTION, ensure_ascii=False) +
        "\nAvailable capabilities (descriptions and schemas; not executable in this phase): " +
        json.dumps(catalog, ensure_ascii=False)
    )


def execution_context(plan: dict[str, Any]) -> str:
    return (
        "Agent task plan (working task contract; it may be a minimal fallback derived from the original request):\n" +
        json.dumps(plan, ensure_ascii=False, indent=2) +
        "\nWork toward every objective and its acceptance criteria. You remain responsible for choosing "
        "tools, arguments, action order, and whether further work is needed. Use only native structured "
        "tool_calls for tools. Never claim a requirement is satisfied without support in the conversation "
        "or tool results. If the plan missed part of the user's request, address that gap rather than "
        "blindly following an incomplete plan. Before presenting a final answer, ensure the verifier can "
        "assess every objective as satisfied with specific evidence."
    )


def verification_prompt(plan: dict[str, Any], candidate: str, successful_tool_calls: list[dict[str, str]] | None = None) -> str:
    successful_tool_calls = successful_tool_calls or []
    return (
        "You are a separate completion-verification phase for a general-purpose AI agent. Compare the original "
        "user request, the full execution transcript, actual tool results and the candidate answer against "
        "the model-generated task plan below. Do not trust the candidate answer merely because it sounds "
        "confident. Every objective must be assessed separately, and every satisfied objective must cite "
        "specific evidence present in the transcript (message content or tool result). Detect omitted "
        "requirements, unsupported factual claims, unresolved dependencies, failed actions, stale or "
        "missing data, and claims that a tool ran when no successful result exists. If more work is possible "
        "with available tools, mark incomplete and recommend concrete next steps. If essential information "
        "is missing and cannot be obtained, use needs_clarification. Compare the original user request "
        "directly with the plan and identify any request requirement omitted from the plan as missing_information. "
        "Declare complete when every required objective is satisfied with valid evidence and no required "
        "information is missing. Do not mark the task incomplete merely because optional follow-up ideas, "
        "offers, or unrelated future work could be done; those are not part of the user's request. Keep "
        "recommended_next_steps empty when the requested task is complete. Evidence must use explicit references: "
        "user_request for facts directly supplied by the user; tool:<tool_call_id> for a successful tool "
        "result (never an error); message:<zero_based_index> for an existing transcript message; or "
        "model_knowledge only for stable general knowledge that does not require current or private external "
        "data. Never invent a tool ID or message index. For tool evidence, use the exact call_id from the "
        "successful tool evidence catalog below, not the function name. A function name is shown only to "
        "help identify what each call did. Return ONLY JSON matching this shape: " +
        json.dumps(VERIFICATION_SCHEMA_DESCRIPTION, ensure_ascii=False) +
        "\nSuccessful tool evidence catalog (only these calls may be cited):\n" +
        json.dumps(successful_tool_calls, ensure_ascii=False, indent=2) +
        "\nTask plan:\n" + json.dumps(plan, ensure_ascii=False, indent=2) +
        "\nCandidate answer:\n" + str(candidate or "")
    )
