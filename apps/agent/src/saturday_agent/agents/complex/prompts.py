PLAN_SYSTEM_PROMPT = "Draft a robust plan for a complex task. Include constraints and risks."
PLAN_USER_PROMPT = "Task: {task}\nContext: {context}"

SYNTHESIZE_SYSTEM_PROMPT = (
    "Use the plan and any tool outputs to generate a complete answer."
    "\nPlan:\n{plan}\nTool Results:\n{tool_results}"
)

VERIFY_SYSTEM_PROMPT = (
    "You are a strict quality gate. Respond with 'OK: ...' or 'FAIL: ...'."
    " Check completeness, correctness, and constraints."
)
VERIFY_USER_PROMPT = "Task: {task}\nPlan: {plan}\nAnswer: {answer}\nTool Results: {tool_results}"
