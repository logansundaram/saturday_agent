PLAN_SYSTEM_PROMPT = "Create a short execution plan. Keep it practical and numbered."
PLAN_USER_PROMPT = "Task: {task}\nContext: {context}"

EXECUTE_SYSTEM_PROMPT = "Use the plan to answer the task accurately. Plan:\n{plan}"

VERIFY_SYSTEM_PROMPT = (
    "You are a strict verifier. Respond with 'OK: ...' or 'FAIL: ...' and explain briefly."
)
VERIFY_USER_PROMPT = "Task: {task}\nPlan: {plan}\nAnswer: {answer}"
