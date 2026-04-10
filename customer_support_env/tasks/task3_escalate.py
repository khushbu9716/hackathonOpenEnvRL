# tasks/task3_escalate.py
# Task 3 — Multi-turn Escalation Handling (Hard)
#
# The agent handles a frustrated customer across multiple turns.
# Must de-escalate, escalate to correct department, and stay consistent.
# Max 10 steps.

import json
from pathlib import Path

from customer_support_env.env.environment import CustomerSupportEnvironment
from customer_support_env.env.graders import get_grader
from customer_support_env.models import SupportAction

# ---------------------------------------------------------------------------
# Task Config
# ---------------------------------------------------------------------------

TASK_ID = "task3_escalate"
DIFFICULTY = "hard"
MAX_STEPS = 10
PASS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a senior customer support agent for an e-commerce company.

You are handling a frustrated customer who has had previous bad experiences
with our support team. The conversation history is provided.

Your responsibilities:
1. Classify the ticket category
2. Acknowledge previous failures — never ignore prior unresolved issues
3. De-escalate the customer's frustration with empathetic language
4. Escalate to the correct department based on the issue:
   - billing_team      : duplicate charges, refund delays, billing disputes
   - logistics_team    : lost packages, delivery failures, wrong address
   - technical_team    : product compatibility, firmware, technical defects
   - legal_team        : legal threats, consumer protection mentions
   - senior_support    : abuse, repeated failures, $500+ product issues
5. Provide a concrete timeline or next step
6. Resolve the ticket

Critical rules:
  - NEVER contradict something you said in an earlier turn
  - NEVER make promises you cannot keep (e.g. "arrives tomorrow")
  - ALWAYS acknowledge if the customer mentions a legal threat
  - ALWAYS offer refund as an option for product compatibility issues

Always respond with valid JSON only. Examples:
{"action_type": "classify", "category": "billing"}
{"action_type": "respond", "message": "I sincerely apologise for the previous..."}
{"action_type": "escalate", "department": "billing_team"}
{"action_type": "resolve", "resolution_note": "Escalated to billing team."}
"""

# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_episode(agent_fn, ticket_id: str = None, verbose: bool = True) -> dict:
    """
    Run a single Task 3 episode.

    Args:
        agent_fn  : Function(system_prompt, observation_text) -> dict (raw action)
        ticket_id : Optional specific ticket ID. Random if None.
        verbose   : Print step-by-step output.

    Returns:
        dict with ticket_id, score, breakdown, feedback, steps_taken
    """
    env = CustomerSupportEnvironment()
    grader = get_grader(TASK_ID)

    obs = env.reset(task_id=TASK_ID, ticket_id=ticket_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK 3 | Ticket: {obs.ticket_id} | Difficulty: {obs.difficulty}")
        print(f"Scenario: {obs.ticket_text}")
        print(f"Conversation history turns: {len(obs.conversation_history)}")
        print(f"{'='*60}")

    classification_value = None
    escalation_department = None
    resolution_note = None
    classified = False
    responded = False
    steps_taken = 0

    for step in range(MAX_STEPS):
        if obs.done:
            break

        observation_text = _format_observation(obs)
        raw_action = agent_fn(SYSTEM_PROMPT, observation_text)

        try:
            action = SupportAction(**raw_action)
        except Exception as e:
            if verbose:
                print(f"Step {step+1}: Invalid action — {e}")
            continue

        if action.action_type == "classify":
            classification_value = action.category
            classified = True
        elif action.action_type == "respond":
            responded = True
        elif action.action_type == "escalate":
            escalation_department = action.department
        elif action.action_type == "resolve":
            resolution_note = action.resolution_note

        obs = env.step(action)
        steps_taken += 1

        if verbose:
            print(f"Step {step+1}: {action.action_type} "
                  f"| feedback: {obs.last_action_feedback} "
                  f"| reward: {obs.reward}")

    # Grade the episode
    ticket_data = _load_ticket(TASK_ID, obs.ticket_id)
    result = grader.grade(
        ticket=ticket_data,
        escalation_department=escalation_department,
        conversation_history=obs.conversation_history,
        resolution_note=resolution_note,
        classified=classified,
        responded=responded,
    )

    if verbose:
        print(f"\nFINAL SCORE: {result.score} | Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")
        print(f"Breakdown: {result.breakdown}")

    return {
        "task_id": TASK_ID,
        "ticket_id": obs.ticket_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
        "passed": result.passed,
        "steps_taken": steps_taken,
        "escalation_department": escalation_department,
    }


def run_all(agent_fn, verbose: bool = True) -> list:
    """Run all Task 3 tickets and return scores."""
    tickets_path = Path(__file__).parent.parent / "data" / "tickets.json"
    with open(tickets_path) as f:
        all_tickets = json.load(f)

    results = []
    for ticket in all_tickets[TASK_ID]["tickets"]:
        result = run_episode(agent_fn, ticket_id=ticket["ticket_id"], verbose=verbose)
        results.append(result)

    avg_score = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"\nTASK 3 AVERAGE SCORE: {avg_score} over {len(results)} tickets")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_observation(obs) -> str:
    lines = [
        f"Ticket ID: {obs.ticket_id}",
        f"Customer: {obs.customer_name}",
    ]
    if obs.order_id:
        lines.append(f"Order ID: {obs.order_id}")
    lines.append(f"Issue: {obs.ticket_text}")

    if obs.conversation_history:
        lines.append("\n--- Conversation History ---")
        for turn in obs.conversation_history:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("--- End of History ---\n")

    lines.append(f"Available actions: {obs.available_actions}")
    if obs.last_action_feedback:
        lines.append(f"Last feedback: {obs.last_action_feedback}")
    return "\n".join(lines)


def _load_ticket(task_id: str, ticket_id: str) -> dict:
    tickets_path = Path(__file__).parent.parent / "data" / "tickets.json"
    with open(tickets_path) as f:
        all_tickets = json.load(f)
    tickets = all_tickets[task_id]["tickets"]
    return next(t for t in tickets if t["ticket_id"] == ticket_id)
