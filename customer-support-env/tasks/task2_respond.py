# tasks/task2_respond.py
# Task 2 — Response Generation (Medium)
#
# The agent must classify the ticket AND draft an appropriate
# customer response that is accurate, complete, and empathetic.
# Max 5 steps.

import json
from pathlib import Path

from env.environment import CustomerSupportEnvironment
from env.graders import get_grader
from models import SupportAction

# ---------------------------------------------------------------------------
# Task Config
# ---------------------------------------------------------------------------

TASK_ID = "task2_respond"
DIFFICULTY = "medium"
MAX_STEPS = 5
PASS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce company.

You must:
1. Classify the ticket into the correct category
2. Draft a professional, empathetic response to the customer
3. Resolve the ticket

Category options:
  - billing, delivery, refund, product_issue, general_inquiry

Company policies:
  - Refund window: 30 days from delivery
  - Standard delivery: 5 business days
  - Express delivery: 2 business days
  - Damaged item: entitled to full refund or replacement within 30 days
  - Billing disputes: refunded within 3-5 business days after verification
  - Returns: refunds processed within 7 business days of return receipt

Response guidelines:
  - Always acknowledge the customer's frustration
  - Reference the ticket ID or order ID in your response
  - Provide a clear timeline or next steps
  - Never say "we cannot help" or "contact your bank"
  - Keep tone empathetic and professional

Always respond with valid JSON only. Examples:
{"action_type": "classify", "category": "billing"}
{"action_type": "respond", "message": "Dear [name], I'm sorry to hear about..."}
{"action_type": "resolve", "resolution_note": "Responded and resolved."}
"""

# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_episode(agent_fn, ticket_id: str = None, verbose: bool = True) -> dict:
    """
    Run a single Task 2 episode.

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
        print(f"TASK 2 | Ticket: {obs.ticket_id} | Difficulty: {obs.difficulty}")
        print(f"Ticket: {obs.ticket_text}")
        print(f"{'='*60}")

    classification_value = None
    response_text = None
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
        elif action.action_type == "respond":
            response_text = action.message

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
        classification_value=classification_value,
        response_text=response_text,
    )

    if verbose:
        print(f"\nFINAL SCORE: {result.score} | Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")
        if response_text:
            print(f"Agent response: {response_text[:200]}...")

    return {
        "task_id": TASK_ID,
        "ticket_id": obs.ticket_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
        "passed": result.passed,
        "steps_taken": steps_taken,
        "response_text": response_text,
    }


def run_all(agent_fn, verbose: bool = True) -> list:
    """Run all Task 2 tickets and return scores."""
    tickets_path = Path(__file__).parent.parent / "data" / "tickets.json"
    with open(tickets_path) as f:
        all_tickets = json.load(f)

    results = []
    for ticket in all_tickets[TASK_ID]["tickets"]:
        result = run_episode(agent_fn, ticket_id=ticket["ticket_id"], verbose=verbose)
        results.append(result)

    avg_score = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"\nTASK 2 AVERAGE SCORE: {avg_score} over {len(results)} tickets")
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
    lines.append(f"Message: {obs.ticket_text}")
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
