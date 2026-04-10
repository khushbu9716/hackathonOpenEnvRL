# tasks/task1_classify.py
# Task 1 — Ticket Classification (Easy)
#
# The agent reads a support ticket and classifies it into the correct category.
# No response required. Max 3 steps.

import json
from pathlib import Path

from customer_support_env.env.environment import CustomerSupportEnvironment
from customer_support_env.env.graders import get_grader
from customer_support_env.models import SupportAction

# ---------------------------------------------------------------------------
# Task Config
# ---------------------------------------------------------------------------

TASK_ID = "task1_classify"
DIFFICULTY = "easy"
MAX_STEPS = 3
PASS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a customer support agent for an e-commerce company.

Your ONLY job is to read the support ticket and classify it into exactly
one of these categories:

  - billing         : payment issues, duplicate charges, invoices
  - delivery        : shipping delays, lost packages, tracking issues
  - refund          : refund requests, return status, money not received
  - product_issue   : broken items, defective products, technical faults
  - general_inquiry : questions about policies, discounts, general info

Steps:
1. Call classify with the correct category
2. Call resolve to close the ticket

Always respond with valid JSON only. Examples:
{"action_type": "classify", "category": "billing"}
{"action_type": "resolve", "resolution_note": "Ticket classified."}
"""

# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_episode(agent_fn, ticket_id: str = None, verbose: bool = True) -> dict:
    """
    Run a single Task 1 episode.

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
        print(f"TASK 1 | Ticket: {obs.ticket_id} | Difficulty: {obs.difficulty}")
        print(f"Ticket: {obs.ticket_text}")
        print(f"{'='*60}")

    classify_attempt_count = 0
    classification_value = None
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
            classify_attempt_count += 1
            classification_value = action.category

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
        classify_attempt_count=classify_attempt_count if classify_attempt_count > 0 else 1,
    )

    if verbose:
        print(f"\nFINAL SCORE: {result.score} | Passed: {result.passed}")
        print(f"Feedback: {result.feedback}")

    return {
        "task_id": TASK_ID,
        "ticket_id": obs.ticket_id,
        "score": result.score,
        "breakdown": result.breakdown,
        "feedback": result.feedback,
        "passed": result.passed,
        "steps_taken": steps_taken,
    }


def run_all(agent_fn, verbose: bool = True) -> list:
    """Run all Task 1 tickets and return scores."""
    tickets_path = Path(__file__).parent.parent / "data" / "tickets.json"
    with open(tickets_path) as f:
        all_tickets = json.load(f)

    results = []
    for ticket in all_tickets[TASK_ID]["tickets"]:
        result = run_episode(agent_fn, ticket_id=ticket["ticket_id"], verbose=verbose)
        results.append(result)

    avg_score = round(sum(r["score"] for r in results) / len(results), 4)
    print(f"\nTASK 1 AVERAGE SCORE: {avg_score} over {len(results)} tickets")
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
