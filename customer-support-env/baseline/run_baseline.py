# baseline/run_baseline.py
# Baseline Inference Script
#
# Runs an OpenAI model against all 3 tasks and produces
# reproducible baseline scores.
#
# Usage:
#   export OPENAI_API_KEY=your_key_here
#   python baseline/run_baseline.py
#
# Optional flags:
#   --task    : Run a single task (task1_classify | task2_respond | task3_escalate)
#   --model   : OpenAI model to use (default: gpt-4o-mini)
#   --verbose : Print step-by-step episode output

import os
import json
import argparse
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI

from tasks.task1_classify import run_all as run_task1
from tasks.task2_respond import run_all as run_task2
from tasks.task3_escalate import run_all as run_task3


# ---------------------------------------------------------------------------
# OpenAI Agent Function
# ---------------------------------------------------------------------------

def make_agent(model: str = "gpt-4o-mini"):
    """
    Returns an agent function that calls the OpenAI API.

    The agent receives:
      - system_prompt : Task instructions
      - observation   : Current state of the episode as a string

    The agent returns:
      - A dict representing the action (e.g. {"action_type": "classify", "category": "billing"})
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def agent_fn(system_prompt: str, observation: str) -> dict:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation},
                ],
                temperature=0.0,  # deterministic — important for reproducibility
                max_tokens=500,
            )

            raw_text = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()

            action_dict = json.loads(raw_text)
            return action_dict

        except json.JSONDecodeError as e:
            print(f"  [Agent] JSON parse error: {e} | Raw: {raw_text[:100]}")
            # Return a safe fallback action
            return {"action_type": "resolve", "resolution_note": "Unable to parse response."}

        except Exception as e:
            print(f"  [Agent] API error: {e}")
            return {"action_type": "resolve", "resolution_note": "API error."}

    return agent_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline inference on all tasks.")
    parser.add_argument(
        "--task",
        choices=["task1_classify", "task2_respond", "task3_escalate", "all"],
        default="all",
        help="Which task to run (default: all)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print step-by-step episode output"
    )
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run: export OPENAI_API_KEY=your_key_here")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"E-Commerce Support Environment — Baseline Inference")
    print(f"Model : {args.model}")
    print(f"Task  : {args.task}")
    print(f"{'='*60}")

    agent = make_agent(model=args.model)
    all_results = {}

    # --- Task 1 ---
    if args.task in ("task1_classify", "all"):
        print("\n>>> Running Task 1: Ticket Classification (Easy)")
        results = run_task1(agent, verbose=args.verbose)
        all_results["task1_classify"] = _summarise(results)

    # --- Task 2 ---
    if args.task in ("task2_respond", "all"):
        print("\n>>> Running Task 2: Response Generation (Medium)")
        results = run_task2(agent, verbose=args.verbose)
        all_results["task2_respond"] = _summarise(results)

    # --- Task 3 ---
    if args.task in ("task3_escalate", "all"):
        print("\n>>> Running Task 3: Multi-turn Escalation (Hard)")
        results = run_task3(agent, verbose=args.verbose)
        all_results["task3_escalate"] = _summarise(results)

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Avg Score':>10} {'Pass Rate':>10} {'Tickets':>8}")
    print(f"{'-'*55}")

    overall_scores = []
    for task_id, summary in all_results.items():
        print(
            f"{task_id:<25} "
            f"{summary['avg_score']:>10.4f} "
            f"{summary['pass_rate']:>9.1f}% "
            f"{summary['total_tickets']:>8}"
        )
        overall_scores.append(summary["avg_score"])

    if overall_scores:
        overall_avg = round(sum(overall_scores) / len(overall_scores), 4)
        print(f"{'-'*55}")
        print(f"{'OVERALL AVERAGE':<25} {overall_avg:>10.4f}")

    print(f"{'='*60}\n")

    # Save results to file
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "results": all_results,
                "overall_avg": overall_avg if overall_scores else 0.0,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(results: list) -> dict:
    if not results:
        return {"avg_score": 0.0, "pass_rate": 0.0, "total_tickets": 0}

    scores = [r["score"] for r in results]
    passed = [r for r in results if r["passed"]]

    return {
        "avg_score": round(sum(scores) / len(scores), 4),
        "pass_rate": round(len(passed) / len(results) * 100, 1),
        "total_tickets": len(results),
        "individual_scores": [
            {"ticket_id": r["ticket_id"], "score": r["score"]}
            for r in results
        ],
    }


if __name__ == "__main__":
    main()
