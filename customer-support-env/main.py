# main.py
# E-Commerce Customer Support Environment — FastAPI Server Entry Point
#
# Wires together the environment, models, and graders into a FastAPI app
# using OpenEnv's create_fastapi_app helper.
#
# Start locally:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Or via Docker:
#   docker build -t ecommerce-support-env .
#   docker run -p 7860:7860 ecommerce-support-env

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_fastapi_app

from models import SupportAction, SupportObservation
from env.environment import CustomerSupportEnvironment
from env.graders import get_grader


# ---------------------------------------------------------------------------
# Instantiate the environment (one instance shared across requests)
# ---------------------------------------------------------------------------

environment = CustomerSupportEnvironment()


# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app
# This gives us /reset, /step, /state endpoints automatically
# ---------------------------------------------------------------------------

app = create_fastapi_app(
    env=environment,
    action_type=SupportAction,
    observation_type=SupportObservation,
)


# ---------------------------------------------------------------------------
# Health check endpoint
# Required by openenv.yaml runtime.health_endpoint
# Also used by Docker HEALTHCHECK and HuggingFace Spaces
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Simple health check — returns 200 if server is running."""
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "ecommerce_support_env",
            "version": "1.0.0",
        }
    )


# ---------------------------------------------------------------------------
# Grade endpoint
# Runs the appropriate grader at the end of an episode
# Called by the baseline script after each episode completes
# ---------------------------------------------------------------------------

@app.post("/grade")
async def grade(payload: dict):
    """
    Grade a completed episode.

    Expected payload:
    {
        "task_id"               : "task1_classify" | "task2_respond" | "task3_escalate",
        "ticket_id"             : "T1-001",
        "classification_value"  : "billing",           // task1 + task2
        "response_text"         : "Dear customer...",  // task2
        "escalation_department" : "billing_team",      // task3
        "conversation_history"  : [...],               // task3
        "resolution_note"       : "Issue resolved.",   // all tasks
        "classified"            : true,
        "responded"             : true,
        "classify_attempt_count": 1                    // task1 only
    }

    Returns:
    {
        "score"     : 0.85,
        "breakdown" : { ... },
        "passed"    : true,
        "feedback"  : "..."
    }
    """
    task_id = payload.get("task_id")
    if not task_id:
        return JSONResponse(status_code=400, content={"error": "task_id is required"})

    # Load the ticket from the dataset
    import json
    from pathlib import Path
    tickets_path = Path(__file__).parent / "data" / "tickets.json"
    with open(tickets_path) as f:
        all_tickets = json.load(f)

    ticket_pool = all_tickets.get(task_id, {}).get("tickets", [])
    ticket_id = payload.get("ticket_id")
    ticket = next((t for t in ticket_pool if t["ticket_id"] == ticket_id), None)

    if not ticket:
        return JSONResponse(
            status_code=404,
            content={"error": f"Ticket {ticket_id} not found in {task_id}"}
        )

    # Get the right grader and run it
    grader = get_grader(task_id)

    if task_id == "task1_classify":
        result = grader.grade(
            ticket=ticket,
            classification_value=payload.get("classification_value"),
            classify_attempt_count=payload.get("classify_attempt_count", 1),
        )

    elif task_id == "task2_respond":
        result = grader.grade(
            ticket=ticket,
            classification_value=payload.get("classification_value"),
            response_text=payload.get("response_text"),
        )

    elif task_id == "task3_escalate":
        result = grader.grade(
            ticket=ticket,
            escalation_department=payload.get("escalation_department"),
            conversation_history=payload.get("conversation_history", []),
            resolution_note=payload.get("resolution_note"),
            classified=payload.get("classified", False),
            responded=payload.get("responded", False),
        )

    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown task_id: {task_id}"}
        )

    return JSONResponse(content={
        "score": result.score,
        "breakdown": result.breakdown,
        "passed": result.passed,
        "feedback": result.feedback,
    })


# ---------------------------------------------------------------------------
# Info endpoint
# Returns environment metadata for discoverability
# ---------------------------------------------------------------------------

@app.get("/info")
async def info():
    """Returns environment metadata from openenv.yaml."""
    return JSONResponse(content={
        "name": "ecommerce_support_env",
        "version": "1.0.0",
        "description": (
            "E-commerce customer support RL environment. "
            "Agent handles tickets across 3 tasks: classify, respond, escalate."
        ),
        "tasks": [
            {
                "id": "task1_classify",
                "difficulty": "easy",
                "max_steps": 3,
                "description": "Classify the support ticket into the correct category."
            },
            {
                "id": "task2_respond",
                "difficulty": "medium",
                "max_steps": 5,
                "description": "Classify and draft an appropriate customer response."
            },
            {
                "id": "task3_escalate",
                "difficulty": "hard",
                "max_steps": 10,
                "description": "Handle a frustrated multi-turn customer conversation."
            },
        ],
        "action_types": [
            "classify", "respond", "escalate", "request_info", "resolve"
        ],
        "reward_range": [0.0, 1.0],
    })


# ---------------------------------------------------------------------------
# Run with uvicorn when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
