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
import json
import asyncio
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app
from customer_support_env.models import SupportAction, SupportObservation
from customer_support_env.env.environment import CustomerSupportEnvironment
from customer_support_env.env.graders import get_grader

# Import fine-tuning functions
from customer_support_env.fine_tune_model import upload_training_file, start_fine_tuning, monitor_fine_tuning


# ---------------------------------------------------------------------------
# Instantiate the environment (one instance shared across requests)
# ---------------------------------------------------------------------------

environment = CustomerSupportEnvironment()


# ---------------------------------------------------------------------------
# Create the OpenEnv FastAPI app
# This gives us /reset, /step, /state endpoints automatically
# ---------------------------------------------------------------------------

app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="ecommerce_support_env"
)


# ---------------------------------------------------------------------------
# Health check endpoint
# Required by openenv.yaml runtime.health_endpoint
# Also used by Docker HEALTHCHECK and HuggingFace Spaces
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "E-commerce Customer Support RL Environment",
        "description": "OpenEnv reinforcement learning environment for support ticket handling",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "grade": "/grade"
        }
    }


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
        "endpoints": {
            "environment": ["/reset", "/step", "/state"],
            "grading": ["/grade"],
            "health": ["/health"],
            "info": ["/info"],
            "fine_tuning": [
                "/finetune/prepare-data",
                "/finetune/start",
                "/finetune/status/{job_id}",
                "/finetune/models",
                "/finetune/jobs"
            ]
        }
    })


# ---------------------------------------------------------------------------
# Fine-tuning endpoints
# Allow users to fine-tune models through the API
# ---------------------------------------------------------------------------

def get_openai_client():
    """Get OpenAI client with error handling."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY environment variable not set"
        )
    return OpenAI(api_key=api_key)


@app.post("/finetune/prepare-data")
async def prepare_finetune_data():
    """Prepare training data for fine-tuning."""
    try:
        # Import and run the data preparation
        from customer_support_env.prepare_finetune_data import prepare_finetune_data as prep_data
        prep_data()

        # Check what files were created
        data_files = [
            "finetune_task1_classification.jsonl",
            "finetune_task2_response.jsonl",
            "finetune_task3_escalation.jsonl",
            "finetune_all_tasks.jsonl"
        ]

        created_files = []
        for filename in data_files:
            if Path(filename).exists():
                created_files.append(filename)

        return JSONResponse(content={
            "status": "success",
            "message": f"Training data prepared successfully. Created {len(created_files)} files.",
            "files": created_files
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare data: {str(e)}")


@app.post("/finetune/start")
async def start_finetuning(
    background_tasks: BackgroundTasks,
    data_file: str = "finetune_all_tasks.jsonl",
    model: str = "gpt-3.5-turbo",
    suffix: str = None,
    monitor: bool = False
):
    """Start a fine-tuning job."""
    try:
        client = get_openai_client()

        # Check if data file exists
        data_path = Path(data_file)
        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Training data file {data_file} not found. Run /finetune/prepare-data first."
            )

        # Upload training file
        training_file_id = upload_training_file(client, str(data_path))

        # Start fine-tuning
        job_id = start_fine_tuning(client, training_file_id, model, suffix)

        # Save job info
        job_info = {
            "job_id": job_id,
            "training_file": data_file,
            "base_model": model,
            "timestamp": time.time()
        }

        with open("finetune_job.json", "w") as f:
            json.dump(job_info, f, indent=2)

        # Start monitoring in background if requested
        if monitor:
            background_tasks.add_task(monitor_fine_tuning_background, client, job_id)

        return JSONResponse(content={
            "status": "success",
            "job_id": job_id,
            "training_file_id": training_file_id,
            "message": "Fine-tuning job started successfully",
            "job_info": job_info
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start fine-tuning: {str(e)}")


async def monitor_fine_tuning_background(client, job_id: str):
    """Monitor fine-tuning job in background."""
    try:
        model_id = monitor_fine_tuning(client, job_id)
        if model_id:
            print(f"Fine-tuning completed! Model: {model_id}")
        else:
            print(f"Fine-tuning job {job_id} ended without success")
    except Exception as e:
        print(f"Error monitoring fine-tuning job {job_id}: {str(e)}")


@app.get("/finetune/status/{job_id}")
async def get_finetuning_status(job_id: str):
    """Get the status of a fine-tuning job."""
    client = get_openai_client()

    try:
        job = client.fine_tuning.jobs.retrieve(job_id)

        return JSONResponse(content={
            "job_id": job_id,
            "status": job.status,
            "model": getattr(job, 'fine_tuned_model', None),
            "created_at": job.created_at,
            "finished_at": getattr(job, 'finished_at', None),
            "error": getattr(job, 'error', None)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/finetune/models")
async def list_finetuned_models():
    """List available fine-tuned models."""
    client = get_openai_client()

    try:
        # Get fine-tuning jobs
        jobs = client.fine_tuning.jobs.list()

        models = []
        for job in jobs.data:
            if job.status == "succeeded" and hasattr(job, 'fine_tuned_model'):
                models.append({
                    "model_id": job.fine_tuned_model,
                    "job_id": job.id,
                    "base_model": job.model,
                    "created_at": job.created_at,
                    "suffix": getattr(job, 'suffix', None)
                })

        return JSONResponse(content={
            "models": models,
            "count": len(models)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/finetune/jobs")
async def list_finetuning_jobs():
    """List all fine-tuning jobs."""
    client = get_openai_client()

    try:
        jobs = client.fine_tuning.jobs.list(limit=20)

        job_list = []
        for job in jobs.data:
            job_list.append({
                "job_id": job.id,
                "status": job.status,
                "model": getattr(job, 'fine_tuned_model', None),
                "base_model": job.model,
                "created_at": job.created_at,
                "finished_at": getattr(job, 'finished_at', None),
                "suffix": getattr(job, 'suffix', None)
            })

        return JSONResponse(content={
            "jobs": job_list,
            "count": len(job_list)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


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
