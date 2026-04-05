import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

def upload_training_file(client: OpenAI, file_path: str) -> str:
    """Upload a training file to OpenAI."""
    print(f"Uploading {file_path}...")
    with open(file_path, "rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"Uploaded file: {file_response.id}")
    return file_response.id

def start_fine_tuning(client: OpenAI, training_file_id: str, model: str = "gpt-3.5-turbo", suffix: str = None) -> str:
    """Start a fine-tuning job."""
    print(f"Starting fine-tuning job with model {model}...")

    fine_tune_params = {
        "training_file": training_file_id,
        "model": model,
        "hyperparameters": {
            "n_epochs": 3
        }
    }

    if suffix:
        fine_tune_params["suffix"] = suffix

    fine_tune_response = client.fine_tuning.jobs.create(**fine_tune_params)

    print(f"Fine-tuning job started: {fine_tune_response.id}")
    return fine_tune_response.id

def monitor_fine_tuning(client: OpenAI, job_id: str):
    """Monitor the fine-tuning job progress."""
    print(f"Monitoring fine-tuning job {job_id}...")

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"Status: {status}")

        if status == "succeeded":
            print(f"Fine-tuning completed! Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif status == "failed":
            print(f"Fine-tuning failed: {job.error}")
            return None
        elif status in ["cancelled", "expired"]:
            print(f"Fine-tuning {status}")
            return None

        time.sleep(60)  # Check every minute

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI model for customer support")
    parser.add_argument("--data-file", default="finetune_all_tasks.jsonl",
                       help="JSONL file with training data")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                       help="Base model to fine-tune")
    parser.add_argument("--suffix", help="Suffix for the fine-tuned model name")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor the fine-tuning job until completion")

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    # Check if data file exists
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: Training data file {args.data_file} not found")
        print("Run 'python prepare_finetune_data.py' first to generate training data")
        return

    try:
        # Upload training file
        training_file_id = upload_training_file(client, str(data_path))

        # Start fine-tuning
        job_id = start_fine_tuning(client, training_file_id, args.model, args.suffix)

        # Save job info
        job_info = {
            "job_id": job_id,
            "training_file": args.data_file,
            "base_model": args.model,
            "timestamp": time.time()
        }

        with open("finetune_job.json", "w") as f:
            json.dump(job_info, f, indent=2)

        print(f"Job information saved to finetune_job.json")

        if args.monitor:
            model_id = monitor_fine_tuning(client, job_id)
            if model_id:
                print(f"\nTo use this model, update your baseline script with:")
                print(f"model='{model_id}'")

    except Exception as e:
        print(f"Error during fine-tuning: {e}")

if __name__ == "__main__":
    main()
    # </content>
# <parameter name="filePath">c:\Users\Khushbu kumari\OneDrive\Desktop\Hackathon\hackathonOpenEnvRL\customer-support-env\fine_tune_model.py