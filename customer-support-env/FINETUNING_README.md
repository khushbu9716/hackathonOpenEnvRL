# Fine-Tuning Guide for Customer Support Environment

This guide explains how to fine-tune language models for improved performance on the customer support tasks.

## Prerequisites

1. **OpenAI API Key**: You need an OpenAI API key with fine-tuning access
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. **Python Dependencies**: The required packages are already in `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```

## Step 1: Prepare Training Data

Run the data preparation script to convert the ticket data into fine-tuning format:

```bash
python prepare_finetune_data.py
```

This creates several JSONL files:
- `finetune_task1_classification.jsonl` - Classification examples
- `finetune_task2_response.jsonl` - Response generation examples  
- `finetune_task3_escalation.jsonl` - Escalation handling examples
- `finetune_all_tasks.jsonl` - Combined dataset

## Step 2: Fine-Tune the Model

Start the fine-tuning process:

```bash
# Fine-tune on all tasks (recommended)
python fine_tune_model.py --data-file finetune_all_tasks.jsonl --model gpt-3.5-turbo --suffix customer-support

# Or fine-tune on specific tasks
python fine_tune_model.py --data-file finetune_task1_classification.jsonl --model gpt-3.5-turbo --suffix classification-only

# Monitor progress in real-time
python fine_tune_model.py --data-file finetune_all_tasks.jsonl --monitor
```

The script will:
1. Upload your training data to OpenAI
2. Start the fine-tuning job
3. Save job information to `finetune_job.json`
4. (Optional) Monitor progress until completion

## Step 3: Evaluate the Fine-Tuned Model

Once fine-tuning is complete, test your model against the baseline:

```bash
# Run baseline with your fine-tuned model
python baseline/run_baseline.py --model ft:gpt-3.5-turbo-0125:your-org:customer-support:abc123

# Compare with base model
python baseline/run_baseline.py --model gpt-3.5-turbo
```

## Step 4: Use in Production

Update your agent code to use the fine-tuned model:

```python
# In your agent implementation
model_id = "ft:gpt-3.5-turbo-0125:your-org:customer-support:abc123"
agent = make_agent(model=model_id)
```

## API-Based Fine-Tuning

The fine-tuning functionality is also available through the FastAPI server endpoints. This allows you to trigger fine-tuning jobs programmatically or through web requests.

### Prerequisites

Make sure the server is running:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

And set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
```

### Step 1: Prepare Training Data via API

```bash
curl -X POST http://localhost:8000/finetune/prepare-data
```

This runs the data preparation script and creates the JSONL training files.

### Step 2: Start Fine-Tuning via API

```bash
# Basic fine-tuning
curl -X POST "http://localhost:8000/finetune/start" \
  -H "Content-Type: application/json" \
  -d '{
    "data_file": "finetune_all_tasks.jsonl",
    "model": "gpt-3.5-turbo",
    "suffix": "customer-support-api"
  }'

# With background monitoring
curl -X POST "http://localhost:8000/finetune/start?monitor=true" \
  -H "Content-Type: application/json" \
  -d '{
    "data_file": "finetune_all_tasks.jsonl",
    "model": "gpt-3.5-turbo"
  }'
```

### Step 3: Monitor Progress

```bash
# Check job status
curl http://localhost:8000/finetune/status/ftjob-abc123

# List all fine-tuning jobs
curl http://localhost:8000/finetune/jobs

# List completed models
curl http://localhost:8000/finetune/models
```

### API Endpoints Summary

- `POST /finetune/prepare-data` - Prepare training data
- `POST /finetune/start` - Start a fine-tuning job
- `GET /finetune/status/{job_id}` - Check job status
- `GET /finetune/jobs` - List all jobs
- `GET /finetune/models` - List fine-tuned models

## Tips for Better Results

1. **Data Quality**: The training data includes ground truth labels and expected response elements
2. **Hyperparameters**: The script uses 3 epochs by default. You can adjust this in the code
3. **Model Selection**: GPT-3.5-turbo is faster and cheaper to fine-tune than GPT-4
4. **Evaluation**: Always compare against the baseline to ensure improvement

## Troubleshooting

- **API Key Issues**: Make sure your OpenAI account has fine-tuning access
- **Data Format**: Ensure your JSONL follows OpenAI's message format
- **Cost**: Fine-tuning has associated costs. Check OpenAI's pricing
- **Time**: Fine-tuning can take 30 minutes to several hours depending on dataset size

## Advanced Usage

For more control, you can modify the fine-tuning parameters in `fine_tune_model.py`:

```python
fine_tune_params = {
    "training_file": training_file_id,
    "model": model,
    "hyperparameters": {
        "n_epochs": 5,  # More epochs for better results
        "batch_size": 16  # Adjust based on your needs
    }
}
```

## Alternative: Local Fine-Tuning

If you prefer not to use OpenAI, you can fine-tune open-source models locally using Hugging Face transformers. Add these to `requirements.txt`:

```
torch>=2.0
transformers
datasets
accelerate
```

Then use the local fine-tuning approach mentioned in the main README.