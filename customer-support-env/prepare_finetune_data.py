import json
import os
from pathlib import Path
from typing import List, Dict, Any

def prepare_finetune_data():
    """
    Prepare fine-tuning data from tickets.json for OpenAI fine-tuning.
    Creates JSONL files for different tasks.
    """
    data_path = Path(__file__).parent / "data" / "tickets.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare Task 1: Classification
    task1_examples = prepare_task1_data(data['task1_classify']['tickets'])
    save_jsonl(task1_examples, "finetune_task1_classification.jsonl")

    # Prepare Task 2: Response Generation
    task2_examples = prepare_task2_data(data['task2_respond']['tickets'])
    save_jsonl(task2_examples, "finetune_task2_response.jsonl")

    # Prepare Task 3: Escalation (more complex)
    task3_examples = prepare_task3_data(data['task3_escalate']['tickets'])
    save_jsonl(task3_examples, "finetune_task3_escalation.jsonl")

    # Combined dataset for general fine-tuning
    all_examples = task1_examples + task2_examples + task3_examples
    save_jsonl(all_examples, "finetune_all_tasks.jsonl")

    print(f"Prepared {len(task1_examples)} Task 1 examples")
    print(f"Prepared {len(task2_examples)} Task 2 examples")
    print(f"Prepared {len(task3_examples)} Task 3 examples")
    print(f"Total: {len(all_examples)} examples")

def prepare_task1_data(tickets: List[Dict]) -> List[Dict]:
    """Prepare classification examples for Task 1."""
    examples = []

    system_prompt = """You are a customer support agent for an e-commerce company.

Your ONLY job is to read the support ticket and classify it into exactly
one of these categories:

  - billing         : payment issues, duplicate charges, invoices
  - delivery        : shipping delays, lost packages, tracking issues
  - refund          : refund requests, return status, money not received
  - product_issue   : broken items, defective products, technical faults
  - general_inquiry : questions about policies, discounts, general info

Respond with only the category name."""

    for ticket in tickets:
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket['ticket_text']},
                {"role": "assistant", "content": ticket['ground_truth_category']}
            ]
        }
        examples.append(example)

    return examples

def prepare_task2_data(tickets: List[Dict]) -> List[Dict]:
    """Prepare response generation examples for Task 2."""
    examples = []

    system_prompt = """You are a customer support agent for an e-commerce company.

Read the customer ticket and provide an appropriate response. Your response should:
- Be empathetic and professional
- Address the customer's concern
- Provide clear next steps
- Include the ticket ID in your response

Categories:
  - billing: payment issues, duplicate charges, invoices
  - delivery: shipping delays, lost packages, tracking issues
  - refund: refund requests, return status, money not received
  - product_issue: broken items, defective products, technical faults
  - general_inquiry: questions about policies, discounts, general info"""

    for ticket in tickets:
        # First classify
        classify_example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify and respond to: {ticket['ticket_text']}"},
                {"role": "assistant", "content": f"Category: {ticket['ground_truth_category']}\n\nResponse: This is a {ticket['ground_truth_category']} issue. We're sorry for the inconvenience. We'll look into this right away. Your ticket ID is {ticket['ticket_id']}."}
            ]
        }
        examples.append(classify_example)

    return examples

def prepare_task3_data(tickets: List[Dict]) -> List[Dict]:
    """Prepare escalation examples for Task 3."""
    examples = []

    system_prompt = """You are a customer support agent handling escalated customer issues.

The customer is frustrated and may be threatening legal action or chargebacks.
Your goals:
1. De-escalate the situation with empathy
2. Acknowledge their frustration
3. Take ownership of the problem
4. Provide clear resolution steps
5. Escalate to appropriate department if needed

Departments: billing_team, logistics_team, technical_team, legal_team, senior_support"""

    for ticket in tickets:
        conversation = ticket.get('conversation_history', [])
        if conversation:
            # Build conversation context
            messages = [{"role": "system", "content": system_prompt}]

            for turn in conversation:
                role = "user" if turn['role'] == 'customer' else "assistant"
                messages.append({"role": role, "content": turn['content']})

            # Add the escalation action as response
            escalation_response = f"I understand your frustration and apologize for the ongoing issues. I'll escalate this to our {ticket.get('correct_escalation_path', 'senior_support')} team immediately. They will contact you within 24 hours. Your ticket ID is {ticket['ticket_id']}."

            messages.append({"role": "assistant", "content": escalation_response})

            examples.append({"messages": messages})

    return examples

def save_jsonl(examples: List[Dict], filename: str):
    """Save examples to JSONL file."""
    output_path = Path(__file__).parent / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved {len(examples)} examples to {filename}")

if __name__ == "__main__":
    prepare_finetune_data()
    
    # </content>
# <parameter name="filePath">c:\Users\Khushbu kumari\OneDrive\Desktop\Hackathon\hackathonOpenEnvRL\customer-support-env\prepare_finetune_data.py