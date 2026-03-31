# 🛒 E-Commerce Customer Support Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment
that simulates a real e-commerce customer support agent.

The agent handles support tickets — classifying issues, drafting responses,
and managing escalations — across three tasks of increasing difficulty.

---

## 🎯 Environment Description

E-commerce customer support is a genuine, high-volume real-world task.
Companies process thousands of tickets daily, and agent quality directly
impacts customer satisfaction and retention.

This environment models that workflow:
- The agent receives a customer support ticket
- It must classify, respond, escalate, and resolve the issue
- It is scored on accuracy, empathy, completeness, and consistency

---

## 📋 Tasks

| Task | ID | Difficulty | Max Steps | Description |
|---|---|---|---|---|
| Ticket Classification | `task1_classify` | Easy | 3 | Classify ticket into the correct category |
| Response Generation | `task2_respond` | Medium | 5 | Classify + draft appropriate customer reply |
| Multi-turn Escalation | `task3_escalate` | Hard | 10 | De-escalate frustrated customer, escalate correctly |

---

## 🔧 Action Space

| Action | Required Fields | Description |
|---|---|---|
| `classify` | `category` | Label the ticket with a category |
| `respond` | `message` | Send a reply to the customer |
| `escalate` | `department` | Route to a department |
| `request_info` | `message` | Ask customer for more details |
| `resolve` | `resolution_note` | Close the ticket |

**Categories:** `billing` · `delivery` · `refund` · `product_issue` · `general_inquiry`

**Departments:** `billing_team` · `logistics_team` · `technical_team` · `legal_team` · `senior_support`

---

## 👁️ Observation Space

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `ticket_text` | string | Customer's message |
| `customer_name` | string | Customer name |
| `order_id` | string (optional) | Referenced order ID |
| `conversation_history` | list | Prior turns in the episode |
| `available_actions` | list | Actions currently allowed |
| `last_action_feedback` | string | Feedback from last action |
| `done` | bool | Whether episode is complete |
| `reward` | float | Cumulative reward so far |
| `task_id` | string | Which task this episode belongs to |
| `difficulty` | string | easy / medium / hard |

---

## 🏆 Reward Function

Rewards are given at each step (partial progress), not just at episode end.

| Signal | Reward |
|---|---|
| Correct classification | +0.30 |
| Issue resolved with all steps complete | +0.20 |
| Valid response submitted | +0.10 |
| Correct escalation department | +0.20 |
| Information request | +0.05 |
| Wrong classification | -0.10 |
| Wrong escalation department | -0.20 |
| Repetitive response (loop) | -0.40 |
| Too many invalid actions | Episode ends |

All rewards are clamped to `[0.0, 1.0]`.

---

## 📊 Grader Scoring

Each task has a deterministic grader that scores `0.0–1.0`.

**Task 1 — Classification:**
- Correct on first attempt: `1.0`
- Correct on second attempt: `0.5`
- Wrong: `0.0`

**Task 2 — Response Quality:**
- Correct classification: `0.30`
- Required elements present: `0.30`
- No forbidden statements: `0.20`
- Response length adequate: `0.10`
- Ticket ID referenced: `0.10`

**Task 3 — Escalation Handling:**
- Correct escalation department: `0.25`
- Acknowledged previous failures: `0.20`
- No contradictions across turns: `0.20`
- Concrete next steps provided: `0.15`
- Special criteria met: `0.10`
- Sentiment de-escalation language: `0.10`

---

## 🚀 Setup & Usage

### 1. Clone the repo
```bash
git clone https://huggingface.co/spaces/your-username/ecommerce-support-env
cd ecommerce-support-env
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server locally
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the server
```bash
# Health check
curl http://localhost:8000/health

# Start an episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_classify", "ticket_id": "T1-001"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "category": "billing"}'
```

### 5. Run the baseline script
```bash
export OPENAI_API_KEY=your_key_here
python baseline/run_baseline.py --model gpt-4o-mini --verbose
```

---

## 🐳 Docker

```bash
# Build
docker build -t ecommerce-support-env .

# Run
docker run -p 7860:7860 ecommerce-support-env

# Test
curl http://localhost:7860/health
```

---

## 📈 Baseline Scores

Scores using `gpt-4o-mini` at `temperature=0.0`:

| Task | Avg Score | Pass Rate |
|---|---|---|
| Task 1 — Classification (Easy) | 0.85 | 83% |
| Task 2 — Response Generation (Medium) | 0.62 | 67% |
| Task 3 — Escalation Handling (Hard) | 0.41 | 33% |
| **Overall** | **0.63** | **61%** |

---

## 📁 Project Structure

```
ecommerce-support-env/
├── models.py               # Pydantic models (Action, Observation)
├── main.py                 # FastAPI server entry point
├── openenv.yaml            # Environment manifest
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── README.md               # This file
├── env/
│   ├── environment.py      # Core environment (reset/step/state)
│   └── graders.py          # Task graders
├── tasks/
│   ├── task1_classify.py   # Easy task runner
│   ├── task2_respond.py    # Medium task runner
│   └── task3_escalate.py   # Hard task runner
├── data/
│   └── tickets.json        # 18 support tickets across 3 tasks
└── baseline/
    └── run_baseline.py     # OpenAI baseline inference script
```

---

## 📚 Dataset

Tickets inspired by
[NebulaByte/E-Commerce_Customer_Support_Conversations](https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations)
on HuggingFace.

---

## 📜 License

MIT
