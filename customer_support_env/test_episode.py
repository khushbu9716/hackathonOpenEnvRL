# test_episode.py
# Run with: python test_episode.py

from customer_support_env.env.environment import CustomerSupportEnvironment
from customer_support_env.models import SupportAction

print("="*50)
print("TEST: Full Task 1 Episode")
print("="*50)

# Start environment
env = CustomerSupportEnvironment(
    task_id='task1_classify',
    ticket_id='T1-001'
)
obs = env.reset()

print("Ticket:", obs.ticket_text[:60])
print("Reward:", obs.reward)
print("Done:", obs.done)
print("Available actions:", obs.available_actions)
print()

# Step 1 — Classify
action = SupportAction(action_type='classify', category='billing')
obs = env.step(action)
print("After classify:")
print("  Reward:", obs.reward)
print("  Feedback:", obs.last_action_feedback)
print("  Available actions:", obs.available_actions)
print()

# Step 2 — Resolve
action = SupportAction(action_type='resolve', resolution_note='Ticket classified and resolved.')
obs = env.step(action)
print("After resolve:")
print("  Reward:", obs.reward)
print("  Done:", obs.done)
print("  Feedback:", obs.last_action_feedback)


# ─────────────────────────────────────────
# TEST: Task 2 Episode
# ─────────────────────────────────────────
print()
print("="*50)
print("TEST: Full Task 2 Episode")
print("="*50)


env = CustomerSupportEnvironment(
    task_id='task2_respond',
    ticket_id='T2-001'
)
obs = env.reset()
print("Ticket:", obs.ticket_text[:60])

# Classify
action = SupportAction(action_type='classify', category='billing')
obs = env.step(action)
print("After classify - Reward:", obs.reward)

# Respond
action = SupportAction(
    action_type='respond',
    message=(
        "Dear Linda, I'm sorry to hear about the duplicate charge on "
        "order ORD-82910. We will process the refund within 3-5 business "
        "days. Your ticket ID is T2-001. Please reach out if you need help."
    )
)
obs = env.step(action)
print("After respond - Reward:", obs.reward)
print("Feedback:", obs.last_action_feedback)

# Resolve
action = SupportAction(action_type='resolve', resolution_note='Responded and resolved.')
obs = env.step(action)
print("After resolve - Reward:", obs.reward)
print("Done:", obs.done)

# ─────────────────────────────────────────
# TEST: Task 3 Episode
# ─────────────────────────────────────────
print()
print("="*50)
print("TEST: Full Task 3 Episode")
print("="*50)

env = CustomerSupportEnvironment(
    task_id='task3_escalate',
    ticket_id='T3-001'
)
obs = env.reset()
print("Conversation history turns loaded:", len(obs.conversation_history))
print("Ticket:", obs.ticket_text[:60])

# Classify
action = SupportAction(action_type='classify', category='billing')
obs = env.step(action)
print("After classify - Reward:", obs.reward)

# Respond
action = SupportAction(
    action_type='respond',
    message=(
        "I sincerely apologise for the previous failed attempts to resolve "
        "this. I understand your frustration with the repeated charges. "
        "I am escalating this to our billing team right now and you will "
        "hear back within 1 business day."
    )
)
obs = env.step(action)
print("After respond - Reward:", obs.reward)

# Escalate
action = SupportAction(action_type='escalate', department='billing_team')
obs = env.step(action)
print("After escalate - Reward:", obs.reward)
print("Feedback:", obs.last_action_feedback)

# Resolve
action = SupportAction(action_type='resolve', resolution_note='Escalated to billing team.')
obs = env.step(action)
print("After resolve - Reward:", obs.reward)
print("Done:", obs.done)