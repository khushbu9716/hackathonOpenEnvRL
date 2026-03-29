from env.environment import CustomerSupportEnv
from env.models import Action

def run():
    env = CustomerSupportEnv()
    obs = env.reset()

    print("Customer:", obs.customer_message)

    # Simple rule-based agent (no API)
    if "order" in obs.customer_message.lower():
        msg = "Here is your tracking link"
    elif "damaged" in obs.customer_message.lower():
        msg = "Sorry for the issue, we will process your refund"
    else:
        msg = "Can you please provide more details?"

    action = Action(action_type="reply", message=msg)

    obs, reward, done, _ = env.step(action)

    print("Agent:", msg)
    print("Reward:", reward.score)