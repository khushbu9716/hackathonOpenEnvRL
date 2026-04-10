from customer_support_env.env.environment import CustomerSupportEnv
from customer_support_env.env.models import Action

def test_env():
    env = CustomerSupportEnv()
    obs = env.reset()

    action = Action(action_type="reply", message="Here is your tracking link")

    obs, reward, done, _ = env.step(action)

    assert done is True