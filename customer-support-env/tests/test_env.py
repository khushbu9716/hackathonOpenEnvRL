from env.environment import CustomerSupportEnv
from env.models import Action

def test_env():
    env = CustomerSupportEnv()
    obs = env.reset()

    action = Action(action_type="reply", message="Here is your tracking link")

    obs, reward, done, _ = env.step(action)

    assert done is True