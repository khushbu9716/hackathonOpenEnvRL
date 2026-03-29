import os
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

client = OpenAI(api_key=os.getenv("OPENAIAPIKEY"))

def run():
    env = CustomerSupportEnv()
    obs = env.reset()

    prompt = f"""
    Customer: {obs.customer_message}
    What is the best response?
    """

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    message = response.choices[0].message.content

    action = Action(action_type="reply", message=message)

    obs, reward, done, _ = env.step(action)

    print("Reward:", reward.score)

if __name__ == "__main__":
    run()