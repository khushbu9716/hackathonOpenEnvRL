import random
from .models import Observation, Action, Reward
from .tasks import TASKS
from .grader import grade_task

class CustomerSupportEnv:

    def __init__(self):
        self.current_task = None
        self.done = False
        self.history = []

    def reset(self):
        self.current_task = random.choice(TASKS)
        self.done = False
        self.history = []

        return Observation(
            ticket_id=self.current_task["id"],
            customer_message=self.current_task["message"],
            conversation_history=[],
            status="open"
        )

    def step(self, action: Action):
        self.history.append(action.message or "")

        score = grade_task(self.current_task, action)

        reward = Reward(score=score, reason="graded")

        self.done = True  # MVP (1-step)

        obs = Observation(
            ticket_id=self.current_task["id"],
            customer_message=self.current_task["message"],
            conversation_history=self.history,
            status="resolved" if score > 0 else "open"
        )

        return obs, reward, self.done, {}

    def state(self):
        return {
            "task": self.current_task,
            "history": self.history
        }