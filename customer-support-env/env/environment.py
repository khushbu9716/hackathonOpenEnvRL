# env/environment.py
# E-Commerce Customer Support Environment — Core Logic
# Implements OpenEnv spec: reset(), step(), state()

import json
import uuid
import random
from pathlib import Path

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import SupportAction, SupportObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = {
    "task1_classify": 3,
    "task2_respond": 5,
    "task3_escalate": 10,
}

# Path to ticket dataset
TICKETS_PATH = Path(__file__).parent.parent / "data" / "tickets.json"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnvironment(Environment):
    """
    E-Commerce Customer Support RL Environment.

    The agent handles support tickets across 3 tasks of increasing difficulty:
      - task1_classify : Classify the ticket category (easy)
      - task2_respond  : Classify + draft an appropriate response (medium)
      - task3_escalate : Multi-turn de-escalation and resolution (hard)

    Episodes end when the agent calls resolve(), hits max_steps,
    or takes an invalid action 3 times in a row.
    """

    def __init__(self):
        # Load ticket dataset once at startup
        with open(TICKETS_PATH, "r") as f:
            self._all_tickets = json.load(f)

        # Internal state — will be properly initialised in reset()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._current_ticket = None
        self._task_id = None
        self._conversation_history = []
        self._cumulative_reward = 0.0
        self._done = False
        self._invalid_action_count = 0

        # Track what the agent has done this episode
        self._classified = False
        self._responded = False
        self._escalated = False
        self._classification_value = None
        self._last_response = None
        self._last_escalation = None

    # -----------------------------------------------------------------------
    # reset() — start a new episode
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "task1_classify", ticket_id: str = None) -> SupportObservation:
        """
        Start a new episode.

        Args:
            task_id   : One of task1_classify | task2_respond | task3_escalate
            ticket_id : Specific ticket to load. If None, picks a random one.

        Returns:
            SupportObservation with the first ticket the agent must handle.
        """
        # Fresh state
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_id = task_id
        self._conversation_history = []
        self._cumulative_reward = 0.0
        self._done = False
        self._invalid_action_count = 0
        self._classified = False
        self._responded = False
        self._escalated = False
        self._classification_value = None
        self._last_response = None
        self._last_escalation = None

        # Pick ticket
        ticket_pool = self._all_tickets[task_id]["tickets"]
        if ticket_id:
            ticket = next((t for t in ticket_pool if t["ticket_id"] == ticket_id), None)
            if not ticket:
                raise ValueError(f"Ticket {ticket_id} not found in {task_id}")
        else:
            ticket = random.choice(ticket_pool)

        self._current_ticket = ticket

        # For task3, seed the conversation history from the dataset
        if task_id == "task3_escalate":
            self._conversation_history = [
                {"role": turn["role"], "content": turn["content"]}
                for turn in ticket.get("conversation_history", [])
            ]

        return self._build_observation(feedback=None)

    # -----------------------------------------------------------------------
    # step() — agent takes one action
    # -----------------------------------------------------------------------

    def step(self, action: SupportAction) -> SupportObservation:
        """
        Execute one agent action and return the next observation.

        Reward is computed incrementally — partial progress is rewarded
        at each step rather than only at episode end.
        """
        if self._done:
            return self._build_observation(
                feedback="Episode is already done. Call reset() to start a new one."
            )

        self._state.step_count += 1
        step_reward = 0.0
        feedback = ""

        # --- Validate action has required fields ---
        validation_error = self._validate_action(action)
        if validation_error:
            self._invalid_action_count += 1
            feedback = f"Invalid action: {validation_error}"
            # Penalise repeated invalid actions
            if self._invalid_action_count >= 3:
                self._done = True
                feedback += " | Episode ended: too many invalid actions."
            return self._build_observation(feedback=feedback)

        self._invalid_action_count = 0  # reset on valid action

        # --- Route to action handler ---
        if action.action_type == "classify":
            step_reward, feedback = self._handle_classify(action)

        elif action.action_type == "respond":
            step_reward, feedback = self._handle_respond(action)

        elif action.action_type == "escalate":
            step_reward, feedback = self._handle_escalate(action)

        elif action.action_type == "request_info":
            step_reward, feedback = self._handle_request_info(action)

        elif action.action_type == "resolve":
            step_reward, feedback = self._handle_resolve(action)
            self._done = True

        # Add to conversation history
        if action.message:
            self._conversation_history.append({
                "role": "agent",
                "content": action.message
            })

        # Accumulate reward
        self._cumulative_reward = round(
            max(0.0, min(1.0, self._cumulative_reward + step_reward)), 4
        )

        # Check max steps
        max_steps = MAX_STEPS.get(self._task_id, 5)
        if self._state.step_count >= max_steps and not self._done:
            self._done = True
            feedback += f" | Episode ended: max steps ({max_steps}) reached."

        return self._build_observation(feedback=feedback)

    # -----------------------------------------------------------------------
    # state property — episode metadata
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_classify(self, action: SupportAction):
        """Reward correct ticket classification."""
        if self._classified:
            return -0.1, "Already classified this ticket. Duplicate classify action penalised."

        self._classified = True
        self._classification_value = action.category
        ground_truth = self._current_ticket.get("ground_truth_category")

        if action.category == ground_truth:
            return 0.3, f"Correct classification: '{action.category}'. +0.3 reward."
        else:
            return -0.1, (
                f"Incorrect classification: '{action.category}'. "
                f"Expected: '{ground_truth}'. -0.1 penalty."
            )

    def _handle_respond(self, action: SupportAction):
        """Reward response quality — checked by grader later, partial reward here."""
        if not action.message or len(action.message.strip()) < 20:
            return -0.1, "Response too short or empty. Minimum 20 characters required."

        if self._responded:
            # Penalise repetitive responses
            if self._last_response and action.message.strip() == self._last_response.strip():
                return -0.4, "Repetitive response detected. -0.4 penalty."

        self._responded = True
        self._last_response = action.message

        # Add customer reply to history (simulated for task3)
        if self._task_id == "task3_escalate":
            self._conversation_history.append({
                "role": "agent",
                "content": action.message
            })

        # Partial reward for a non-empty, non-repetitive response
        # Full grader scoring happens in graders.py
        return 0.1, "Response recorded. Full quality score applied at episode end."

    def _handle_escalate(self, action: SupportAction):
        """Reward correct escalation path."""
        if self._escalated:
            return -0.1, "Already escalated this ticket. Duplicate escalation penalised."

        self._escalated = True
        self._last_escalation = action.department

        # Only task3 has a defined correct escalation path
        if self._task_id == "task3_escalate":
            correct_dept = self._current_ticket.get("correct_escalation_path")
            if action.department == correct_dept:
                return 0.2, f"Correct escalation to '{action.department}'. +0.2 reward."
            else:
                return -0.2, (
                    f"Wrong escalation path: '{action.department}'. "
                    f"Expected: '{correct_dept}'. -0.2 penalty."
                )

        # For task1/task2, escalation is valid but not required
        return 0.05, f"Escalated to '{action.department}'."

    def _handle_request_info(self, action: SupportAction):
        """Small reward for requesting relevant information."""
        if not action.message or len(action.message.strip()) < 10:
            return -0.05, "Request for information too vague."
        return 0.05, "Information request sent to customer."

    def _handle_resolve(self, action: SupportAction):
        """
        Final reward when agent resolves the ticket.
        Checks whether all required steps were completed.
        """
        bonus = 0.0
        notes = []

        # Must have classified before resolving
        if self._classified:
            bonus += 0.1
            notes.append("classified (+0.1)")
        else:
            bonus -= 0.1
            notes.append("never classified (-0.1)")

        # Must have responded before resolving (task2 and task3)
        if self._task_id in ("task2_respond", "task3_escalate"):
            if self._responded:
                bonus += 0.2
                notes.append("responded (+0.2)")
            else:
                bonus -= 0.1
                notes.append("never responded (-0.1)")

        # Must have escalated for task3
        if self._task_id == "task3_escalate":
            if self._escalated:
                bonus += 0.1
                notes.append("escalated (+0.1)")
            else:
                bonus -= 0.1
                notes.append("never escalated (-0.1)")

        summary = ", ".join(notes)
        return bonus, f"Episode resolved. Summary: {summary}"

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _validate_action(self, action: SupportAction) -> str | None:
        """Return an error string if action is invalid, else None."""
        if action.action_type == "classify" and not action.category:
            return "'classify' action requires a 'category' field."
        if action.action_type == "respond" and not action.message:
            return "'respond' action requires a 'message' field."
        if action.action_type == "escalate" and not action.department:
            return "'escalate' action requires a 'department' field."
        if action.action_type == "request_info" and not action.message:
            return "'request_info' action requires a 'message' field."
        return None

    def _build_observation(self, feedback: str | None) -> SupportObservation:
        """Construct the observation the agent sees after each step."""
        ticket = self._current_ticket
        return SupportObservation(
            ticket_id=ticket["ticket_id"],
            ticket_text=ticket["ticket_text"],
            customer_name=ticket["customer_name"],
            order_id=ticket.get("order_id"),
            conversation_history=list(self._conversation_history),
            available_actions=self._get_available_actions(),
            last_action_feedback=feedback,
            done=self._done,
            reward=self._cumulative_reward,
            task_id=self._task_id,
            difficulty=ticket["difficulty"],
        )

    def _get_available_actions(self) -> list[str]:
        """
        Restrict available actions based on episode state.
        Prevents the agent from taking nonsensical sequences.
        """
        if self._done:
            return []

        actions = ["respond", "request_info", "resolve"]

        # Can only classify once
        if not self._classified:
            actions.insert(0, "classify")

        # Can only escalate once
        if not self._escalated:
            actions.append("escalate")

        return actions