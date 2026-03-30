# models.py
# E-Commerce Customer Support Environment — Typed Models
# Extends OpenEnv core Action, Observation, and uses core State

from typing import Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class SupportAction(Action):
    """
    An action the agent can take during a customer support episode.

    Available action types:
    - classify   : Label the ticket with a category
    - respond    : Send a reply message to the customer
    - escalate   : Route the ticket to a specific department
    - request_info: Ask the customer for more information
    - resolve    : Mark the ticket as resolved and close the episode
    """

    action_type: Literal[
        "classify",
        "respond",
        "escalate",
        "request_info",
        "resolve"
    ] = Field(
        ...,
        description="The type of action the agent wants to take"
    )

    # Used when action_type = "classify"
    category: Optional[Literal[
        "billing",
        "delivery",
        "refund",
        "product_issue",
        "general_inquiry"
    ]] = Field(
        default=None,
        description="Ticket category — required when action_type is 'classify'"
    )

    # Used when action_type = "respond" or "request_info"
    message: Optional[str] = Field(
        default=None,
        description="The text message to send to the customer"
    )

    # Used when action_type = "escalate"
    department: Optional[Literal[
        "billing_team",
        "logistics_team",
        "technical_team",
        "legal_team",
        "senior_support"
    ]] = Field(
        default=None,
        description="Department to escalate to — required when action_type is 'escalate'"
    )

    # Used when action_type = "resolve"
    resolution_note: Optional[str] = Field(
        default=None,
        description="Summary of how the issue was resolved"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class SupportObservation(Observation):
    """
    What the agent sees at each step of the episode.

    Includes the current ticket, conversation history,
    available actions, and any status updates.
    """

    # The original customer ticket text
    ticket_id: str = Field(
        ...,
        description="Unique identifier for this support ticket"
    )

    ticket_text: str = Field(
        ...,
        description="The customer's message or complaint"
    )

    customer_name: str = Field(
        ...,
        description="Name of the customer who submitted the ticket"
    )

    order_id: Optional[str] = Field(
        default=None,
        description="Order ID referenced in the ticket, if any"
    )

    # Conversation history so far in this episode
    conversation_history: list[dict] = Field(
        default_factory=list,
        description=(
            "List of prior turns in this episode. "
            "Each entry is a dict with 'role' (agent/customer) and 'content'."
        )
    )

    # What actions are currently available to the agent
    available_actions: list[str] = Field(
        default_factory=lambda: [
            "classify", "respond", "escalate", "request_info", "resolve"
        ],
        description="List of action_types the agent may use at this step"
    )

    # Feedback from the last action taken
    last_action_feedback: Optional[str] = Field(
        default=None,
        description="System message about the result of the last action taken"
    )

    # Whether the episode has ended
    done: bool = Field(
        default=False,
        description="True if the episode is complete (resolved or max steps reached)"
    )

    # Current reward accumulated so far
    reward: float = Field(
        default=0.0,
        description="Cumulative reward earned so far in this episode"
    )

    # Task metadata
    task_id: str = Field(
        ...,
        description="Which task this episode belongs to: task1, task2, or task3"
    )

    difficulty: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="Difficulty level of this episode"
    )