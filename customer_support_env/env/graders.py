# env/graders.py
# E-Commerce Customer Support Environment — Task Graders
#
# Each grader:
#   - Takes the completed episode state as input
#   - Returns a score between 0.0 and 1.0
#   - Is fully deterministic — no randomness, no LLM calls
#   - Provides a breakdown dict explaining how the score was computed

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Grader Result — returned by every grader
# ---------------------------------------------------------------------------

@dataclass
class GraderResult:
    """
    Standardised result returned by every grader.

    score      : Final score 0.0 – 1.0
    breakdown  : Per-signal scores that add up to final score
    passed     : True if score >= 0.5 (considered a pass)
    feedback   : Human-readable explanation of the score
    """
    score: float
    breakdown: dict = field(default_factory=dict)
    passed: bool = False
    feedback: str = ""

    def __post_init__(self):
        self.score = round(max(0.0, min(1.0, self.score)), 4)
        self.passed = self.score >= 0.5


# ---------------------------------------------------------------------------
# Task 1 Grader — Ticket Classification (Easy)
# ---------------------------------------------------------------------------

class Task1Grader:
    """
    Grades the agent on correct ticket classification only.

    Scoring:
      - Correct category on first attempt  : 1.0
      - Correct category after 1 retry     : 0.5
      - Correct category after 2+ retries  : 0.25
      - Never classified / wrong category  : 0.0
    """

    def grade(
        self,
        ticket: dict,
        classification_value: str | None,
        classify_attempt_count: int = 1,
    ) -> GraderResult:
        ground_truth = ticket.get("ground_truth_category")
        breakdown = {}

        if not classification_value:
            return GraderResult(
                score=0.0,
                breakdown={"classified": 0.0},
                feedback="Agent never called classify action."
            )

        if classification_value == ground_truth:
            if classify_attempt_count == 1:
                score = 1.0
                breakdown["correct_first_attempt"] = 1.0
                feedback = f"Correct classification '{classification_value}' on first attempt."
            elif classify_attempt_count == 2:
                score = 0.5
                breakdown["correct_second_attempt"] = 0.5
                feedback = f"Correct classification '{classification_value}' on second attempt."
            else:
                score = 0.25
                breakdown["correct_late_attempt"] = 0.25
                feedback = f"Correct classification '{classification_value}' after {classify_attempt_count} attempts."
        else:
            score = 0.0
            breakdown["wrong_category"] = 0.0
            feedback = (
                f"Wrong classification: agent said '{classification_value}', "
                f"expected '{ground_truth}'."
            )

        return GraderResult(score=score, breakdown=breakdown, feedback=feedback)


# ---------------------------------------------------------------------------
# Task 2 Grader — Response Generation (Medium)
# ---------------------------------------------------------------------------

class Task2Grader:
    """
    Grades the agent on classification accuracy AND response quality.

    Scoring breakdown (total = 1.0):
      - Correct classification             : 0.30
      - Required response elements present : 0.30  (split equally per element)
      - No forbidden statements used       : 0.20
      - Response length adequate (>50 chars): 0.10
      - Ticket ID mentioned in response    : 0.10
    """

    def grade(
        self,
        ticket: dict,
        classification_value: str | None,
        response_text: str | None,
    ) -> GraderResult:
        breakdown = {}
        total = 0.0

        ground_truth = ticket.get("ground_truth_category")
        expected_elements = ticket.get("expected_response_elements", [])
        forbidden = ticket.get("forbidden_statements", [])
        ticket_id = ticket.get("ticket_id", "")
        order_id = ticket.get("order_id", "")

        # --- Signal 1: Correct classification (0.30) ---
        if classification_value and classification_value == ground_truth:
            breakdown["correct_classification"] = 0.30
            total += 0.30
        else:
            breakdown["correct_classification"] = 0.0

        if not response_text:
            breakdown["response_elements"] = 0.0
            breakdown["no_forbidden_statements"] = 0.20
            breakdown["response_length"] = 0.0
            breakdown["ticket_id_mentioned"] = 0.0
            return GraderResult(
                score=round(total, 4),
                breakdown=breakdown,
                feedback="Agent never sent a response."
            )

        response_lower = response_text.lower()

        # --- Signal 2: Required response elements (0.30) ---
        # Each element is worth an equal share of 0.30
        if expected_elements:
            element_score_per = 0.30 / len(expected_elements)
            element_total = 0.0
            missing = []

            for element in expected_elements:
                if self._check_element(element, response_lower, ticket):
                    element_total += element_score_per
                else:
                    missing.append(element)

            breakdown["response_elements"] = round(element_total, 4)
            total += element_total

            if missing:
                missing_str = ", ".join(missing)
            else:
                missing_str = "none"
        else:
            breakdown["response_elements"] = 0.30
            total += 0.30
            missing_str = "none"

        # --- Signal 3: No forbidden statements (0.20) ---
        used_forbidden = [f for f in forbidden if f.lower() in response_lower]
        if not used_forbidden:
            breakdown["no_forbidden_statements"] = 0.20
            total += 0.20
        else:
            breakdown["no_forbidden_statements"] = 0.0

        # --- Signal 4: Response length adequate (0.10) ---
        if len(response_text.strip()) >= 50:
            breakdown["response_length"] = 0.10
            total += 0.10
        else:
            breakdown["response_length"] = 0.0

        # --- Signal 5: Ticket ID or Order ID mentioned (0.10) ---
        if ticket_id.lower() in response_lower or (order_id and order_id.lower() in response_lower):
            breakdown["ticket_id_mentioned"] = 0.10
            total += 0.10
        else:
            breakdown["ticket_id_mentioned"] = 0.0

        feedback_parts = [
            f"Classification: {'correct' if breakdown['correct_classification'] > 0 else 'wrong'}.",
            f"Missing elements: {missing_str}.",
            f"Forbidden statements used: {used_forbidden if used_forbidden else 'none'}.",
            f"Response length: {'ok' if breakdown['response_length'] > 0 else 'too short'}.",
            f"Ticket ID referenced: {'yes' if breakdown['ticket_id_mentioned'] > 0 else 'no'}.",
        ]

        return GraderResult(
            score=round(total, 4),
            breakdown=breakdown,
            feedback=" ".join(feedback_parts)
        )

    def _check_element(self, element: str, response_lower: str, ticket: dict) -> bool:
        """
        Check whether a required response element is present.
        Each element maps to a set of keyword signals.
        """
        element_keywords = {
            "acknowledgment": [
                "sorry", "apologise", "apologize", "understand",
                "i see", "thank you for", "we understand", "i'm sorry"
            ],
            "timeline": [
                "business day", "within", "hours", "days",
                "by", "processed", "expect", "soon"
            ],
            "ticket_id": [
                ticket.get("ticket_id", "").lower(),
                ticket.get("order_id", "").lower() if ticket.get("order_id") else None,
            ],
            "tracking_advice": [
                "track", "tracking", "tracking number",
                "courier", "carrier", "shipment status"
            ],
            "escalation_option": [
                "escalat", "senior", "specialist", "manager",
                "team", "department", "transfer"
            ],
            "policy_reference": [
                "policy", "30 day", "30-day", "within 30",
                "refund policy", "return policy", "entitled"
            ],
            "next_steps": [
                "next step", "will", "going to", "we'll",
                "our team", "you will", "expect", "contact"
            ],
            "troubleshooting_or_replacement": [
                "reset", "restart", "update", "firmware", "troubleshoot",
                "replacement", "replace", "exchange", "new unit"
            ],
            "accurate_answer": [
                # General presence check — if response is long enough
                # and not forbidden, assume accuracy
                # (deterministic proxy for correctness)
            ],
        }

        keywords = element_keywords.get(element, [])

        # Filter out None values (e.g. missing order_id)
        keywords = [k for k in keywords if k]

        if not keywords:
            # If no keywords defined, give benefit of the doubt if response is substantial
            return len(response_lower) > 100

        return any(kw in response_lower for kw in keywords)


# ---------------------------------------------------------------------------
# Task 3 Grader — Multi-turn Escalation Handling (Hard)
# ---------------------------------------------------------------------------

class Task3Grader:
    """
    Grades the agent on multi-turn escalation handling.

    Scoring breakdown (total = 1.0):
      - Correct escalation department      : 0.25
      - Acknowledged previous failures     : 0.20
      - Did not contradict prior turns      : 0.20
      - Provided concrete next steps        : 0.15
      - Special criteria met               : 0.10  (e.g. legal threat acknowledged)
      - Sentiment improvement signal       : 0.10
    """

    def grade(
        self,
        ticket: dict,
        escalation_department: str | None,
        conversation_history: list[dict],
        resolution_note: str | None,
        classified: bool,
        responded: bool,
    ) -> GraderResult:
        breakdown = {}
        total = 0.0

        criteria = ticket.get("resolution_criteria", {})
        agent_turns = [
            t["content"].lower()
            for t in conversation_history
            if t.get("role") == "agent"
        ]
        all_agent_text = " ".join(agent_turns)

        # --- Signal 1: Correct escalation department (0.25) ---
        correct_dept = ticket.get("correct_escalation_path")
        if escalation_department and escalation_department == correct_dept:
            breakdown["correct_escalation"] = 0.25
            total += 0.25
        elif escalation_department and escalation_department != correct_dept:
            breakdown["correct_escalation"] = 0.0
        else:
            breakdown["correct_escalation"] = 0.0

        # --- Signal 2: Acknowledged previous failures (0.20) ---
        if criteria.get("must_acknowledge_previous_failures"):
            acknowledgment_keywords = [
                "previous", "earlier", "last time", "before",
                "i understand your frustration", "apologise for",
                "apologize for", "failed", "not resolved", "should have"
            ]
            if any(kw in all_agent_text for kw in acknowledgment_keywords):
                breakdown["acknowledged_failures"] = 0.20
                total += 0.20
            else:
                breakdown["acknowledged_failures"] = 0.0
        else:
            breakdown["acknowledged_failures"] = 0.20
            total += 0.20

        # --- Signal 3: No contradictions across turns (0.20) ---
        contradiction_score = self._check_no_contradictions(agent_turns)
        breakdown["no_contradictions"] = round(contradiction_score * 0.20, 4)
        total += breakdown["no_contradictions"]

        # --- Signal 4: Concrete next steps provided (0.15) ---
        if criteria.get("must_provide_timeline") or criteria.get("must_provide_concrete_next_steps"):
            next_step_keywords = [
                "within", "business day", "by", "today", "tomorrow",
                "will contact", "will process", "will investigate",
                "escalating now", "our team will", "expect"
            ]
            if any(kw in all_agent_text for kw in next_step_keywords):
                breakdown["concrete_next_steps"] = 0.15
                total += 0.15
            else:
                breakdown["concrete_next_steps"] = 0.0
        else:
            breakdown["concrete_next_steps"] = 0.15
            total += 0.15

        # --- Signal 5: Special criteria (0.10) ---
        special_score = self._check_special_criteria(criteria, all_agent_text)
        breakdown["special_criteria"] = round(special_score * 0.10, 4)
        total += breakdown["special_criteria"]

        # --- Signal 6: Sentiment improvement proxy (0.10) ---
        # Proxy: agent used de-escalation language
        deescalation_keywords = [
            "understand", "hear you", "priority", "important to us",
            "personally", "ensure", "make this right", "resolve",
            "take responsibility", "sincerely"
        ]
        deescalation_count = sum(
            1 for kw in deescalation_keywords if kw in all_agent_text
        )
        sentiment_score = min(1.0, deescalation_count / 3)
        breakdown["sentiment_improvement"] = round(sentiment_score * 0.10, 4)
        total += breakdown["sentiment_improvement"]

        # Build feedback
        feedback_parts = [
            f"Escalation: {'correct' if breakdown['correct_escalation'] > 0 else 'wrong or missing'}.",
            f"Acknowledged failures: {'yes' if breakdown['acknowledged_failures'] > 0 else 'no'}.",
            f"Contradictions: {'none detected' if breakdown['no_contradictions'] >= 0.15 else 'possible contradiction found'}.",
            f"Concrete next steps: {'yes' if breakdown['concrete_next_steps'] > 0 else 'no'}.",
            f"Special criteria: {breakdown['special_criteria']}/0.10.",
            f"Sentiment de-escalation signals: {deescalation_count} found.",
        ]

        return GraderResult(
            score=round(total, 4),
            breakdown=breakdown,
            feedback=" ".join(feedback_parts)
        )

    def _check_no_contradictions(self, agent_turns: list[str]) -> float:
        """
        Simple contradiction detector.
        Looks for direct negations of statements made in earlier turns.
        Returns 1.0 if no contradictions, 0.0 if contradiction detected.
        """
        if len(agent_turns) <= 1:
            return 1.0

        # Contradiction signal pairs — if both appear across turns, flag it
        contradiction_pairs = [
            ("will refund", "cannot refund"),
            ("can replace", "cannot replace"),
            ("will be delivered", "cannot guarantee delivery"),
            ("refund approved", "refund denied"),
            ("replacement approved", "no replacement"),
            ("will escalate", "cannot escalate"),
        ]

        all_text = " ".join(agent_turns)
        for phrase_a, phrase_b in contradiction_pairs:
            if phrase_a in all_text and phrase_b in all_text:
                return 0.0

        return 1.0

    def _check_special_criteria(self, criteria: dict, all_agent_text: str) -> float:
        """
        Check ticket-specific special criteria.
        Returns 1.0 if all applicable criteria met, partial otherwise.
        """
        checks = []

        if criteria.get("must_acknowledge_legal_threat"):
            legal_keywords = [
                "understand your concern", "take this seriously",
                "escalate", "senior", "legal", "specialist",
                "investigate", "review"
            ]
            checks.append(any(kw in all_agent_text for kw in legal_keywords))

        if criteria.get("must_not_dismiss_legal_threat"):
            dismissal_keywords = [
                "no need for legal", "unnecessary", "overreacting",
                "not a legal matter"
            ]
            # Pass if no dismissal language found
            checks.append(not any(kw in all_agent_text for kw in dismissal_keywords))

        if criteria.get("must_offer_refund_as_option"):
            refund_keywords = ["refund", "money back", "reimburse", "full refund"]
            checks.append(any(kw in all_agent_text for kw in refund_keywords))

        if criteria.get("must_acknowledge_product_page_claim"):
            product_keywords = [
                "product page", "description", "listed", "advertised",
                "compatibility", "our website"
            ]
            checks.append(any(kw in all_agent_text for kw in product_keywords))

        if criteria.get("must_reopen_investigation"):
            reopen_keywords = [
                "reopen", "re-open", "investigate", "look into",
                "review your case", "escalate to"
            ]
            checks.append(any(kw in all_agent_text for kw in reopen_keywords))

        if not checks:
            return 1.0  # No special criteria defined — full score

        passed = sum(checks)
        return passed / len(checks)


# ---------------------------------------------------------------------------
# Grader Factory — single entry point
# ---------------------------------------------------------------------------

def get_grader(task_id: str):
    """
    Returns the correct grader instance for a given task_id.

    Usage:
        grader = get_grader("task1_classify")
        result = grader.grade(ticket, classification_value="billing")
    """
    graders = {
        "task1_classify": Task1Grader(),
        "task2_respond": Task2Grader(),
        "task3_escalate": Task3Grader(),
    }
    if task_id not in graders:
        raise ValueError(
            f"Unknown task_id: '{task_id}'. "
            f"Must be one of: {list(graders.keys())}"
        )
    return graders[task_id]