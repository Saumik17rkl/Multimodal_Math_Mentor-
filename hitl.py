from datetime import datetime
from pymongo.collection import Collection


def init_hitl_fields():
    """
    Default HITL field structure.
    Use when inserting a new solution.
    """
    return {
        "approved": None,               # None = not reviewed
        "human_feedback": None,
        "corrected_solution": None,
        "reviewed_at": None
    }


def save_hitl_review(
    collection: Collection,
    question: str,
    original_solution: str,
    decision: str,
    feedback: str | None = None,
    corrected_solution: str | None = None
):
    """
    Persist HITL decision into MongoDB.

    decision must be one of:
    - "Approve"
    - "Reject"
    - "Edit & Approve"
    """

    update_data = {
        "approved": decision in ("Approve", "Edit & Approve"),
        "human_feedback": feedback,
        "reviewed_at": datetime.utcnow()
    }

    if decision == "Edit & Approve":
        update_data["corrected_solution"] = corrected_solution

    collection.update_one(
        {
            "question": question,
            "solution": original_solution
        },
        {
            "$set": update_data
        }
    )


def get_final_solution(record: dict) -> str:
    """
    Return corrected solution if present,
    otherwise return original model solution.
    """
    if record.get("corrected_solution"):
        return record["corrected_solution"]
    return record["solution"]


def is_human_approved(record: dict) -> bool:
    """
    Check if solution is approved by human.
    """
    return record.get("approved") is True
