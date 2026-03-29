def grade_task(task, action):
    msg = (action.message or "").lower()

    if task["expected"] in msg:
        return 1.0

    return 0.0