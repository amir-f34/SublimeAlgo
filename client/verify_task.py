from pathlib import Path
import os.path
import configuration


def verify_task(task_id):
    Path(f"./tasks/{task_id}.verify").touch()
    print(f"Verified task: {task_id}")


def is_verified(task_id):
    if os.path.exists(f"./tasks/{task_id}.verify"):
        return True
    if configuration.config["Configuration"]["autoverify"] == "true":
        return False
