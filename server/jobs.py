import database


def get_tasks():
    return database.get_all_tasks()


def get_new_job(task_id, pubkey):
    return database.get_new_job_for_task(task_id, pubkey)


def download_task(task_id):
    doc = database.get_task(task_id)
    return {"filename": doc["name"], "path": f"./tasks/{doc['_id']}.zip"}
