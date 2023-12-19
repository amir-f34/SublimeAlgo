import configuration
import time
from tabulate import tabulate
import server
from job_runner import JobRunner
import encrypt


class Client:
    def __init__(self):
        self.host = configuration.config["Configuration"]["host"]
        self.auto = configuration.config["Configuration"]["autoVerify"]
        self.retry_wait = 0.1
        self.no_jobs_wait = 10

    def show_tasks(self):
        pubkey = encrypt.pubkey()
        print(f"Public key: {pubkey}")
        tasks = server.fetch_tasks()
        task_id = None
        table = [["name", "population", "generations"]]
        for task in tasks:
            table.append([task["_id"], task["population"], task["generations"]])
        print(tabulate(table))

    def run_loop(self, task_id):
        tasks = server.fetch_tasks()
        task_name = None
        gen = 0
        for task in tasks:
            if task["_id"] == task_id:
                task_name = task["name"]
                gen = task["generations"]
                break

        if task_name is None:
            print(f"Task not found: {task_name}")
            return

        while True:
            job = server.fetch_job(task_id)

            if isinstance(job, dict) and job.get("new_task"):
                task_id = job["new_task"]
            elif job:
                print(f"Task: {task_id}, Gen: {gen} Job: {job}, ")
                jr = JobRunner(task_id, job)
                starting_time = time.time()
                ret = jr.run_job()
                print(f"Returned: {ret} Took: {time.time()-starting_time}")
                for i in range(5):
                    if server.post_job(task_id, job, ret):
                        break
                    time.sleep(1)
            elif job is None:
                print("Request failed, trying again.")
            else:
                print("No available jobs, waiting.")
                time.sleep(self.no_jobs_wait)
            time.sleep(self.retry_wait)

            tasks = server.fetch_tasks()
            for task in tasks:
                if task["name"] == task_name or task["name"].split(".")[0] == task_name:
                    task_id = task["_id"]
                    gen = task["generations"]
                    break
