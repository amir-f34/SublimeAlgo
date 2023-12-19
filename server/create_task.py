import os.path
import sys
import database
import json
import hashlib

from zipfile import ZipFile


class TaskCreator:
    def __init__(self, task_name):
        self.task_name = task_name

    def create(self):
        if not os.path.exists(f"./tasks/{self.task_name}"):
            raise Exception("task doesn't exist")
        with ZipFile(f"./tasks/{self.task_name}.zip", "w") as myzip:
            for fname in os.listdir(f"./tasks/{self.task_name}"):
                if os.path.isfile(f"./tasks/{self.task_name}/{fname}"):
                    myzip.write(f"./tasks/{self.task_name}/{fname}", fname)
        md = json.loads(open(f"./tasks/{self.task_name}/meta.json").read())
        pop = md["population"]

        hashstring = hashlib.sha256(
            open(f"./tasks/{self.task_name}.zip", "rb").read()
        ).hexdigest()
        new_task_name = f"{self.task_name}-{hashstring[:8]}"
        os.rename(f"./tasks/{self.task_name}.zip", f"./tasks/{new_task_name}.zip")
        task_id = new_task_name
        if not database.get_task(task_id):
            database.add_task(self.task_name, task_id, pop)


def main():
    task_name = sys.argv[1]
    TaskCreator(task_name).create()


if __name__ == "__main__":
    main()
