import subprocess
import sys
import os
import os.path
import json
from zipfile import ZipFile
import server
import configuration
import sys
import verify_task


class JobRunner:
    def __init__(self, task_id, job):
        self.task_id = task_id
        self.job = job

    def run_job(self):
        fname = self.task_id
        if not os.path.exists(fname):
            server.download_task(fname)
        if not os.path.exists(f"./tasks/{self.task_id}"):
            self.unzip_task()
        if verify_task.is_verified(self.task_id):
            self.install_packages()
            ret = self.run_proc()["ret"]
            return ret
        else:
            print(f"Task downloaded to ./tasks/{self.task_id}")
            return ret

    def unzip_task(self):
        try:
            os.mkdir(f"./tasks/{self.task_id}")
        except FileExistsError:
            pass

        with ZipFile(f"./tasks/{self.task_id}.zip", "r") as zf:
            zf.extractall(f"./tasks/{self.task_id}/")

    def install_packages(self):
        req = f"./tasks/{self.task_id}/requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])

    def run_proc(self):
        fname = f"./tasks/{self.task_id}/task.py"
        ret = subprocess.check_output([sys.executable, fname, self.job])
        return json.loads(ret)
