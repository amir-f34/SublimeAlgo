import database


class Reputation:
    def __init__(self):
        self.min_total = 2

    def add_task_rep(self, task_id, job_id, ret, pubkey):
        database.add_ret(task_id, job_id, ret, pubkey)

    def update_task_rep(self, task_id, job_id):
        matches = dict()
        task_rets = database.get_ret(task_id, job_id)
        if task_rets is None:
            database.create_ret(task_id, job_id)
            task_rets = database.get_ret(task_id, job_id)

        total = 0
        for pubkey, value in task_rets["ret"].items():
            rep = database.get_rep(pubkey)
            if rep["rep"] / ((1 + rep["tasks"] - rep["rep"]) ** 2) > 1:
                solved = 1
            else:
                solved = 0
            total += solved
            trusted = self.is_trusted(pubkey)
            if value in matches:
                matches[value]["matches"] += solved
                if trusted:
                    matches[value]["trusted"] = True
            else:
                matches[value] = {"matches": solved, "trusted": trusted}

        if len(task_rets["ret"]) >= self.min_total:
            for val, data in matches.items():
                if data["trusted"] or (total > 0 and data["matches"] / total > 0.5):
                    database.update_ret(task_id, job_id, value)
                    for pubkey, value in task_rets["ret"].items():
                        database.update_rep(pubkey, value == val)
                    database.update_ret(task_id, job_id, val)
                    break

    def is_trusted(self, pkey):
        if pkey in database.get_trusted():
            return True
        return False
