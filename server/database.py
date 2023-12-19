import pymongo
import random
import sys
import os

connection = pymongo.MongoClient("localhost", 27017)
db = connection.server


def wipe_database():
    db.generations.drop()
    db.nonces.drop()
    db.tasks.drop()
    db.task_rep.drop()
    db.pub_rep.drop()
    db.trusted.drop()
    for fn in os.listdir("./tasks"):
        if ".zip" in fn:
            os.remove(f"./tasks/{fn}")


def get_database():
    return db


## Tasks
def add_task(task_name, task_id, population):
    db.tasks.insert_one(
        {"_id": task_id, "name": task_name, "population": population, "generations": 0}
    )


def get_task_id_by_name(task_name):
    return db.tasks.find_one({"name": task_name})["_id"]


def get_task(task_id):
    return db.tasks.find_one({"_id": task_id})


def update_task(task_id, generations):
    db.tasks.update_one({"_id": task_id}, {"$set": {"generations": generations}})


def get_all_tasks():
    tasks = [x for x in db.tasks.find()]
    return tasks


## Jobs
def get_new_job_for_task(task_id, client_pubkey):
    task = get_task(task_id)
    generation = task["generations"]
    individuals = db.generations.find_one(
        {"task_id": task["_id"], "generation": generation}
    )["individuals"]

    current_assignment = get_assigned_job(client_pubkey, task_id)
    if current_assignment:
        return current_assignment

    for key, value in random.sample(individuals.items(), len(individuals)):
        rets = get_ret(task_id, key)
        if value is None and (not rets or client_pubkey not in rets["ret"]):
            assign_job(client_pubkey, task_id, key)
            return key

    assign_job(client_pubkey, task_id, {})
    return {}


def assign_job(client_pubkey, task_id, job_id):
    db.assigned_jobs.update_one(
        {"_id": client_pubkey + ":" + task_id}, {"$set": {"job": job_id}}, upsert=True
    )


def get_assigned_job(client_pubkey, task_id):
    data = db.assigned_jobs.find_one({"_id": client_pubkey + ":" + task_id})
    if data:
        return data["job"]
    return {}


def remove_assigned_job(client_pubkey, task_id):
    try:
        db.assigned_jobs.delete_one({"_id": client_pubkey + ":" + task_id})
    except:
        import traceback

        traceback.print_exc()


## Generations
def get_generation(task_id, generation):
    return db.generations.find_one({"task_id": task_id, "generation": generation})


def create_generation(task_id, generation, dna_strings):
    individuals = dict([[x, None] for x in dna_strings])
    db.generations.insert_one(
        {"task_id": task_id, "generation": generation, "individuals": individuals}
    )


def update_generation(task_id, generation, update):
    update_doc = {"$set": {}}
    for key in update:
        update_doc["$set"]["individuals." + key] = update[key]
    db.generations.update_one(
        {"task_id": task_id, "generation": generation}, update_doc
    )


## Signing nonces
def new_nonce():
    return hex(int(random.random() * (2**64)))[2:]


def get_client_nonce(pubkey):
    doc = db.nonces.find_one({"_id": pubkey})
    if doc:
        return doc["nonce"]
    else:
        update_client_nonce(pubkey)
        return db.nonces.find_one({"_id": pubkey})["nonce"]


def update_client_nonce(pubkey):
    db.nonces.update_one({"_id": pubkey}, {"$set": {"nonce": new_nonce()}}, upsert=True)


## Pubkeys Reputations
def get_rep(pubkey):
    doc = db.pub_rep.find_one({"_id": pubkey})
    if not doc:
        doc = {"_id": pubkey, "rep": 0, "tasks": 0}
        db.pub_rep.insert_one(doc)
    return doc


def update_rep(pubkey, matched):
    amt = int(matched)
    db.pub_rep.update_one(
        {"_id": pubkey}, {"$inc": {"tasks": 1, "rep": amt}}, upsert=True
    )


## Pubkeys trusted
def get_trusted():
    trusted = list()
    for doc in db.trusted.find():
        trusted.append(doc["_id"])
    return trusted


def make_trusted(pubkey):
    db.trusted.insert_one({"_id": pubkey})


## Task Returns
def create_ret(task_id, job_id):
    db.task_rep.insert_one({"_id": task_id + "-" + job_id, "ret": {}})


def get_ret(task_id, job_id):
    return db.task_rep.find_one({"_id": task_id + "-" + job_id})


def add_ret(task_id, job_id, ret, pubkey):
    db.task_rep.update_one(
        {"_id": task_id + "-" + job_id}, {"$set": {"ret." + pubkey: ret}}, upsert=True
    )


def update_ret(task_id, job_id, ret):
    db.task_rep.update_one(
        {"_id": task_id + "-" + job_id}, {"$set": {"final": ret}}, upsert=True
    )


def main():
    if sys.argv[1] == "WIPE":
        wipe_database()


if __name__ == "__main__":
    main()
