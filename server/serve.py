#!/usr/bin/env python3
from typing import Annotated, Any
from fastapi import Request, FastAPI, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jobs
import database
import encrypt
from reputation import Reputation
import json

__author__ = "Amir F."
__copyright__ = "Copyright 2023, Amir F."
__credits__ = ["Amir F."]
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "amir.f34@proton.me"
__status__ = "Completed"

app = FastAPI()

with open("manifest.json", "r") as read_content:
    manifest = json.load(read_content)


@app.get("/manifest.json")
async def manifest_file():
    return manifest


@app.get("/tasks")
async def tasks_get():
    tasks = jobs.get_tasks()
    return tasks


@app.get("/download_task/{task_id}")
async def download_task(task_id: str):
    data = jobs.download_task(task_id)
    return FileResponse(
        path=data["path"],
        media_type="application/octet-stream",
        filename=data["filename"],
    )


@app.get("/nonce/{pubkey}")
async def nonce(pubkey: str):
    nonce = database.get_client_nonce(pubkey)
    return {"nonce": nonce}


@app.get("/new_job/{task_id}")
async def new_job(request: Request):
    body = await request.body()
    public_key = request.headers["pubkey"]
    signature = request.headers["signature"]
    nonce = request.headers["nonce"]
    task_id = request.path_params["task_id"]
    rnonce = database.get_client_nonce(public_key)
    if rnonce != nonce:
        return {"error": "Wrong nonce"}
    if not encrypt.verify(nonce.encode("utf-8"), signature, public_key):
        return {"error": "Wrong signature"}
    database.update_client_nonce(public_key)
    job = jobs.get_new_job(task_id, public_key)
    return job


@app.post("/post_job/{task_id}/{job_id}")
async def post_job(request: Request):
    body = await request.json()
    task_id = request.path_params["task_id"]
    job_id = request.path_params["job_id"]
    public_key = request.headers["pubkey"]
    signature = request.headers["signature"]
    nonce = request.headers["nonce"]
    rnonce = database.get_client_nonce(public_key)
    if rnonce != nonce:
        return {"error": "Wrong nonce"}
    if not encrypt.verify(nonce.encode("utf-8"), signature, public_key):
        return {"error": "Wrong signature"}
    database.update_client_nonce(public_key)

    ret = body["returned"]
    rep = Reputation()
    rep.add_task_rep(task_id, job_id, ret, public_key)
    rep.update_task_rep(task_id, job_id)
    database.remove_assigned_job(public_key, task_id)
