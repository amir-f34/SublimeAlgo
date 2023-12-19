import requests
import configuration
import encrypt
import json
from torpy.http.requests import TorRequests


def base_request(uri, method="GET", headers=None, data=None, stream=False):
    host = configuration.config["Configuration"]["host"]

    if host.startswith("http://localhost:"):
        if method == "GET":
            return requests.get(host + uri, headers=headers, stream=stream)
        elif method == "POST":
            return requests.post(host + uri, headers=headers, data=data)
    else:
        # Use Tor
        with TorRequests() as tor_requests:
            with tor_requests.get_session() as sess:
                if method == "GET":
                    return sess.get(host + uri, headers=headers)
                elif method == "POST":
                    return sess.post(host + uri, headers=headers, data=data)


def fetch_tasks():
    data = base_request("/tasks")
    return data.json()


def fetch_nonce(pubkey):
    data = base_request("/nonce/" + pubkey)
    data = data.json()
    return data["nonce"]


def fetch_job(task_id):
    pubkey = encrypt.pubkey()
    nonce = fetch_nonce(pubkey)
    signature = encrypt.sign(nonce.encode("utf-8"))
    headers = {
        "content-type": "application/json",
        "signature": signature,
        "pubkey": pubkey,
        "nonce": nonce,
    }
    try:
        data = base_request(f"/new_job/{task_id}", headers=headers)
        data = data.json()
        return data
    except:
        return None


def post_job(task_id, job, ret):
    pubkey = encrypt.pubkey()
    nonce = fetch_nonce(pubkey)

    signature = encrypt.sign(nonce.encode("utf-8"))
    headers = {
        "content-type": "application/json",
        "signature": signature,
        "pubkey": pubkey,
        "nonce": nonce,
    }
    data = json.dumps({"returned": ret})
    try:
        base_request(
            f"/post_job/{task_id}/{job}", method="POST", data=data, headers=headers
        )
        return True
    except:
        return False


def download_task(task_id):
    local_filename = f"./tasks/{task_id}.zip"
    uri = f"/download_task/{task_id}"

    with base_request(uri, stream=True) as req:
        req.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in req.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename
