#!/usr/bin/env python3
from typing import Annotated, Any
from fastapi import Request, FastAPI, Header, Response
from fastapi.responses import FileResponse
from contextlib import redirect_stdout
import json
import sys
import sublime_cli
import io

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


@app.post("/sublime_cli")
async def post_command(request: Request):
    body = await request.json()
    args = body["args"]
    sys.argv = args

    with redirect_stdout(io.StringIO()) as f:
        sublime_cli.main()

    output = f.getvalue()
    return Response(content=output, media_type="application/text")
