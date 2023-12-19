==Overview

This is the repo for the OpenD/I Hackathon 2023, HyperCycle challenge 1/2. While the primary development of this repo was designed for the decentralized incremental learning challenge, the anonymity protections in the solution (via Tor support) may qualify for the privacy focus of challenge 1.

During the first half of the hackathon, my focus was on making the network mode functional, and during the second half, it was to create example genetic algorithms applied to neural architectures to show that they can rediscover classic findings in deep learning. However, while having issues with the skip connection task, I had the idea for the GAS algorithm, and my focus shifted towards that instead. 

As such, most of the genetic algorithm tasks were implemented in the single task mode instead of the network mode (with clients and servers), so they could be quicker to develop and test. The reference implementation for the decentralized challenge is in the `task01` task and is referenced below in the network mode section.


==Installation

Before running, be sure to have git, python3, pip, and virtualenv installed. Also make sure you have mongodb installed and are running a database on port 27017 (default).

```
sudo apt install python3 python3-dev virtualenv
```

clone the repo:

`git clone https://github.com/amir-f34/SublimeAlgo`


==Running in Network Mode

=SERVER:

To run a server:

1.) Start a mongodb instance
2.) Run:
`virtualenv venv`
`source ./venv/bin/activate`
`cd SublimeAlgo`
`pip install -r requirements.txt`
`cd server`
`uvicorn serve:app --reload`

To create a new task (for example: `task09`):

1.) Add `task09` to tasks/
2.) Add a corresponding task manager to `task_manager.py` for genetic algorithm specifics. 
3.) Run `python3 create_task.py task09`
4.) Run/Restart the task manager: `python3 task_manager.py`. Be sure to add the task to the manager loop in `main()`.

To run the example (task01) in the demo, just run `python3 task_manager.py`

To create a trusted key (for reputation bootstrapping)

`python3 create_trusted.py <Public Key>`


=CLIENT

To run a client:

`virtualenv venv`
`source ./venv/bin/activate`
`cd SublimeAlgo`
`pip install -r requirements.txt`
`cd client`
`python3 sublime_cli.py ...`

For help options:

`python3 sublime_cli.py -h`

To specify a different config file, or create a new one if it doesn't exist:

`python3 sublime_cli.py -f subl2.conf`

To show current tasks:

`python3 sublime_cli.py -s`

To download a task (with autoverify off):

`python3 sublime_cli.py -t task01-18ed0f86

To mark a task as verified:

`python3 sublime_cli.py -v task01-18ed0f86`

To run a verified task:

`python3 sublime_cli.py -t task01-18ed0f86`


You can also use the CLI with the API server:

`uvicorn client_serve:app --reload --port=8001`

And then access it with:

`curl -XPOST http://localhost:8001/sublime_cli -d '{"args": ["-s"]}'`

==Running in Single Task Mode

Install the requirements as in the network mode, and then run

`python3 task_single.py` followed by either `--task1` `--task4` `--task5` or `--task8`


=Contact Information

Amir F.
amir.f34@proton.me
0x1D7e4845f19c801651280CBDD84d70ea511bF954

