#!/usr/bin/env python3
import client
import argparse
import configuration
import verify_task

__author__ = "Amir F."
__copyright__ = "Copyright 2023, Amir F."
__credits__ = ["Amir F."]
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "amir.f34@proton.me"
__status__ = "Completed"


def main():
    parser = argparse.ArgumentParser(description="Sing CLI")

    parser.add_argument(
        "-s", "--show", action="store_true", help="Show availables tasks."
    )

    parser.add_argument(
        "-t", "--task", nargs=1, metavar="taskName", type=str, help="Task to run."
    )

    parser.add_argument(
        "-f",
        "--config",
        nargs=1,
        metavar="configFile",
        type=str,
        help="Configuration file to use (default: subl.conf).",
    )
    parser.add_argument(
        "-v",
        "--verify",
        nargs=1,
        metavar="verifyTask",
        type=str,
        help="TaskName to verify.",
    )

    args = parser.parse_args()
    if args.config != None:
        configuration.parse_config(args.config[0])
    else:
        configuration.parse_config()

    if args.verify != None:
        verify_task.verify_task(args.verify[0])
        return None
    app = client.Client()

    if args.task != None:
        app.run_loop(args.task[0])
    else:
        app.show_tasks()


if __name__ == "__main__":
    main()
