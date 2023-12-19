import database
import sys


def main():
    pubkey = sys.argv[1]
    database.make_trusted(pubkey)
    print(f"{pubkey} now trusted.")


if __name__ == "__main__":
    main()
