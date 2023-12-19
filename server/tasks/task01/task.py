import sys
import json


class Task1:
    def __init__(self, dna_string):
        self.dna_string = dna_string

    def run(self):
        top = 1 * self.dna_string.count("a") + 4 * self.dna_string.count("dd")
        bottom = (len(self.dna_string) ** 2) / 256 + 1
        return top / bottom


def main(dna_string):
    net = Task1(dna_string)
    ret = net.run()
    print(ret)


if __name__ == "__main__":
    main(sys.argv[1])
