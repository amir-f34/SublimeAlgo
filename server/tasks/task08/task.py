import sys
import json


class Task8:
    def __init__(self, dna_string):
        self.dna_string = dna_string

    def run(self):
        top1 = (
            1 * self.dna_string.count("a")
            + 4 * self.dna_string.count("ab")
            + 8 * self.dna_string.count("abc")
        )
        top2 = (
            0 * self.dna_string.count("d")
            + 0 * self.dna_string.count("de")
            + 2 * self.dna_string.count("def")
            + 32 * self.dna_string.count("defg")
        )
        top = top1 + top2
        bottom = (len(self.dna_string) ** 2) / 256 + 1
        return top / bottom


def main(dna_string):
    net = Task8(dna_string)
    ret = net.run()
    print(json.dumps({"ret": ret}))


if __name__ == "__main__":
    main(sys.argv[1])
