#!/usr/bin/env python3
import os
import sys
import hashlib
import random
import time

__author__ = "Amir F."
__copyright__ = "Copyright 2023, Amir F."
__credits__ = ["Amir F."]
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "amir.f34@proton.me"
__status__ = "Completed"


random.seed(0)


class TaskSingle:
    def __init__(self):
        self.task_name = ""
        self.population = 50
        self.cull_rate = 1 / 5
        self.default_length = 5
        self.dna_chars = "abcde"
        self.starting_chromosomes = 1
        self.uneven_crossover = 0.1
        self.mutation_chance = 0.4
        self.generations = []
        self.reverse = False

        self.gen_count = 0
        self.fit_lookup = {}

    def fill_generation_zero(self):
        self.generations = [
            {
                "individuals": dict(
                    [
                        (self.generate_random(self.starting_chromosomes), None)
                        for x in range(self.population)
                    ]
                )
            }
        ]

    def generate_random(self, chromosomes=1):
        output = [
            "".join([random.choice(self.dna_chars) for x in range(self.default_length)])
            for x in range(chromosomes)
        ]
        return "|".join(output)

    def start(self):
        self.fill_generation_zero()

        while True:
            self.step()
            print(f"{self.gen_count}")

    def get_fit(self, dna):
        if dna not in self.fit_lookup:
            self.fit_lookup[dna] = self.run(dna)
        return self.fit_lookup[dna]

    def step(self):
        generation = self.generations[-1]
        update = {}
        flag = True
        best = ""
        best_fit = 999999999

        for key, data in generation["individuals"].items():
            if data is None:
                fitness = self.get_fit(key)
                generation["individuals"][key] = fitness
            else:
                fitness = data
            if fitness < best_fit:
                best_fit = fitness
                best = key

        print(
            f"task: {self.task_name}, generation: {self.gen_count} best: {best} fit: {best_fit} individuals: {generation} "
        )
        self.create_next_generation(generation)
        self.gen_count += 1

    def create_next_generation(self, generation):
        individual_scores = [
            (x, generation["individuals"][x]) for x in generation["individuals"]
        ]
        random.shuffle(individual_scores)
        best_individuals = sorted(
            individual_scores, key=lambda x: x[1], reverse=self.reverse
        )
        best_individuals = best_individuals[: int(self.population * self.cull_rate)]
        new_individuals = best_individuals.copy()
        random.shuffle(best_individuals)
        best_length = len(best_individuals)
        while len(new_individuals) < self.population:
            ind1 = random.choice(range(best_length))
            ind2 = random.choice(range(best_length - 1))
            if ind2 >= ind1:
                ind2 += 1
            new_individuals.extend(
                self.reproduce(best_individuals[ind1], best_individuals[ind2])
            )
        dna_strings = [x[0] for x in new_individuals]
        new_gen = {"individuals": dict()}
        for dna in dna_strings:
            new_gen["individuals"][dna] = None
        self.generations.append(new_gen)

    def reproduce(self, ind1, ind2):
        strings1 = ind1[0].split("|")
        strings2 = ind2[0].split("|")
        max_chromo = max(len(strings1), len(strings2))

        while len(strings1) < max_chromo:
            strings1.append("")
        while len(strings2) < max_chromo:
            strings2.append("")

        output1 = []
        output2 = []
        for string1, string2 in zip(strings1, strings2):
            if len(string1) > 0:
                pos1 = random.choice(range(len(string1)))
            else:
                pos1 = 0
            if len(string2) > 0:
                if random.random() < self.uneven_crossover:
                    pos2 = random.choice(range(len(string2)))
                else:
                    pos2 = pos1
            else:
                pos2 = 0
            output1.append(string1[:pos1] + string2[pos2:])
            output2.append(string2[:pos2] + string1[pos1:])
        output1 = "|".join(output1)
        output2 = "|".join(output2)
        for i in range(200):
            output1 = self.mutate(output1)
            if output1 not in self.fit_lookup:
                break
        for i in range(200):
            output2 = self.mutate(output2)
            if output2 not in self.fit_lookup:
                break

        return [[output1, None], [output2, None]]

    def mutate(self, string):
        if random.random() < self.mutation_chance:
            string_array = [x for x in string]
            choice = random.choice(["dup", "insert", "inplace", "delete", "multiplace"])
            if string:
                char = random.choice(range(len(string)))
            else:
                choice = "insert"
                char = 0

            if choice == "dup":
                string_array = (
                    string_array[:char] + [string_array[char]] + string_array[char:]
                )
            elif choice == "insert":
                string_array = (
                    string_array[:char]
                    + [random.choice(self.dna_chars)]
                    + string_array[char:]
                )
            elif choice == "inplace":
                string_array[char] = random.choice(self.dna_chars)
            elif choice == "delete" and len(string) > 0:
                string_array = string_array[:char] + string_array[char + 1 :]
            elif choice == "multiplace":
                if random.random() < 0.5:
                    string_array = (
                        string_array[:char]
                        + [random.choice(self.dna_chars) for x in range(2)]
                        + string_array[char:]
                    )
                else:
                    string_array = string_array[:char] + string_array[char + 2 :]

            string = "".join(string_array)
        return string


class Task1(TaskSingle):
    def __init__(self):
        super(Task1, self).__init__()
        self.task_name = "task1"
        self.population = 50
        self.cull_rate = 1 / 5
        self.default_length = 5
        self.dna_chars = "0123456789abcdef"
        self.uneven_crossover = 0.1
        self.mutation_chance = 0.4
        self.reverse = True

    def run(self, dna_string):
        import tasks.task01.task

        ret = tasks.task01.task.Task1(dna_string).run()
        time.sleep(0.01)
        return ret


class Task4(TaskSingle):
    def __init__(self):
        super(Task4, self).__init__()
        self.task_name = "task4"
        self.dna_chars = "0123456789abcdef|"
        self.default_length = 4
        self.population = 50
        self.mutation_chance = 0.5
        self.starting_chromosomes = 2

    def run(self, dna_string):
        import tasks.task04.task

        ret = tasks.task04.task.Task4(dna_string).run(epochs=50)
        return ret


class Task5:
    def __init__(self):
        self.task_name = ""
        self.population = 50
        self.cull_rate = 1 / 5
        self.default_length = 5
        self.dna_chars = "abcdefghijk"

        self.starting_chromosomes = 1
        self.total_chromosomes = 11
        self.split_chance = 0.5
        self.uneven_crossover = 0.1
        self.mutation_chance = 0.66
        self.generations = []

        self.gen_count = 0
        self.fit_lookup = {}

    def random_gene(self, no_split=False):
        if random.random() < self.split_chance and no_split == False:
            gene = (
                "s"
                + random.choice(self.dna_chars).upper()
                + random.choice(self.dna_chars)
            )
        else:
            gene = random.choice(self.dna_chars)
        return gene

    def fill_generation_zero(self):
        self.generations = [
            {
                "individuals": dict(
                    [
                        (self.generate_random(self.starting_chromosomes), None)
                        for x in range(self.population)
                    ]
                )
            }
        ]

    def generate_random(self, chromosomes=1):
        output = [
            "".join(
                [self.random_gene(no_split=True) for x in range(self.default_length)]
            )
            for x in range(chromosomes)
        ]
        output = "|".join(output)
        while len(output.split("|")) < self.total_chromosomes:
            output += "|"
        return output

    def start(self):
        self.fill_generation_zero()
        while True:
            self.step()
            print(f"{self.gen_count}")

    def get_fit(self, dna):
        if dna not in self.fit_lookup:
            self.fit_lookup[dna] = self.run(dna)
        return self.fit_lookup[dna]

    def step(self):
        generation = self.generations[-1]
        update = {}
        flag = True
        best = ""
        best_fit = 999999999

        for key, data in generation["individuals"].items():
            if data is None:
                fitness = self.get_fit(key)
                generation["individuals"][key] = fitness
            else:
                fitness = data
            if fitness < best_fit:
                best_fit = fitness
                best = key

        print(
            f"task: {self.task_name}, generation: {self.gen_count} best: {best} fit: {best_fit} individuals: {generation} "
        )
        self.create_next_generation(generation)
        self.gen_count += 1

    def create_next_generation(self, generation):
        individual_scores = [
            (x, generation["individuals"][x]) for x in generation["individuals"]
        ]
        random.shuffle(individual_scores)
        best_individuals = sorted(individual_scores, key=lambda x: x[1])
        best_individuals = best_individuals[: int(self.population * self.cull_rate)]
        new_individuals = best_individuals.copy()
        random.shuffle(best_individuals)
        best_length = len(best_individuals)
        while len(new_individuals) < self.population:
            ind1 = random.choice(range(best_length))
            ind2 = random.choice(range(best_length - 1))
            if ind2 >= ind1:
                ind2 += 1
            new_individuals.extend(
                self.reproduce(best_individuals[ind1], best_individuals[ind2])
            )
        dna_strings = [x[0] for x in new_individuals]
        new_gen = {"individuals": dict()}
        for dna in dna_strings:
            new_gen["individuals"][dna] = None
        self.generations.append(new_gen)

    def gene_list(self, chromosome):
        genes = list()
        i = 0
        while i < len(chromosome):
            char = chromosome[i]
            if char in "abcdefghijk":
                genes.append(char)
                i += 1
            elif char == "s":
                genes.append(chromosome[i : i + 3])
                i += 3
            else:
                i += 1
        return genes

    def reproduce(self, ind1, ind2):
        strings1 = ind1[0].split("|")
        strings2 = ind2[0].split("|")
        max_chromo = max(len(strings1), len(strings2))

        while len(strings1) < max_chromo:
            strings1.append("")
        while len(strings2) < max_chromo:
            strings2.append("")

        output1 = []
        output2 = []
        for string1, string2 in zip(strings1, strings2):
            gstring1 = self.gene_list(string1)
            gstring2 = self.gene_list(string2)

            if len(string1) > 0:
                pos1 = random.choice(range(len(gstring1)))
            else:
                pos1 = 0
            if len(gstring2) > 0:
                # uneven cross-over chance
                if random.random() < self.uneven_crossover:
                    pos2 = random.choice(range(len(gstring2)))
                else:
                    pos2 = pos1
            else:
                pos2 = 0
            output1.append("".join(gstring1[:pos1] + gstring2[pos2:]))
            output2.append("".join(gstring2[:pos2] + gstring1[pos1:]))
        output1 = "|".join(output1)
        output2 = "|".join(output2)
        for i in range(200):
            output1 = self.mutate(output1)
            if output1 not in self.fit_lookup:
                break
        for i in range(200):
            output2 = self.mutate(output2)
            if output2 not in self.fit_lookup:
                break

        return [[self.fix_output(output1), None], [self.fix_output(output2), None]]

    def fix_output(self, string):
        if string.startswith("s"):
            string = "a" + string
        try:
            chromosomes = string.split("|")
            output = []
            for chromo in chromosomes:
                string_array = self.gene_list(chromo)
                for i, gene in enumerate(string_array):
                    if gene.startswith("s"):
                        if gene[1] in ["K"]:
                            string_array[i] = ""
                            continue
                        cc = 0
                        for gg in string_array[i:]:
                            if len(gg) == 1:
                                cc + +1
                        value = self.dna_chars.index(gene[2]) + i
                        if value > cc:
                            gene = [x for x in gene]
                            gene[2] = self.dna_chars[cc]
                            gene = "".join(gene)
                        string_array[i] = gene
                chromosome = "".join(string_array)
                output.append(chromosome)
            return "|".join(output)
        except:
            import traceback

            traceback.print_exc()
            return "|" * string.count("|")

    def mutate(self, string):
        if random.random() < self.mutation_chance:
            try:
                string_array = [x for x in string]
                choice = random.choice(
                    [
                        "dup",
                        "insert",
                        "insert",
                        "insert",
                        "inplace",
                        "delete",
                        "multiplace",
                    ]
                )
                char = 0
                isempty = string.replace("|", "") == ""
                if isempty:
                    choice = "insert"
                while isempty is False:
                    char = random.choice(range(len(string)))
                    if string[char] not in "s|":
                        break
                act_char = string[char].lower()
                if string[char].isupper():
                    string_array[char] = random.choice(self.dna_chars).upper()
                else:
                    if choice == "dup":
                        string_array = (
                            string_array[:char] + [act_char] + string_array[char:]
                        )
                    elif choice == "insert":
                        if char > 0 and string_array[char - 1] != "|":
                            gene = self.random_gene()
                        else:
                            gene = random.choice(self.dna_chars)
                        string_array = (
                            string_array[:char] + [gene] + string_array[char:]
                        )
                        if gene[0] == "s":
                            chromo = self.dna_chars.index(gene[1].lower())
                            chromos = "".join(string_array).split("|")
                            if not chromos[chromo]:
                                chromos[chromo] = random.choice(self.dna_chars)
                            string_array = [x for x in "|".join(chromos)]
                    elif choice == "inplace":
                        string_array[char] = random.choice(self.dna_chars)
                    elif choice == "delete" and len(string) > 0:
                        if char == 0 or not string[char - 1].isupper():
                            string_array = (
                                string_array[:char] + string_array[char + 1 :]
                            )
                    elif choice == "multiplace":
                        if random.random() < 0.5:
                            string_array = (
                                string_array[:char]
                                + [random.choice(self.dna_chars) for x in range(2)]
                                + string_array[char:]
                            )
                        else:
                            string_array = (
                                string_array[:char] + string_array[char + 2 :]
                            )

                string = "".join(string_array)
            except IndexError:
                pass
        return string

    def run(self, dna_string):
        import tasks.task05.task

        try:
            ret = tasks.task05.task.Task5(dna_string).run(epochs=250)
        except:
            import traceback

            traceback.print_exc()
            ret = 9999999
        return ret


"""



"""


class Task8:
    def __init__(self, use_star=True):
        self.task_name = "GAS"
        self.population = 10
        self.default_length = 5
        self.dna_chars = "abcdefg"
        self.split_chance = 0.5
        self.uneven_crossover = 0.1
        self.mutation_chance = 0.66
        self.individuals = []
        self.fit_lookup = {}
        self.max_dist = 0
        self.best = ""
        self.best_fit = 0
        self.steps = 0
        self.use_star = use_star

    def random_gene(self):
        gene = random.choice(self.dna_chars)
        return gene

    def fill_generation_zero(self):
        self.individuals = [["", {"mutation_distance": 0, "reproduced": 0, "fit": 0}]]

    def generate_random(self):
        output = "".join([self.random_gene() for x in range(self.default_length)])
        return output

    def start(self):
        self.fill_generation_zero()
        while True:
            self.step()
            print(f"Individuals: {len(self.individuals)}, MaxDistance: {self.max_dist}")

    def get_fit(self, dna):
        if dna not in self.fit_lookup:
            self.fit_lookup[dna] = self.run(dna)
        return self.fit_lookup[dna]

    def step(self):
        self.individuals.sort(
            key=lambda x: x[1]["fit"]
            / ((20 + x[1]["mutation_distance"]) * (20 + x[1]["reproduced"])),
            reverse=True,
        )
        if self.use_star == False:
            self.individuals.sort(key=lambda x: x[1]["fit"], reverse=True)
            current_gen_length = min(10, len(self.individuals))
        else:
            current_gen_length = len(self.individuals)

        for i in range(self.population):
            if self.use_star:
                ind1 = int((random.random() ** 2) * current_gen_length)
                ind2 = int((random.random() ** 2) * current_gen_length)
            else:
                ind1 = int(random.random() * current_gen_length)
                ind2 = int(random.random() * current_gen_length)

            mutate_extra = 0
            if self.individuals[ind1][1]["reproduced"] > 500:
                mutate_extra = 5

            new_individuals = self.reproduce(
                self.individuals[ind1][0], self.individuals[ind2][0], mutate_extra
            )
            for entry in new_individuals:
                if entry not in self.fit_lookup:
                    print(entry)
                    fit = self.get_fit(entry)
                    data = {
                        "mutation_distance": max(
                            self.individuals[ind1][1]["mutation_distance"],
                            self.individuals[ind2][1]["mutation_distance"],
                        )
                        + 1,
                        "reproduced": 0,
                        "fit": fit,
                    }
                    if data["mutation_distance"] > self.max_dist:
                        self.max_dist = data["mutation_distance"]
                    if data["fit"] > self.best_fit:
                        self.best_fit = data["fit"]
                        self.best = entry
                    self.individuals.append([entry, data])
            self.individuals[ind1][1]["reproduced"] += 1
            self.individuals[ind2][1]["reproduced"] += 1
        self.steps += 1
        print(
            f"task: {self.task_name}, gen: {self.steps}, individuals: {len(self.individuals)}, best: {self.best}, fit: {self.best_fit}"
        )
        print(self.individuals[:10])
        # time.sleep(0.01)

    def reproduce(self, ind1, ind2, mutate_extra=0):
        output = []
        if len(ind1) > 0:
            pos1 = random.choice(range(len(ind1)))
        else:
            pos1 = 0
        if len(ind2) > 0:
            # uneven cross-over chance
            if random.random() < self.uneven_crossover:
                pos2 = random.choice(range(len(ind2)))
            else:
                pos2 = pos1
        else:
            pos2 = 0
        output.append("".join(ind1[:pos1] + ind2[pos2:]))
        output.append("".join(ind2[:pos2] + ind1[pos1:]))
        for i in range(2):
            for j in range(200):
                output[i] = self.mutate(output[i])
                if output[i] not in self.fit_lookup and j >= mutate_extra:
                    break
        return output

    def mutate(self, string):
        while random.random() < self.mutation_chance:
            choice = random.choice(["dup", "insert", "inplace", "delete"])
            char = 0
            if string == "":
                choice = "insert"
            else:
                char = random.choice(range(len(string)))

            if choice == "dup":
                string_array = string[:char] + string[char] + string[char:]
            elif choice == "insert":
                gene = random.choice(self.dna_chars)
                string = string[:char] + gene + string[char:]
            elif choice == "inplace":
                string = (
                    string[:char] + random.choice(self.dna_chars) + string[char + 1 :]
                )
            elif choice == "delete" and len(string) > 0:
                string = string[:char] + string[char + 1 :]
        return string

    def run(self, dna_string):
        from tasks.task08 import task

        print(dna_string)
        ret = task.Task8(dna_string).run()
        return ret


def main():
    if "--task1" in sys.argv:
        Task1().start()
    elif "--task4" in sys.argv:
        Task4().start()
    elif "--task5" in sys.argv:
        Task5().start()
    elif "--task8" in sys.argv:
        if "--GA" in sys.argv:
            Task8(False).start()
        else:  # --GAS
            Task8(True).start()


if __name__ == "__main__":
    main()
