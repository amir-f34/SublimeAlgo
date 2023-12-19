#!/usr/bin/env python3
import database
import os
import sys
import hashlib
import random
import time
from reputation import Reputation

__author__ = "Amir F."
__copyright__ = "Copyright 2023, Amir F."
__credits__ = ["Amir F."]
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "amir.f34@proton.me"
__status__ = "Completed"


class Task1:
    def __init__(self):
        self.task_name = "task01"
        self.task_id = None
        self.population = 50
        self.cull_rate = 1 / 5
        self.default_length = 5
        self.dna_chars = "abcde"

        self.uneven_crossover = 0.1
        self.mutation_chance = 0.4
        if not self.task_id:
            self.task_id = database.get_task_id_by_name(self.task_name)
        if not database.get_task(self.task_id):
            database.add_task(self.task_name, self.task_id, self.population)
        self.fill_generation_zero()

    def fill_generation_zero(self):
        if not database.get_generation(self.task_id, 0):
            database.create_generation(
                self.task_id,
                0,
                [self.generate_random() for x in range(self.population)],
            )

    def generate_random(self):
        return "".join(
            [random.choice(self.dna_chars) for x in range(self.default_length)]
        )

    def step(self):
        task = database.get_task(self.task_id)
        generation = task["generations"]
        generation_obj = database.get_generation(self.task_id, generation)
        update = {}
        flag = True
        best = ""
        best_fit = 0
        for key, data in generation_obj["individuals"].items():
            if data is None:
                status = database.get_ret(self.task_id, key)

                if status and status.get("final") is not None:
                    update[key] = status
                elif status:
                    rep = Reputation()
                    rep.update_task_rep(self.task_id, key)
                flag = False
            elif data.get("final", 0) > best_fit:
                best_fit = data["final"]
                best = key

        if update:
            database.update_generation(self.task_id, generation, update)
        if flag:
            print(
                f"task: {self.task_name}, gen: {generation}, best: {best} fit: {best_fit}"
            )
            self.create_next_generation(self.task_id, generation)
            generation += 1
            database.update_task(self.task_id, generation)

    def create_next_generation(self, task_id, generation):
        generation_obj = database.get_generation(task_id, generation)
        individuals = [x for x in generation_obj["individuals"].items()]
        best_individuals = sorted(
            individuals, key=lambda x: x[1]["final"], reverse=True
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
        database.create_generation(task_id, generation + 1, dna_strings)

    def reproduce(self, ind1, ind2):
        string1 = ind1[0]
        string2 = ind2[0]
        pos1 = random.choice(range(len(string1)))
        if random.random() < self.uneven_crossover:
            pos2 = random.choice(range(len(string2)))
        else:
            pos2 = pos1
        new1 = self.mutate(string1[:pos1] + string2[pos2:])
        new2 = self.mutate(string2[:pos2] + string1[pos1:])

        return [[new1, None], [new2, None]]

    def mutate(self, string):
        if random.random() < self.mutation_chance:
            string_array = [x for x in string]
            choice = random.choice(["dup", "insert", "inplace", "delete"])
            char = random.choice(range(len(string)))
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
                for i in range(2):
                    string_array[char + i] = random.choice(self.dna_chars)
            string = "".join(string_array)
        return string


def main():
    tasks = [Task1()]
    while True:
        for task in tasks:
            task.step()
        time.sleep(1)


if __name__ == "__main__":
    main()
