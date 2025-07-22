import numpy as np
import random

from evolution.genome import create_genome
from environment.lunarlander_runner import evaluate_genome

class GeneticAlgorithm:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, population_size, mutation_rate, render):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genome_length = self.input_size * self.hidden1_size + self.hidden1_size + \
            self.hidden1_size * self.hidden2_size + self.hidden2_size + \
            self.hidden2_size * self.output_size + self.output_size
        self.population = self.initialize_population()
        self.render = render

    def initialize_population(self):
        return [create_genome(self.input_size, self.hidden1_size, self.hidden2_size, self.output_size)
                for _ in range(self.population_size)]

    def mutate(self, genome):
        mutation = np.clip(np.random.randn(len(genome)) * self.mutation_rate, -0.1, 0.1)
        #mutation = np.clip(mutation, -0.2, 0.2)
        return genome + mutation

    def evaluate_population(self, generation):
        fitness_scores = []
        seeds = [random.randint(0, 1000) for _ in range(1)]
        for i, genome in enumerate(self.population):
            fitness = evaluate_genome(genome, self.input_size, self.hidden1_size, self.hidden2_size, self.output_size, i ,generation, seeds, self.render)
            fitness_scores.append((genome, fitness))

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores

    def select_parents(self, sorted_population):
        num_parents = max(2, int(self.population_size * 0.2))
        parents = [genome for genome, _ in sorted_population[:num_parents]]
        return parents

    def crossover(self, parent1, parent2):
        child = np.array([np.random.choice([g1, g2])
                          for g1, g2 in zip(parent1, parent2)])
        return child
        #alpha = 0.5
        #return alpha * parent1 + (1 - alpha) * parent2

    def next_generation(self, sorted_population):
        parents = self.select_parents(sorted_population)
        new_population = []
        
        #fitness_values = [fitness for _, fitness, in sorted_population]
        #avg_fitness = sum(fitness_values) / len(fitness_values)

        #elites = [genome for genome, fitness in sorted_population if fitness > 200]
        #new_population.extend(elites)

        #if parents[0] not in new_population:
        new_population.append(parents[0])
        

        while len(new_population) < self.population_size:
            parent1, parent2 = random.choices(parents, k=2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population


