import pygad as pg
import numpy as np

distance = int(input("How many kilometers the distance is: "))
while(distance <= 0 or distance > 5000):
    distance = int(input("Distance has to be from 1 up to 5000 kilometers. Give again: "))
length_array = []
rods_array = []

with open("inputs.txt") as f:
    for line in f.readlines():
        inputs = line.split(",")
        length_array.append(int(inputs[0]))
        rods_array.append(int(inputs[1]))

"""
i = 1

while(True): 
    length = float(input(f"How many kilometers are the cable number {i}: "))
    while(length < 0 or length > 200):
        length = float(input("Length has to be from 1 up to 200 kilometers. Give again: "))
    rods = float(input(f"How many {length} kilometres rods there are: "))
    while(rods < 0 or rods > 100):
        rods = int(input("The number of rods has to be from 1 up to 100. Give again: "))
    i += 1
    if length != 0 and rods != 0:
        length_array.append(length)
        rods_array.append(rods)
    else:
        break            
"""

def initialize_population():
    population = []
    for _ in range(140):
        individual = []
        for i in range(len(length_array)):
            gene = np.random.randint(0, rods_array[i])
            individual.append(gene)

        population.append(individual)

    return np.array(population)

def fitness_func(ga_instance, solution, solution_idx):
    total_length = sum(solution[i]*length_array[i] for i in range(len(length_array)))
    
    total_connections = max(0,sum(solution) - 1) 
    connections_penalty = total_connections ** 1.5

    penalty = abs(total_length - distance) ** 2

    overage_penalty = 0
    if total_length > distance:
        overage_penalty = abs(total_length - distance) ** 2.5

    downrange_penalty = 0
    if total_length < distance:
        downrange_penalty = abs(total_length - distance) ** 3

    total_penalty = penalty + connections_penalty + overage_penalty + downrange_penalty
    return -total_penalty

def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        random_split_point = np.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]
        offspring.append(parent1)
        idx += 1
    return np.array(offspring)
            
def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))
        upper_limit_of_gene = ga_instance.gene_space[random_gene_idx]
        new_gene_value = np.random.randint(0, upper_limit_of_gene + 1)
        offspring[chromosome_idx, random_gene_idx] = new_gene_value
    return offspring

parent_selection_type = "tournament"

initial_population = initialize_population()
num_genes = len(length_array)

gene_space = []
for i in range(len(rods_array)):
    gene_space.append(rods_array[i])

ga_instance = pg.GA(num_generations=140,
                    sol_per_pop=140,
                    num_parents_mating=140,
                    keep_elitism = 2,
                    parent_selection_type=parent_selection_type,
                    K_tournament = 2,
                    num_genes= num_genes,
                    initial_population=initial_population,
                    fitness_func=fitness_func,
                    crossover_type=crossover_func,
                    mutation_type=mutation_func,
                    gene_space=gene_space,
                    mutation_probability=0.1
)

ga_instance.run()

best_solution, best_fitness, _ =  ga_instance.best_solution()
total_length = sum(best_solution[i] * length_array[i] for i in range(len(length_array)))

print("Best solution (number of rods):", best_solution)
print("Parent selection method:", ga_instance.parent_selection_type)
print("Best fitness:", best_fitness)
print("Total length of given robs", sum(length_array[i] * rods_array[i] for i in range(len(length_array)))) 
print("Total length of robs:", total_length)
print("Target distance:", distance)
    


    
