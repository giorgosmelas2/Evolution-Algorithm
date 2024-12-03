import pygad as pg
import numpy as np

distance = int(input("How many kilometers the distance is: "))
while(distance < 0 or distance > 5000):
    distance = int(input("Distance has to be from 1 up to 5000 kilometers. Give again: "))

length_array = []
rods_array = []
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

def initialize_population():
    population = []

    for _ in range(6):
        individual = []

        for i in range(len(length_array)):
            gene = np.random.randint(0, rods_array[i])
            individual.append(gene)

        population.append(individual)

    return np.array(population)


def fitness_func(ga_instance, solution, solution_idx):
    total_length = sum(solution[i]*length_array[i] for i in range(len(length_array)))
    total_connections = sum(solution) - 1 
    penalty = 0

    if total_length != distance:

        difference = total_length - distance
        difference_percentage = (difference / distance) * 100 
        
        if difference_percentage < -20:
            penalty = difference * 1000
        elif difference_percentage < -10:
            penalty = difference * 700
        elif difference_percentage < -5:
            penalty = difference * 500
        elif difference_percentage < 0:
            penalty = difference * 250
        elif difference_percentage < 5:
            penalty = difference * 50
        elif difference_percentage < 10:
            penalty = difference * 100
        elif difference_percentage < 20:
            penalty = difference * 150
    
    total_penalty = penalty + total_connections * 10
    return total_penalty

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
    upper_limit_of_geme = ga_instance.gene_space

    for chromosome_idx in range(offspring.shape[0]):
        random_gene_idx = np.random.choice(range(offspring.shape[1]))

        offspring[chromosome_idx, random_gene_idx] += np.random.randint(0, upper_limit_of_geme[random_gene_idx])

    return offspring

print(f"length_array: {length_array}")
print(f"rods_array: {rods_array}")
print(f"Population:\n{initialize_population()}")

num_genes = len(length_array)
initial_population = initialize_population()
gene_space = []
for i in range(len(rods_array)):
    gene_space.append(rods_array[i])

ga_instance = pg.GA(num_generations=50,
                    sol_per_pop=6,
                    num_parents_mating=2,
                    num_genes= num_genes,
                    initial_population=initial_population,
                    fitness_func=fitness_func,
                    crossover_type=crossover_func,
                    mutation_type=mutation_func,
                    gene_space=gene_space
)

ga_instance.run()

print("Best solution:", ga_instance.best_solution())
    





