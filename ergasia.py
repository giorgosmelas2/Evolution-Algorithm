import pygad as pg
import numpy as np

distance = int(input("How many kilometers the distance is: "))
while(distance < 0 or distance > 5000):
    distance = int(input("Distance has to be from 1 up to 5000 kilometers. Give again: "))

length_and_rods = []
length = -1
rods = -1
i = 1

while(length != 0 and rods != 0):

    length = float(input(f"How many kilometers are the cable number {i}: "))
    while(length < 0 or length > 200):
        length = float(input("Length has to be from 1 up to 200 kilometers. Give again: "))
    
    rods = float(input(f"How many {length} kilometres rods there are: "))
    while(rods < 0 or rods > 100):
        rods = float(input("The number of rods has to be from 1 up to 100. Give again: "))

    i = i + 1
    length_and_rods.append([length, rods])



def initialize_population():
    population = []
    for i in range(5):
        individual = []
        for _ in range(len(length_and_rods)-1):
            gene = np.random.randint(0, length_and_rods[_][1])
            individual.append(gene)
        population.append(individual)
    return np.array(population)


def fitness(solution):
    total_length = sum(solution[i]*length_and_rods[_][1] for _ in range(len(length_and_rods)-1))
    total_connections = sum(solution) - 1 
    penalty = 0

    if total_length != distance:

        difference = total_length - distance
        difference_percentage = ((difference) / distance) * 100 
        
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





    


#print(f"Array length_and_rods: {length_and_rods}")
print(f"Population: {initialize_population(length_and_rods)}")


