from DATA.container import container

import random

import time

from array import array

samplesInputContainer = container("math_test/INPUT_SAMPLES")
samplesLengthsContainer = container("math_test/LENGTHS_SAMPLES")
samplesStartContainer = container("math_test/START_SAMPLES")

samplesContainer = container("math_test/STRING_SAMPLES")
correctionsContainer = container("math_test/CORRECTIONS")

operations = ["+", "-", "*", "/"]

def get_random_number(min = 0, max = 100):
    return random.randint(min,max)/10

def add_something():
    operation = operations[random.randint(0, len(operations)-1)]
    if operation == "/":
        return " " + operation + " " + get_random_number(1, 10).__str__()
    if operation == "*":
        return " " + operation + " " + get_random_number(0, 10).__str__()
    return " " + operation + " " + get_random_number().__str__()

def generate_sample():
    sample = get_random_number().__str__()

    for i in range(random.randint(0, 5)):
        sample += add_something()

    return sample

def calculate_sample(sample):
    return float(eval(sample))

def turn_sample_to_instructions(sample):
    splitedSample = str.split(sample, " ")

    input = array("f", [])

    for i,v in enumerate(splitedSample):
        if v == "+":
            input.extend(array("f", [1,0]))
        elif v == "-":
            input.extend(array("f", [1,1]))
        elif v == "*":
            input.extend(array("f", [1,2]))
        elif v == "/":
            input.extend(array("f", [1,3]))
        else:
            input.extend(array("f", [0,float(v)]))

    return len(splitedSample), input

samplesAmount = 10000

samplesInput = array("f", [])

samples = [None] * samplesAmount
sampleLengths = [None] * samplesAmount
sampleStarts = [None] * samplesAmount
corrections = [None] * samplesAmount

for i in range(samplesAmount):
    samples[i] = generate_sample()
    sampleLengths[i], sampleInput = turn_sample_to_instructions(samples[i])
    sampleStarts[i] = len(samplesInput)
    samplesInput.extend(sampleInput)
    corrections[i] = calculate_sample(samples[i])

samplesInputContainer.write(samplesInput.tolist())
samplesLengthsContainer.write(sampleLengths)
samplesStartContainer.write(sampleStarts)

samplesContainer.write(samples)
correctionsContainer.write(corrections)