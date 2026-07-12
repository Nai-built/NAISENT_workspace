# NAISENT LSTM Math Calculator in CPU

import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

from datetime import datetime

import random

import time

from array import array

pre_string = "MATH_TEST"

samplesInputContainer = container(pre_string+"/INPUT_SAMPLES")
samplesLengthsContainer = container(pre_string+"/LENGTHS_SAMPLES")
samplesStartContainer = container(pre_string+"/START_SAMPLES")

samplesContainer = container(pre_string+"/STRING_SAMPLES")
correctionsContainer = container(pre_string+"/CORRECTIONS")

def generate_samples(n=1000):
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

    samplesAmount = n

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

def train_model(e=1000, lr=0.0005):

    structureId = ONL.Build(ONL.CHAIN([
        ONL.LSTM_RECURSIVE(2, 64, True),
        ONL.LSTM_RECURSIVE(64, 64),

        ONL.DENSE(64, 256
                , activation=ONL.ReLU(0.001)),
        ONL.DENSE(256, 1),
    ]))

    modelContainer = container(pre_string+"/MODEL_3")

    samplesInputData = samplesInputContainer.read()

    samplesData = samplesContainer.read()
    sampleLengthsData = samplesLengthsContainer.read()
    sampleStartsData = samplesStartContainer.read()
    correctionsData = correctionsContainer.read()

    def captureSample(index):
        sample = samplesData[index]
        correction = array("f", [correctionsData[index]])

        inputStart = sampleStartsData[index]
        sampleLength: int = sampleLengthsData[index]

        input = array("f", samplesInputData[inputStart:inputStart+(sampleLength*2)])

        return sample, input, sampleLength, correction

    def assembleBatch(size):
        inputs = array("f", [])
        seriesLengths = array("i", [])
        corrections = array("f", [])

        for i in range(size):
            #random.randint(0, len(samplesData)-1)
            sample, input, seriesLength, correction = captureSample(random.randint(0, len(samplesData)-1))

            inputs.extend(input)
            seriesLengths.append(seriesLength)
            corrections.extend(correction)
        
        return inputs, seriesLengths, corrections


    batchSize = 64 # 128

    lr = lr

    try:
        ONL.WriteStructureInfo(structureId, modelContainer.read())
    except:
        print("FAILED TO LOAD MODEL")

    startTime = datetime.now()

    epoch = e #20000 #100000

    for i in range(epoch):
        inputs, seriesLengths, corrections = assembleBatch(batchSize)

        totalSamples = (int)(len(inputs)/2)

        outputs = array("f", [0.0] * batchSize)
        propagation = array("f", [0.0] * batchSize)
        inputPropagation = array("f", [0.0] * len(inputs))

        startStepTime = datetime.now()

        ONL.Activate(structureId, inputs, outputs, batchSize, seriesLengths, totalSamples)
        ONL.MSE(outputs, corrections, propagation)
        ONL.Adjust(structureId, propagation, inputPropagation, lr, batchSize, seriesLengths, totalSamples)

        if (i+1) % 100 == 0:
            lr = max(.0001, lr-.00000001)
            print(
                list(zip(corrections.tolist(), outputs.tolist())), lr)
            print((datetime.now()-startTime).total_seconds(), (datetime.now()-startStepTime).total_seconds(), i+1, "/", epoch)

    modelContainer.write(ONL.ReadStructureInfo(structureId))

# generate_samples()
train_model(e=pow(10, 4), lr=0.001)