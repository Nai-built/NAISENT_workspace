from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

import random

import time

from datetime import datetime

from array import array

structureId = ONL.Build(ONL.CHAIN([
    ONL.LSTM_RECURSIVE(2, 64, True),
    ONL.LSTM_RECURSIVE(64, 64),

    ONL.DENSE(64, 64
              , activation=ONL.ReLU(0.001)),
    # ONL.DENSE(64, 64
    #           , activation=ONL.ReLU(0.001)),
    ONL.DENSE(64, 1),
]))

modelContainer = container("math_test/MODEL_3")

samplesInputContainer = container("math_test/INPUT_SAMPLES")
samplesLengthsContainer = container("math_test/LENGTHS_SAMPLES")
samplesStartContainer = container("math_test/START_SAMPLES")

samplesContainer = container("math_test/STRING_SAMPLES")
correctionsContainer = container("math_test/CORRECTIONS")

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


batchSize = 256 # 128

lr = 0.0005

ONL.WriteStructureInfo(structureId, modelContainer.read())

startTime = datetime.now()

epoch = 10000 #20000 #100000

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
        # lr = max(.0001, lr-.0000001)
        print(
            # inputs, "\n\n",
            corrections, "\n---\n", outputs, lr)
        print((datetime.now()-startTime).total_seconds(), (datetime.now()-startStepTime).total_seconds(), i+1, "/", epoch)

modelContainer.write(ONL.ReadStructureInfo(structureId))