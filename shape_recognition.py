from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

from array import array

from datetime import datetime

import random

import time

ARRAY_TYPE = "f"

samplesAmount = 1000-1

sampleSize = 80*60*3
correctionSize = 3

collectedSamples = container("SAMPLES").read()[0:(samplesAmount+1)*sampleSize]
collectedCorrections = container("CORRECTIONS").read()[0:(samplesAmount+1)*correctionSize]

modelStorage = container("IMAGE_REC_SHOWCASE")

batchSize = 64

# probably works, but just try to give it more training time

# IT LEARNT!!!!!!!!
# i am so glad and happy / proud like a father somehow
# now.. we need to make it actually save our model so all that training doesn't go to waste each time [ DONE ]

structureId = ONL.Build(ONL.CHAIN([
    ONL.CONVOLUTIONAL(3, 32, kernelSize='3x3', channelSize='80x60', padding=1, stride=2
                      , activation=ONL.ReLU(.001)
                      , pooling=ONL.MAX_POOL(poolingSize="2x2", stride=2)),

    ONL.CONVOLUTIONAL(32, 64, kernelSize='3x3', channelSize='20x15', padding=1, stride=2
                      , activation=ONL.ReLU(.001)
                      , pooling=ONL.MAX_POOL(poolingSize="2x2", stride=2)),
    

    ONL.DENSE(64*5*4, 256
              , activation=ONL.ReLU(.001)),
    ONL.DENSE(256, 256
              , activation=ONL.ReLU(.001)),
    ONL.DENSE(256, 3),
]))

ONL.WriteStructureInfo(structureId, modelStorage.read())

output = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

propagation = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

# non-important
inputPropagation = array(ARRAY_TYPE, [0.0] * batchSize * sampleSize)

startTIme = datetime.now()

def getHighestNumberIndex(list):
    highestNumber = None
    highestnumberIndex = 0
    i = 0
    for v in list:
        if highestNumber:
            if v > highestNumber:
                highestNumber = v
                highestnumberIndex = i
        else:
            highestNumber = v
            highestnumberIndex = i
        i += 1
    return highestnumberIndex

def indexToText(index):
    if index == 0:
        return "Circle"
    elif index == 1:
        return "Triangle"
    elif index == 2:
        return "Square"

def listToText(list):
    return indexToText(getHighestNumberIndex(list))

def compareToString(output, correction):
    string = ""

    successfulAmount = 0

    for i in range(batchSize):
        outputShape = listToText(output[i*correctionSize:i*correctionSize + correctionSize])
        correctionShape = listToText(correction[i*correctionSize:i*correctionSize + correctionSize])

        if outputShape == correctionShape:
            string += outputShape + " == " + correctionShape + " ----- SUCCESS\n"
            successfulAmount += 1
        else:
            string += outputShape + " == " + correctionShape + " ----- F\n"
            pass

    string += "( " + successfulAmount.__str__() + " / " + batchSize.__str__() + " )"
    print(string)

epoch = 10000
for i in range(epoch):
    output = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

    inputPropagation = array(ARRAY_TYPE, [0.0] * batchSize * sampleSize)

    # selectedSamples = array("f", collectedSamples[selectedBatchStart*sampleSize:selectedBatchEnd*sampleSize])
    # selectedCorrections = array("f", collectedCorrections[selectedBatchStart*correctionSize:selectedBatchEnd*correctionSize])
    
    selectedSamples = array("f")
    selectedCorrections = array("f")

    for _ in range(batchSize):
        sampleTarget = random.randint(0, samplesAmount)

        selectedSamples.extend(collectedSamples[sampleTarget*sampleSize:sampleTarget*sampleSize + sampleSize])
        selectedCorrections.extend(collectedCorrections[sampleTarget*correctionSize:sampleTarget*correctionSize + correctionSize])

    # print(len(selectedSamples)/batchSize, sampleSize)

    startStepTIme = datetime.now()

    ONL.Activate(structureId, selectedSamples, output, batchSize=batchSize)
    ONL._softmax(output, correctionSize)
    ONL.MSE(output, selectedCorrections, propagation)
    ONL.Adjust(structureId, propagation, inputPropagation, lr=.0003, batchSize=batchSize)

    print("<---" + i.__str__() + "--->")
    compareToString(output, selectedCorrections)

    print("time passed:", (datetime.now()-startTIme).total_seconds(), "-", (datetime.now()-startStepTIme).total_seconds(), "step:", i+1, "/", epoch)

    # time.sleep(.00001)

modelStorage.write(ONL.ReadStructureInfo(structureId))

# CONCLUDED!