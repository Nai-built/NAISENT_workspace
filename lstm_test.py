from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

from array import array

structureId = ONL.Build(ONL.CHAIN([
    ONL.LSTM_RECURSIVE(2, 64, True),
    ONL.LSTM_RECURSIVE(64, 64),

    ONL.DENSE(64, 1),
]))

batchSize = 4

input = array("f", [
    # sample 1:
        # element 1:
        0, 15, # number 15
        # element 2:
        1, 0, # operation +
        # element 3:
        0, 2, # number 2
    # sample 2:
        # element 1:
        0, 5, # number 5
        # element 2:
        1, 1, # operation -
        # element 3:
        0, 2, # number 2
    # sample 3:
        # element 1:
        0, 5, # number 5
    # sample 4:
        # element 1:
        0, 5, # number 5
        # element 2:
        1, 1, # operation -
        # element 3:
        0, 2, # number 2
        # element 4:
        1, 0, # operation +
        # element 5:
        0, 7, # number 2
])
seriesLengths = array("i", [
    3,
    3,
    1,
    5,
])
correction = array("f", [
    17,
    3,
    5,
    10,
])

totalSamples = (int)(len(input)/2)
print(totalSamples)

for i in range(2000):
    output = array("f", [0.0] * batchSize)
    propagation = array("f", [0.0] * batchSize)
    inputPropagation = array("f", [0.0] * len(input))

    ONL.Activate(structureId, input, output, batchSize, seriesLengths, totalSamples)
    print(output)
    ONL.MSE(output, correction, propagation)
    ONL.Adjust(structureId, propagation, inputPropagation, .001, batchSize, seriesLengths, totalSamples)

output = array("f", [0.0] * batchSize)
propagation = array("f", [0.0] * batchSize)
inputPropagation = array("f", [0.0] * len(input))

ONL.Activate(structureId, input, output, batchSize, seriesLengths, totalSamples)

print(output)