from NeurologicalLibrary.bridge.pythonConversion import NeurologicalLogic as NL
import time
from pathlib import Path
import json

structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
    # EXPECTED INPUT: 16x16
    NL.CONVOLUTIONAL(1, 4, padding=1, stride=1, kernelSize='3x3'),
    NL.AF_ReLU(.001),
    # EXPECTED INPUT: 16x16
    NL.CONVOLUTIONAL(4, 4, padding=0, stride=1, kernelSize='3x3'),
    NL.AF_ReLU(.001),

    # EXPECTED INPUT: 14x14
    NL.MAX_POOLING(stride=2, poolSize='2x2'),

    # EXPECTED INPUT: 7x7
    NL.CONVOLUTIONAL(4, 4, padding=1, stride=1, kernelSize='3x3'),
    NL.AF_ReLU(.001),
    # EXPECTED INPUT: 7x7
    NL.CONVOLUTIONAL(4, 2, padding=0, stride=2, kernelSize='3x3'),
    NL.AF_ReLU(.001),

    # EXPECTED INPUT: 3x3
    NL.MIN_POOLING(stride=2, poolSize='2x2'),

    # EXPECTED INPUT: 1x1
    NL.FLATTENER(),

    # EXPECTED INPUT: 2 * 1x1 = 2
    NL.DENSE(2, 10),
]))

input1 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]
input2 = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]

desiredOutput1 = [0,1,0,0,0,0,0,0,0,0]
desiredOutput2 = [0,0,1,0,0,0,0,0,0,0]

path = Path(Path(__file__).resolve().parent / "CONVOLUTIONAL_test_data.json")

with path.open("r", encoding="utf-8") as f:
    NL.Write(structureId, json.load(f))

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
        i += 1
    return highestnumberIndex

stepIndex = 0
for i in range(100):
    stepIndex += 1
    inputName = None
    input = None
    desiredOutput = None
    if i % 2 == 0:
        inputName = "TWO"
        input = input2
        desiredOutput = desiredOutput2
    else:
        inputName = "ONE"
        input = input1
        desiredOutput = desiredOutput1
    output = NL.Activate(structureId, input=[input], sampleIndex="TEST -s" + stepIndex.__str__())
    print(inputName)
    print(getHighestNumberIndex(output))
    print(output)
    NL.Adjust(structureId, NL.MSE_LOSS_GRADIENT(output, desiredOutput), sampleIndex="TEST -s" + stepIndex.__str__())
    if stepIndex % 2 == 0:
        stepIndex = 0
        NL.ApplyAdjustments(structureId, .1, 2)


with path.open("w", encoding="utf-8") as f:
    json.dump(NL.Read(structureId), f, indent=2)