from NeurologicalLibrary.bridge.pythonConversion import NeurologicalLogic as NL
import time
from pathlib import Path
import json

structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
    NL.LSTM_RECURSIVE(2, 32),
    NL.DENSE(32, 32),
    NL.AF_ReLU(.01),
    NL.DENSE(32, 1),
]))

steps = [
    {
        "input": [10, 2],
        "desiredOutput": [0],
    },
    {
        "input": [10, 3],
        "desiredOutput": [1],
    },
    {
        "input": [10, 6],
        "desiredOutput": [3],
    },
    {
        "input": [10, 10],
        "desiredOutput": [4],
    },
    {
        "input": [10, 9],
        "desiredOutput": [-1],
    },
    {
        "input": [10, 8],
        "desiredOutput": [-2],
    },
    {
        "input": [10, 6],
        "desiredOutput": [-2],
    },
    {
        "input": [10, 3],
        "desiredOutput": [-3],
    },
]

path = Path(Path(__file__).resolve().parent / "RECURSIVE_test_data.json")

with path.open("r", encoding="utf-8") as f:
    NL.Write(structureId, json.load(f))

for __ in range(100):
    outputs = [0.0] * len(steps)

    print("~!~")
    print("SESSION: " + __.__str__())
    print("~-~")
    for i in range(len(steps)):
        stepInfo = steps[i]

        outputs[i] = NL.Activate(structureId, input=stepInfo["input"], sampleIndex="TEST -s" + i.__str__())
        
        print("input:", stepInfo["input"])
        print("output:", outputs[i], "correction:", stepInfo["desiredOutput"])
        time.sleep(0.001)
    for i in reversed(range(len(steps))):
        stepInfo = steps[i]
        output = outputs[i]

        NL.Adjust(structureId, gradient
                  =NL.MSE_LOSS_GRADIENT(output, stepInfo["desiredOutput"])
                , sampleIndex="TEST -s" + i.__str__())
        
    NL.ApplyAdjustments(structureId, .001, 1)

with path.open("w", encoding="utf-8") as f:
    json.dump(NL.Read(structureId), f, indent=2)