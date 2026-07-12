from NeurologicalLibrary.bridge.pythonConversion import multiply, add, nextInteger, NeurologicalLogic as NL

from python.NAISENT_Structure import NAISENT_Structure
from python.NAISENT_Client import NAISENT_Client
from python.NAISENT_Storage import NAISENT_Storage

import time
import json
from pathlib import Path

# import python.simpleTest;

# NAISENT_id = NL.CreateNeurologicalStructure(NL.CHAIN([
#     NL.DENSE(2, 256),
#     NL.AF_ReLU(.01),
#     NL.DENSE(256, 256),
#     NL.AF_ReLU(.01),
#     NL.DENSE(256, 256),
#     NL.AF_ReLU(.01),
#     NL.DENSE(256, 3),
#     NL.AF_Softmax(True),
# ])
# # , optimizer=NL.OPTIMIZER_ADAM(1e-8, .999, .99)
# )

# input = [10, -2.5]

# desiredOutput = [0,1,0]

# for i in range(10000):
#     output = NL.Activate(NAISENT_id, input=input, sampleIndex="TEST")
#     print("activation: ", output)

#     NL.Adjust(NAISENT_id, gradient=NL.MSE_LOSS_GRADIENT(output, desiredOutput), sampleIndex="TEST")
#     NL.ApplyAdjustments(NAISENT_id, learningRate=.01, averagingDivisor=1)

# output = NL.Activate(NAISENT_id, input=input, sampleIndex="TEST")
# print("activation: ", output)

# print(add(multiply(.5, 100), -5), nextInteger(), nextInteger(), nextInteger())

SCRIPT_DIR = Path(__file__).resolve().parent

structure = NAISENT_Structure()
client = NAISENT_Client()
storage = NAISENT_Storage(SCRIPT_DIR / "temporaryData.json")

structure.write(storage.read())

def interactionCallback(state, stepIndex):
    return structure.interact(state, stepIndex=stepIndex)

interactions = {}
desiredActions = {}

amountOfEpisodes = 0

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

client.start()

while True:
    if amountOfEpisodes >= 100:
        break

    # print("stepping")

    interaction, response, stepIndex, hadRestarted = client.shapeTestClientStep(interactionCallback=interactionCallback)

    if hadRestarted:
        print("CONCLUDED EPISODE:", amountOfEpisodes+1)
        for i in reversed(range(len(interactions))):
            structure.correct(interaction=interactions[i]
                              , correction=desiredActions[i]
                              , stepIndex=i)
        structure.applyCorrections(16)
        interactions.clear()
        desiredActions.clear()
        amountOfEpisodes += 1
        structure.episodeIndex = amountOfEpisodes+1
    
    print(stepIndex, interaction, response.tolist())
    predictionText = listToText(interaction)
    correctionText = listToText(response)
    print(stepIndex, predictionText, correctionText)
    if predictionText == correctionText:
        print("--", "SUCCESS")
    else:
        print("--", "F")

    interactions[stepIndex] = interaction
    desiredActions[stepIndex] = response.tolist()

    # time.sleep(2) # .05

    client.procceed()

storage.write(structure.read())
