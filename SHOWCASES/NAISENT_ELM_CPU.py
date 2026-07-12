# NAISENT Experimental Language Model in CPU

import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

from array import array
from datetime import datetime

import tokenizer

modelContainer = container("NAISENT_experimental")

lr = 0.003

n = tokenizer.vocabulary_size()

structureId = ONL.Build(ONL.DECODER_ONLY_TRANSFORMER(n, n, 256, [
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
]))

input = array("f", [])
seriesLengths = array("i", [])
responsePositions = array("i", [])
correction = array("f", [])

def add(text):
    global input
    global seriesLengths
    global correction

    generalTokens, generalAmount, responsePosition, _ = tokenizer.tokenize(text)
    input.extend(generalTokens[0:len(generalTokens)-n])
    seriesLengths.append(generalAmount-1)
    responsePositions.append(responsePosition[0])
    correction.extend(generalTokens[n:len(generalTokens)])

add("Hello, who are you?\n<BOS>I am NAISENT.<EOS>\n")
add("Hi.\n<BOS>Hello there, I am NAISENT.<EOS>\n")
add("Hi?\n<BOS>Hello there, I am NAISENT.<EOS>\n")
add("Hi\n<BOS>Hello there, I am NAISENT.<EOS>\n")
add("Hello.\n<BOS>Hi, I am NAISENT.<EOS>\n")
add("hello.\n<BOS>Hi, I am NAISENT.<EOS>\n")
add("hello\n<BOS>Hi, I am NAISENT.<EOS>\n")
add("Hello, who are you?\n<BOS>I am NAISENT.<EOS>\n")
add("hello, who are you\n<BOS>I am NAISENT.<EOS>\n")
add("Hello there, who you are ?\n<BOS>Hello, I am NAISENT.<EOS>\n")

totalSamples = (int)(len(input)/n)

batchSize = 10

ONL.WriteStructureInfo(structureId, modelContainer.read())

# training loop
for _ in range(300):
    output = array("f", [0.0] * len(correction))

    inputPropagation = array("f", [0.0] * len(input))
    propagation = array("f", [0.0] * len(correction))

    startTime = datetime.now()

    ONL.Activate(structureId, input, output, batchSize, seriesLengths, totalSamples)
    ONL._softmax(output, n)
    ONL.MSE(output, correction, propagation)
    s = 0
    for i in range(len(seriesLengths)):
        for j in range(responsePositions[i]*n):
            propagation[s*n + j] = 0.0
            output[s*n + j] = correction[s*n + j]

        s += seriesLengths[i]
    ONL.Adjust(structureId, propagation, inputPropagation, lr, batchSize, seriesLengths, totalSamples)

    print(tokenizer.formSequence(input, totalSamples))
    print("==")
    print(tokenizer.formSequence(correction, totalSamples))
    print("------------------------------------------")
    print(tokenizer.formSequence(output, totalSamples))

    print((datetime.now()-startTime).total_seconds()) # 0.0316

modelContainer.write(ONL.ReadStructureInfo(structureId))