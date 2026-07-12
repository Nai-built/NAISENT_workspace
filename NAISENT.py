from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container

from array import array
from datetime import datetime

import tokenizer

modelContainer = container("NAISENT_test5") # test4 container was the one with 6 stacks

lr = 0.003

n = tokenizer.vocabulary_size()

structureId = ONL.Build(ONL.DECODER_ONLY_TRANSFORMER(n, n, 256, [
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    # ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    # ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    # ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    # ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
    # ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(0.001))]),
]))

input = array("f", [])
seriesLengths = array("i", [])
responsePositions = array("i", [])
correction = array("f", [])

def add(text):
    global input
    global seriesLengths
    global correction

    generalTokens, generalAmount, responsePosition = tokenizer.tokenize(text)
    input.extend(generalTokens[0:len(generalTokens)-n])
    seriesLengths.append(generalAmount-1)
    responsePositions.append(responsePosition)
    correction.extend(generalTokens[n:len(generalTokens)])

# add("USER: Hello, who are you?\nMODEL: NAISENT<END OF SEQUENCE>")
# add("USER: Hi\nMODEL: NAISENT<END OF SEQUENCE>")
add("USER: Hello, who are you?\nMODEL: I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hi.\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hi?\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hi\nMODEL: Hello there, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hello.\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: hello.\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: hello\nMODEL: Hi, I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hello, who are you?\nMODEL: I am NAISENT.<END OF SEQUENCE>\n")
add("USER: hello, who are you\nMODEL: I am NAISENT.<END OF SEQUENCE>\n")
add("USER: Hello there, who you are ?\nMODEL: Hello, I am NAISENT.<END OF SEQUENCE>\n")

# print(responsePositions)

totalSamples = (int)(len(input)/n)

# print(totalSamples, len(input), len(correction))
# print(input[0:n])
# print(correction[0:n])

# print(input)
# print(correction)
# print(seriesLengths)

batchSize = 10

ONL.WriteStructureInfo(structureId, modelContainer.read())

for _ in range(1):
    output = array("f", [0.0] * len(correction))

    inputPropagation = array("f", [0.0] * len(input))
    propagation = array("f", [0.0] * len(correction))

    startTime = datetime.now()

    ONL.Activate(structureId, input, output, batchSize, seriesLengths, totalSamples)
    ONL._softmax(output, n)
    ONL.MSE(output, correction, propagation)
    s = 0
    for i in range(len(seriesLengths)):
        # print("AAA")
        # print(vocabulary.formSequence(correction[s*n:s*n + responsePositions[i]*n], responsePositions[i]))
        for j in range(responsePositions[i]*n):
            propagation[s*n + j] = 0.0
            output[s*n + j] = correction[s*n + j]

        s += seriesLengths[i]
    ONL.Adjust(structureId, propagation, inputPropagation, lr, batchSize, seriesLengths, totalSamples)

    # print([output[i:i + n] for i in range(0, len(output), n)])

    print(tokenizer.formSequence(input, totalSamples))
    print("==")
    print(tokenizer.formSequence(correction, totalSamples))
    print("------------------------------------------")
    print(tokenizer.formSequence(output, totalSamples))

    print((datetime.now()-startTime).total_seconds()) # 0.0316

# modelContainer.write(ONL.ReadStructureInfo(structureId))