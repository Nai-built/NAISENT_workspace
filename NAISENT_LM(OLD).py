from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL
from DATA.container import container

from array import array
from datetime import datetime

import tokenizer

modelContainer = container("CUDA/NAISENT") # test4 container was the one with 6 stacks

# modelContainer = container("NAISENT_test5") # test4 container was the one with 6 stacks

n = tokenizer.vocabulary_size()

modelId = CNL.Build(CNL.CHAIN([
    CNL.SCC_POSITIONAL_EMBEDDING(n,256),

    # STACK 1
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(1024,256),
    ], residual=True),
    # STACK 2
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(1024,256),
    ], residual=True),
    # STACK 3
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(1024,256),
    ], residual=True),
    # STACK 4
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(1024,256),
    ], residual=True),

    CNL.RMSNORM(256),
    CNL.DENSE(256,n),
]), optimization=CNL.OPT_ADAM())

CNL.Initialize(modelId, 0)

input = array("f", [])
seriesLengths = array("i", [])
responsePositions = array("i", [])
correction = array("f", [])

def add(text):
    global input
    global seriesLengths
    global correction

    generalTokens, generalAmount, responsePosition = tokenizer.tokenize(text)
    print(generalTokens, len(generalTokens), generalAmount)
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

# CNL.Insert(modelId, CNL.DictToSaveInfo(modelContainer.read()))

for _ in range(1000):
    output = array("f", [0.0] * len(correction))

    inputPropagation = array("f", [0.0] * len(input))
    propagation = array("f", [0.0] * len(correction))

    startTime = datetime.now()

    CNL.Activate(modelId, input, output, batchSize, seriesLengths, totalSamples)
    CNL._softmax(output, n)
    CNL.MSE(output, correction, propagation)
    s = 0
    for i in range(len(seriesLengths)):
        # print("AAA")
        # print(vocabulary.formSequence(correction[s*n:s*n + responsePositions[i]*n], responsePositions[i]))
        for j in range(responsePositions[i]*n):
            propagation[s*n + j] = 0.0
            output[s*n + j] = correction[s*n + j]
            pass

        s += seriesLengths[i]
    CNL.Adjust(modelId, propagation, inputPropagation, batchSize, seriesLengths, totalSamples)
    CNL.Update(modelId, .01/batchSize)

    # print([output[i:i + n] for i in range(0, len(output), n)])

    print(tokenizer.formSequence(input, totalSamples))
    print("==")
    print(tokenizer.formSequence(correction, totalSamples))
    print("------------------------------------------")
    print(tokenizer.formSequence(output, totalSamples))

    print((datetime.now()-startTime).total_seconds()) # 0.0316

# modelContainer.write(CNL.SaveInfoToDict(CNL.Extract(modelId)))

# modelContainer.write(ONL.ReadStructureInfo(structureId))