from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL
from DATA.container import container
import tokenizer

from array import array
from datetime import datetime

from random import randint

with open("pretrain_text.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

for i in range(len(lines)):
    tokenizer.BPE_learn(lines[i])
for i in range(len(lines)):
    tokenizer.BPE_learn(lines[i])
for i in range(len(lines)):
    tokenizer.BPE_learn(lines[i])
for i in range(len(lines)):
    tokenizer.BPE_learn(lines[i])
# for i in range(len(lines)):
#     tokenizer.BPE_learn(lines[i])

print(tokenizer.vocabulary_list, len(tokenizer.vocabulary_list)) # :D

n = tokenizer.vocabulary_size()

modelContainer = container("CUDA/test_language_model2")

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

CNL.Insert(modelId, CNL.DictToSaveInfo(modelContainer.read()))

# CNL.Initialize(modelId, 0)

batchSize = 0

input = []
seriesLengths = array("i", [])
correction = []

def add(text):
    global input
    global seriesLengths
    global correction

    global batchSize

    generalTokens, generalAmount = tokenizer.tokenize(text)
    # print(generalTokens, len(generalTokens), generalAmount)
    input.append(array("f", generalTokens[0:len(generalTokens)-n]))
    seriesLengths.append(generalAmount-1)
    batchSize += 1
    correction.append(array("f", generalTokens[n:len(generalTokens)]))

for i in range(len(lines)):
    add(lines[i])

for _ in range(100):
    for sample_index in range(len(lines)):
        sample_length = seriesLengths[sample_index]

        input_sample = input[sample_index]
        correction_sample = correction[sample_index]

        output_sample = array("f", [0.0] * sample_length*n)

        inputPropagation = array("f", [0.0] * sample_length*n)
        propagation = array("f", [0.0] * sample_length*n)

        startTime = datetime.now()

        CNL.Activate(modelId, input_sample, output_sample, 1, array("i", [sample_length]), sample_length)
        CNL._softmax(output_sample, n)
        CNL.MSE(output_sample, correction_sample, propagation)
        # s = 0
        # for i in range(len(seriesLengths)):
        #     # print("AAA")
        #     # print(vocabulary.formSequence(correction[s*n:s*n + responsePositions[i]*n], responsePositions[i]))
        #     for j in range(responsePositions[i]*n):
        #         propagation[s*n + j] = 0.0
        #         output[s*n + j] = correction[s*n + j]
        #         pass

        #     s += seriesLengths[i]
        CNL.Adjust(modelId, propagation, inputPropagation, 1, array("i", [sample_length]), sample_length)
        CNL.Update(modelId, .0002)

        # print([output[i:i + n] for i in range(0, len(output), n)])

        print(tokenizer.formSequence(input_sample, sample_length))
        print("==")
        print(tokenizer.formSequence(correction_sample, sample_length))
        print("------------------------------------------")
        print(tokenizer.formSequence(output_sample, sample_length))

        print((datetime.now()-startTime).total_seconds(), "epoch:", _) # 0.0316


modelContainer.write(CNL.SaveInfoToDict(CNL.Extract(modelId)))

# :DDDDDDDDDD IT LEARNS!!!!!!!!!