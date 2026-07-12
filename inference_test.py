from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL
from DATA.container import container
import tokenizer

from array import array
from datetime import datetime

from random import randint

# PREPARE TOKENIZER SAVE FILE
vocabularyContainer = container("NAISENT/NAISENT_VOCAB")

def load_vocabulary():
    tokenizer.vocabulary_list = vocabularyContainer.read()

load_vocabulary()

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
    # # STACK 3
    # CNL.CHAIN([
    #     CNL.RMSNORM(256),
    #     CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
    #     CNL.DENSE(256,256),
    # ], residual=True),
    # CNL.CHAIN([
    #     CNL.RMSNORM(256),
    #     CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
    #     CNL.DENSE(1024,256),
    # ], residual=True),
    # # STACK 4
    # CNL.CHAIN([
    #     CNL.RMSNORM(256),
    #     CNL.MH_MASKED_SELF_ATTENTION(256, 64, 4),
    #     CNL.DENSE(256,256),
    # ], residual=True),
    # CNL.CHAIN([
    #     CNL.RMSNORM(256),
    #     CNL.DENSE(256,1024, activation=CNL.AF_ReLU(0.001)),
    #     CNL.DENSE(1024,256),
    # ], residual=True),

    CNL.RMSNORM(256),
    CNL.DENSE(256,n),
]), optimization=CNL.OPT_ADAM())
CNL.Initialize(modelId, 0)

def load_model(target):
    CNL.Insert(modelId, CNL.DictToSaveInfo(container(target).read()))

def inference_test(text):
    input, sequenceLength, _, _ = tokenizer.tokenize(text)

    output = array("f", [0.0] * len(input))

    CNL.Infer(modelId, input, output, sequenceLength)
    CNL._softmax(output, n)

    # print(output)

    for i in range(sequenceLength):
        print(tokenizer.formSequence(output[i*n:i*n+n], 1))

    print(tokenizer.formSequence(output, sequenceLength))

def activation_test(text):
    input, sequenceLength, _, _ = tokenizer.tokenize(text)

    output = array("f", [0.0] * len(input))

    CNL.Activate(modelId, input, output, 1, array("i", [sequenceLength]), sequenceLength)
    CNL._softmax(output, n)

    # print(output)

    for i in range(sequenceLength):
        print(tokenizer.formSequence(output[i*n:i*n+n], 1))

    print(tokenizer.formSequence(output, sequenceLength))

load_model("NAISENT/NAISENT_LM_v9_FT")
# inference_test()
# inference_test()
# inference_test()
# inference_test()
inference_test("N")
print("---")
inference_test("A")
print("---")
inference_test("I")
print("---")
inference_test("S")
print("---")
inference_test("E")
print("---")
inference_test("N")
print("---")
inference_test("T")
print("---B___")
# inference_test("NAISENT")
activation_test("NAISENT")

# INFERENCE IS WORKING!!!

# there is a kind of big problem though, masked self attention doesn't seem to be.. "masking" make sure to fix it so we can use inference reliably
# DONE DONE DONE!!!!!

# INFERENCE IS OFFICIALLY WORKING PERFECTLY FINE AND READY FOR USE!