# NAISENT Language Model in CUDA

import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL
from DATA.container import container
import tokenizer

from array import array
from datetime import datetime
import time

from random import randint

# PREPARE TOKENIZER SAVE FILE
vocabularyContainer = container("NAISENT/NAISENT_VOCAB")

# PREPARE PRE-TRAINING DATA
with open("pretrain_text.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# TRAIN TOKENIZER
def train_tokenizer(epochs):
    for _ in range(epochs): # for multiple epochs
        for i in range(len(lines)):
            tokenizer.BPE_learn(lines[i])

def save_vocabulary():
    vocabularyContainer.write(tokenizer.vocabulary_list)
def load_vocabulary():
    tokenizer.vocabulary_list = vocabularyContainer.read()

# train the model's tokenizer with Byte-Pair Encoding (just merging frequent letters with eachother for better training)
train_tokenizer(2)
# save the trained vocabulary by the tokenizer into json
save_vocabulary()
# loading it back from json (yes, its redundent in this showcase IK)
load_vocabulary()
print(tokenizer.vocabulary_list, len(tokenizer.vocabulary_list)) # :D

n = tokenizer.vocabulary_size()

# CONSTRUCT MODEL
modelId = CNL.Build(CNL.CHAIN([
    # positional encoding to distinguish between the positions of letters (genuinely important for the model)
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

    CNL.RMSNORM(256),
    CNL.DENSE(256,n),
]), optimization=CNL.OPT_ADAM())
CNL.Initialize(modelId, 0)

def save_model(target):
    container(target).write(CNL.SaveInfoToDict(CNL.Extract(modelId)))
def load_model(target):
    try:
        CNL.Insert(modelId, CNL.DictToSaveInfo(container(target).read()))
    except:
        print("COULDN'T LOAD MODEL", target)

def pre_train(learning_rate, epochs, miniBatchingSize):
    input = []
    seriesLengths = array("i", [])
    correction = []

    def add(text):
        generalTokens, generalAmount, _, _ = tokenizer.tokenize(text)
        input.append(array("f", generalTokens[0:len(generalTokens)-n]))
        seriesLengths.append(generalAmount-1)
        correction.append(array("f", generalTokens[n:len(generalTokens)]))

    for i in range(len(lines)):
        add(lines[i])

    def assembleBatch(batchSize):
        lines_indices = list(range(len(lines)))

        _input = array("f")
        _lengths = array("i")
        _correction = array("f")
        _amount = 0

        for i in range(batchSize):
            rand = randint(0, len(lines_indices)-1)
            random_sample = lines_indices[rand]
            # print(random_sample, lines_indices)
            lines_indices.remove(random_sample)

            sample_length = seriesLengths[random_sample]

            _input.extend(input[random_sample])
            _lengths.append(sample_length)
            _correction.extend(correction[random_sample])
            _amount += sample_length

        return _input, _correction, _lengths, _amount

    for _ in range(epochs):
        batchInput, batchCorrection, batchLengths, batchTotalSamples = assembleBatch(miniBatchingSize)
        
        batchOutput = array("f", [0.0] * batchTotalSamples*n)

        batchInputPropagation = array("f", [0.0] * batchTotalSamples*n)
        batchPropagation = array("f", [0.0] * batchTotalSamples*n)

        startTime = datetime.now()

        CNL.Activate(modelId, batchInput, batchOutput, miniBatchingSize, batchLengths, batchTotalSamples)
        CNL._softmax(batchOutput, n)
        CNL.MSE(batchOutput, batchCorrection, batchPropagation)
        CNL.Adjust(modelId, batchPropagation, batchInputPropagation, miniBatchingSize, batchLengths, batchTotalSamples)
        CNL.Update(modelId, learning_rate/miniBatchingSize)

        if _ >= epochs-1 or _ <= 0:
            print("INPUT &:\n", tokenizer.formSequence(batchInput, batchTotalSamples))
            print("CORRECTION *:\n", tokenizer.formSequence(batchCorrection, batchTotalSamples))
            print("OUTPUT ^:\n", tokenizer.formSequence(batchOutput, batchTotalSamples))

        if _ % 100 == 0:
            print((datetime.now()-startTime).total_seconds(), "epoch:", _) # 0.0316

# PREPARE FINE-TUNING DATA
with open("finetune_text.txt", "r", encoding="utf-8") as f:
    QnA = [line.strip() for line in f if line.strip()]

def fine_tune(learning_rate, epochs, miniBatchingSize):
    input = []
    seriesLengths = array("i", [])
    correction = []
    
    BOS_positions = []
    EOS_positions = []

    def add(text):
        generalTokens, generalAmount, _BOS_positions, _EOS_positions = tokenizer.tokenize(text)
        input.append(array("f", generalTokens[0:len(generalTokens)-n]))
        seriesLengths.append(generalAmount-1)
        correction.append(array("f", generalTokens[n:len(generalTokens)]))

        BOS_positions.append(_BOS_positions)
        EOS_positions.append(_EOS_positions)

    for i in range(len(QnA)):
        add(QnA[i])

    def assembleBatch(batchSize):
        QnA_indices = list(range(len(QnA)))

        _input = array("f")
        _lengths = array("i")
        _correction = array("f")
        _amount = 0

        _BOS_positions = []
        _EOS_positions = []

        for i in range(batchSize):
            rand = randint(0, len(QnA_indices)-1)
            random_sample = QnA_indices[rand]
            # print(random_sample, QnA_indices)
            QnA_indices.remove(random_sample)

            sample_length = seriesLengths[random_sample]

            _input.extend(input[random_sample])
            _lengths.append(sample_length)
            _correction.extend(correction[random_sample])
            _amount += sample_length

            _BOS_positions.append(BOS_positions[random_sample])
            _EOS_positions.append(EOS_positions[random_sample])

        return _input, _correction, _lengths, _amount, _BOS_positions, _EOS_positions

    for _ in range(epochs):
        batchInput, batchCorrection, batchLengths, batchTotalSamples, batch_BOS_positions, batch_EOS_positions \
        = assembleBatch(miniBatchingSize)
        
        batchOutput = array("f", [0.0] * batchTotalSamples*n)

        batchInputPropagation = array("f", [0.0] * batchTotalSamples*n)
        batchPropagation = array("f", [0.0] * batchTotalSamples*n)

        startTime = datetime.now()

        CNL.Activate(modelId, batchInput, batchOutput, miniBatchingSize, batchLengths, batchTotalSamples)
        CNL._softmax(batchOutput, n)
        CNL.MSE(batchOutput, batchCorrection, batchPropagation)
        s = 0
        for i in range(miniBatchingSize):
            for bos in range(len(batch_BOS_positions[i])):
                bos_position = batch_BOS_positions[i][bos]
                eos_position = 1
                if bos > 0:
                    eos_position = batch_EOS_positions[i][bos-1]
                # print(bos_position, i, s, len(batchPropagation), eos_position)
                for j in range((bos_position-(eos_position-1))*n):
                    # print(len(batchCorrection), s*n + (eos_position-1)*n + j, s*n + (eos_position-1)*n, (bos_position)*n)
                    batchPropagation[s*n + (eos_position-1)*n + j] = 0.0
                        # print(s*n + (eos_position-1)*n + j, len(batchPropagation), s*n, j, (eos_position-1)*n, (bos_position+1-(eos_position-1))*n)
                    batchOutput[s*n + (eos_position-1)*n + j] = batchCorrection[s*n + (eos_position-1)*n + j]
                    pass
                pass
            s += batchLengths[i]
        CNL.Adjust(modelId, batchPropagation, batchInputPropagation, miniBatchingSize, batchLengths, batchTotalSamples)
        CNL.Update(modelId, learning_rate/miniBatchingSize)

        if _ >= epochs-1 or _ <= 0:
            print("INPUT &:\n", tokenizer.formSequenceSplit(batchInput, batchLengths))
            print("CORRECTION *:\n", tokenizer.formSequenceSplit(batchCorrection, batchLengths))
            print("OUTPUT ^:\n", tokenizer.formSequenceSplit(batchOutput, batchLengths))
                #   , batchOutput[seriesLengths[0]*n - n:seriesLengths[0]*n])

        if _ % 100 == 0:
            print((datetime.now()-startTime).total_seconds(), "epoch:", _) # 0.0316

def inference(maxTokens):
    CNL.ClearInferMemory(modelId)
    print("\n ---- INFERENCE ----")
    _input = input() + "<BOS>"
    
    def infer(text):
        tokens, sequenceLength, _, _ = tokenizer.tokenize(text)

        output = array("f", [0.0] * len(tokens))

        CNL.Infer(modelId, tokens, output, sequenceLength)
        CNL._softmax(output, n)

        return output[len(output)-n:len(output)]
    
    print(_input)
    for i in range(maxTokens):
        # print("INPUT:", _input, end="\n")
        out = infer(_input)
        _input = tokenizer.formSequence(out, 1)
        time.sleep(.02)
        print(_input, end="", flush=True)
        if _input == "<EOS>": break
    print("\n")

startTime = datetime.now()
load_model("NAISENT/NAISENT_LM")
print("MODEL-LOADING TIME:", (datetime.now()-startTime).total_seconds())

# you can comment these "pre_train" and "fine_tune" lines of code
# if the model is already trained and you want to skip to inference

startTime = datetime.now()
pre_train(.002, 20000, 1)
print("TOTAL PRE-TRAINING TIME:", (datetime.now()-startTime).total_seconds())

startTime = datetime.now()
fine_tune(.0002, 12000, 1)
print("TOTAL FINE-TUNING TIME:", (datetime.now()-startTime).total_seconds())

startTime = datetime.now()
for i in range(50):
    inference(300)
print("TOTAL INFERENCE SESSION TIME:", (datetime.now()-startTime).total_seconds())

startTime = datetime.now()
save_model("NAISENT/NAISENT_LM")
print("MODEL-SAVING TIME:", (datetime.now()-startTime).total_seconds())

# the transformer setup seems to not work properly with batching -_- make sure to come back to it later!

# TIME FOR INFERENCE! more investigation on the whole batching problem will be done later. (currently kept as "requires more training time" problem)

# yeah.. I genuinely didn't fix that yet, you can give it a go if you want to though :>