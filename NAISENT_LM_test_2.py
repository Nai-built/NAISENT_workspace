from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL
from DATA.container import container
import tokenizer

from array import array
from datetime import datetime
import time

from random import randint

# PREPARE TOKENIZER SAVE FILE
vocabularyContainer = container("NAISENT/NAISENT_VOCAB_30M_testing2")

# PREPARE PRE-TRAINING DATA
with open("pretrain_text.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# TRAIN TOKENIZER
def train_tokenizer(epochs):
    full_data: str = ""
    for i in range(len(lines)):
        full_data += lines[i]
    for _ in range(epochs): # for multiple epochs
        tokenizer.BPE_learn(full_data)

def save_vocabulary():
    vocabularyContainer.write(tokenizer.vocabulary_list)
def load_vocabulary():
    tokenizer.vocabulary_list = vocabularyContainer.read()

# train_tokenizer(90)
# save_vocabulary()
load_vocabulary()
print(tokenizer.vocabulary_list, len(tokenizer.vocabulary_list)) # :D


# CONSTRUCT MODEL
n = tokenizer.vocabulary_size()

d_model = 256*2
d_ff = 1024*2
d_head = 64
n_heads = 4*2

_dropout = 16*2

def stack():
    return [
        CNL.CHAIN([
            CNL.RMSNORM(d_model),
            CNL.MH_MASKED_SELF_ATTENTION(d_model, d_head,n_heads),
            CNL.DENSE(d_head*n_heads, d_model,
                      max_dropout=_dropout),
        ], residual=True),
        CNL.CHAIN([
            CNL.RMSNORM(d_model),
            CNL.DENSE(d_model, d_ff,activation=CNL.AF_ReLU(0.001)),
            CNL.DENSE(d_ff, d_model,
                      max_dropout=_dropout),
        ], residual=True),
    ]

modelId = CNL.Build(CNL.CHAIN([
    CNL.SCC_POSITIONAL_EMBEDDING(n,d_model),

    *stack(),
    *stack(),
    *stack(),
    *stack(),
    *stack(),
    *stack(),

    CNL.RMSNORM(d_model),
    CNL.DENSE(d_model,n),
]), optimization=CNL.OPT_ADAM())
CNL.Initialize(modelId, 0)

def save_model(targetContainer):
    try:
        save_info = CNL.Extract(modelId)
        targetContainer.write(CNL.SaveInfoToDict(save_info))
    except Exception as e:
        print(e)
        print("SAVE HAVE FAILED!")
        def _request():
            print("save again?")
            request = input()
            if request == "save":
                save_model(targetContainer)
            elif request == "quit":
                pass
            else:
                print("please enter 'save' or 'quit'")
                _request()
        _request()
def load_model(targetContainer):
    success = CNL.Insert(modelId, CNL.DictToSaveInfo(targetContainer.read()))
    if not success:
        raise ValueError("CRY!")

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

        _input = []
        _lengths = []
        _correction = []
        _amount = 0

        for i in range(batchSize):
            _max = len(lines_indices)-1
            rand = randint(0, _max)
            random_sample = lines_indices[rand]
            # print(random_sample, lines_indices)
            lines_indices.remove(random_sample)

            sample_length = seriesLengths[random_sample]

            _input.append(input[random_sample])
            _lengths.append(sample_length)
            _correction.append(correction[random_sample])
            _amount += sample_length

        return _input, _correction, _lengths, _amount

    for _ in range(epochs):
        batchInput, batchCorrection, batchLengths, batchTotalSamples = assembleBatch(miniBatchingSize)
        
        startTime = datetime.now()
        
        for i in range(miniBatchingSize):
            sampleOutput = array("f", [0.0] * batchLengths[i]*n)

            sampleInputPropagation = array("f", [0.0] * batchLengths[i]*n)
            samplePropagation = array("f", [0.0] * batchLengths[i]*n)

            CNL.Activate(modelId, batchInput[i], sampleOutput, 1, array("i", [batchLengths[i]]), batchLengths[i])
            CNL._softmax(sampleOutput, n)
            CNL.MSE(sampleOutput, batchCorrection[i], samplePropagation)
            CNL.Adjust(modelId, samplePropagation, sampleInputPropagation, 1, array("i", [batchLengths[i]]), batchLengths[i])
            
            if _ >= epochs-1 or _ <= 0:
                print("INPUT &:\n", tokenizer.formSequenceSplit(batchInput[i], array("i", [batchLengths[i]])))
                print("CORRECTION *:\n", tokenizer.formSequenceSplit(batchCorrection[i], array("i", [batchLengths[i]])))
                print("OUTPUT ^:\n", tokenizer.formSequenceSplit(sampleOutput, array("i", [batchLengths[i]])))

        
        CNL.Update(modelId, learning_rate/miniBatchingSize)

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

        _input = []
        _lengths = []
        _correction = []
        _amount = 0

        _BOS_positions = []
        _EOS_positions = []

        for i in range(batchSize):
            rand = randint(0, len(QnA_indices)-1)
            random_sample = QnA_indices[rand]
            # print(random_sample, QnA_indices)
            QnA_indices.remove(random_sample)

            sample_length = seriesLengths[random_sample]

            _input.append(input[random_sample])
            _lengths.append(sample_length)
            _correction.append(correction[random_sample])
            _amount += sample_length

            _BOS_positions.append(BOS_positions[random_sample])
            _EOS_positions.append(EOS_positions[random_sample])

        return _input, _correction, _lengths, _amount, _BOS_positions, _EOS_positions

    for _ in range(epochs):
        batchInput, batchCorrection, batchLengths, batchTotalSamples, batch_BOS_positions, batch_EOS_positions \
        = assembleBatch(miniBatchingSize)
        
        startTime = datetime.now()

        for i in range(miniBatchingSize):
            sampleOutput = array("f", [0.0] * batchLengths[i]*n)

            sampleInputPropagation = array("f", [0.0] * batchLengths[i]*n)
            samplePropagation = array("f", [0.0] * batchLengths[i]*n)

            CNL.Activate(modelId, batchInput[i], sampleOutput, 1, array("i", [batchLengths[i]]), batchLengths[i])
            CNL._softmax(sampleOutput, n)
            CNL.MSE(sampleOutput, batchCorrection[i], samplePropagation)

            for bos in range(len(batch_BOS_positions[i])):
                bos_position = batch_BOS_positions[i][bos]
                eos_position = 1
                if bos > 0:
                    eos_position = batch_EOS_positions[i][bos-1]
                # print(bos_position, i, s, len(batchPropagation), eos_position)
                for j in range((bos_position-(eos_position-1))*n):
                    # print(len(batchCorrection), s*n + (eos_position-1)*n + j, s*n + (eos_position-1)*n, (bos_position)*n)
                    samplePropagation[(eos_position-1)*n + j] = 0.0
                        # print(s*n + (eos_position-1)*n + j, len(batchPropagation), s*n, j, (eos_position-1)*n, (bos_position+1-(eos_position-1))*n)
                    sampleOutput[(eos_position-1)*n + j] = batchCorrection[i][(eos_position-1)*n + j]
                    pass
                pass

            CNL.Adjust(modelId, samplePropagation, sampleInputPropagation, 1, array("i", [batchLengths[i]]), batchLengths[i])
            
            if _ >= epochs-1 or _ <= 0:
                print("INPUT &:\n", tokenizer.formSequenceSplit(batchInput[i], array("i", [batchLengths[i]])))
                print("CORRECTION *:\n", tokenizer.formSequenceSplit(batchCorrection[i], array("i", [batchLengths[i]])))
                print("OUTPUT ^:\n", tokenizer.formSequenceSplit(sampleOutput, array("i", [batchLengths[i]])))
                    #   , batchOutput[seriesLengths[0]*n - n:seriesLengths[0]*n])

        CNL.Update(modelId, learning_rate/miniBatchingSize)

        print((datetime.now()-startTime).total_seconds(), "epoch:", _) # 0.0316

def inference(maxTokens):

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

laod_container = container("NAISENT/NAISENT_LM_30M_good6")
load_model(laod_container)

pre_train(.001, 1000, 4)
# fine_tune(.0003, 400, 16)

save_container = container("NAISENT/NAISENT_LM_30M_good6")
save_model(save_container)

# inference(300)

print("TOTAL TIME:", (datetime.now()-startTime).total_seconds())

# the transformer setup seems to not work properly with batching -_- make sure to come back to it later!

# TIME FOR INFERENCE! more investigation on the whole batching problem will be done later. (currently kept as "requires more training time" problem)

# TRY AGAIN WITHOUT DROPOUT! see if batching still doesn't work.
# actually batching was actually working... huh <>;