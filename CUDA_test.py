from CudaNeurologicalLibrary.python import CudaNeurologicalLibrary as CNL

from array import array

from datetime import datetime

modelId = CNL.Build(CNL.CHAIN([
    CNL.SCC_POSITIONAL_EMBEDDING(2,256),

        # CNL.RMSNORM(256),
        # CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
        # CNL.RMSNORM(256),
        # CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),

    # STACK 1
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,256*4, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(256*4,256),
    ], residual=True),
    
    # STACK 2
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,256*4, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(256*4,256),
    ], residual=True),
    
    # STACK 3
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,256*4, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(256*4,256),
    ], residual=True),
    
    # STACK 4
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
        CNL.DENSE(256,256),
    ], residual=True),
    CNL.CHAIN([
        CNL.RMSNORM(256),
        CNL.DENSE(256,256*4, activation=CNL.AF_ReLU(0.001)),
        CNL.DENSE(256*4,256),
    ], residual=True),

    # CNL.CHAIN([
    #     CNL.RMSNORM(256),
    #     CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
    #     CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # ]),

    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.DENSE(256,256, activation=CNL.AF_ReLU(0.001)),
    # CNL.RMSNORM(256),
    # CNL.MH_MASKED_SELF_ATTENTION(256, (int)(256/4), 4),
    CNL.RMSNORM(256),
    CNL.DENSE(256,2),
]), optimization=CNL.OPT_ADAM())

batchSize = 512

input = array("f", [5,2, -7,-3] * (int)(batchSize/2))
output = array("f", [0.0] * batchSize)
desired = array("f", [1.0, 0.0, 0.0, 1.0] * (int)(batchSize/2))
propagation = array("f", [0.0] * batchSize)

print(input, desired)

inputPropagation = array("f", [0.0] * len(input))

CNL.Initialize(modelId, 0)

t = datetime.now()

for i in range(200):
    output = array("f", [0.0] * batchSize)
    inputPropagation = array("f", [0.0] * len(input))

    # print(len(inputPropagation))

    startTime = datetime.now()

    CNL.Activate(modelId, input, output, 1, array("i", [batchSize]*1), batchSize)
    CNL._softmax(output, 2)
    print("activation", (datetime.now()-startTime).total_seconds())
    startTime = datetime.now()

    CNL.MSE(output, desired, propagation)
    print("loss", (datetime.now()-startTime).total_seconds())
    startTime = datetime.now()

    CNL.Adjust(modelId, propagation, inputPropagation, 1, array("i", [batchSize]*1), batchSize)
    print("adjusted", (datetime.now()-startTime).total_seconds())
    startTime = datetime.now()

    CNL.Update(modelId, .001)
    print("updated", (datetime.now()-startTime).total_seconds())
    
    # print(len(inputPropagation))

print(output)
print("full", (datetime.now()-t).total_seconds())