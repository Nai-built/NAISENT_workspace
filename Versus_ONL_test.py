from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL

from array import array

from datetime import datetime

modelId = ONL.Build(ONL.CHAIN([
    ONL.DENSE(2,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    ONL.DENSE(256,256, activation=ONL.ReLU(0.001)),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    # ONL.DENSE(256,256),
    ONL.DENSE(256,1),
]))

batchSize = 512

input = array("f", [5,2, 7,3] * (int)(batchSize/2))
output = array("f", [0.0] * batchSize)
desired = array("f", [1.0, -2.0] * (int)(batchSize/2))
propagation = array("f", [0.0] * batchSize)

inputPropagation = array("f", [0.0] * len(input))

t = datetime.now()

for i in range(1000):
    output = array("f", [0.0] * batchSize)
    inputPropagation = array("f", [0.0] * len(input))

    # print(len(inputPropagation))

    startTime = datetime.now()

    ONL.Activate(modelId, input, output, batchSize, array("i"), batchSize)
    print("activation", (datetime.now()-startTime).total_seconds())
    startTime = datetime.now()

    ONL.MSE(output, desired, propagation)
    print("loss", (datetime.now()-startTime).total_seconds())
    startTime = datetime.now()

    ONL.Adjust(modelId, propagation, inputPropagation, .0001, batchSize, array("i"), batchSize)
    print("adjusted", (datetime.now()-startTime).total_seconds())
    
    # print(len(inputPropagation))
    
print(output)
print("full", (datetime.now()-t).total_seconds())