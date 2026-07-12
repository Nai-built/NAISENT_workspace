from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL

from array import array
from datetime import datetime

import time

structureId = ONL.Build(ONL.CHAIN([
    # ONL.DENSE(2, 1, activation=ONL.ReLU(.001))
    ONL.DENSE(2, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 512, activation=ONL.ReLU(.001)),
    ONL.DENSE(512, 1),
]))

input = array("f", [
    5,-2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,
    5,-2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,

    5,2
,   4,6
,   6,3
,   1,-6
,   3,-3
,   11,5
,   3,4
,   9,0,
])

desired_output = array("f", [0.0] * (int)(len(input)/2) * 1)
for i in range(len(desired_output)):
    desired_output[i] = input[i*2] * input[(i*2) + 1]
    
output = array("f", [0.0] * (int)(len(input)/2) * 1)

inputPropagation = array("f", [0.0] * (int)(len(input)))

epoch = 10000

propagation = array("f", [2.0] * (int)(len(input)/2) * 1)

# passed removing blocking

# now test it after adding the gradient buffers PASSED

startTIme = datetime.now()
for i in range(epoch):
    output = array("f", [0.0] * (int)(len(input)/2) * 1)
    inputPropagation = array("f", [0.0] * (int)(len(input)))

    startStepTIme = datetime.now()

    ONL.Activate(structureId, input, output)
    ONL.MSE(output, desired_output, propagation)
    ONL.Adjust(structureId, propagation, inputPropagation, lr=.001)

    print("time passed:", (datetime.now()-startTIme).total_seconds(), "-", (datetime.now()-startStepTIme).total_seconds(), "step:", i, "/", epoch)

    time.sleep(.00001)

print(output)
print(desired_output)