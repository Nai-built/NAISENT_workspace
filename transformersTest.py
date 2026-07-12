from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL

from array import array

from datetime import datetime

lr = 0.0001

# structureId = ONL.Build(ONL.DECODER_ONLY_TRANSFORMER(3, 3, 256, [
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
#     ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001)),ONL.FFL(256*4, ONL.ReLU(.001))]),
# ]))
structureId = ONL.Build(ONL.DECODER_ONLY_TRANSFORMER(3, 3, 256, [
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001))]),
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001))]),
    ONL.TRANSFORMER_STACK(4, [ONL.FFL(256*4, ONL.ReLU(.001))]),
]))

input = array("f", [
    0,1,0,
    0,1,0,
    0,0,1,
    1,0,0,
    1,0,0,
    0,1,0,

    0,1,0,
    0,1,0,
    0,1,0,
    0,0,1,
    
    1,0,0,
    1,0,0,

    1,0,0,
    0,0,1,
    0,0,1,
    0,1,0,
])
correction = array("f", [
    0,1,0,
    0,0,1,
    1,0,0,
    1,0,0,
    0,1,0,
    0,0,1,

    0,1,0,
    0,1,0,
    0,1,0,
    0,0,1,
    
    1,0,0,
    1,0,0,

    1,0,0,
    0,0,1,
    0,0,1,
    0,1,0,
])

totalSamples = (int)(len(input)/3)

batchSize = 4

seriesLengths = array("i", [
    6,
    4,
    2,
    4,
])

n = 3

for _ in range(10000):
    output = array("f", [0.0] * len(input))

    inputPropagation = array("f", [0.0] * len(input))
    propagation = array("f", [0.0] * len(correction))

    startTime = datetime.now()

    ONL.Activate(structureId, input, output, batchSize, seriesLengths, totalSamples)
    ONL._softmax(output, 3)
    ONL.MSE(output, correction, propagation)
    ONL.Adjust(structureId, propagation, inputPropagation, lr, batchSize, seriesLengths, totalSamples)

    print([output[i:i + n] for i in range(0, len(output), n)])

    print((datetime.now()-startTime).total_seconds()) # 0.0316