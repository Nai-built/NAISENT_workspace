from bridge.pythonConversion import NeurologicalLogic as NL

from datetime import datetime

structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
    NL.DENSE(2, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 512),
    NL.AF_ReLU(.001),
    NL.DENSE(512, 1),
]))

input = [
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
    [3,3],
]

startTIme = datetime.now()

for i in range(len(input)):
    NL.Activate(structureId, input[i], "tt: " + i.__str__())
    
print("time passed:", (datetime.now()-startTIme).total_seconds())
