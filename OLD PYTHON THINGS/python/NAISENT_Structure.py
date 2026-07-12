from bridge.pythonConversion import NeurologicalLogic as NL

LEARNING_RATE = 1e-3/2

class NAISENT_Structure:
    def __init__(self):
        
        self.structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
            NL.SPREADER(3, 80, 60),

            # EXPECTED INPUT: 80x60
            NL.CONVOLUTIONAL(3, 8, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),
            # EXPECTED INPUT: 80x60
            NL.CONVOLUTIONAL(8, 8, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),

            # EXPECTED INPUT: 80x60
            NL.MAX_POOLING(stride=2, poolSize='2x2'),

            # EXPECTED INPUT: 40x30
            NL.CONVOLUTIONAL(8, 16, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),
            # EXPECTED INPUT: 40x30
            NL.CONVOLUTIONAL(16, 16, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),

            # EXPECTED INPUT: 40x30
            NL.MAX_POOLING(stride=2, poolSize='2x2'),

            # EXPECTED INPUT: 20x15
            NL.CONVOLUTIONAL(16, 32, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),
            # EXPECTED INPUT: 20x15
            NL.CONVOLUTIONAL(32, 32, padding=1, stride=1, kernelSize='3x3'),
            NL.AF_ReLU(.01),

            # EXPECTED INPUT: 20x15
            NL.MAX_POOLING(stride=2, poolSize='2x2'),

            # EXPECTED INPUT: 10x7
            NL.FLATTENER(),
            
            # EXPECTED INPUT: 32 * 10x7 = 2,240
            NL.DENSE(2240, 256),
            NL.AF_ReLU(.01),
            NL.DENSE(256, 3),
            NL.AF_Softmax(True),
        ]))
        # self.structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
        #     NL.SPREADER(3, 80, 60),

        #     # EXPECTED INPUT: 80x60
        #     NL.CONVOLUTIONAL(3, 16, padding=1, stride=1, kernelSize='3x3'),
        #     NL.AF_ReLU(.01),

        #     # EXPECTED INPUT: 80x60
        #     NL.MIN_POOLING(stride=2, poolSize='2x2'),

        #     # EXPECTED INPUT: 40x30
        #     NL.CONVOLUTIONAL(16, 16, padding=1, stride=1, kernelSize='3x3'),
        #     NL.AF_ReLU(.01),

        #     # EXPECTED INPUT: 40x30
        #     NL.MIN_POOLING(stride=2, poolSize='2x2'),

        #     # EXPECTED INPUT: 20x15
        #     NL.CONVOLUTIONAL(16, 2, padding=1, stride=1, kernelSize='3x3'),
        #     NL.AF_ReLU(.01),

        #     # EXPECTED INPUT: 20x15
        #     NL.FLATTENER(),
            
        #     # EXPECTED INPUT: 2 * 20x15 = 600
        #     NL.DENSE(600, 32),
        #     NL.AF_ReLU(.01),
        #     NL.DENSE(32, 3),
        #     NL.AF_Softmax(True),
        # ]))

        # self.structureId = NL.CreateNeurologicalStructure(NL.CHAIN([
        #     NL.LSTM_RECURSIVE(3, 128),
        #     NL.DENSE(128, 128),
        #     NL.AF_ReLU(.001),
        #     # NL.DENSE(64, 64),
        #     # NL.AF_ReLU(.01),
        #     NL.DENSE(128, 3),
        #     NL.AF_Softmax(True),

        #     # NL.DENSE(3, 256),
        #     # NL.AF_ReLU(.01),
        #     # NL.DENSE(256, 256),
        #     # NL.AF_ReLU(.01),
        #     # NL.DENSE(256, 256),
        #     # NL.AF_ReLU(.01),
        #     # NL.DENSE(256, 3),
        #     # NL.AF_Softmax(True),
        # ]))

        self.episodeIndex = 1

    def interact(self, state, stepIndex):
        interaction = NL.Activate(self.structureId, input=state, sampleIndex="ep" + self.episodeIndex.__str__() + " -s" + stepIndex.__str__())
        # print(interaction)
        return interaction

    def correct(self, interaction, correction, stepIndex):
        NL.Adjust(self.structureId, gradient
                  =NL.MSE_LOSS_GRADIENT(interaction, correction), sampleIndex="ep" + self.episodeIndex.__str__() + " -s" + stepIndex.__str__())
        
    def applyCorrections(self, divisor):
        NL.ApplyAdjustments(self.structureId, learningRate=LEARNING_RATE, averagingDivisor=divisor)

    def read(self):
        return NL.Read(self.structureId)
    def write(self, data):
        return NL.Write(self.structureId, data)