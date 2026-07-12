# NAISENT Shape Recognition Model in CPU

import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from OptimizedNeurologicalLibrary.python import OptimizedNeurologicalLibrary as ONL
from DATA.container import container
import json
import time
from array import array
from datetime import datetime
import random

pre_string = "SHAPE_RECOGNITION"

import subprocess
import ctypes

def wait_for_window(window_title, timeout=30):
    """Waits until a window with the given title appears."""
    start_time = time.time()
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFOCALBACK if hasattr(ctypes, 'WINFOCALBACK') else ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    GetWindowText = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible

    print(f"Waiting for '{window_title}' to launch...")
    
    while time.time() - start_time < timeout:
        titles = []
        def foreach_window(hwnd, lParam):
            if IsWindowVisible(hwnd):
                length = GetWindowTextLength(hwnd)
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowText(hwnd, buff, length + 1)
                titles.append(buff.value)
            return True

        EnumWindows(EnumWindowsProc(foreach_window), 0)
        
        if window_title in titles:
            print("App is fully running and visible!")
            return True
        time.sleep(0.5) # Check every half second
        
    print("Timeout reached. App took too long to start.")
    return False

def collect_samples(n):
    # 1. Start the app in the background
    project_dir = "cs/Display_WinForms"
    process = subprocess.Popen(["powershell", "-Command", "dotnet run"], cwd=project_dir)

    # 2. Wait until the window exists (Change "Form1" to your actual WinForms Window Title)
    if wait_for_window("NAISENT ENVIRONMENT", timeout=45):
        # 3. Your remaining Python code goes here
        print("Continuing with the rest of the Python script...")

    import socket
    import struct

    def recv_exact(sock, n):
        buf = bytearray(n)
        pos = 0
        while pos < n:
            r = sock.recv(n - pos)
            if not r:
                raise ConnectionError()
            buf[pos:pos+len(r)] = r
            pos += len(r)
        return buf

    def recv_float_array(sock):
        length = struct.unpack("<i", recv_exact(sock, 4))[0]
        raw = recv_exact(sock, length * 4)
        arr = array("f")
        arr.frombytes(raw)
        return arr

    def send_float_array(sock, arr):
        # arr: iterable of floats
        sock.sendall(struct.pack("<i", len(arr)))
        sock.sendall(array("f", arr).tobytes())


    _socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _socket.connect(("127.0.0.1", 5555))

    stepIndex = -1

    def start():
        _socket.sendall(b'\x01')

    def shapeTestClientStep():
        # 2) receive state
        
        # n = 60 * 80 * 3
        obs = recv_float_array(_socket)

        # 3) request correction
        _socket.sendall(b'\x01')

        # 4) receive correction (float[])
        correction = recv_float_array(_socket)
        return obs, correction
        
    def proceed():
        _socket.sendall(b'\x01')

    samplesContainer = container(pre_string+"/SAMPLES")
    correctionsContainer = container(pre_string+"/CORRECTIONS")

    collectedSamples = array("f")
    collectedCorrections = array("f")

    start()

    for i in range(n):
        sample, correction = shapeTestClientStep()

        collectedSamples.extend(sample)
        collectedCorrections.extend(correction)

        proceed()

    samplesContainer.write(collectedSamples.tolist())
    correctionsContainer.write(collectedCorrections.tolist())

def train_model(n, e=1000, lr=0.001):
    ARRAY_TYPE = "f"

    samplesAmount = n-1

    sampleSize = 80*60*3
    correctionSize = 3

    collectedSamples = container(pre_string+"/SAMPLES").read()[0:(samplesAmount+1)*sampleSize]
    collectedCorrections = container(pre_string+"/CORRECTIONS").read()[0:(samplesAmount+1)*correctionSize]

    modelStorage = container(pre_string+"/MODEL_SHOWCASE")

    batchSize = 64

    # probably works, but just try to give it more training time

    # IT LEARNT!!!!!!!!
    # i am so glad and happy / proud like a father somehow
    # now.. we need to make it actually save our model so all that training doesn't go to waste each time [ DONE ]

    structureId = ONL.Build(ONL.CHAIN([
        ONL.CONVOLUTIONAL(3, 32, kernelSize='5x5', channelSize='80x60', padding=2, stride=4
                        , activation=ONL.ReLU(.001)),

        ONL.CONVOLUTIONAL(32, 64, kernelSize='5x5', channelSize='20x15', padding=2, stride=4
                        , activation=ONL.ReLU(.001)),
        

        ONL.DENSE(64*5*4, 256
                , activation=ONL.ReLU(.001)),
        ONL.DENSE(256, 256
                , activation=ONL.ReLU(.001)),
        ONL.DENSE(256, 3),
    ]))

    try:
        ONL.WriteStructureInfo(structureId, modelStorage.read())
    except:
        print("FAILED TO LOAD MODEL!")

    output = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

    propagation = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

    # non-important
    inputPropagation = array(ARRAY_TYPE, [0.0] * batchSize * sampleSize)

    startTIme = datetime.now()

    def getHighestNumberIndex(list):
        highestNumber = None
        highestnumberIndex = 0
        i = 0
        for v in list:
            if highestNumber:
                if v > highestNumber:
                    highestNumber = v
                    highestnumberIndex = i
            else:
                highestNumber = v
                highestnumberIndex = i
            i += 1
        return highestnumberIndex

    def indexToText(index):
        if index == 0:
            return "Circle"
        elif index == 1:
            return "Triangle"
        elif index == 2:
            return "Square"

    def listToText(list):
        return indexToText(getHighestNumberIndex(list))

    def compareToString(output, correction):
        string = ""

        successfulAmount = 0

        for i in range(batchSize):
            outputShape = listToText(output[i*correctionSize:i*correctionSize + correctionSize])
            correctionShape = listToText(correction[i*correctionSize:i*correctionSize + correctionSize])

            if outputShape == correctionShape:
                string += outputShape + " == " + correctionShape + " ----- SUCCESS\n"
                successfulAmount += 1
            else:
                string += outputShape + " == " + correctionShape + " ----- F\n"
                pass

        string += "( " + successfulAmount.__str__() + " / " + batchSize.__str__() + " )"
        print(string)

    # runs a training session with the amount of steps as e (epochs)
    for i in range(e):
        output = array(ARRAY_TYPE, [0.0] * batchSize * correctionSize)

        inputPropagation = array(ARRAY_TYPE, [0.0] * batchSize * sampleSize)

        # selectedSamples = array("f", collectedSamples[selectedBatchStart*sampleSize:selectedBatchEnd*sampleSize])
        # selectedCorrections = array("f", collectedCorrections[selectedBatchStart*correctionSize:selectedBatchEnd*correctionSize])
        
        selectedSamples = array("f")
        selectedCorrections = array("f")

        for _ in range(batchSize):
            sampleTarget = random.randint(0, samplesAmount)

            selectedSamples.extend(collectedSamples[sampleTarget*sampleSize:sampleTarget*sampleSize + sampleSize])
            selectedCorrections.extend(collectedCorrections[sampleTarget*correctionSize:sampleTarget*correctionSize + correctionSize])

        # print(len(selectedSamples)/batchSize, sampleSize)

        startStepTIme = datetime.now()

        ONL.Activate(structureId, selectedSamples, output, batchSize=batchSize)
        ONL._softmax(output, correctionSize)
        ONL.MSE(output, selectedCorrections, propagation)
        # lower the Learning-Rate gradually between each training session
        ONL.Adjust(structureId, propagation, inputPropagation, lr=lr, batchSize=batchSize)

        print("<---" + i.__str__() + "--->")
        compareToString(output, selectedCorrections)

        print("time passed:", (datetime.now()-startTIme).total_seconds(), "-", (datetime.now()-startStepTIme).total_seconds(), "step:", i+1, "/", e)

    modelStorage.write(ONL.ReadStructureInfo(structureId))

# CONCLUDED!

collect_samples(1000)
train_model(1000, 3000, 0.0002)