# Beta testing for NAISENT: Ballseeker
# A newer version has been made with PyTorch, if you want to check it out (has temporal recognition with LSTM)
# Go to my reddit u/nai-official or this youtube video to learn more
# https://www.youtube.com/watch?v=-Nxd4MPhm6E

import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from NeurologicalLibrary.bridge.pythonConversion import multiply, add, nextInteger, NeurologicalLogic as NL

from BASIC_CLASSES.NAISENT_Structure import NAISENT_Structure
from BASIC_CLASSES.NAISENT_Client import NAISENT_Client
from BASIC_CLASSES.NAISENT_Storage import NAISENT_Storage

import time
import json

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

# 1. Start the app in the background
project_dir = "cs/Display_WinForms"
process = subprocess.Popen(["powershell", "-Command", "dotnet run"], cwd=project_dir)

# 2. Wait until the window exists (Change "Form1" to your actual WinForms Window Title)
if wait_for_window("NAISENT ENVIRONMENT", timeout=45):
    # 3. Your remaining Python code goes here
    print("Continuing with the rest of the Python script...")

SCRIPT_DIR = Path(__file__).resolve().parent

structure = NAISENT_Structure()
client = NAISENT_Client()
storage = NAISENT_Storage(SCRIPT_DIR / "temporaryData__ball.json")

try:
    structure.write(storage.read())
except:
    print("FAILED TO LOAD MODEL!")

def interactionCallback(state, stepIndex):
    return structure.interact(state, stepIndex=stepIndex)

interactions = {}
desiredActions = {}

amountOfEpisodes = 0

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
        return "Left"
    elif index == 1:
        return "Right"
    elif index == 2:
        return "Idle"

def listToText(list):
    return indexToText(getHighestNumberIndex(list))

# START THE MODEL'S TRAINING LOOP
client.start()
while True:
    if amountOfEpisodes >= 120:
        break

    # print("stepping")

    interaction, response, stepIndex, hadRestarted, episodeLength = client.ballTestClientStep(interactionCallback=interactionCallback)

    if hadRestarted:
        print("CONCLUDED EPISODE:", amountOfEpisodes+1)
        print(episodeLength)
        for i in reversed(range(len(interactions))):
            structure.correct(interaction=interactions[i]
                              , correction=desiredActions[i]
                              , stepIndex=i)
        structure.applyCorrections(1)
        interactions.clear()
        desiredActions.clear()
        amountOfEpisodes += 1
        structure.episodeIndex = amountOfEpisodes+1
    
    print(stepIndex, interaction, response.tolist())
    predictionText = listToText(interaction)
    correctionText = listToText(response)
    print(stepIndex, predictionText, correctionText)
    if predictionText == correctionText:
        print("--", "SUCCESS")
    else:
        print("--", "F")

    interactions[stepIndex] = interaction
    desiredActions[stepIndex] = response.tolist()

    # time.sleep(2) # .05

    client.proceed()

storage.write(structure.read())
