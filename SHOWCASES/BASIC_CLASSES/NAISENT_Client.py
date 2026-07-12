import socket
import json
import struct
import time
from array import array

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

class NAISENT_Client:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(("127.0.0.1", 5555))

        self.stepIndex = -1
#
    def ballTestClientStep(self, interactionCallback):
        # 1) request state
        self.socket.sendall(b'\x01')

        # 2) receive state
        hadRestarted = recv_exact(self.socket, 1)[0] != 0
        
        episodeLength = self.stepIndex+1

        if hadRestarted:
            self.stepIndex = -1
        self.stepIndex += 1

        # n = 60 * 80 * 3
        obs = recv_float_array(self.socket)

        # 3) compute action (float[])
        action = interactionCallback(obs, self.stepIndex)  # returns list/array of floats
        send_float_array(self.socket, action)

        # 4) receive correction (float[])
        correction = recv_float_array(self.socket)
        return action, correction, self.stepIndex, hadRestarted, episodeLength

    def start(self):
        self.socket.sendall(b'\x01')

    def shapeTestClientStep(self, interactionCallback):
        # 2) receive state
        
        hadRestarted = self.stepIndex == 16-1

        if hadRestarted:
            self.stepIndex = -1
        self.stepIndex += 1

        # n = 60 * 80 * 3
        obs = recv_float_array(self.socket)

        # 3) compute action (float[])
        action = interactionCallback(obs, self.stepIndex)  # returns list/array of floats

        # 3) request correction
        self.socket.sendall(b'\x01')

        # 4) receive correction (float[])
        correction = recv_float_array(self.socket)
        
        return action, correction, self.stepIndex, hadRestarted
    
    def proceed(self):
        self.socket.sendall(b'\x01')