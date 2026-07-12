from pathlib import Path
import json

class NAISENT_Storage:
    def __init__(self, path: Path):
        self.path = path
    
    def write(self, data):
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def read(self):
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data