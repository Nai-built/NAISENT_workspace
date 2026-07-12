from pathlib import Path
import json

CONTAINER_FOLDER = Path(__file__).resolve().parent / "CONTAINER_FOLDER"

class container:
    def __init__(self, name: str):
        self.path = CONTAINER_FOLDER / (name + ".json")
    
    def write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=5)

    def read(self):
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data