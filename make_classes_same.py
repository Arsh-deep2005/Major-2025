import os
from pathlib import Path
import shutil

CLIENTS = [
    "data/disease/rice/farm_1",
    "data/disease/rice/farm_2"
]

def ensure_same_classes():
    all_labels = set()

    # collect all labels from all farms
    for client in CLIENTS:
        if not os.path.exists(client): continue
        for name in os.listdir(client):
            if os.path.isdir(os.path.join(client, name)):
                all_labels.add(name)

    print("All detected labels:", all_labels)

    # ensure every client has same folder structure
    for client in CLIENTS:
        os.makedirs(client, exist_ok=True)
        for label in all_labels:
            path = os.path.join(client, label)
            os.makedirs(path, exist_ok=True)
            print("Created:", path)

if __name__ == "__main__":
    ensure_same_classes()
