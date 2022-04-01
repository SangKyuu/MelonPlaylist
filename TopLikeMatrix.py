import json
from tqdm import tqdm
import pandas as pd
import os


class TopLikeMatrix:
    def __init__(self, FILE_PATH):
        self.FILE_PATH = FILE_PATH

        self.train = pd.read_json(os.path.join(FILE_PATH, 'train.json'))
        self.val = pd.read_json(os.path.join(FILE_PATH, 'val.json'))
        self.results = pd.read_json(os.path.join(FILE_PATH, 'results.json'))

        # with open(os.path.join(FILE_PATH, 'train.json'), encoding="utf-8") as f:
        #     self.train = json.load(f)
        # with open(os.path.join(FILE_PATH, 'val.json'), encoding="utf-8") as f:
        #     self.val = json.load(f)
        # with open(os.path.join(FILE_PATH, 'results.json'), encoding="utf-8") as f:
        #     self.results = json.load(f)

    def topk(self):
        self.t