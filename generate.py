import os
import json
import time
import argparse

import music21

from utils.common import set_seed
from mustransformer.data import DataHandler
from mustransformer.generator import MusicGenerator
from mustransformer.model import build_model


dir_path = os.path.dirname(os.path.realpath(__file__))

with open('config.json', "r", encoding='utf8') as fh:
    config = json.load(fh)
GENERATE_LEN = config["GENERATE_LEN"]

us = music21.environment.Environment()
us['musescoreDirectPNGPath'] = config["musescoreDirectPNGPath"]

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=1)
parser.add_argument("-l", "--length", type=float, default=1.0)
parser.add_argument("-t", "--temperature", type=float, default=1.4)
parser.add_argument("-s", "--seed", type=int, default=42)
parser.add_argument("-d", "--draw", type=int, default=0)


class MusicSampleGenerator(MusicGenerator):
    def __init__(self, model, index_to_note, index_to_duration):
        super().__init__(index_to_note, index_to_duration)
        self.model = model

    def generate_samples(self, n=1, max_tokens=GENERATE_LEN, temperature=1.0, seed=42, show_score=0):
        for i in range(1, n+1):
            set_seed(seed + i)
            info = self.generate(
                ["START"], ["0.0"], max_tokens=max_tokens, temperature=temperature
            )
            midi_stream = info[-1]["midi"].chordify()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            midi_stream.write("midi", fp=os.path.join(dir_path, "data/generated", f"output-{i}-{timestr}.mid"))
            if show_score > 0:
                midi_stream.show()


if __name__ == "__main__":
    args = parser.parse_args()
    dh = DataHandler()
    model, attn_model = build_model(dh.notes_vocab_size, dh.durations_vocab_size)
    model.load_weights(os.path.join(dir_path, "checkpoint/checkpoint.ckpt"))

    music_generator = MusicSampleGenerator(model, dh.notes_vocab, dh.durations_vocab)
    music_generator.generate_samples(n=args.number,
                                     max_tokens=GENERATE_LEN*args.length,
                                     temperature=args.temperature,
                                     seed=args.seed,
                                     show_score=args.draw)

