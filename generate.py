import os
import json
import time
import numpy as np
import music21
from keras import models
from utils.midi_tools import get_midi_note
from mustransformer.data import DataHandler
from mustransformer.generator import MusicGenerator
from mustransformer.model import build_model


dir_path = os.path.dirname(os.path.realpath(__file__))

with open('config.json', "r", encoding='utf8') as fh:
    config = json.load(fh)
GENERATE_LEN = config["GENERATE_LEN"]

us = music21.environment.Environment()
us['musescoreDirectPNGPath'] = config["musescoreDirectPNGPath"]


class MusicSampleGenerator2:
    def __init__(self, model, index_to_note, index_to_duration):
        self.model = model
        self.index_to_note = index_to_note
        self.note_to_index = {
            note: index for index, note in enumerate(index_to_note)
        }
        self.index_to_duration = index_to_duration
        self.duration_to_index = {
            duration: index for index, duration in enumerate(index_to_duration)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def get_note(self, notes, durations, temperature):
        sample_note_idx = 1
        while sample_note_idx == 1:
            sample_note_idx, note_probs = self.sample_from(
                notes[0][-1], temperature
            )
            sample_note = self.index_to_note[sample_note_idx]
        sample_duration_idx = 1
        while sample_duration_idx == 1:
            sample_duration_idx, duration_probs = self.sample_from(
                durations[0][-1], temperature
            )
            sample_duration = self.index_to_duration[sample_duration_idx]
        new_note = get_midi_note(sample_note, sample_duration)
        return (
            new_note,
            sample_note_idx,
            sample_note,
            note_probs,
            sample_duration_idx,
            sample_duration,
            duration_probs,
        )

    def generate(self, start_notes, start_durations, max_tokens, temperature):
        attention_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("attention").output,
        )

        start_note_tokens = [self.note_to_index.get(x, 1) for x in start_notes]
        start_duration_tokens = [
            self.duration_to_index.get(x, 1) for x in start_durations
        ]
        sample_note = None
        sample_duration = None
        info = []
        midi_stream = music21.stream.Stream()
        midi_stream.append(music21.clef.TrebleClef())
        for sample_note, sample_duration in zip(start_notes, start_durations):
            new_note = get_midi_note(sample_note, sample_duration)
            if new_note is not None:
                midi_stream.append(new_note)
        while len(start_note_tokens) < max_tokens:
            x1 = np.array([start_note_tokens])
            x2 = np.array([start_duration_tokens])
            notes, durations = self.model.predict([x1, x2], verbose=0)
            repeat = True
            while repeat:
                (
                    new_note,
                    sample_note_idx,
                    sample_note,
                    note_probs,
                    sample_duration_idx,
                    sample_duration,
                    duration_probs,
                ) = self.get_note(notes, durations, temperature)

                if (
                    isinstance(new_note, music21.chord.Chord)
                    or isinstance(new_note, music21.note.Note)
                    or isinstance(new_note, music21.note.Rest)
                ) and sample_duration == "0.0":
                    repeat = True
                else:
                    repeat = False

            if new_note is not None:
                midi_stream.append(new_note)

            _, att = attention_model.predict([x1, x2], verbose=0)

            info.append(
                {
                    "prompt": [start_notes.copy(), start_durations.copy()],
                    "midi": midi_stream,
                    "chosen_note": (sample_note, sample_duration),
                    "note_probs": note_probs,
                    "duration_probs": duration_probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_note_tokens.append(sample_note_idx)
            start_duration_tokens.append(sample_duration_idx)
            start_notes.append(sample_note)
            start_durations.append(sample_duration)
            if sample_note == "START":
                break
        return info

    def generate_samples(self, n, max_tokens=GENERATE_LEN, temperature=1.0, show_score=False):
        for i in range(1, n+1):
            info = self.generate(
                ["START"], ["0.0"], max_tokens=max_tokens, temperature=temperature
            )
            midi_stream = info[-1]["midi"].chordify()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            midi_stream.write("midi", fp=os.path.join(dir_path, "data/generated", f"output-{i}-{timestr}.mid"))
            if show_score:
                midi_stream.show()


class MusicSampleGenerator(MusicGenerator):
    def __init__(self, model, index_to_note, index_to_duration):
        super().__init__(index_to_note, index_to_duration)
        self.model = model

    def generate_samples(self, n, max_tokens=GENERATE_LEN, temperature=1.0, show_score=False):
        for i in range(1, n+1):
            info = self.generate(
                ["START"], ["0.0"], max_tokens=max_tokens, temperature=temperature
            )
            midi_stream = info[-1]["midi"].chordify()
            timestr = time.strftime("%Y%m%d-%H%M%S")
            midi_stream.write("midi", fp=os.path.join(dir_path, "data/generated", f"output-{i}-{timestr}.mid"))
            if show_score:
                midi_stream.show()


dh = DataHandler()
model, attn_model = build_model(dh.notes_vocab_size, dh.durations_vocab_size)
model.load_weights(os.path.join(dir_path, "checkpoint/checkpoint.ckpt"))

music_generator = MusicSampleGenerator(model, dh.notes_vocab, dh.durations_vocab)
music_generator.generate_samples(n=3, max_tokens=2*GENERATE_LEN)
