import json
import music21

import tensorflow as tf
from keras import layers

from utils.common import get_files
from utils.midi_tools import parse_midi_files, load_parsed_files


with open('config.json', "r", encoding='utf8') as fh:
    config = json.load(fh)


BATCH_SIZE = config["BATCH_SIZE"]
MIDI_DATA_PATH = config["MIDI_DATA_PATH"]
PARSE_MIDI_FILES = config["PARSE_MIDI_FILES"]
PARSED_DATA_PATH = config["PARSED_DATA_PATH"]
SEQ_LEN = config["SEQ_LEN"]
DATASET_REPETITIONS = config["DATASET_REPETITIONS"]


class DataHandler:
    def __init__(self) -> None:
        notes, durations = self.get_notes_durations()
        self.notes = notes 
        self.durations = durations
        self.get_notes_durations_ds_vocab()
        assert self.seq_ds is not None
        assert self.notes_vocab_size is not None
        assert self.durations_vocab_size is not None
        assert self.notes_vectorize_layer is not None
        assert self.durations_vectorize_layer is not None
        assert self.notes_vocab is not None
        assert self.durations_vocab is not None
        self.ds = self.get_datasets()

    @staticmethod
    def get_notes_durations():
        parser = music21.converter
        file_list = get_files(MIDI_DATA_PATH, ".mid")
        if PARSE_MIDI_FILES:
            notes, durations = parse_midi_files(
                file_list, parser, SEQ_LEN + 1, PARSED_DATA_PATH
            )
            print("File parsing is completed.")
        else:
            notes, durations = load_parsed_files(PARSED_DATA_PATH)
        return notes, durations 

    @staticmethod
    def create_dataset(elements, batch_size):
        ds = (
            tf.data.Dataset.from_tensor_slices(elements)
            .batch(batch_size, drop_remainder=True)
            .shuffle(1000)
        )
        vectorize_layer = layers.TextVectorization(
            standardize=None, output_mode="int"
        )
        vectorize_layer.adapt(ds)
        vocab = vectorize_layer.get_vocabulary()
        return ds, vectorize_layer, vocab

    def get_notes_durations_ds_vocab(self):
        notes_seq_ds, notes_vectorize_layer, notes_vocab = self.create_dataset(self.notes, BATCH_SIZE)
        durations_seq_ds, durations_vectorize_layer, durations_vocab = self.create_dataset(self.durations, BATCH_SIZE)
        self.seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))
        self.notes_vocab_size = len(notes_vocab)
        self.durations_vocab_size = len(durations_vocab)
        self.notes_vectorize_layer = notes_vectorize_layer
        self.durations_vectorize_layer = durations_vectorize_layer
        self.notes_vocab = notes_vocab
        self.durations_vocab = durations_vocab

        # Display some token:note mappings
        print(f"\nNOTES_VOCAB: length = {len(notes_vocab)}")
        for i, note in enumerate(notes_vocab[:10]):
            print(f"{i}: {note}")

        print(f"\nDURATIONS_VOCAB: length = {len(durations_vocab)}")
        # Display some token:duration mappings
        for i, note in enumerate(durations_vocab[:10]):
            print(f"{i}: {note}")

    def prepare_inputs(self, notes, durations):
        # Create the training set of sequences and the same sequences shifted by one note
        notes = tf.expand_dims(notes, -1)
        durations = tf.expand_dims(durations, -1)
        tokenized_notes = self.notes_vectorize_layer(notes)
        tokenized_durations = self.durations_vectorize_layer(durations)
        x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
        y = (tokenized_notes[:, 1:], tokenized_durations[:, 1:])
        return x, y

    def get_datasets(self):
        return self.seq_ds.map(self.prepare_inputs).repeat(DATASET_REPETITIONS)


if __name__ == "__main__":
    dh = DataHandler()
    print(dh.notes_vocab, dh.notes_vocab_size)