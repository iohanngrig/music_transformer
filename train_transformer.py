import os
import json
import music21
from keras import callbacks

from utils.common import get_files, set_seed
from utils.midi_tools import parse_midi_files, load_parsed_files
from mustransformer.data import DataHandler
from mustransformer.model import build_model
from mustransformer.generator import MusicGenerator


us = music21.environment.Environment()
us['musescoreDirectPNGPath'] = r'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'

dir_path = os.path.dirname(os.path.realpath(__file__))
CONFIG = os.path.join(dir_path, 'config.json')
with open(CONFIG, "r", encoding='utf8') as fh:
    config = json.load(fh)

EPOCHS = config["EPOCHS"]
LOAD_MODEL = config["LOAD_MODEL"]

set_seed(42)

dh = DataHandler()

# Create a model save checkpoint
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./runs/logs")

# Tokenize starting prompt
music_generator = MusicGenerator(dh.notes_vocab, dh.durations_vocab)
model, attn_model = build_model(dh.notes_vocab_size, dh.durations_vocab_size)

print("\n", model.summary())

if LOAD_MODEL:
    model.load_weights("./checkpoint/checkpoint.ckpt")
    # model = models.load_model('./models/model', compile=True)

model.fit(
    dh.ds,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint_callback,
        tensorboard_callback,
        music_generator
    ],
)

# Save the final model
model.save("models/model_bach")
