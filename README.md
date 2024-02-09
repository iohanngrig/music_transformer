# music_transformer

## Music Generation

* Model learns `notes` and `durations` from the midi files
* Difference from the original algorithm in Ref [1] is that instead of keeping only the last note from each chord, we keep the highest 4 dropping the same lower notes that are in different octaves.
* The drawback is that, if notes in a chord have different durations, longer notes won't be sustained, but replayed. This will result in a characteristic hammering pattern. Also, durations are not learned as good as notes, resulting in a jazz like pattern.
* The vocabulary consists not only of single notes (singlets) but of duplets, triplets and quadruplets of notes. However, since not all combinations of notes are likely the total size of the dictionary is 7,567 instead of ${89 \choose 4} = 2,441,626$, where 89 is the number of notes on a piano, plus an empty note. 
* This repo comes with a pre-trained model
* The midi output is generated inside the `data/generated` folder.

## Installation

Clone this repo via
```bash
https://github.com/iohanngrig/music_transformer.git
```

Create a conda environment from .yml file

```bash
conda env create -f environment.yml
```

or

Create a new conda environment (versions matter)

```bash
conda create -n env python=3.9
```

Install Tensorflow specific packages (select your version of cuda)

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.1.0
pip install "tensorflow<2.11"
```

Install `MuseScore` desktop software, and after `music21` library:

```bash
pip install music21
python -m music21.configure 
```
wait until the configuration finds the path to `MuseScore4.exe` file.

Install the rest of the packages (repetitions are present):

```bash
pip install -r requirements.txt
```

## Generation 

You can generate via `generate.py` script

```bash
python generate.py
```

## Dataset

The dataset consists of works by J.S. Bach. Midi files for training are placed inside the data/midi folder. You can try different/larger datasets.

## Training

You have to train transformer by running

```bash
python train_transformer.py
```
monitor training by running `tensorboard --logdir runs/logs/train`.

If training from scratch or on a new dataset, modify `config.json`, and set `PARSE_MIDI_FILES: true` and `LOAD_MODEL: false`. 

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

## References

[1] This work is based on the original code developed by [David Foster](https://github.com/davidADSP) inside the following repository [Generative_Deep_Learning_2nd_Edition](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) that contains codes for the eponymous textbook.

## License

This project is licensed under the MIT license