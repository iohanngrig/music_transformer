import json 
import tensorflow as tf
from keras import layers, models, losses
from mustransformer.layers.position_encoding import TokenAndPositionEmbedding
from mustransformer.layers.transformer_block import TransformerBlock


with open('config.json', "r", encoding='utf8') as fh:
    config = json.load(fh)


EMBEDDING_DIM = config["EMBEDDING_DIM"]
N_HEADS = config["N_HEADS"]
KEY_DIM = config["KEY_DIM"]
FEED_FORWARD_DIM = config["FEED_FORWARD_DIM"]
DROPOUT_RATE = config["DROPOUT_RATE"]
B1, B2 = config["BETA_LOW"], config["BETA_UP"]
LR = config["LEARNING_RATE"]
DR = config["DECAY_RATE"]
DS = config["DECAY_STEPS"]


def build_model(notes_vocab_size, durations_vocab_size):
    note_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    durations_inputs = layers.Input(shape=(None,), dtype=tf.int32)
    note_embeddings = TokenAndPositionEmbedding(
        notes_vocab_size, EMBEDDING_DIM // 2
    )(note_inputs)
    duration_embeddings = TokenAndPositionEmbedding(
        durations_vocab_size, EMBEDDING_DIM // 2
    )(durations_inputs)
    embeddings = layers.Concatenate()([note_embeddings, duration_embeddings])
    x, attention_scores = TransformerBlock(
        N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, DROPOUT_RATE, name="attention"
    )(embeddings)
    note_outputs = layers.Dense(
        notes_vocab_size, activation="softmax", name="note_outputs"
    )(x)
    duration_outputs = layers.Dense(
        durations_vocab_size, activation="softmax", name="duration_outputs"
    )(x)
    model = models.Model(
        inputs=[note_inputs, durations_inputs],
        outputs=[note_outputs, duration_outputs],  # attention_scores
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR, decay_steps=DS, decay_rate=DR)
    model.compile(
        tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=B1, beta_2=B2),
        loss=[
            losses.SparseCategoricalCrossentropy(),
            losses.SparseCategoricalCrossentropy(),
        ],
    )
    att_model = models.Model(
        inputs=[note_inputs, durations_inputs], outputs=attention_scores
    )
    return model, att_model
