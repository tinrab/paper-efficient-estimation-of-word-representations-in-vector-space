import os

EMBEDDING_DIM = 60
CONTEXT_SIZE = (
    2  # Number of words on each side of the target word (2 words before, 2 words after)
)
EPOCHS = 1
LEARNING_RATE = 0.01  # 0.001
BATCH_SIZE = 128
DATASET_SIZE = 5_000

MODEL_DIR = "checkpoint"
