import os
import re
from collections import deque, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from config import *
from utils import word_analogy

MODEL_FILE = os.path.join(MODEL_DIR, "cbow_model.pth")
VOCAB_FILE = os.path.join(MODEL_DIR, "cbow_vocab.pth")


def load_data():
    """
    Loads and preprocesses the data.
    """

    # Purposefully small vocab dataset. It's easier to experiment with.
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Combine and tokenize the text.
    text = " ".join([item for item in dataset[:DATASET_SIZE]["text"]])
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    tokens = re.findall(r"\b\w+\b", text)  # Split into words

    # Build vocabulary.
    word_counts = Counter(tokens)
    frequent_word_list = sorted(
        [word for word, count in word_counts.items() if count > 5]
    )
    vocab = {word: i for i, word in enumerate(frequent_word_list)}
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Create context-target pairs for training.
    data = []
    window = deque(maxlen=2 * CONTEXT_SIZE + 1)
    for word in tokens:
        word_idx = vocab.get(word)
        if word_idx is None:
            # Skip words not in our filtered vocabulary.
            continue

        window.append(word_idx)
        if len(window) == 2 * CONTEXT_SIZE + 1:
            target = window[CONTEXT_SIZE]
            context = [
                window[i] for i in range(2 * CONTEXT_SIZE + 1) if i != CONTEXT_SIZE
            ]
            data.append((context, target))

    print(f"Created {len(data)} context-target pairs.")

    return data, vocab, vocab_size


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        # The embedding layer maps word indices to dense vectors.
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The linear layer predicts the target word from the averaged context embeddings.
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: (batch_size, 2 * CONTEXT_SIZE)
        embeds = self.embeddings(inputs)
        # shape: (batch_size, 2 * CONTEXT_SIZE, embedding_dim)

        # Average the embeddings of the context words.
        avg_embeds = torch.mean(embeds, dim=1)
        # shape: (batch_size, embedding_dim)

        out = self.linear(avg_embeds)
        # shape: (batch_size, vocab_size)

        return out


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data, vocab, vocab_size = load_data()
    if not data or vocab_size == 0:
        print(
            "No training data or vocabulary generated. Check DATASET_SIZE and filtering criteria."
        )
        return None, None

    model = CBOWModel(vocab_size, EMBEDDING_DIM).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Ensure the model directory exists.
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        total_loss = 0

        # Recalculate number of batches in case of leftover data.
        num_batches = len(data) // BATCH_SIZE
        if num_batches == 0:
            print(
                "Not enough data to form a single batch. Consider reducing BATCH_SIZE."
            )
            break

        for i in range(0, len(data) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            context_words, target_word = zip(*batch)

            context_tensors = torch.LongTensor(context_words).to(device)
            target_tensor = torch.LongTensor(target_word).to(device)

            model.zero_grad()
            logits = model(context_tensors)
            loss = loss_function(logits, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_model_path = os.path.join(MODEL_DIR, f"cbow_model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / num_batches:.4f}, Model saved to {epoch_model_path}"
        )

    # Save the final model and the vocabulary
    torch.save(model.state_dict(), MODEL_FILE)
    torch.save(vocab, VOCAB_FILE)
    print(f"Training finished. Final model saved to {MODEL_FILE}")
    print(f"Vocabulary saved to {VOCAB_FILE}")

    return model, vocab


def load_model(model_path, vocab_path):
    try:
        vocab = torch.load(vocab_path)
        vocab_size = len(vocab)

        model = CBOWModel(vocab_size, EMBEDDING_DIM)

        model.load_state_dict(torch.load(model_path))

        model.eval()

        print(f"Model loaded successfully from {model_path}")
        return model, vocab
    except FileNotFoundError:
        print(
            f"Error: Model or vocabulary not found. Searched for '{model_path}' and '{vocab_path}'."
        )
        return None, None


def main():
    train()

    model, vocab = load_model(MODEL_FILE, VOCAB_FILE)
    for a, b, c in [["boy", "he", "she"]]:
        print(
            f"\nAnalogy: {a} - {b} + {c} = {word_analogy(model.embeddings.weight.data, vocab, a, b, c)}"
        )


if __name__ == "__main__":
    main()
