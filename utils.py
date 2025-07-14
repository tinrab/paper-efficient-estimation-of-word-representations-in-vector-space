import torch


def word_analogy(word_embeddings, vocab, word1, word2, word3, top_n=5):
    """
    Performs word analogy task (e.g., "king - man + woman").
    """
    if not vocab:
        print("Cannot perform analogy: vocabulary is empty.")
        return

    inv_vocab = {i: word for word, i in vocab.items()}
    for word in [word1, word2, word3]:
        if word not in vocab:
            print(f"Word '{word}' not in vocabulary.")
            return

    # Get the vectors for the input words.
    vec1 = word_embeddings[vocab[word1]]
    vec2 = word_embeddings[vocab[word2]]
    vec3 = word_embeddings[vocab[word3]]

    # Perform the vector arithmetic.
    result_vec = vec1 - vec2 + vec3

    # Find the most similar words to the result vector.
    distances = torch.nn.functional.cosine_similarity(
        result_vec.unsqueeze(0), word_embeddings
    )
    top_results = torch.topk(distances, top_n + 3)

    count = 0
    words = []
    for i in top_results.indices:
        word = inv_vocab[i.item()]
        if word not in [word1, word2, word3]:
            words.append(word)
            count += 1
            if count == top_n:
                break

    return words
