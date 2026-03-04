from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


MODEL = "all-MiniLM-L6-v2"

def similarity_labels(concepts, X, model_name = MODEL):
    model = SentenceTransformer(model_name)

    # Flatten
    concept_keys = []
    all_phrases = []

    for k, vlist in concepts.items():
        concept_keys.extend([k] * len(vlist))
        all_phrases.extend(vlist)

    # Batch encoding
    phrase_embeddings = model.encode(
        all_phrases,
        normalize_embeddings=True,
        batch_size=64,
        convert_to_numpy=True
    )

    # Mean emb per concept
    concept_embeddings = dict()
    for key, emb in zip(concept_keys, phrase_embeddings):
        concept_embeddings.setdefault(key, []).append(emb)

    concept_embeddings = {
        k: np.mean(v, axis=0) for k, v in concept_embeddings.items()
    }

    # Flatten
    concept_matrix = np.vstack(list(concept_embeddings.values()))
    concept_names = list(concept_embeddings.keys())

    # Cosine similarity
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    scores_matrix = X_norm @ concept_matrix.T


    return {
        concept_names[i]: scores_matrix[:, i]
        for i in range(len(concept_names))
    }
