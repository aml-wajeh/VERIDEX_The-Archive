"""04 - Vector representation: encode text into dense embeddings.

Production logic lives in ``src.embeddings``. Needs the embedding model
(downloaded automatically on first run).

Run from the project root::

    python 04_vector_representation.py
"""

from src.config import get_settings
from src.embeddings import EmbeddingGenerator

_TEXTS = [
    {
        "chunk_id": "c1",
        "document_id": "d1",
        "text": "Paris is the capital of France.",
        "metadata": {},
    }
]


def main() -> None:
    """Encode the sample chunk and print the vector dimension."""
    try:
        generator = EmbeddingGenerator(get_settings().embedding)
        records = generator.encode_chunks(_TEXTS, show_progress=False)
        print("vectors:", len(records), "dim:", records[0].dimension)
    except Exception as exc:  # model / network unavailable offline
        print("Embedding step needs the model/network:", exc)


if __name__ == "__main__":
    main()
