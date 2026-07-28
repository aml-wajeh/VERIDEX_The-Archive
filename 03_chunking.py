"""03 - Chunking: split processed text into overlapping chunks.

Production logic lives in ``src.chunker``.

Run from the project root::

    python 03_chunking.py
"""

from src.chunker import Chunker
from src.config import get_settings
from src.text_processor import TextProcessor

_SAMPLES = [
    {
        "document_id": "d1",
        "text": "First sentence. Second sentence. Third sentence. Fourth.",
        "metadata": {
            "title": "T",
            "question": "Q",
            "answer": "A",
            "dataset_split": "train",
            "source_dataset": "squad_v2",
        },
    }
]


def main() -> None:
    """Process then chunk the sample document and print the chunks."""
    settings = get_settings().chunking
    processed = TextProcessor(settings).process_documents(_SAMPLES)
    chunks = Chunker(settings).chunk_documents(processed)
    print("chunks:", len(chunks))
    for chunk in chunks:
        print(chunk.chunk_id, chunk.start_index, chunk.end_index)
        print(repr(chunk.text))


if __name__ == "__main__":
    main()
