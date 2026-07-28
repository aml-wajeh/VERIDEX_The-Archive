"""02 - Preprocessing: clean & Unicode-normalise text.

Production logic lives in ``src.text_processor``.

Run from the project root::

    python 02_preprocessing.py
"""

from src.config import get_settings
from src.text_processor import TextProcessor

_SAMPLES = [
    {"document_id": "d1", "text": "  Hello   world  ", "metadata": {}},
    {"document_id": "d2", "text": "Line one.\n\n\nLine   two.", "metadata": {}},
]


def main() -> None:
    """Clean and normalise the sample documents, then print them."""
    processor = TextProcessor(get_settings().chunking)
    for doc in processor.process_documents(_SAMPLES):
        print(doc.document_id, "->", repr(doc.text))


if __name__ == "__main__":
    main()
