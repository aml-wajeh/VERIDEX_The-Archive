"""01 - Documents: ingest & validate the SQuAD v2 dataset.

Production logic lives in ``src.data_loader``. This file is a thin
spec-named entry point that imports from ``src`` so the required file
names from the project brief exist without duplicating any logic.

Run from the project root::

    python 01_documents.py
"""

from src.config import get_settings
from src.data_loader import DataLoader, DatasetLoadingError

_FALLBACK = {
    "train": [
        {
            "id": "1",
            "title": "Demo",
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answers": {"text": ["Paris"], "answer_start": [0]},
        }
    ]
}


def main() -> None:
    """Load (or fall back) and print dataset statistics."""
    settings = get_settings()
    loader = DataLoader(settings.dataset, settings.paths)
    try:
        loader.load()
    except DatasetLoadingError as exc:
        print("Hugging Face unavailable, using fallback:", exc)
        loader.load_from_records(_FALLBACK, validate=True)
    print("splits:", loader.splits(), "sizes:", loader.split_sizes())
    print(loader.compute_statistics())


if __name__ == "__main__":
    main()
