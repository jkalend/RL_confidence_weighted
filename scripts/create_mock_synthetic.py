"""Create a small mock synthetic dataset for testing without GPU."""

import json
from pathlib import Path

from src.config import DATA_DIR


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "synthetic_target.json"

    mock = [
        {
            "input": "Apple Inc. was founded by Steve Jobs in Cupertino.",
            "pseudo_label": '[{"text": "Apple Inc.", "type": "organization-company"}, {"text": "Steve Jobs", "type": "person-founder"}, {"text": "Cupertino", "type": "location-city"}]',
            "confidence_score": 0.875,
            "generations": ["[]"] * 8,
            "tokens": ["Apple", "Inc.", "was", "founded", "by", "Steve", "Jobs", "in", "Cupertino", "."],
        },
        {
            "input": "The conference will be held at MIT in Boston next March.",
            "pseudo_label": '[{"text": "MIT", "type": "organization-university"}, {"text": "Boston", "type": "location-city"}, {"text": "March", "type": "date-month"}]',
            "confidence_score": 0.75,
            "generations": ["[]"] * 8,
            "tokens": ["The", "conference", "will", "be", "held", "at", "MIT", "in", "Boston", "next", "March", "."],
        },
        {
            "input": "Barack Obama served as the 44th President of the United States.",
            "pseudo_label": '[{"text": "Barack Obama", "type": "person-politician"}, {"text": "United States", "type": "location-country"}]',
            "confidence_score": 1.0,
            "generations": ["[]"] * 8,
            "tokens": ["Barack", "Obama", "served", "as", "the", "44th", "President", "of", "the", "United", "States", "."],
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(mock, f, indent=2, ensure_ascii=False)

    print(f"Created mock dataset at {path} with {len(mock)} samples")


if __name__ == "__main__":
    main()
