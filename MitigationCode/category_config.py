"""Shared configuration for isolated FairMT category experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "FairMT-bench" / "FairMT_10K" / "Stereotype" / "Scattered_Questions"
RESULTS_DIR = PROJECT_ROOT / "results" / "mistral_categories"

CATEGORY_VOCABULARY = {
    "age": ("old", "elderly", "senior", "aged"),
    "appearance": ("attractive", "unattractive", "beautiful", "ugly"),
    "disability": ("disabled", "disability", "wheelchair", "blind"),
    "gender": ("woman", "man", "female", "male"),
    "race": ("race", "racial", "black", "white"),
    "religion": ("muslim", "christian", "jewish", "religious"),
}

GOLDEN_LABELS = ("person", "individual", "human", "adult")

INPUT_FILENAMES = {
    "age": "prompt_age.json",
    "appearance": "prompt_appearance.json",
    "disability": "prompt_disable.json",
    "gender": "prompt_gender.json",
    "race": "prompt_race.json",
    "religion": "prompt_religion.json",
}


def get_category() -> str:
    """Return the requested category and reject unknown values early."""
    category = os.getenv("MITIGATION_CATEGORY", "age").strip().lower()
    if category not in CATEGORY_VOCABULARY:
        choices = ", ".join(CATEGORY_VOCABULARY)
        raise ValueError(f"Unknown MITIGATION_CATEGORY '{category}'. Choose one of: {choices}")
    return category


def get_category_paths(category: str | None = None) -> dict[str, Path]:
    """Return isolated input and stage output paths for one category."""
    category = category or get_category()
    if category not in CATEGORY_VOCABULARY:
        raise ValueError(f"Unknown category '{category}'")

    category_root = RESULTS_DIR / category
    return {
        "input_file": DATASET_DIR / INPUT_FILENAMES[category],
        "root": category_root,
        "stage1": category_root / "stage1",
        "stage2": category_root / "stage2",
        "stage3": category_root / "stage3",
        "stage4": category_root / "stage4",
    }