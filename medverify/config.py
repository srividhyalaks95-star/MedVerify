"""Central configuration. Paths are relative to the Final_Project root so the
same code runs on the local Mac and on Colab."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT / "external" / "CliniFact" / "data" / "processed"
    / "primary_outcome_publication_dataset"
)
RESULTS_DIR = ROOT / "results"

SEED = 42

# Official labels (unchanged from CliniFact)
LABELS = {0: "Inconclusive", 1: "Evidence", 2: "NEI"}
NUM_LABELS = len(LABELS)

SPLITS = ("train", "validation", "test")

# Models (configurable)
BIOBERT_CHECKPOINT = "dmis-lab/biobert-base-cased-v1.2"
MAX_LENGTH = 512

# LLM (only for optional decomposition and correction; used in later phases)
LLM_MODEL = "gpt-4.1-mini"

# Columns that carry or derive from the label. Never fed to the model,
# retrieval or ranking.
LEAKY_COLUMNS = ("label", "pValues", "statisticalMethods", "nonInferiorityTypes")
