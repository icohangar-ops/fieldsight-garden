"""Prompts. FT_PROMPT is the short prompt the LoRA model is trained and served with.
Zero-shot baselines (base Qwen, Inkling) get the same prompt plus the closed class list,
otherwise they cannot know the label vocabulary."""
from fieldsight.labels import CLASSES

FT_PROMPT = "Plant and disease in this leaf photo? Answer with the class only."

ZS_PROMPT = (
    "Identify the plant and its disease (or 'healthy') in this leaf photo.\n"
    "Answer with exactly one class from this list and nothing else:\n"
    + "\n".join(CLASSES)
)
