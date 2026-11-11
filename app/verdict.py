"""
Turn P(AI) and a threshold into the verdict shown to users (no TensorFlow needed).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    label: str          # "AI-generated" | "Real"
    p_ai: float
    confidence: float   # probability of the predicted label
    threshold: float

    @property
    def uncertain(self):
        return 0.5 <= self.p_ai < self.threshold


def decide(p_ai, threshold):
    is_ai = p_ai >= threshold
    return Verdict("AI-generated" if is_ai else "Real", p_ai,
                   p_ai if is_ai else 1.0 - p_ai, threshold)
