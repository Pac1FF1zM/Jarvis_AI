"""From-scratch video gesture-recognition components for Jarvis.

This package deliberately contains no downloaded checkpoints, pretrained
backbones, Hugging Face components, or landmark extractors.  The models learn
directly from RGB clips supplied by the user-owned training workspace.
"""

from .labels import IPN_LABELS, NO_GESTURE_LABEL

__all__ = ["IPN_LABELS", "NO_GESTURE_LABEL"]
