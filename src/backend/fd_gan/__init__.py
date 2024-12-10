# This marks fd_gan as a Python package

# Import key components for external use
from .generators import BlendGenerator, TransferGenerator
from .discriminator import Discriminator
from .losses import classification_loss, forgery_similarity_loss, diversity_loss
from .training import train_fd_gan

# Optional: Define package-level metadata
__version__ = "1.0.0"
__all__ = [
    "BlendGenerator",
    "TransferGenerator",
    "Discriminator",
    "classification_loss",
    "forgery_similarity_loss",
    "diversity_loss",
    "train_fd_gan"
]
