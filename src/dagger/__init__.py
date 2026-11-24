"""dagger: independent Jagger-style morphological analysis in Cython."""
from .tagger import Tagger, Token

__version__ = "0.1.0"
__all__ = ["Tagger", "Token"]

