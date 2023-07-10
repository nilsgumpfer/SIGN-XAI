"""
SIGN-XAI
=====
SIGNed explanations: Unveiling relevant features by reducing bias
=====
Author: Nils Gumpfer
"""

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Error importing tensorflow - it is required to run the SIGN-XAI package.")