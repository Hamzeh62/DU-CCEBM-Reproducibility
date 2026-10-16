"""Make ``import du_ccebm_ipix`` in tests/ resolve to the package's own
self-contained copy in ``code/`` (not the parent repository)."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.join(_HERE, "..", "code")
if os.path.isdir(_CODE) and _CODE not in sys.path:
    sys.path.insert(0, _CODE)
