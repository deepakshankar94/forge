"""Visualization tools for Forge."""

from forge.visualize.unified_viewer import UnifiedBackend, UnifiedViewer, unified_visualize
from forge.visualize.web_viewer import WebBackend, WebViewer, web_visualize

__all__ = [
    "UnifiedBackend",
    "UnifiedViewer",
    "unified_visualize",
    "WebBackend",
    "WebViewer",
    "web_visualize",
]

# Optional fast viewer (requires opencv-python)
try:
    from forge.visualize.cv_viewer import CVBackend, CVViewer, cv_visualize
    __all__.extend(["CVBackend", "CVViewer", "cv_visualize"])
except ImportError:
    pass
