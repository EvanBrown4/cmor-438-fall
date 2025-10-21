"""
rice2025: A minimal machine learning toolkit.

Usage:
    from rice2025 import ...
"""

try:
    __version__ = version("rice2025")
except PackageNotFoundError:  # when running from source without install
    __version__ = "0.1.0"