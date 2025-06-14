import os
import sys

from qf import VERBOSITY


def make_cache_dir(cache_dir, use_cache):
    """
    Create a cache directory if it doesn't exist and return its path.
    If use_cache is False, return None.
    """
    def get_base_dir():
        if getattr(sys, 'frozen', False):  # Running as bundled executable (e.g. PyInstaller)
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))


    if use_cache and cache_dir is not None:
        base_dir = get_base_dir()
        # Move up 3 directories to reach the root of the project
        for _ in range(3):
            base_dir = os.path.dirname(base_dir)

        cache_dir = os.path.join(base_dir, cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print("Cache directory:", cache_dir) if VERBOSITY > 0 else None


    return cache_dir if use_cache else None