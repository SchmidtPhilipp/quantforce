import time

from tqdm.auto import tqdm


def wait(seconds, progress=True):
    """
    Waits for a specified amount of time while showing a progress bar.

    Parameters:
        seconds (int): The number of seconds to wait.
    """
    if progress:
        for _ in tqdm(range(seconds), desc="Waiting", unit="s"):
            time.sleep(1)
    else:
        time.sleep(seconds)
