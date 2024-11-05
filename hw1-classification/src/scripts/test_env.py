import sys

def test_env():
    """ Check if the environment is correctly set up. Try to import the necessary modules. """
    try:
        import torch
        import sklearn
        import nbformat
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("INFO: Environment is correctly set up.")

if __name__ == "__main__":
    test_env()