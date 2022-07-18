from pathlib import Path

import cv2
import numpy as np


def load_image(
  path: str | Path,
) -> np.ndarray:
  if (isinstance(path, Path)):
    path = str(path)

  img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

  return img
