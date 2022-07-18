import cv2
import numpy as np


def compress_image(
  img: np.ndarray,
  n_channels: int,
  quality: int
) -> np.ndarray:
  if n_channels == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)        

  _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  img = cv2.imdecode(encimg, 0) if n_channels == 1 else cv2.imdecode(encimg, 3)

  if n_channels == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img