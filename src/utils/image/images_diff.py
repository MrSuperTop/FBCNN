import cv2
import numpy as np

from skimage.metrics import structural_similarity


def images_diff(
  first: np.ndarray,
  second: np.ndarray
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
  if first.ndim == 3 or second.ndim == 3:
    first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

  # Compute SSIM between the two images
  score: float
  diff: np.ndarray
  score, diff = structural_similarity(first, second, full=True)

  # The diff image contains the actual image differences between the two images
  # and is represented as a floating point data type in the range [0,1] 
  # so we must convert the array to 8-bit unsigned integers in the range
  # [0,255] before we can use it with OpenCV
  diff = (diff * 255).astype("uint8")

  # Threshold the difference image, followed by finding contours to
  # obtain the regions of the two input images that differ
  thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  mask = np.zeros(first.shape, dtype='uint8')
  filled_after = second.copy()

  for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
      cv2.drawContours(mask, [c], 0, (255,255,255), -1)
      cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

  return score, (mask, diff, filled_after)
