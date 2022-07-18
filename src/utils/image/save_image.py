from pathlib import Path
import cv2


def save_image(img, img_path: str | Path):
  if (isinstance(img_path, Path)):
    img_path = str(img_path)

  cv2.imwrite(img_path, img)
