from itertools import zip_longest
from typing import Iterable
import cv2

from src.typings import image
from ..screen_size import screen_size

ESCAPE_CODE = 27
_, screen_h = screen_size()

def show_images(
  images: Iterable[image],
  names: list[str]=[]
) -> None:
  windows: list[str] = []
  startX = -50

  for i, (name, image) in enumerate(zip_longest(names, images)):
    if name is None:
      name = f'image_{i}'

    windows.append(name)

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(name, startX, 0)
    cv2.imshow(name, image)

    h, w = image.shape
    ratio = screen_h / h
    window_w = int(w * ratio)

    startX += window_w

  try:
    while True:
      key = cv2.waitKey(50)

      if key == ESCAPE_CODE:
        break

      if any([cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 for window in windows]):        
        break     
  except KeyboardInterrupt as _:
    exit(0)
  finally:
    cv2.destroyAllWindows()
