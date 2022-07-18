from pathlib import Path

from .classes.Model import Model
from .utils.image.compress_image import compress_image
from .utils.image.images_diff import images_diff
from .utils.image.load_image import load_image
from .utils.image.show_image import show_images


def main():
  model_path = Path("./_models/fbcnn_gray.pth")
  input_dir = Path("./images/input")

  image = load_image(next(input_dir.iterdir()))
  before = compress_image(image, 1, 50)

  model = Model(model_path)
  after = model.predict(image)

  score, images = images_diff(before, after)

  print(f"Similarity: {score}")
  show_images(images)


if __name__ == "__main__":
  main()

# TODO: Make a lookalike https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing#scrollTo=lHNHoP8PZJQ7
