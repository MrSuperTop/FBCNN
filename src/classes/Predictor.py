import os
from pathlib import Path

from ..constants import DIFF_IMAGE_NAMES, IMG_EXTENSIONS
from ..utils.get_logger import get_logger
from ..utils.get_timestamp import get_timestamp
from ..utils.image.images_diff import images_diff
from ..utils.image.load_image import load_image
from ..utils.image.save_image import save_image
from .Model import Model


class Predictor():
  def __init__(
    self,
    input_dir: Path | str,
    output_dir: Path | str,
    model_path: Path | str,
    color=True
  ) -> None:
    self.model_path = Path(model_path)
    self.input_dir = Path(input_dir)
    self.output_dir = Path(output_dir)

    self.logger = get_logger(
      __name__
    )

    self.model = Model(
      model_path,
      color
    )

  def predict(
    self,
    save_diff=False
  ):
    out_dir = self.output_dir.joinpath(get_timestamp())

    if not out_dir.exists():
      os.makedirs(out_dir)

    img_files = [f for f in self.input_dir.iterdir() if f.suffix in IMG_EXTENSIONS]
    self.logger.info(f'Found {len(img_files)} image(s) in {self.input_dir}')

    for i, image in enumerate(img_files):
      # * Load
      self.logger.info(f'\n[{i}] Starting to manipulate: {image}')

      self.logger.info('Loading...')
      img_L = load_image(
        image
      )

      self.logger.info(f'Processing started')
      img_E = self.model.predict(img_L)

      result_path = out_dir.joinpath(f'{image.stem}.png')
      if save_diff:
        result_folder = out_dir.joinpath(image.stem)
        diff_folder = result_folder.joinpath('diff')
        result_path = result_folder.joinpath(f'{image.stem}.png')

        diff_folder.mkdir(parents=True)

        score, diff_images = images_diff(img_L, img_E)
        self.logger.info(f"Images before and after artefact removal are {round(score * 100, 2)}% simillar")

        for name, image in zip(DIFF_IMAGE_NAMES, diff_images):
          save_image(
            image,
            diff_folder.joinpath(f'{name}.png')
          )

      save_image(
        img_E,
        result_path
      )
