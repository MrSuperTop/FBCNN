import os
from pathlib import Path

from src.classes.Model import Model
from src.constants import IMG_EXTENSIONS

from ..utils.get_logger import get_logger
from ..utils.get_timestamp import get_timestamp
from ..utils.image.compress_image import compress_image
from ..utils.image.load_image import load_image
from ..utils.image.save_image import save_image

N_CHANNELS = 3


class TestRunner():
  def __init__(
    self,
    input_dir: Path,
    output_dir: Path,
    model_path: Path
  ) -> None:
    self.quality_factor_list = [i for i in range(10, 100, 10)]

    self.model_path = model_path
    self.input_dir = input_dir
    self.output_dir = output_dir

    self.logger = get_logger(
      __name__
    )

    self.model = Model(model_path)

  def run_tests(
    self
  ):
    parent_out_dir = self.output_dir.joinpath(get_timestamp())
    for quality_factor in self.quality_factor_list:
      out_dir = parent_out_dir.joinpath(str(quality_factor))

      if not out_dir.exists():
        os.makedirs(out_dir)

      logger_name = f'log_qf_{quality_factor}'
      run_logger = get_logger(
        logger_name,
        job_dir=str(out_dir.joinpath(f'{logger_name}.log'))
      )

      run_logger.info('\n--------------- quality factor: {:d} ---------------'.format(quality_factor))

      img_files = [f for f in self.input_dir.iterdir() if f.suffix in IMG_EXTENSIONS]
      for i, img in enumerate(img_files):
        # * Load and compress
        img_L = load_image(
          img
        )

        run_logger.info('[{}] Compressing image ({})'.format(i, img))
        img_L = compress_image(
          img_L,
          N_CHANNELS,
          quality_factor
        )

        img_E = self.model.predict(img_L)

        save_image(
          img_E,
          out_dir.joinpath(f'{img.stem}.png')
        )
