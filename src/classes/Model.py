from pathlib import Path
from time import time

import numpy as np
import torch

from ..models.network_fbcnn import FBCNN
from ..utils.array_to_tensor import array_to_tensor
from ..utils.get_logger import get_logger
from ..utils.tensor_to_array import tensor_to_array


class Model(FBCNN):
  def __init__(
    self,
    path: Path | str,
    color=True
  ) -> None:
    self.path = Path(path)
    self.logger = get_logger(
      __name__
    )

    channels = 3 if color else 1

    super().__init__(
      in_nc=channels,
      out_nc=channels,
      act_mode='R'
    )

    self._setup()

  def _setup(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not self.path.exists():
      self.logger.error(f'Model doesn\'t exis at path {self.path}, aborting...')

      exit(1)

    self.logger.info(f'Loading model from {self.path}')
    self.load_state_dict(
      torch.load(self.path),
      strict=True
    )

    self.eval()

    for _, v in self.named_parameters():
      v.requires_grad = False

    self = self.to(self.device)

  def predict(
    self,
    image: np.ndarray
  ) -> np.ndarray:
    img_L = array_to_tensor(image)

    # * Process image
    start = time()

    img_L = img_L.to(self.device)
    img_E, QF = self(img_L)
    QF = 1 - QF

    end = time()

    self.logger.info(f"Processing took {end - start}s")
    self.logger.info('Predicted quality factor: {:d}'.format(round(float(QF * 100))))

    img_E = tensor_to_array(img_E)

    return img_E
