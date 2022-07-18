import numpy as np
import torch


def tensor_to_array(
  tensor: torch.Tensor
) -> np.ndarray:
  """
  tensor_to_array converting torch.Tensor after processing back to np.ndarray which contains image data

  Args:
      tensor (torch.Tensor): network result tensor

  Returns:
      np.ndarray: proper image data ready for manipulations or saving
  """  

  result: np.ndarray = tensor.cpu().numpy().squeeze()

  if result.ndim == 3:
    result = np.transpose(result, (1, 2, 0))

  result = np.rint((result.clip(0, 1) * 255.)).astype(np.uint8)

  return result
