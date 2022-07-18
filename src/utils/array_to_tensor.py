import numpy as np
import torch


def array_to_tensor(
  array: np.ndarray
) -> torch.Tensor:
  """
  array_to_tensor coverts np.ndarray into a torch.Tensor with values suitable for the network

  Args:
      array (np.ndarray): input np array to be converted into a torch.Tensor

  Returns:
      torch.Tensor: converted array in a needed form
  """

  if array.ndim == 2:
    array = np.expand_dims(array, axis=2)

  return torch.from_numpy(
    np.ascontiguousarray(array)
  ).permute(2, 0, 1).float().div(255.).unsqueeze(0)
