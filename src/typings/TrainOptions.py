from typing import Any, Literal, TypedDict


class Dataset(TypedDict):
  name: str
  dataroot_H: str
  dataroot_L: str
  dataset_type: Literal['jpeg', 'jpeggray', 'jpeggraydouble']
  phase: Literal['test', 'train']
  scale: int


class TrainDataset(Dataset):
  H_size: int
  dataloader_batch_size: int
  dataloader_num_workers: int
  dataloader_shuffle: bool


class Datasets(TypedDict):
  test: Dataset
  train: TrainDataset


class NetG(TypedDict):
  act_mode: str
  downsample_mode: str
  in_nc: int
  init_bn_type: str
  init_gain: int
  init_type: str
  nb: int
  nc: list[int]
  net_type: str
  out_nc: int
  upsample_mode: str


class Paths(TypedDict):
  images: str
  log: str
  models: str
  options: str
  pretrained_netG: str | None
  root: str
  task: str


class Train(TypedDict):
  G_lossfn_type: str
  G_lossfn_weight: int
  G_optimizer_clipgrad: Any
  G_optimizer_lr: int
  G_optimizer_type: str
  G_regularizer_clipstep: Any
  G_regularizer_orthstep: Any
  G_scheduler_gamma: int
  G_scheduler_milestones: list[int]
  G_scheduler_type: str
  QF_lossfn_type: str
  QF_lossfn_weight: int
  manual_seed: int
  checkpoint_print: int
  checkpoint_save: int
  checkpoint_test: int


class TrainOptions(TypedDict):
  gpu_ids: list[int]
  merge_bn: bool
  merge_bn_startpoint: int
  model: Literal['fbcnn']
  n_channels: Literal[1, 3]
  is_train: bool
  opt_path: str
  scale: int
  task: str
  path: Paths
  datasets: Datasets
  netG: NetG
  train: Train
