import argparse
import glob
from pathlib import Path

from .classes.Model import Model
from .constants import DIFF_IMAGE_NAMES, IMG_EXTENSIONS
from .utils.ask_confirmation import ask_confirmation
from .utils.image.images_diff import images_diff
from .utils.image.load_image import load_image
from .utils.image.save_image import save_image


def main():
  parser = argparse.ArgumentParser(
    description='Run JPEG artefact removal on a folder of an image files'
  )

  parser.add_argument(
    'source_path',
    type=Path,
    help='path to the source folder or an image file'
  )

  parser.add_argument(
    'out_path',
    type=Path,
    help='directory to output processed files',
    default='./images/output'
  )

  parser.add_argument(
    '--model_path',
    type=Path,
    help='path to the trained model weights (.pth)',
    default='./_models/fbcnn_color.pth'
  )

  parser.add_argument(
    '--generate_diff',
    action='store_true',
    help='whether the script should create diff files (this can be used to compare the result with original image)',
  )

  parser.add_argument(
    '--suffix',
    type=str,
    default=None,
    help='suffix for out folders or images'
  )

  args = parser.parse_args()

  if not args.source_path.exists():
    naming = 'folder' if len(args.source_path.suffix) == 0 else 'file'
    print(f'Source {naming} doesn\'t exist')

    exit(1)

  files_grabbed = [args.source_path]
  isDir = args.source_path.is_dir()

  if isDir:
    files_grabbed = []

    for type in IMG_EXTENSIONS:
      glob_string = f'{args.source_path}/**/*{type}'
      files_grabbed.extend(
        glob.glob(glob_string, recursive=True)
      )

  if len(files_grabbed) == 0:
    print(f'No files to process found in {args.source_path}')

    exit(1)

  if args.out_path.exists():
    answer = ask_confirmation('Output folder already exists... Proceed? (Some data can be overwriten)')

    if not answer:
      exit(1)

  args.out_path.mkdir(parents=True, exist_ok=True)

  model = Model(
    args.model_path,
    color=True
  )

  for i, path in enumerate(map(Path, files_grabbed)):
    print(f'({i}) processing {path.name}')

    input = load_image(path)
    result = model.predict(input)

    base_dir = args.out_path
    output_name = path.stem if args.suffix is None else f'{path.stem}_{args.suffix}'

    if args.generate_diff:
      base_dir = base_dir.joinpath(output_name)

      diff_folder = base_dir.joinpath('diff')
      diff_folder.mkdir(parents=True, exist_ok=True)

      score, diff_images = images_diff(input, result)
      print(f"Images before and after artefact removal are {round(score * 100, 2)}% simillar")

      for name, image in zip(DIFF_IMAGE_NAMES, diff_images):
        save_image(
          image,
          diff_folder.joinpath(f'{name}.png')
        )

    save_image(
      result,
      f'{base_dir.joinpath(output_name)}.png'
    )


if __name__ == "__main__":
  main()
