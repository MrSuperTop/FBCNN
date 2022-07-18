from pathlib import Path
from src.classes.Predictor import Predictor


def main():
  model_path = Path('./_models/fbcnn_color.pth')
  input_dir = Path('./images/input')
  output_path = Path('./images/output')

  runner = Predictor(
    input_dir,
    output_path,
    model_path
  )

  runner.predict(
    save_diff=True
  )


if __name__ == '__main__':
  main()
