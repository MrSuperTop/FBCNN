from pathlib import Path

from src.classes.TestRunner import TestRunner


def main():
  model_path = Path('./_models/fbcnn_color.pth')
  input_dir = Path('./images/input')
  output_path = Path('./images/test_output')

  runner = TestRunner(
    input_dir,
    output_path,
    model_path
  )

  runner.run_tests()


if __name__ == '__main__':
  main()
