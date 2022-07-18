YES_ANSWERS = {'yes', 'y', 'ye', ''}
NO_ANSWERS = {'no', 'n'}
PROMPTS = {
  'y': '[Y/n]',
  'n': '[y/N]'
}


def ask_confirmation(
  question: str,
  default: str = "y"
):
  prompt = PROMPTS[default]
  choice = input(f'{question} {prompt} ').lower()

  if choice in YES_ANSWERS:
    return True
  elif choice in NO_ANSWERS:
    return False
