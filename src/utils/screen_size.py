from screeninfo import get_monitors

def screen_size() -> tuple[int, int]:
  for monitor in get_monitors():
    if not monitor.is_primary:
      continue

    return monitor.width, monitor.height

  return (0, 0)
