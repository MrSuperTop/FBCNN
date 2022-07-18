from datetime import datetime


def get_timestamp():
  ts = datetime.now()
  time_stamp = '{y}-{m:02d}-{d:02d}_{h:02d}-{mm:02d}-{s:02d}'.format(
    y=ts.year, m=ts.month, d=ts.day, h=ts.hour, mm=ts.minute, s=ts.second
  )

  return time_stamp
