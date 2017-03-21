# Some utility functions
from datetime import *


def log(*args):
    now = datetime.now()
    display_now = str(now).split(" ")[1][:-3]
    print(display_now, *args)


def last_timewindow(timestamp, timewindow=20*60) -> int:
    return timestamp - timewindow
