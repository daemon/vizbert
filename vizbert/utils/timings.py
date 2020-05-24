from collections import defaultdict
import time


__all__ = ['Timer']


class Timer:
    def __init__(self):
        self.timings = defaultdict(lambda: 0.0)
        self.curr_context = []
        self.curr_name = None

    def time(self, name):
        self.curr_name = name
        return self

    def enter(self, name):
        self.curr_context.append((time.time(), name))

    def exit(self):
        b = time.time()
        a, name = self.curr_context.pop()
        self.curr_name = self.curr_context[-1][1] if self.curr_context else None
        self.timings[name] += b - a

    def __enter__(self):
        self.enter(self.curr_name)
        return self

    def __exit__(self, *args, **kwargs):
        self.exit()
        return self

    def __str__(self):
        return str(self.timings)

    @staticmethod
    def singleton():
        if not hasattr(Timer, '_singleton'):
            Timer._singleton = Timer()
        return Timer._singleton

    @property
    def proportions(self):
        total = sum(self.timings.values())
        return {k: v / total for k, v in self.timings.items()}
