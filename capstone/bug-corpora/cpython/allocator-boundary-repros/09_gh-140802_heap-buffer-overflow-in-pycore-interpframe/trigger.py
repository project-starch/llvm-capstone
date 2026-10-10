import sys
import asyncio

class JumpTracer:
    def __init__(self, func, jump_to):
        self.code = func.__code__
        self.jump_to = jump_to
        self.first_line = None

    def trace(self, frame, event, arg):
        if self.first_line is None and event == 'line' and frame.f_code is self.code:
            self.first_line = frame.f_lineno - 1
            try:
                frame.f_lineno = self.first_line - self.jump_to
            except TypeError:
                frame.f_lineno = self.jump_to
        return self.trace

async def target():
    # Keep a couple of lines so the tracer has places to land.
    x = 0
    x += 1
    return x

if __name__ == "__main__":
    tracer = JumpTracer(target, jump_to=1)
    sys.settrace(tracer.trace)
    asyncio.run(target())