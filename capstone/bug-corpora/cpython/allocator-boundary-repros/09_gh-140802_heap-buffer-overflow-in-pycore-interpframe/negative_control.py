"""Negative control for case 09: the jump target made valid.

The defect is a frame jump to a line outside the code object, which reads past
the end of the line table. The tracer still installs, still fires on the first
line, and still assigns frame.f_lineno -- to the line it is already on, which is
in range. The asyncio run and the frame traffic are unchanged.
"""
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
                # The defect assigns first_line - jump_to, which is out of range.
                frame.f_lineno = frame.f_lineno
            except (TypeError, ValueError):
                pass
        return self.trace

async def target():
    x = 0
    x += 1
    return x

if __name__ == "__main__":
    tracer = JumpTracer(target, jump_to=1)
    sys.settrace(tracer.trace)
    asyncio.run(target())
    sys.settrace(None)
    print("NEGATIVE-CONTROL no defect performed")
