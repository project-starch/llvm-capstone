# MPY-T25 / upstream 10402, verbatim.
import io
class TestStream(io.StringIO):
    def __init_(self, alloc_size):
        super().__init__(alloc_size)
test = TestStream(100)
print("Now the seg fault...")
print("hello", file=test)
print("T25 survived")
