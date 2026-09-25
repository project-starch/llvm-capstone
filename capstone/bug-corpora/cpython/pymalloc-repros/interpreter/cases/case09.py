# gh-139210: xml.etree.ElementTree.iterparse() frees the events sequence before
# formatting the "unknown event" error, so event_name (a char* into a freed
# item) is read by PyErr_Format. From CPython's regression test
# (test_xml_etree, unknown-events path). The events must be a generator, not a
# tuple/list: PySequence_Fast then builds a throwaway list that is the sole
# owner of the event-name string, so the internal Py_DECREF frees it. The fix
# reorders the two lines; 3.13.7 (our pin) reads the freed char* (UAF).
import io
import xml.etree.ElementTree as ET

def events():
    yield "start"
    yield "".join(["bo", "gus"]) + str(id(object()))   # sole ref lives in the list

try:
    ET.iterparse(io.BytesIO(b"<r/>"), events())
except ValueError:
    pass                             # "unknown event '...'" is formatted from freed memory
print("trigger-09: reached")
