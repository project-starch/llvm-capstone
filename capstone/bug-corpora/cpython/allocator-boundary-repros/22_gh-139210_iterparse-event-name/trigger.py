#!/usr/bin/env python3
"""Trigger for gh139210 -- the reproducer from upstream issue #139210.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (4-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import gettext
import io
import xml.etree.ElementTree as ET

A_context_unicode = '\u6309\u94ae'
A_message_unicode = '\u786e\u5b9a'
A_result_unicode = gettext.pgettext(A_context_unicode, A_message_unicode)
fusion = A_result_unicode
B_xml_data = b'<?xml version="1.0"?>\n<root>\n  <child name="a">Text1</child>\n  <child name="b">Text2</child>\n</root>'
B_f = io.BytesIO(B_xml_data)
B_result = list(ET.iterparse(B_f, events=fusion))

