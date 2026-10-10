#!/usr/bin/env python3
"""Trigger for gh144475 -- the reproducer from upstream issue #144475.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (48-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import gc 
from functools import partial 

g_partial = None 

class EvilObject: 
    def __init__(self, name, is_trigger=False): 
        self.name = name 
        self.is_trigger = is_trigger 
        self.triggered = False 
    
    def __repr__(self): 
        global g_partial 
        if self.is_trigger and not self.triggered and g_partial is not None: 
            self.triggered = True 
            # Replace args via __setstate__, this frees the old tuple 
            new_state = (lambda x: x, ("replaced",), {}, None) 
            g_partial.__setstate__(new_state) 
            gc.collect() 
        return f"EvilObject({self.name})" 
 
evil1 = EvilObject("trigger", is_trigger=True) 
evil2 = EvilObject("victim1") 
evil3 = EvilObject("victim2") 
evil4 = EvilObject("victim3") 
evil5 = EvilObject("victim4") 

p = partial(lambda: None, evil1, evil2, evil3, evil4, evil5) 
g_partial = p 

del evil1, evil2, evil3, evil4, evil5 

repr(p) 

