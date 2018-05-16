#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:48:25 2018

@author: Harshvardhan
"""


def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v
