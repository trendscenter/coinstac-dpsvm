#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:50:08 2020

@author: hgazula
"""
import os
import numpy as np


def saveBin(path, arr):
    with open(path, 'wb+') as fh:
        header = '%s' % str(arr.dtype)
        for index in arr.shape:
            header += ' %d' % index
        header += '\n'
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, 'rb') as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)