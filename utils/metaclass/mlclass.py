#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This is an abstract class template for the ml_from_scratch methods"""
from abc import ABCMeta, abstractmethod


class MLClass(object):
    """The abstract class and interfaces for the ml_from_scratch scripts"""
    __metaClass__ = ABCMeta

    @abstractmethod
    def fit(self, input_data, label):
        """the interface for the fit method"""
        return False
    @abstractmethod
    def predict(self, input_data):
        """the interface for the transform method"""
        return False
