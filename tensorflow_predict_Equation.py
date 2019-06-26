# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:17:09 2019

@author: dgupta
"""
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([5.0]))

