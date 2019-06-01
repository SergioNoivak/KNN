import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import const as const
import controller as controller

# resultado VIRGINICA

# ler do csv
df =  controller.csv2df()
df = controller.randomize_indexes(df)

# separar conjunto de testes do conjunto de treino
train, test = controller.split(df)

controller.knn(train,test)







