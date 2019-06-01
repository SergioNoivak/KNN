
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import const as const
import math
import sys

INT_MAX = sys.maxsize  

INT_MIN = -sys.maxsize-1



def csv2df():
    # ler do csv
    url = './iris.csv'
    df = pd.read_csv(url)
    return df

def randomize_indexes(df):
    #randomizar posicoes
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split(df):
    size_split = round(len(df['sepal.length'])*(1-const.test_size))
    last_index = len(df['sepal.length'])-1
    print('size_split',size_split)
    print('last_index',last_index)
    train = df[last_index-size_split:last_index+1]
    train = train.reset_index(drop=True)
    print("train:",train)
    test = df[0:last_index-size_split]
    return train, test

def knn(train,test):
    results = []
    for index_test in test.T:
        distance_queue = []        
        for index_population in train.T:
           distance = 0
           for feature in train:
                if(feature !="variety" and index_population!=index_test):
                       distance+=math.pow(train[feature][index_test]-train[feature][index_population],2)
                if( index_population!=index_test):
                    distance = math.sqrt(distance)
           if(index_population!=index_test):
                distance_queue.append({"element":index_population,"distance":distance})
           
        distance_queue.sort(key=lambda x:x["distance"])
        vector_k = distance_queue[0:const.k]
        range_map = {}
        for neighbor in vector_k:
            if(range_map.get(train['variety'][neighbor['element']])==None):
                range_map[train['variety'][neighbor['element']]] = 1
            else:
                range_map[train['variety'][neighbor['element']]]+=1          
        min_distance = INT_MAX
        result = ""
        for i in range_map:
            if(range_map[i]<min_distance):
                min_distance = range_map[i]
                result = i
        results.append(result)
    
    numero_acertos = 0
    for index_test in test.T:
        print(train['variety'][index_test],"==",results[index_test] )
        if(train['variety'][index_test]==results[index_test]):
            numero_acertos+=1
            
    print(numero_acertos/len(results))