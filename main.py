# fastapi-example/app.py
# -*- coding: utf-8 -*-


from typing import Optional
from fastapi import FastAPI
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import KNeighborsTransformer
from math import radians
from scipy.sparse import coo_matrix, hstack
from scipy.sparse import save_npz, load_npz
import pandas as pd
import numpy as np
from pickle import dump,load

dataset = pd.read_csv('comparable_estates.txt', sep='|')
#dataset = dataset[['latitude', 'longitude', 'area', 'bathrooms', 'garages', 'rooms', 'type']]
#dataset['latitude'] = dataset['latitude'].apply(radians)
#dataset['longitude'] = dataset['longitude'].apply(radians)
#radianes = np.array(dataset[['latitude', 'longitude']])
#transformer = KNeighborsTransformer(n_neighbors=20, mode='connectivity', metric='haversine')
#transf_vecinos2 = transformer.fit(radianes)
#matriz_vecinos_cercanos = transf_vecinos2.transform(radianes)
#dump(transf_vecinos2, open('transf_vecinos.pkl', 'wb'))

transf_vecinos = load(open('transf_vecinos.pkl', 'rb'))
st_fit = load(open('st_fit.pkl', 'rb'))
modelo_ajust_ = load(open('modelo_ajust.pkl', 'rb'))
model_ = load(open('neighs.pkl', 'rb'))

app = FastAPI()


@app.get("/400")
def vecinos_cercanos(latitud: float, longitud: float,
                     area: float, bathrooms: float,
                     garages:float, rooms:float, tipo:float):
    latitude = radians(latitud)
    longitude = radians(longitud)
    unic_dist = np.array([[latitude, longitude]])
    vecinos_distancia = modelo_ajust_.transform(unic_dist) * 2
    features = pd.DataFrame([{'area': area, 'bathrooms': bathrooms,
                              'garages': garages, 'rooms': rooms,
                              'type': tipo}])

    matriz_variables = hstack([vecinos_distancia, coo_matrix(features)]).tocsr()
    matriz_variables = st_fit.transform(matriz_variables)
    vecinos = model_.kneighbors(matriz_variables)
    vecinos = pd.DataFrame(np.concatenate(vecinos).T)
    M = vecinos[1]
    resultado = dataset.loc[M, ['latitude', 'longitude', 'area', 'bathrooms', 'garages', 'rooms', 'type']]
    js = resultado.to_json(orient='columns')
    return js

