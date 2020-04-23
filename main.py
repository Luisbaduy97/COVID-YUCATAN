# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:38:10 2020

@author: DELL
"""


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
#import plotly.graph_objects as go

covid = pd.read_csv('http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip')
coords = pd.read_csv('coordenadas.csv')
yuc_coords = coords[coords['Num_Ent'] == 31]

d = {n:m for n, m in zip(yuc_coords.Num_Mun, yuc_coords.Municipio)}
numb = [1,2,3]
diag = ['Positivo', 'Negativo', 'Por confirmar']
d2 = {n2:m2 for n2, m2 in zip(numb, diag)}


yucatan = covid[covid['ENTIDAD_RES'] == 31]

yuc2 = yucatan.groupby(by=['MUNICIPIO_RES'])['RESULTADO'].value_counts().rename(columns={'RESULTADO':'SUMA'}).reset_index().rename(columns={0:'SUMA'})
yuc2['MUNICIPIO'] = [d.get(m) for m in yuc2['MUNICIPIO_RES'].values.tolist()]
yuc2['RESULTADO2'] = [d2.get(m) for m in yuc2['RESULTADO'].values.tolist()]
#yuc2

muni = np.unique(yuc2['MUNICIPIO'])
pos = []
neg = []
conf = []
for i in muni:
    prov1 = yuc2[(yuc2['MUNICIPIO'] == i) & (yuc2['RESULTADO2'] == 'Positivo')]
    prov2 = yuc2[(yuc2['MUNICIPIO'] == i) & (yuc2['RESULTADO2'] == 'Negativo')]
    prov3 = yuc2[(yuc2['MUNICIPIO'] == i) & (yuc2['RESULTADO2'] == 'Por confirmar')]
    if len(prov1['SUMA'].values.tolist()) == 0:
        pos.append(0)
    else:
        pos.append(prov1['SUMA'].iloc[0])
    if len(prov2['SUMA'].values.tolist()) == 0:
        neg.append(0)
    else:
        neg.append(prov2['SUMA'].iloc[0])
    if len(prov3['SUMA'].values.tolist()) == 0:
        conf.append(0)
    else:
        conf.append(prov3['SUMA'].iloc[0])


lat = {v:t for v, t in zip(yuc_coords.Municipio, yuc_coords.lat)}
lon = {v:t for v, t in zip(yuc_coords.Municipio, yuc_coords.lon)}
lat_muni = [lat.get(v) for v in muni]
lon_muni = [lon.get(v) for v in muni]
#dataf = 

data_f = pd.DataFrame()
data_f['Municipio'] = muni
data_f['Positivos'] = pos
data_f['Negativos'] = neg
data_f['Por confirmar'] = conf
data_f['Latitud'] = lat_muni
data_f['Longitud'] = lon_muni
data_f['Tamaño'] = np.asarray(pos) + 100



import plotly.express as px

fig = px.scatter_mapbox(data_f, lat="Latitud", lon="Longitud", hover_name="Municipio",
                        color_discrete_sequence=["red"], zoom=8, height=500, size='Tamaño', hover_data=["Positivos", "Negativos", "Por confirmar"],
                       center={'lat':lat.get('Mérida'), 'lon':lon.get('Mérida')})
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

sexo_dict = {1:'Mujer', 2:'Hombre', 3:'No especificado'}

cols = ['NEUMONIA', 'DIABETES', 'EPOC', 'INMUSUPR', 'ASMA', 'HIPERTENSION', 'CARDIOVASCULAR', 'OBESIDAD',
        'RENAL_CRONICA', 'OTRA_COM', 'OTRO_CASO', 'SEXO', 'RESULTADO']
#yucatan[cols]
yuc3 = yucatan[cols]
sex = [sexo_dict.get(v) for v in yuc3['SEXO'].values.tolist()]
yuc3['Gender'] = sex


data3 = yuc3[yuc3['RESULTADO'] == 1]
neumonia = data3[data3['NEUMONIA'] == 1]
diabetes = data3[data3['DIABETES'] == 1]
epoc = data3[data3['EPOC'] == 1]
hiper = data3[data3['HIPERTENSION'] == 1]
ob = data3[data3['OBESIDAD'] == 1]
car = data3[data3['CARDIOVASCULAR'] == 1]

fig2 = go.Figure()
fig2.add_trace(go.Histogram(histfunc="count", y=neumonia['NEUMONIA'], x=neumonia['Gender'], name="NEUMONIA"))
fig2.add_trace(go.Histogram(histfunc="count", y=diabetes['DIABETES'], x=diabetes['Gender'], name="DIABETES"))
fig2.add_trace(go.Histogram(histfunc="count", y=epoc['EPOC'], x=epoc['Gender'], name="EPOC"))
fig2.add_trace(go.Histogram(histfunc="count", y=hiper['HIPERTENSION'], x=hiper['Gender'], name="HIPERTENSION"))
fig2.add_trace(go.Histogram(histfunc="count", y=ob['OBESIDAD'], x=ob['Gender'], name="OBESIDAD"))
fig2.add_trace(go.Histogram(histfunc="count", y=car['CARDIOVASCULAR'], x=car['Gender'], name="CARDIOVASCULAR"))
fig2.update_layout(title="Número de casos por enfermedad y sexo",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"), title_x=0.5)
#fig2.show()

app = dash.Dash()

app.layout = html.Div([html.Div(children = [html.H1('Mapa de datos en Yucatán'), dcc.Graph(id='example', figure = fig)]),
    html.Div(children = [html.H1('Casos confirmados por enfermedad y género'), dcc.Graph(id='disease', figure = fig2)])])


if __name__ == '__main__':
    app.run_server(debug=True)











