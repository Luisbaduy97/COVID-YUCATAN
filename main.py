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
#import flask
#import plotly.graph_objects as go

#server = flask.Flask(__name__) # define flask app.server

#app = dash.Dash(__name__, server=server) # call flask server

covid = pd.read_csv('http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip', encoding='ANSI')
coords = pd.read_csv('coordenadas.csv')
yuc_coords = coords[coords['Num_Ent'] == 31]

d = {n:m for n, m in zip(yuc_coords.Num_Mun, yuc_coords.Municipio)}
numb = [1,2,3]
diag = ['Positivo', 'Negativo', 'Por confirmar']
d2 = {n2:m2 for n2, m2 in zip(numb, diag)}


yucatan = covid[covid['ENTIDAD_RES'] == 31]

#yuc2 = yucatan.groupby(by=['MUNICIPIO_RES'])['RESULTADO'].value_counts().rename(columns={'RESULTADO':'SUMA'}).reset_index().rename(columns={0:'SUMA'})
#yuc2['MUNICIPIO'] = [d.get(m) for m in yuc2['MUNICIPIO_RES'].values.tolist()]
#yuc2['RESULTADO2'] = [d2.get(m) for m in yuc2['RESULTADO'].values.tolist()]
#yuc2

yuc2 = yucatan.groupby(by=['MUNICIPIO_RES'])['RESULTADO'].value_counts()
#yuc2.rename(columns={0: "SUMA"})#.reset_index()
yuc2 = pd.Series(yuc2.values, index = yuc2.index).reset_index().rename(columns={0: 'SUMA'})
yuc2['MUNICIPIO'] = [d.get(m) for m in yuc2['MUNICIPIO_RES'].values.tolist()]
yuc2['RESULTADO2'] = [d2.get(m) for m in yuc2['RESULTADO'].values.tolist()]


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
fig.update_layout(margin={"r":100,"t":0,"l":100,"b":0})

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
fig2.update_layout(title="Número de casos por enfermedad y sexo",title_x=0.5, margin={"r":350,"t":100,"l":400,"b":50})
#fig2.show()



####################################
fi = yucatan[(yucatan['RESULTADO'] == 1) & (yucatan['FECHA_INGRESO'] != '9999-99-99')]
fi['SEX'] = [sexo_dict.get(v) for v in fi['SEXO'].values.tolist()]
fi2 = fi.groupby(['FECHA_INGRESO'])['RESULTADO'].value_counts()
fii = pd.Series(fi2.values, index = fi2.index).reset_index().rename(columns={0: 'SUMA'})
fii['ACUMULADO'] = fii['SUMA'].cumsum() #acumulado de casos positivos


#acumalo de casos negativos
ni = yucatan[(yucatan['RESULTADO'] == 2) & (yucatan['FECHA_INGRESO'] != '9999-99-99')]
ni['SEX'] = [sexo_dict.get(v) for v in ni['SEXO'].values.tolist()]
ni2 = ni.groupby(['FECHA_INGRESO'])['RESULTADO'].value_counts()
nii = pd.Series(ni2.values, index = ni2.index).reset_index().rename(columns={0: 'SUMA'})
nii['ACUMULADO'] = nii['SUMA'].cumsum() #acumulado de casos negativos

#acumulado de casos por confirmar
#pi = yucatan[(yucatan['RESULTADO'] == 3) & (yucatan['FECHA_INGRESO'] != '9999-99-99')]
#pi['SEX'] = [sexo_dict.get(v) for v in pi['SEXO'].values.tolist()]
#pi2 = pi.groupby(['FECHA_INGRESO'])['RESULTADO'].value_counts()
#pii = pd.Series(pi2.values, index = pi2.index).reset_index().rename(columns={0: 'SUMA'})
#pii['ACUMULADO'] = pii['SUMA'].cumsum() #acumulado de casos por confirmar


figx = go.Figure()
figx.add_trace(go.Scatter(
    x=fii['FECHA_INGRESO'].values.tolist(),
    y=fii['ACUMULADO'].values.tolist(), mode='lines+markers', name = 'Casos acumulados positivos'))
figx.add_trace(go.Scatter(
    x=nii['FECHA_INGRESO'].values.tolist(),
    y=nii['ACUMULADO'].values.tolist(), mode='lines+markers', name = 'Casos acumulados negativos'))
#figx.add_trace(go.Scatter(
#    x=pii['FECHA_INGRESO'].values.tolist(),
#    y=pii['ACUMULADO'].values.tolist(), mode='lines+markers', name = 'Casos por confirmar'))
figx.update_layout(title="Casos acumulados en el estado",xaxis_title="Fecha de ingreso",
                   yaxis_title="Acumulado", title_x=0.5, margin={"r":350,"t":100,"l":400,"b":50})
figx.update_xaxes(rangeslider_visible=True)
#figx.show()

#app = dash.Dash()
app = dash.Dash(__name__)

        

app.layout = html.Div([html.Div(children = [html.H1('Casos acumulados en Yucatán'), dcc.Graph(id='acumulado', figure = figx)]),
    html.Div(children = [html.H1('Mapa de datos en Yucatán'), dcc.Graph(id='mapa', figure = fig)]),
    html.Div(children = [html.H1('Casos confirmados por enfermedad y género'), dcc.Graph(id='disease', figure = fig2)])], style = {'background-color': '#2a1a5e', 
                                                                                                                                   'text-align': 'center', 'color': 'white', 'margin-left': 'auto', 'margin-right':'auto'})



# local
if __name__ == '__main__':
     app.run_server(debug=True)

# gunicorn
#app = app.server










