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
import plotly.express as px
import flask


#server = flask.Flask(__name__) # define flask app.server

#app = dash.Dash(__name__, server=server) # call flask server

server = flask.Flask(__name__)
external = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
app = dash.Dash(__name__, external_stylesheets = external, server=server)

app.title = 'COVID-19 Yucatán'

###################### datos modelo matematico ########################

data_sis = pd.read_csv('resultadosyuca.csv', header=None)
date = np.array('2020-03-13', dtype=np.datetime64)
date_p = date + np.arange(180)
data_sis['Fecha'] = date_p
activos = pd.read_excel('activos.xlsx')
date_a = date + np.arange(activos.shape[0])
data_sis["Fecha2"] = data_sis['Fecha'].dt.strftime("%Y-%m-%d")
activos['Fecha'] = date_a

mean_tpr = np.mean(data_sis, axis=1)
std_tpr = np.std(data_sis, axis=1)
tprs_upper = mean_tpr + std_tpr
tprs_lower = mean_tpr - std_tpr


mensaje = 'Se trata de un ejercicio académico para estudiar la evolución del COVID-19 en el Estado de Yucatán, México. Para este estudio se ha considerado diferentes modelos epidemiológicos y el registro de datos reales a partir del 13 de marzo. Como todo modelo matemático lo que se brinda es la estimación que arroja el modelo pero que no es una verdad absoluta, de cualquier manera, pudiera ser una alerta útil de prevención para la población. Investigadores y estudiantes asociados continúan trabajando en otros modelos matemáticos más generales que contemplen otras variables y métodos de solución.'


#### Modelo Dr Jorge, Dr Julián

data_sir_j = pd.read_csv('model_data/modeloSIR.csv')
date_sir_j = date + np.arange(data_sir_j.shape[0])
data_sir_j['Fecha'] = date_sir_j

fig_sir_j = go.Figure()
#fig_sir_j.add_trace(go.Scatter(x=data_sir_j['Fecha'], y= data_sir_j['Susceptibles'], mode='lines',line_color='blue', name = 'Susceptibles'))
fig_sir_j.add_trace(go.Scatter(x=data_sir_j['Fecha'], y= data_sir_j['Infectados'], mode='lines',line_color='orange', name = 'Infectados'))
#fig_sir_j.add_trace(go.Scatter(x=data_sir_j['Fecha'], y= data_sir_j['Recuperados'], mode='lines',line_color='green', name = 'Recuperados'))
fig_sir_j.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Activos'], mode='lines+markers',name = 'Casos activos reales', line_color = 'red'))

fig_sir_j.update_xaxes(rangeslider_visible=True)
fig_sir_j.update_layout(title="Modelo SIR",yaxis_title="Casos activos",title_x=0.43,template = 'plotly_dark')

#fig_sir_j_s = go.Figure()
#fig_sir_j_s.add_trace(go.Scatter(x=data_sir_j['Fecha'], y= data_sir_j['Susceptibles'], mode='lines',line_color='blue', name = 'Susceptibles'))
#fig_sir_j_s.update_xaxes(rangeslider_visible=True)
#fig_sir_j_s.update_layout(title="Modelo SIR",yaxis_title="Susceptibles",title_x=0.43,template = 'plotly_dark')


mensaje_sir = html.P(['El modelo SIR (susceptible-infectado-recuperado), también conocido como el modelo de Kermack y McKendrick por su famoso artículo, es un modelo clásico que, junto al teorema del umbral epidemiológico derivado de este, ha jugado un papel fundamental en desarrollos posteriores en el estudio de la dinámica de transmisión de enfermedades infecciosas. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatanSIR.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})

### Modelo SIS
data_sis_n = pd.read_csv('model_data/modelo_SIS.csv')
date_sis_n = date + np.arange(data_sis_n.shape[0])
data_sis_n['Fecha'] = date_sis_n

fig_sis = go.Figure()
#fig_sis.add_trace(go.Scatter(x=data_sis_n['Fecha'], y= data_sis_n['S(t)'], mode='lines',line_color='blue', name = 'Susceptibles'))
fig_sis.add_trace(go.Scatter(x=data_sis_n['Fecha'], y= data_sis_n['acumulados'], mode='lines',line_color='orange', name = 'Infectados acumulados'))
fig_sis.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Confirmados'], mode='lines+markers',name = 'Casos acumulados reales', line_color = 'red'))

fig_sis.add_trace(go.Scatter(x=data_sis_n['Fecha'], y= data_sis_n['activos'], mode='lines',name = 'Infectados activos'))
fig_sis.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Activos'], mode='lines+markers',name = 'Casos activos reales', line_color = 'yellow'))

fig_sis.update_xaxes(rangeslider_visible=True)
fig_sis.update_layout(title="Modelo SIS",yaxis_title="Casos",title_x=0.43,template = 'plotly_dark')

#fig_sis_s = go.Figure()
#fig_sis_s.add_trace(go.Scatter(x=data_sis_n['Fecha'], y= data_sis_n['S(t)'], mode='lines',line_color='blue', name = 'Susceptibles'))
#fig_sis_s.update_xaxes(rangeslider_visible=True)
#fig_sis_s.update_layout(title="Modelo SIS",yaxis_title="Susceptibles",title_x=0.43,template = 'plotly_dark')


mensaje_sis = html.P(['Este es un modelo simple de compartimentos del tipo Susceptibles - Infectados - Susceptibles (SIS), se pretende encontrar la estimación de los parámetros de las tasas de infección y recuperación a partir del registro de datos reales. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatanSISrv.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})


### Modelo Cajas
date_c = np.array('2020-01-06', dtype=np.datetime64)

mensaje_gompertz = html.P(['Se construye una aproximación analítica para la evolución de la curva epidémica de covid-19. Partiendo de la observación de que el número de infectados es mucho menor que la población total susceptible, se reduce el modelo susceptible-infectado-recuperado (SIR) y se obtiene una solución analítica de tipo Gompertz proponiendo una forma dependiente del tiempo para el parámetro de crecimiento. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatanSIRGpmpertz.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})

cajas_activo = pd.read_csv('model_data/YUCATAN-predict-activos.csv')
cajas_activo['Fecha'] = date_c + np.arange(cajas_activo.shape[0])

fig_cajas_activo = go.Figure()
fig_cajas_activo.add_trace(go.Scatter(x=cajas_activo['Fecha'], y= cajas_activo[' Infectados'], mode='lines',line_color='orange', name = 'Infectados'))
fig_cajas_activo.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Activos'], mode='lines+markers',name = 'Casos activos reales', line_color = 'red'))

fig_cajas_activo.update_xaxes(rangeslider_visible=True)
fig_cajas_activo.update_layout(title="Modelo SIR Gompertz casos activos",yaxis_title="Casos activos",title_x=0.43,template = 'plotly_dark')

cajas_acumulado = pd.read_csv('model_data/YUCATAN-predict.csv')
cajas_acumulado['Fecha'] = date_c + np.arange(cajas_acumulado.shape[0])

fig_cajas_acumulado = go.Figure()
fig_cajas_acumulado.add_trace(go.Scatter(x=cajas_acumulado['Fecha'], y= cajas_acumulado[' Infectados '], mode='lines',line_color='orange', name = 'Infectados'))
fig_cajas_acumulado.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Confirmados'], mode='lines+markers',name = 'Casos acumulados reales', line_color = 'red'))

fig_cajas_acumulado.update_xaxes(rangeslider_visible=True)
fig_cajas_acumulado.update_layout(title="Modelo SIR Gompertz casos acumulados",yaxis_title="Casos acumulados",title_x=0.43,template = 'plotly_dark')

#Modelo feno diarios
modelo_epi = pd.read_csv('model_data/feno_diarios.csv')
modelo_epi['Fecha'] = date + np.arange(modelo_epi.shape[0])

fig_epi = go.Figure()
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCL'], mode='lines',line_color='orange', name = 'MCL'))
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCLG'], mode='lines',line_color='green', name = 'MCLG'))
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCR'], mode='lines',line_color='yellow', name = 'MCR'))
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCRG'], mode='lines',line_color='blue', name = 'MCRG'))
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCG'], mode='lines',line_color='pink', name = 'MCG'))
fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['MCGG'], mode='lines',line_color='gray', name = 'MCGG'))
#fig_epi.add_trace(go.Scatter(x=modelo_epi['Fecha'], y= modelo_epi['CASOS'], mode='lines',line_color='red', name = 'Casos diarios acumulados'))
##fig_epi.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Confirmados'], mode='lines+markers',name = 'Casos acumulados reales', line_color = 'red'))

fig_epi.update_xaxes(rangeslider_visible=True)
fig_epi.update_layout(title="Modelo fenomenológico",yaxis_title="Casos acumulados",title_x=0.43,template = 'plotly_dark')


### Casos acumulados

#Modelo feno acumulados
feno_d = pd.read_csv('model_data/feno_diarios.csv')
feno_d['Fecha'] = date + np.arange(feno_d.shape[0])

fig_feno_d = go.Figure()
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCL'], mode='lines',line_color='orange', name = 'MCL'))
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCLG'], mode='lines',line_color='green', name = 'MCLG'))
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCR'], mode='lines',line_color='yellow', name = 'MCR'))
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCRG'], mode='lines',line_color='blue', name = 'MCRG'))
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCG'], mode='lines',line_color='pink', name = 'MCG'))
fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['MCGG'], mode='lines',line_color='gray', name = 'MCGG'))
#fig_feno_d.add_trace(go.Scatter(x=feno_d['Fecha'], y= feno_d['CASOS'], mode='lines',line_color='red', name = 'Casos diarios reales'))
##fig_epi.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Confirmados'], mode='lines+markers',name = 'Casos acumulados reales', line_color = 'red'))

fig_feno_d.update_xaxes(rangeslider_visible=True)
fig_feno_d.update_layout(title="Modelo fenomenológico",yaxis_title="Casos diarios",title_x=0.43,template = 'plotly_dark')

mensaje_epi = html.P(['Los Modelos de Crecimiento que se emplean son: el exponencial (MCE), el logístico (MCL), el de Richards (MCR), el de Gompertz (MCG) y sus variantes generalizadas denotadas con MCEG, MCLG, MCRG y MCGG respectivamente, con datos proporcionados por las autoridades del estado de Yucatán hasta el día 17 de mayo de 2020. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatanFenomenologicos.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})

##### Modelo SEIR
seir = pd.read_csv('model_data/modeloSEIR.csv',encoding="ISO-8859-1")
seir['Fecha'] = date + np.arange(seir.shape[0])

fig_seir = go.Figure()
#fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Susceptibles'], mode='lines',line_color='blue', name = 'Susceptibles'))
fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Expuestos'], mode='lines',line_color='yellow', name = 'Expuestos'))
fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Infectados'], mode='lines',line_color='orange', name = 'Infectados'))
fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Recuperados'], mode='lines',line_color='green', name = 'Recuperados'))
fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Infectados acumulados'], mode='lines',line_color='red', name = 'Infectaods acumulados'))

fig_seir.update_xaxes(rangeslider_visible=True)
fig_seir.update_layout(title="Modelo SEIR",yaxis_title="Número de casos",title_x=0.43,template = 'plotly_dark')

##susceptibles
#fig_seir_s = go.Figure()
#fig_seir_s.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Susceptibles'], mode='lines',line_color='blue', name = 'Susceptibles'))
#fig_seir_s.update_xaxes(rangeslider_visible=True)
#fig_seir_s.update_layout(title="Modelo SEIR",yaxis_title="Susceptibles",title_x=0.43,template = 'plotly_dark')


mensaje_seir = html.P(['El modelo SEIR es un modelo compartimental que incorpora a los susceptibles: personas que fueron expuestas y contagiadas a la enfermedad y que están incubando la enfermedad y no son aún contagiosos. Este modelo lo utilizamos para estimar parámetros a partir de la información de la reducción de movilidad que se ha observado en el estado de Yucatán. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatanSEIR.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})


#Modelo SIRD

sird = pd.read_csv('model_data/ecuaciones en diferencias.csv')
sird['Fecha'] = date + np.arange(sird.shape[0])

fig_sird = go.Figure()
#fig_seir.add_trace(go.Scatter(x=seir['Fecha'], y= seir['Susceptibles'], mode='lines',line_color='blue', name = 'Susceptibles'))
fig_sird.add_trace(go.Scatter(x=sird['Fecha'], y= sird['infectados acumulados'], mode='lines',line_color='yellow', name = 'Infectados acumulados'))
fig_sird.add_trace(go.Scatter(x=sird['Fecha'], y= sird['infectados activos'], mode='lines',line_color='orange', name = 'Infectados activos'))
fig_sird.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Activos'], mode='lines+markers',name = 'Casos activos reales', line_color = 'red'))
fig_sird.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Confirmados'], mode='lines+markers',name = 'Casos acumulados reales', line_color = 'pink'))


fig_sird.update_xaxes(rangeslider_visible=True)
fig_sird.update_layout(title="Modelo SIR por diferencias",yaxis_title="Número de casos",title_x=0.43,template = 'plotly_dark')


mensaje_sird = html.P(['Para este estudio se ha considerado un modelo SIRD simplificado a una ecuación, pero con tasas dependientes del tiempo, y el registro de datos reales dentro del periodo del 13 de marzo al 20 de mayo 2020. Ver resumen ', html.A('aquí.', href = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/resumenes/ResumenYucatan_Diferencias.pdf', target="_blank")], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})
########################################################################


#covid = pd.read_csv('http://187.191.75.115/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip', encoding="ISO-8859-1") #auto

##covid = pd.read_csv('https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/historical_db/200613COVID19MEXICO.csv?raw=true', encoding="ISO-8859-1") # manual

covid = pd.read_csv('historical_db/200613COVID19MEXICO.csv', encoding="ISO-8859-1") # manual


coords = pd.read_csv('coordenadas.csv')
yuc_coords = coords[coords['Num_Ent'] == 31]

d = {n:m for n, m in zip(yuc_coords.Num_Mun, yuc_coords.Municipio)}
numb = [1,2,3]
diag = ['Positivo', 'Negativo', 'Por confirmar']
d2 = {n2:m2 for n2, m2 in zip(numb, diag)}


yucatan = covid[covid['ENTIDAD_RES'] == 31]



yuc2 = yucatan.groupby(by=['MUNICIPIO_RES'])['RESULTADO'].value_counts()

yuc2 = pd.Series(yuc2.values, index = yuc2.index).reset_index().rename(columns={0: 'SUMA'})

#yuc2['MUNICIPIO_RES'] = yuc2['MUNICIPIO_RES'].replace(999,50) #Los no especificado los pongo en Mérida

yuc2 = yuc2[yuc2['MUNICIPIO_RES'] != 999] #Quito a los no especificados

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


##margin={"r":200,"t":0,"l":200,"b":100}


fig = px.scatter_mapbox(data_f, lat="Latitud", lon="Longitud", hover_name="Municipio",color='Positivos', color_continuous_scale=px.colors.diverging.Portland, zoom=7, height=500, size='Tamaño', hover_data=["Positivos", "Negativos", "Por confirmar"],center={'lat':lat.get('Mérida'), 'lon':lon.get('Mérida')})
fig.update_layout(mapbox_style="open-street-map")
#fig.update_layout(template = 'plotly_dark', title = 'Mapa de casos en Yucatán')
fig.update_layout(template = 'plotly_dark')

sexo_dict = {1:'Mujer', 2:'Hombre', 3:'No especificado'}

cols = ['NEUMONIA', 'DIABETES', 'EPOC', 'INMUSUPR', 'ASMA', 'HIPERTENSION', 'CARDIOVASCULAR', 'OBESIDAD','RENAL_CRONICA', 'OTRA_COM', 'OTRO_CASO', 'SEXO', 'RESULTADO']
##yucatan[cols]
#yuc3 = yucatan[cols]
yuc3 = pd.DataFrame(data=yucatan[cols].values, columns =cols)
sex = [sexo_dict.get(v) for v in yuc3['SEXO'].values.tolist()]
yuc3['Gender'] = np.asarray(sex)


data3 = yuc3[yuc3['RESULTADO'] == 1]
neumonia = data3[data3['NEUMONIA'] == 1]
diabetes = data3[data3['DIABETES'] == 1]
epoc = data3[data3['EPOC'] == 1]
hiper = data3[data3['HIPERTENSION'] == 1]
ob = data3[data3['OBESIDAD'] == 1]
car = data3[data3['CARDIOVASCULAR'] == 1]
asma = data3[data3['ASMA'] == 1]
inmu = data3[data3['INMUSUPR'] == 1]

fig2 = go.Figure()
fig2.add_trace(go.Histogram(histfunc="count", y=neumonia['NEUMONIA'], x=neumonia['Gender'], name="NEUMONIA"))
fig2.add_trace(go.Histogram(histfunc="count", y=diabetes['DIABETES'], x=diabetes['Gender'], name="DIABETES"))
fig2.add_trace(go.Histogram(histfunc="count", y=epoc['EPOC'], x=epoc['Gender'], name="EPOC"))
fig2.add_trace(go.Histogram(histfunc="count", y=hiper['HIPERTENSION'], x=hiper['Gender'], name="HIPERTENSION"))
fig2.add_trace(go.Histogram(histfunc="count", y=ob['OBESIDAD'], x=ob['Gender'], name="OBESIDAD"))
fig2.add_trace(go.Histogram(histfunc="count", y=car['CARDIOVASCULAR'], x=car['Gender'], name="CARDIOVASCULAR"))
fig2.add_trace(go.Histogram(histfunc="count", y=asma['ASMA'], x=asma['Gender'], name="ASMA"))
fig2.add_trace(go.Histogram(histfunc="count", y=inmu['INMUSUPR'], x=inmu['Gender'], name="INMUNOSUPR"))
fig2.update_layout(title="Casos positivos que presentan otra enfermedad",title_x=0.5, template = 'plotly_dark')



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




figx = go.Figure()
figx.add_trace(go.Scatter(x=fii['FECHA_INGRESO'].values.tolist(),y=fii['ACUMULADO'].values.tolist(), mode='lines+markers', name = 'Casos acumulados positivos'))
figx.add_trace(go.Scatter(x=nii['FECHA_INGRESO'].values.tolist(),y=nii['ACUMULADO'].values.tolist(), mode='lines+markers', name = 'Casos acumulados negativos'))


figx.update_layout(title="Casos acumulados en el estado",xaxis_title="Fecha de ingreso",yaxis_title="Acumulado", title_x=0.5, legend_orientation="h", template = 'plotly_dark')
figx.update_xaxes(rangeslider_visible=True)



####################### Agregando los modelo matemáticos #############

fig_m1 = go.Figure()
fig_m1.add_trace(go.Scatter(x=data_sis['Fecha'], y= tprs_upper, fill = None,mode='lines',line_color='lightskyblue', name = 'Rango máximo'))
fig_m1.add_trace(go.Scatter(x=data_sis['Fecha'], y= tprs_lower, fill = 'tonexty',mode='lines',line_color='lightskyblue', name = 'Rango mínimo'))
fig_m1.add_trace(go.Scatter(x=data_sis['Fecha'], y= mean_tpr, fill = None,mode='lines',line_color='blue', name = 'Rango promedio'))
fig_m1.add_trace(go.Scatter(x = activos['Fecha'], y = activos['Casos Activos'], mode='lines+markers',name = 'Casos activos reales', line_color = 'red'))

fig_m1.update_xaxes(rangeslider_visible=True)
fig_m1.update_layout(title="Modelo SIR",yaxis_title="Casos activos",title_x=0.43,template = 'plotly_dark')


#Modelo Dr Jorge y Dr. Julian

######################################################################

#html.Div(children = [html.H2('Modelos SIR'), html.P(mensaje, style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'}),dcc.Graph(id='m1', figure = fig_m1)])


################# Indice #################

index = html.Ol(children = [html.Li(children = [html.A('Información de Yucatán', href = '#casos-yuc')]),
                            html.Li(children = [html.A('Modelos matemáticos', href = '#mode_m'),
                            html.Ol(children = [html.Li(html.A('SIS', href='#sis')),
                                                html.Li(html.A('SIR', href='#sir_n')),
                                                html.Li(html.A('SIR (I<<S) soluciones tipo Gompertz', href='#sir_g')),
                                                html.Li(html.A('SEIR', href='#seir')),
                                                html.Li(html.A('SIRD (simplificado con tasas dependientes del tiempo)', href='#sird')),
                                                html.Li(html.A('Modelos fenomenológicos', href='#feno'))])]),
                            html.Li(children = [html.A('Integrantes y colaboradores', href = '#colab')])], style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})

##########################################

########### Titulo

table = html.Div(children = [html.Table(children = [
    html.Tr(children = [
        html.Td(html.Img(src = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/logos/unam.png?raw=true', style = {'width':'100px'})),
        html.Td(html.H1('Proyecto COVID-19'), style = {'color' : 'black'})])
    ], style = {'margin-left': 'auto', 'margin-right': 'auto'})], style = {'background-color':'white'})


############### Contenido ###################

intro = html.Div(children = [
    html.P('Es un esfuerzo dirigido a modelar la evolución del COVID-19 en el estado de Yucatán. Este trabajo es un proyecto totalmente académico que presenta resultados de la posible evolución de la pandemia. Utilizamos diferentes modelos matemáticos y datos reales  publicados por las autoridades sanitarias a partir del 13 de marzo de 2020. Este proyecto es sólo informativo y no se busca que estos resultados sean considerados ni reportados como una información confirmada para guiar decisiones clínicas.', style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})
    ])


########### Colaboradores ###################

colab = html.Div(children = [html.H2('Integrantes y colaboradores', id='colab'),
                             html.Img(src = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/logos/Grupo_colaboradores2.png?raw=true')],
                 style = {'margin-left': 'auto', 'margin-right': 'auto'})


#############################################
#html.Div(children = [html.Img(src = 'https://github.com/Luisbaduy97/COVID-YUCATAN/blob/master/logos/logos_finales.png?raw=true', style = {'height': '100px', 'width':'1280px'})], style = {'background-color': 'white'})

app.layout = html.Div([
    table,
    intro,
    html.Div(children = [html.H2('Contenido'), html.P('En este sitio encontrarás gráficas interactivas sobre la evolución del COVID-19 en Yucatán. Se presentan diferentes modelos matemáticos y podrás descargar un resumen detallado de ellos.', style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'}),index]),
    html.Div(children = [html.H2('COVID-19 Yucatán', id = 'casos-yuc'),html.Div([html.Div([dcc.Graph(id='acumulado', figure = figx)], className = 'six columns'),html.Div([dcc.Graph(id='d2', figure = fig2)], className = 'six columns')], className = "row")]),
    html.Div(children = [html.H3('Mapa de casos en Yucatán por municipio'), dcc.Graph(id='mapa', figure = fig)]),
    html.Div(children = [html.H2('Modelos matemáticos', id = 'mode_m'), html.P(mensaje, style = {'margin-left':'20%', 'margin-right':'20%', 'text-align':'justify'})]),
    html.Div(children = [html.H3('Modelo SIS', id = 'sis'), mensaje_sis ,dcc.Graph(id='sis_p', figure = fig_sis)]),
    html.Div(children = [html.H3('Modelo SIR', id='sir_n'), mensaje_sir ,dcc.Graph(id='sir_j', figure = fig_sir_j)]),
    html.Div(children = [html.H3('Modelo SIR (I<<S) soluciones tipo Gompertz',id='sir_g'), mensaje_gompertz ,dcc.Graph(id='sir_g_ac', figure = fig_cajas_acumulado), dcc.Graph(id='sir_g_act', figure = fig_cajas_activo)]),
    html.Div(children = [html.H3('Modelo SEIR',id='seir'), mensaje_seir ,dcc.Graph(id='seir_plot', figure = fig_seir)]),
    html.Div(children = [html.H3('SIRD (simplificado con tasas dependientes del tiempo)',id='sird'), mensaje_sird ,dcc.Graph(id='sird_p', figure = fig_sird)]),
    html.Div(children = [html.H3('Modelos Fenomenológicos',id='feno'), mensaje_epi ,dcc.Graph(id='feno_p', figure = fig_epi),dcc.Graph(id='feno_diario', figure = fig_feno_d)]),
    colab],style = {'background-color': '#121212', 'text-align': 'center','color': 'white'})



## local
#if __name__ == '__main__':
#     app.run_server(debug=True)


## gunicorn
app = app.server

