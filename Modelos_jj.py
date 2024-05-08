import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



tabla2=pd.read_excel("datos.xlsx",sheet_name="Datos Cualitativos")
condicion1=(tabla2["Puntos Kaptar"]=="RutaN")
condicion2=(tabla2["Puntos Kaptar"]=="Plaza Mayor")
condicion3=(tabla2["Puntos Kaptar"]=="Edificio Coltejer")
tabla2=tabla2[(condicion1| (condicion2) | (condicion3))]
#Ruta N, Plaza Mayor y el Edificio Coltejer
tabla1=pd.read_csv("bd_modificado.csv")

tabla1.isnull().sum()
tabla1.fillna(tabla1["volumen_generado"].mean(),inplace=True)

#Punto de ubicacion Ruta N
filtro = tabla1["ubicacion_recoleccion"].str.contains(r'\bRuta\s+N\b', case=False, regex=True)
tabla_filtrada1 = tabla1[filtro]
def contar_valores(fila):
    return len(fila.split(','))
tabla_filtrada1['conteo'] = tabla_filtrada1['ubicacion_recoleccion'].apply(contar_valores)
tabla_filtrada1["visitas_usuario_RutaN"]=tabla_filtrada1["visitas_usuario"]/tabla_filtrada1['conteo']
tabla_filtrada1["volumen_generado_RutaN"]=tabla_filtrada1["volumen_generado"]/tabla_filtrada1['conteo']
tabla_filtrada1.drop(columns=["id","fecha_nacimiento","direccion","visitas_usuario","volumen_generado","conteo","ubicacion_recoleccion"], inplace=True)

#Punto de ubicacion colteger
filtro2 = tabla1["ubicacion_recoleccion"].str.contains(r'\bColtejer\b', case=False, regex=True)
tabla_filtrada2 = tabla1[filtro2]
tabla_filtrada2['conteo'] = tabla_filtrada2['ubicacion_recoleccion'].apply(contar_valores)
tabla_filtrada2["visitas_usuario_coltejer"]=tabla_filtrada2["visitas_usuario"]/tabla_filtrada2['conteo']
tabla_filtrada2["volumen_generado_coltejer"]=tabla_filtrada2["volumen_generado"]/tabla_filtrada2['conteo']
tabla_filtrada2.drop(columns=["id","fecha_nacimiento","direccion","visitas_usuario","volumen_generado","conteo","ubicacion_recoleccion"], inplace=True)

#Punto de ubicación plaza mayor
filtro3 = tabla1["ubicacion_recoleccion"].str.contains(r'\bPlaza\s+Mayor\b', case=False, regex=True)
tabla_filtrada3 = tabla1[filtro3]
tabla_filtrada3['conteo'] = tabla_filtrada3['ubicacion_recoleccion'].apply(contar_valores)
tabla_filtrada3["visitas_usuario_plazamayor"]=tabla_filtrada3["visitas_usuario"]/tabla_filtrada3['conteo']
tabla_filtrada3["volumen_generado_plazamayor"]=tabla_filtrada3["volumen_generado"]/tabla_filtrada3['conteo']
tabla_filtrada3.drop(columns=["id","fecha_nacimiento","direccion","visitas_usuario","volumen_generado","conteo","ubicacion_recoleccion"], inplace=True)




## MODELO RANDOM FOREST

###################
## Ruta N
tabla_cod1 = pd.get_dummies(tabla_filtrada1, columns=['genero'])

X_1 = tabla_cod1.drop(columns=["visitas_usuario_RutaN"])
y_1 = tabla_cod1["visitas_usuario_RutaN"]

random_forest1 = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest1.fit(X_1, y_1)
tabla_cod1["visitas_predichas_RutaN"] = random_forest1.predict(X_1)
## Métricas
mse_1 = mean_squared_error(y_1, tabla_cod1["visitas_predichas_RutaN"])
r2_1 = r2_score(y_1, tabla_cod1["visitas_predichas_RutaN"])
print("Mean Squared Error _ Ruta N:", mse_1)
print("R^2 Score _ Ruta N:", r2_1)


###################
## Coltejer
tabla_cod2 = pd.get_dummies(tabla_filtrada2, columns=['genero'])

X_2 = tabla_cod2.drop(columns=["visitas_usuario_coltejer"])
y_2 = tabla_cod2["visitas_usuario_coltejer"]

random_forest2 = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest2.fit(X_2, y_2)
tabla_cod2["visitas_predichas_coltejer"] = random_forest2.predict(X_2)
## Métricas
mse_2 = mean_squared_error(y_2, tabla_cod2["visitas_predichas_coltejer"])
r2_2 = r2_score(y_2, tabla_cod2["visitas_predichas_coltejer"])
print("Mean Squared Error _ coltejer:", mse_2)
print("R^2 Score _ coltejer:", r2_2)


###################
## Plaza mayor
tabla_cod3 = pd.get_dummies(tabla_filtrada3, columns=['genero'])

X_3 = tabla_cod3.drop(columns=["visitas_usuario_plazamayor"])
y_3 = tabla_cod3["visitas_usuario_plazamayor"]

random_forest3 = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest3.fit(X_3, y_3)
tabla_cod3["visitas_predichas_plazamayor"] = random_forest3.predict(X_3)
## Métricas
mse_3 = mean_squared_error(y_3, tabla_cod3["visitas_predichas_plazamayor"])
r2_3 = r2_score(y_3, tabla_cod3["visitas_predichas_plazamayor"])
print("Mean Squared Error _ plazamayor:", mse_3)
print("R^2 Score _ plazamayor:", r2_3)


# Variables más importantes

###################
## Ruta N
importances1 = random_forest1.feature_importances_
indices1 = np.argsort(importances1)[::-1]
top_variables1 = X_1.columns[indices1[:4]]
top_importances1 = importances1[indices1[:4]]
for variable, importancia in zip(top_variables1, top_importances1):
    print(f"Variable: {variable}, Importancia: {importancia}")

##################
## Coltejer
importances2 = random_forest2.feature_importances_
indices2 = np.argsort(importances2)[::-1]
top_variables2 = X_2.columns[indices2[:4]]
top_importances2 = importances2[indices2[:4]]
for variable, importancia in zip(top_variables2, top_importances2):
    print(f"Variable: {variable}, Importancia: {importancia}")

###################
## Plaza mayor
importances3 = random_forest3.feature_importances_
indices3 = np.argsort(importances3)[::-1]
top_variables3 = X_3.columns[indices3[:4]]
top_importances3 = importances3[indices3[:4]]
for variable, importancia in zip(top_variables3, top_importances3):
    print(f"Variable: {variable}, Importancia: {importancia}")


tabla2







