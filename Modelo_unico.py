import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

tabla1=pd.read_csv("bd_modificado.csv")

tabla1.isnull().sum()
tabla1.fillna(tabla1["volumen_generado"].mean(),inplace=True)

#Limpieza de datos
def contar_valores(fila):
    return len(fila.split(','))
tabla1_mod=tabla1.copy()
tabla1_mod['conteo'] = tabla1_mod['ubicacion_recoleccion'].apply(contar_valores)
tabla1_mod["visitas_usuario2"]=tabla1_mod["visitas_usuario"]/tabla1_mod['conteo']
tabla1_mod["volumen_generado2"]=tabla1_mod["volumen_generado"]/tabla1_mod['conteo']
tabla1_mod.drop(columns=["id","fecha_nacimiento","direccion","visitas_usuario","volumen_generado","conteo","ubicacion_recoleccion"], inplace=True)


# Modelo Random Forest
tabla_cod = pd.get_dummies(tabla1_mod, columns=['genero'])

X = tabla_cod.drop(columns=["visitas_usuario2"])
y = tabla_cod["visitas_usuario2"]

random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X, y)
tabla_cod["visitas_predichas"] = random_forest.predict(X)
## Métricas
mse_bc = mean_squared_error(y, tabla_cod["visitas_predichas"])
r2_bc = r2_score(y, tabla_cod["visitas_predichas"])
print("Mean Squared Error:", mse_bc)
print("R^2 Score:", r2_bc)


# Importancia de las variables
importances1 = random_forest.feature_importances_
indices1 = np.argsort(importances1)[::-1]
top_variables1 = X.columns[indices1[:4]]
top_importances1 = importances1[indices1[:4]]
for variable, importancia in zip(top_variables1, top_importances1):
    print(f"Variable: {variable}, Importancia: {importancia}")


### Tabla final
tabla_cod2=tabla_cod.copy()
tabla_cod2['ubicacion_recoleccion']=tabla1['ubicacion_recoleccion']

# Separar los elementos de 'ubicacion_recoleccion' por coma y desglosarlos en filas separadas
tabla_cod3 = tabla_cod2.copy()
tabla_cod3['ubicacion_recoleccion'] = tabla_cod3['ubicacion_recoleccion'].str.split(', ')
tabla_cod3 = tabla_cod3.explode('ubicacion_recoleccion')

# Agrupar por 'ubicacion_recoleccion' y sumar el número de visitas por persona
visitas_por_ubicacion2 = tabla_cod3.groupby('ubicacion_recoleccion')['visitas_predichas'].sum()

visitas_por_ubicacion2 = visitas_por_ubicacion2.sort_values(ascending=False)
# Convertir la serie a DataFrame
df_visitas_por_ubicacion2 = visitas_por_ubicacion2.reset_index()
df_visitas_por_ubicacion2.columns = ['ubicacion_recoleccion', 'visitas_predichas_mes']
df_visitas_por_ubicacion2 = pd.DataFrame(df_visitas_por_ubicacion2)
df_visitas_por_ubicacion2.head(10)


## GRAFICAR LOS LUGARES POTENCIALES SEGÚN EL NÚMERO DE VIVSITAS MENSUALES
import folium

# Crear un mapa centrado en Medellín
mapa = folium.Map(location=[6.244203, -75.581211], zoom_start=12)

# Agregar marcadores con nombres para las coordenadas proporcionadas
marcadores = [
    {"nombre": "Centro comercial Aventura", "latitud": 6.264033, "longitud": -75.567149},
    {"nombre": "Centro comercial Puerta del Norte", "latitud": 6.3394476, "longitud": -75.5442995},
    {"nombre": "Universidad CES", "latitud": 6.208516, "longitud": -75.553145}
]

for marcador in marcadores:
    folium.Marker(
        location=[marcador["latitud"], marcador["longitud"]],
        tooltip=marcador["nombre"]
    ).add_to(mapa)


# Mostrar el mapa
mapa














