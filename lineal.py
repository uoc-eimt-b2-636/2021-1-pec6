import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Usaremos el dataset 'mtcars'. Lo cargamos como un dataframe de pandas
mtcars=pd.read_csv('mtcars.csv')

# Mostramos varios valores de ejemplo
mtcars.head()

# Genera un modelo lineal del consumo del motor de un coche en relación a su potencia.
X = np.expand_dims(mtcars['hp'], axis=1)
y = mtcars['mpg']
reg = LinearRegression().fit(X, y)
reg.score(X, y)  # Coeficiente R2 0.602437
reg.coef_  
reg.intercept_ 

# Predice el consumo basado en el modelo lineal generado para valores determinados de potencia del motor
reg.predict(np.array([[100], [170], [250]]))

# Genera las predicciones para todo el rango de datos.
reg.predict(X)


