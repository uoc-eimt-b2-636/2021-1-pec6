import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

### K-means
# Ejemplo del material sobre aprendizaje no supervisado de la PEC6.

# Define la semilla de pseudoaleatoriedad para reproducibilidad.
np.random.seed(10)

# Genera una matrix de 2x100 valores, siguiendo una distribución normal con media 0 (valor por defecto) y desviación estándar 1 (valor por defecto).
x = np.random.normal(loc=0, scale=1, size=(100,2))

# Genera una matrix de 2x4 valores, siguiendo una distribución normal con media 0 (valor por defecto) y desviación estándar 2.
xmean = np.random.normal(loc=0, scale=2, size=(4,2))

# Genera un vector de 100 valores enteros aleatorios entre 0 y 3.
which = np.random.randint(4, size=100)

# Genera el conjunto final de valores.
x = x + xmean[which,]

# Genera un vector de 100 valores enteros aleatorios entre 1 y 4, para la asignación inicial de grupos.
col_ini = np.random.randint(4, size=100)

# Crea la figura en la que plotearemos
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(12,5))

# Visualiza los datos y la asignación inicial alteatoria.
ax1.scatter(x[:,0], x[:,1], c=col_ini, alpha = 0.6, s=10)
ax1.set_title("Asignación inicial aleatoria")

# Calcula los centroides de las muestras sin agrupar y los añade.
centroids_init = np.zeros((4,2))
for i in range(4):
  centroids_init[i,:] = np.mean(x[col_ini==i,], axis=0)

ax1.scatter(centroids_init[:,0], centroids_init[:,1], c=[0, 1, 2, 3], marker='x', s=200)

# Realiza una agrupación en 4 conjuntos con el algoritmo K-means, usando una iteración.
kmeans = KMeans(n_clusters=4, random_state=0, max_iter=1, n_init=1).fit(x)

# Visualiza la agrupación y los nuevos centroides.
ax2.scatter(x[:,0], x[:,1], c=kmeans.labels_, alpha = 0.6, s=10)
ax2.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=[0,1,2,3], marker='x', s=200)
ax2.set_title("K-means tras 1 iteración")

# Realiza una agrupación en 4 conjuntos con el algoritmo K-means, usando dos iteraciones.
kmeans = KMeans(n_clusters=4, random_state=0, max_iter=2, n_init=1).fit(x)

# Visualiza la agrupación y los nuevos centroides.
ax3.scatter(x[:,0], x[:,1], c=kmeans.labels_, alpha = 0.6, s=10)
ax3.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=[0,1,2,3], marker='x', s=200)
ax3.set_title("K-means tras 2 iteraciones")

# Visualiza los datos de la clasificación.
plt.show()

