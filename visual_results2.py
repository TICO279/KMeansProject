import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar los resultados
df = pd.read_csv("experiments_results.csv")

# 2. Elegir el número de puntos que deseas graficar
nPoints_to_plot = 800000

# 3. Filtrar y ordenar los datos
subset = df[df["nPoints"] == nPoints_to_plot].sort_values("nThreads")

# 4. Graficar el tiempo serial (en función de nThreads)
plt.plot(
    subset["nThreads"], 
    subset["avgTimeSerial"], 
    marker='o', 
    label="Serial"
)

# 5. Graficar el tiempo paralelo (en función de nThreads)
plt.plot(
    subset["nThreads"], 
    subset["avgTimeParallel"], 
    marker='x', 
    label="Paralelo"
)

# 6. Personalizar la gráfica
plt.title(f"Tiempo vs. Número de Hilos ({nPoints_to_plot} puntos)")
plt.xlabel("Número de Hilos")
plt.ylabel("Tiempo (segundos)")
plt.grid(True)
plt.legend()

# 7. Mostrar la gráfica
plt.show()
