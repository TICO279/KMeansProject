import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments_results.csv")

# Ordenar lista de tamaños de datos
all_nPoints = sorted(df["nPoints"].unique())

for nPoints in all_nPoints:
    subset = df[df["nPoints"] == nPoints].sort_values("nThreads")
    plt.plot(subset["nThreads"], subset["speedUp"], marker='o', label=f"{nPoints} points")

plt.title("Speed-up vs Number of Threads (varios tamaños de datos)")
plt.xlabel("Number of Threads")
plt.ylabel("Speed-up (Serial / Paralelo)")
plt.grid(True)
plt.legend()
plt.show()
