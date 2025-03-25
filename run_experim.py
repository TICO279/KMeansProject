import subprocess
import re
import statistics
import os

# Configuraciones
points_list = [100000, 200000, 300000, 400000, 600000, 800000, 1000000]
# Ajusta según tu máquina. Ejemplo:
#   - vCores = 8  => [1, 4, 8, 16]
threads_list = [1, 4, 8, 16]
num_clusters = 5
num_iterations = 10
max_file_prefix = "data"  # Base de archivos CSV, e.g. "100000_data.csv"

# Nombre del ejecutable C++
EXEC_NAME = "./K_MeansParalelo"

# Expresión regular para extraer tiempos
# Ej: "[Serial] Convergio en 10 iteraciones. Tiempo: 2.12345 seg."
time_pattern = re.compile(r"Tiempo:\s+([\d\.]+)\s+seg")

# Asegúrate de compilar antes de ejecutar este script:
# g++ -fopenmp kmeans_compare.cpp -o kmeans_compare

results = []  # Para guardar (nPoints, nThreads, avgTimeSerial, avgTimeParallel, speedUp)

for nPoints in points_list:
    # Nombre del archivo con los puntos
    input_csv = f"{nPoints}_data.csv"
    # (Opcional) Generar CSV con la libreta o verificar que ya existe
    if not os.path.exists(input_csv):
        print(f"ERROR: No existe {input_csv}. Genera primero con tu libreta.")
        continue

    for nThreads in threads_list:
        print(f"\nProbando {nPoints} puntos, {nThreads} hilos...")

        times_serial = []
        times_parallel = []

        for i in range(num_iterations):
            # Ejecutar kmeans_compare en modo BOTH para obtener tiempos de serial y paralelo
            cmd = [
                EXEC_NAME, 
                input_csv, 
                "results", 
                str(num_clusters), 
                "both", 
                str(nThreads)
            ]
            # Llamada al proceso
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = proc.stdout

            # Extraer los dos tiempos con la regex
            # Esperamos algo como:
            # [Serial] ...
            # Tiempo: ...
            # [Paralelo] ...
            # Tiempo: ...
            matches = time_pattern.findall(output)
            # matches debería ser una lista con dos strings: [timeSerial, timeParallel]
            if len(matches) == 2:
                t_serial_str, t_parallel_str = matches
                t_serial = float(t_serial_str)
                t_parallel = float(t_parallel_str)
                times_serial.append(t_serial)
                times_parallel.append(t_parallel)
            else:
                print("ERROR al parsear la salida:\n", output)

        # Calcular promedios y speed-up
        if len(times_serial) == num_iterations and len(times_parallel) == num_iterations:
            avg_serial = statistics.mean(times_serial)
            avg_parallel = statistics.mean(times_parallel)
            speed_up = avg_serial / avg_parallel if avg_parallel > 0 else 0

            print(f" => Promedio Serial: {avg_serial:.4f} seg")
            print(f" => Promedio Paralelo: {avg_parallel:.4f} seg")
            print(f" => Speed-up: {speed_up:.2f}")

            # Guardar en lista de resultados
            results.append((nPoints, nThreads, avg_serial, avg_parallel, speed_up))
        else:
            print("ERROR: No se obtuvieron tiempos suficientes.")

# Guardar resultados en CSV
with open("experiments_results.csv", "w") as f:
    f.write("nPoints,nThreads,avgTimeSerial,avgTimeParallel,speedUp\n")
    for row in results:
        f.write("{},{},{:.6f},{:.6f},{:.3f}\n".format(*row))

print("\nSe guardaron los resultados en experiments_results.csv")
