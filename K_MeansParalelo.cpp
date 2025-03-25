#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

// --------------------------------------------------
// Estructuras y funciones auxiliares
// --------------------------------------------------

struct Point {
    double x, y;
    int cluster;
};

struct Centroid {
    double x, y;
};

// Calcula la distancia al cuadrado para evitar sqrt()
double distanceSquared(const Point &p, const Centroid &c) {
    double dx = p.x - c.x;
    double dy = p.y - c.y;
    return dx * dx + dy * dy;
}

// --------------------------------------------------
// K-Means (Versión Serial)
// --------------------------------------------------
double runKMeansSerial(vector<Point> &points, int numClusters, int maxIterations) {
    int numPoints = points.size();
    // Inicializar centroides (aleatorio entre los puntos)
    srand(time(NULL));
    vector<Centroid> centroids(numClusters);
    for (int i = 0; i < numClusters; i++) {
        int idx = rand() % numPoints;
        centroids[i].x = points[idx].x;
        centroids[i].y = points[idx].y;
    }

    bool changed = true;
    int iterations = 0;

    // Medir tiempo
    auto start = chrono::high_resolution_clock::now();

    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        // Asignación de cada punto al centroide más cercano (serial)
        for (int i = 0; i < numPoints; i++) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;
            for (int c = 0; c < numClusters; c++) {
                double d = distanceSquared(points[i], centroids[c]);
                if (d < minDist) {
                    minDist = d;
                    bestCluster = c;
                }
            }
            if (points[i].cluster != bestCluster) {
                points[i].cluster = bestCluster;
                changed = true;
            }
        }

        // Actualizar centroides
        vector<double> sumX(numClusters, 0.0);
        vector<double> sumY(numClusters, 0.0);
        vector<int> count(numClusters, 0);

        for (int i = 0; i < numPoints; i++) {
            int cluster = points[i].cluster;
            sumX[cluster] += points[i].x;
            sumY[cluster] += points[i].y;
            count[cluster]++;
        }

        for (int c = 0; c < numClusters; c++) {
            if (count[c] > 0) {
                centroids[c].x = sumX[c] / count[c];
                centroids[c].y = sumY[c] / count[c];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "[Serial] Convergio en " << iterations << " iteraciones. "
         << "Tiempo: " << elapsed.count() << " seg.\n";

    return elapsed.count();
}

// --------------------------------------------------
// K-Means (Versión Paralela con OpenMP)
// --------------------------------------------------
double runKMeansParallel(vector<Point> &points, int numClusters, int maxIterations, int numThreads) {
    // Fijar número de hilos
    omp_set_num_threads(numThreads);

    int numPoints = points.size();
    // Inicializar centroides (aleatorio entre los puntos)
    srand(time(NULL));
    vector<Centroid> centroids(numClusters);
    for (int i = 0; i < numClusters; i++) {
        int idx = rand() % numPoints;
        centroids[i].x = points[idx].x;
        centroids[i].y = points[idx].y;
    }

    bool changed = true;
    int iterations = 0;

    // Medir tiempo
    auto start = chrono::high_resolution_clock::now();

    while (changed && iterations < maxIterations) {
        changed = false;
        iterations++;

        // Asignación de cada punto al centroide más cercano (paralelo)
        #pragma omp parallel for schedule(static) reduction(||:changed)
        for (int i = 0; i < numPoints; i++) {
            double minDist = numeric_limits<double>::max();
            int bestCluster = -1;
            for (int c = 0; c < numClusters; c++) {
                double d = distanceSquared(points[i], centroids[c]);
                if (d < minDist) {
                    minDist = d;
                    bestCluster = c;
                }
            }
            if (points[i].cluster != bestCluster) {
                points[i].cluster = bestCluster;
                changed = true;
            }
        }

        // Actualizar centroides
        vector<double> sumX(numClusters, 0.0);
        vector<double> sumY(numClusters, 0.0);
        vector<int> count(numClusters, 0);

        #pragma omp parallel
        {
            vector<double> localSumX(numClusters, 0.0);
            vector<double> localSumY(numClusters, 0.0);
            vector<int> localCount(numClusters, 0);

            #pragma omp for nowait
            for (int i = 0; i < numPoints; i++) {
                int cluster = points[i].cluster;
                localSumX[cluster] += points[i].x;
                localSumY[cluster] += points[i].y;
                localCount[cluster]++;
            }

            #pragma omp critical
            {
                for (int c = 0; c < numClusters; c++) {
                    sumX[c] += localSumX[c];
                    sumY[c] += localSumY[c];
                    count[c] += localCount[c];
                }
            }
        }

        for (int c = 0; c < numClusters; c++) {
            if (count[c] > 0) {
                centroids[c].x = sumX[c] / count[c];
                centroids[c].y = sumY[c] / count[c];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "[Paralelo] " << numThreads << " hilos, convergio en " << iterations
         << " iteraciones. Tiempo: " << elapsed.count() << " seg.\n";

    return elapsed.count();
}

// --------------------------------------------------
// Función para leer datos CSV
// --------------------------------------------------
bool readCSV(const string &filename, vector<Point> &points) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error al abrir el archivo: " << filename << endl;
        return false;
    }

    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        Point p;
        getline(ss, token, ',');
        p.x = stod(token);
        getline(ss, token, ',');
        p.y = stod(token);
        p.cluster = -1;  // Sin asignar
        points.push_back(p);
    }
    infile.close();
    return true;
}

// --------------------------------------------------
// Función para escribir resultados CSV
// --------------------------------------------------
bool writeCSV(const string &filename, const vector<Point> &points) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error al abrir el archivo de salida: " << filename << endl;
        return false;
    }
    for (auto &p : points) {
        outfile << p.x << "," << p.y << "," << p.cluster << "\n";
    }
    outfile.close();
    return true;
}

// --------------------------------------------------
// main
// --------------------------------------------------
int main(int argc, char* argv[]) {
    /*
      Uso:
         kmeans_compare <archivo_entrada.csv> <archivo_salida_base> <num_clusters> <modo> [num_threads]

      Donde:
         - <archivo_entrada.csv>: datos de entrada
         - <archivo_salida_base>: base para nombrar salidas (ej. "results" -> "results_serial.csv", "results_parallel.csv")
         - <num_clusters>: número de clusters
         - <modo>: "serial", "parallel" o "both"
         - [num_threads]: solo se usa si el modo es "parallel" o "both" (por defecto 4)
    */

    if (argc < 5) {
        cerr << "Uso: " << argv[0] << " <input.csv> <output_base> <num_clusters> <mode> [num_threads]\n";
        cerr << "   mode = serial | parallel | both\n";
        return 1;
    }

    string inputFile    = argv[1];
    string outputBase   = argv[2];
    int numClusters     = atoi(argv[3]);
    string mode         = argv[4];
    int numThreads      = 4; // por defecto

    if ((mode == "parallel" || mode == "both") && argc >= 6) {
        numThreads = atoi(argv[5]);
    }

    // Leer puntos
    vector<Point> points;
    if (!readCSV(inputFile, points)) {
        return 1; // error al leer
    }

    // Para comparar, a menudo necesitamos COPIAS de "points", 
    // porque la asignación de clusters modifica el vector.
    // Dejamos un backup "originalPoints" sin modificar.
    vector<Point> originalPoints = points;

    // Parametros
    const int maxIterations = 100;

    // Variables para medir tiempos
    double timeSerial   = 0.0;
    double timeParallel = 0.0;

    // --------------------------------------------------
    // Modo Serial
    // --------------------------------------------------
    if (mode == "serial" || mode == "both") {
        // Copiamos "originalPoints" para la versión serial
        vector<Point> pointsSerial = originalPoints;
        cout << "Ejecutando K-Means Serial..." << endl;
        timeSerial = runKMeansSerial(pointsSerial, numClusters, maxIterations);

        // Guardar resultados en un CSV
        string serialFile = outputBase + "_serial.csv";
        writeCSV(serialFile, pointsSerial);
    }

    // --------------------------------------------------
    // Modo Paralelo
    // --------------------------------------------------
    if (mode == "parallel" || mode == "both") {
        // Copiamos "originalPoints" para la versión paralela
        vector<Point> pointsParallel = originalPoints;
        cout << "Ejecutando K-Means Paralelo con " << numThreads << " hilos..." << endl;
        timeParallel = runKMeansParallel(pointsParallel, numClusters, maxIterations, numThreads);

        // Guardar resultados en un CSV
        string parallelFile = outputBase + "_parallel.csv";
        writeCSV(parallelFile, pointsParallel);
    }

    // --------------------------------------------------
    // Comparar tiempos si ejecutamos ambas
    // --------------------------------------------------
    if (mode == "both") {
        if (timeParallel > 0.0) {
            double speedup = timeSerial / timeParallel;
            cout << "Speed-up = " << speedup << " (Serial/Paralelo)\n";
        } else {
            cout << "No se pudo calcular Speed-up (tiempo paralelo = 0?).\n";
        }
    }

    return 0;
}
