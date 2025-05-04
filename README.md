# Parallel K-Means Clustering

This project implements a **parallel version of the K-Means algorithm** in C++ using **OpenMP**, and compares its performance with the serial version. It includes tools for **data generation**, **performance measurement**, and **result visualization** using Python.

## Features

- **Parallel K-Means in C++**:
  - Leverages **OpenMP** to speed up the assignment and update steps.
  - Configurable number of clusters (`k`) and data points.
  - Measures execution time for different thread counts.

- **Performance Benchmarking**:
  - Includes a Python script (`run_experim.py`) to run experiments across datasets and thread counts.
  - Serial and parallel timings are recorded for comparison.

- **Visualization Tools**:
  - `visual_results.py`: Plots speedup vs. number of threads across various dataset sizes.
  - `visual_results2.py`: Compares serial and parallel execution times for a fixed dataset size.

- **Data Generation Notebook**:
  - `synthetic_clusters.ipynb`: Creates synthetic datasets to test clustering scalability and performance.

- **Result Exporting**:
  - Results are saved to `experiments_results.csv`, `results_parallel.csv`, and `results_serial.csv`.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Parallel-KMeans-Clustering.git
   cd Parallel-KMeans-Clustering
   ```

2. **Compile the C++ Code with OpenMP**:
   Ensure you have `g++` with OpenMP support installed.
   ```bash
   g++ -fopenmp K_MeansParalelo.cpp -o kmeans_parallel
   ```

3. **Run the Parallel K-Means**:
   Execute the binary with arguments:
   ```bash
   ./kmeans_parallel <num_points> <num_clusters> <num_threads>
   ```
   Example:
   ```bash
   ./kmeans_parallel 1000000 10 4
   ```

4. **Benchmark with Python (Optional)**:
   Use `run_experim.py` to automate testing for multiple thread counts and dataset sizes.
   ```bash
   python run_experim.py
   ```

5. **Visualize Results**:
   - General speedup across datasets:
     ```bash
     python visual_results.py
     ```
   - Serial vs. parallel time comparison:
     ```bash
     python visual_results2.py
     ```

## Notes and Limitations

- The parallel implementation is optimized for **multi-core systems** using OpenMP. Performance gains depend on available hardware and thread scaling.
- **Input data is generated synthetically**. For real-world clustering, further data preprocessing would be required.
- The number of iterations for K-Means is fixed; convergence-based stopping criteria are not implemented.
- The program expects command-line arguments. Missing or malformed inputs may cause runtime errors.
- Some Python scripts assume that results are saved in `experiments_results.csv`. Ensure consistent naming when exporting data.
- Results and graphs are intended for **academic performance evaluation** rather than production deployment.
- The OpenMP schedule policy is static by default. For highly unbalanced workloads, manual tuning might be needed.
