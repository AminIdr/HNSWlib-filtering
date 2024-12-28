# HNSW Filtering Approaches: Bitset vs Roaring Bitmap

This project evaluates two filtering approaches for optimizing multi-attribute filtering in the HNSW (Hierarchical Navigable Small World) algorithm: **Bitset** and **Roaring Bitmap**. The project benchmarks these approaches in terms of **search latency** and **memory usage**, specifically under different levels of sparsity.

## Project Structure

- `example.cpp`: Contains the implementation of the **Bitset** approach for filtering data during nearest neighbor search in HNSW.
- `optimized_example.cpp`: Contains the implementation of the **Roaring Bitmap** approach for filtering, optimized for sparse data handling.
- `comparison.ipynb`: A Python script that processes benchmark results saved in CSV files and generates performance comparison plots between the two filtering approaches.
- `Database Report.pdf`: A detailed report that summarizes the project, including background, methodology, results, and performance analysis.

### Benchmarking

The C++ code generates benchmark results saved in CSV files. These CSV files contain the following fields:
- `probability`: The probability used in the Bernoulli distribution to simulate data sparsity.
- `sparsity`: The sparsity level of the data.
- `search_latency`: The time taken to perform the nearest neighbor search.
- `memory`: The memory usage for the data filtering process.

