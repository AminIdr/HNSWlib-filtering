#include "hnswlib.h"
#include <bitset>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

const size_t attribute_count = 1000;
int max_elements = 10000;   // Maximum number of elements, should be known beforehand
// Use a vector to store attribute bitsets for each label_id
std::vector<std::bitset<attribute_count>> label_attributes(max_elements);

// Filter that checks binary attributes using bitsets
class AttributeFilter : public hnswlib::BaseFilterFunctor {
    const std::bitset<attribute_count>& query_attributes;
 public:
    AttributeFilter(const std::bitset<attribute_count>& query_attributes)
        : query_attributes(query_attributes) {}

    bool operator()(hnswlib::labeltype label_id) {
        // Check if the point's attributes include all query attributes
        return (label_attributes[label_id] & query_attributes) == query_attributes;
    }
};

struct BenchmarkResult {
    double search_latency;
    double sparsity;
    size_t memory_usage; 
};

BenchmarkResult runBenchmark(double probability) {
    BenchmarkResult result;

    int dim = 16;
    int M = 16;
    int ef_construction = 200;
    
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate and add data
    std::mt19937 rng(48);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Attribute assignment
    std::bernoulli_distribution distrib_bit(probability);
    size_t total_set_bits = 0;
    // Skip the first point (as it is the query point)
    for (int i = 1; i < max_elements; i++) {
        std::bitset<attribute_count> attributes;
        for (size_t j = 0; j < attribute_count; j++) {
            bool bit = distrib_bit(rng);
            attributes[j] = bit;
            if(bit) total_set_bits++;
        }
        label_attributes[i] = attributes;
        alg_hnsw->addPoint(data + i * dim, i);
    }

    std::bitset<attribute_count> query_attributes;
    // Set 2 bits to 1
    query_attributes.set(50);
    query_attributes.set(100);
    
    // I assume that the first point is the query point
    int query_label = 0;
    float* query_point = data + query_label * dim;
    AttributeFilter attributeFilter(query_attributes);
    
    int k = 2;
    // Measure only seach latency
    auto search_start = std::chrono::high_resolution_clock::now();
    auto result_set = alg_hnsw->searchKnnCloserFirst(query_point, k, &attributeFilter);
    auto search_end = std::chrono::high_resolution_clock::now();

    // Calculate latency and sparsity
    result.search_latency = std::chrono::duration<double>(search_end - search_start).count();
    result.sparsity = 1.0 - (double)total_set_bits / (max_elements * attribute_count);

    // Calculate memory usage (size of all bitsets)
    result.memory_usage = max_elements * (attribute_count / 8);  // size in bytes

    delete[] data;
    delete alg_hnsw;
    return result;
}

int main() {
    std::vector<double> probabilities = {0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5};
    
    std::ofstream results("benchmark_results.csv", std::ios::app);
    results << "probability,sparsity,search_latency,memory\n";
    
    
    for (double p : probabilities) {
        BenchmarkResult avg_result = {0, 0, 0};
        
        auto result = runBenchmark(p);
        avg_result.search_latency += result.search_latency;
        avg_result.sparsity += result.sparsity;
        avg_result.memory_usage += result.memory_usage;
    
        results << p << ","
                << avg_result.sparsity << ","
                << avg_result.search_latency << ","
                << avg_result.memory_usage << "\n";
                
        std::cout << "Completed probability: " << p << std::endl;
    }
    
    results.close();
    return 0;
}