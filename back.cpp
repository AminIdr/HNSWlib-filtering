#include "hnswlib.h"
#include <bitset>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <iostream>
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

int main() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int dim = 16;               // Dimension of the elements
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    // M * 8-10 bytes per stored element. The higher the M, the more memory is consumed
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(48);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Assign random binary attributes to data points
    // Start attribute assignment timing
    auto attr_start = std::chrono::high_resolution_clock::now();
    std::bernoulli_distribution distrib_bit(1.0 / 15.0); // Updated distribution
    for (int i = 0; i < max_elements; i++) {
        std::bitset<attribute_count> attributes;
        // Set each attribute to 0 or 1 with probability 1/50 for 1
        for (size_t j = 0; j < attribute_count; j++) {
            attributes[j] = distrib_bit(rng);
        }
        label_attributes[i] = attributes;
    }
    auto attr_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> attr_duration = attr_end - attr_start;
    std::cout << "Bitset Attribute Assignment Time: " << attr_duration.count() << " seconds\n";

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Define query attributes
    std::bitset<attribute_count> query_attributes;
    // Set only the first attribute
    query_attributes.set(0);
    query_attributes.set(5);
    // Specify the query point (e.g., the first element)
    int query_label = 0; // Change this to the desired query label
    float* query_point = data + query_label * dim;

    // Create the attribute filter for the query without excluding the query point
    AttributeFilter attributeFilter(query_attributes);

    int k = 2; // Number of nearest neighbors to retrieve
    std::vector<std::pair<float, hnswlib::labeltype>> result =
        alg_hnsw->searchKnnCloserFirst(query_point, k, &attributeFilter);

    for (auto item : result) {
        if ((label_attributes[item.second] & query_attributes) == query_attributes)
            // Print the label attributes for matching vectors
            std::cout << "Label ID: " << item.second << ", Attributes: "
                      << label_attributes[item.second] << "\n";
        else
            std::cout << "Error: attributes do not match\n";
    }
    std::cout << "####################\n";

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // Estimate memory usage by counting set bits
    size_t total_bits = 0;
    
    for (const auto& bits : label_attributes) {
        total_bits += bits.size(); // Add the size of each bitset
    }

    // Or simply max_elements * attribute_count / 8
    size_t memory_usage = total_bits / 8; // Convert bits to bytes
    std::cout << "Bitset Approach Memory Usage: " << memory_usage << " bytes\n";
    std::cout << "Bitset Approach Total Time: " << duration.count() << " seconds\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}