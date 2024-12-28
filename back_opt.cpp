#include "hnswlib.h"
#include "roaring.hh"
#include "roaring.c"
#include <unordered_map>
#include <vector>
#include <chrono>
#include <iostream>

const size_t attribute_count = 1000;
int max_elements = 10000;   // Maximum number of elements, should be known beforehand
// Use a vector to store Roaring Bitmaps for each label_id
std::vector<roaring::Roaring> label_attributes(max_elements);

// Filter that checks binary attributes using Roaring Bitmaps
class AttributeFilter : public hnswlib::BaseFilterFunctor {
    const roaring::Roaring& query_attributes;
 public:
    AttributeFilter(const roaring::Roaring& query_attributes)
        : query_attributes(query_attributes) {}

    bool operator()(hnswlib::labeltype label_id) {
        // Check if the point's attributes include all query attributes
        roaring::Roaring intersection;
        intersection = label_attributes[label_id] & query_attributes;
        return intersection == query_attributes;
    }
};

int main() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    int dim = 16;               // Dimension of the elements
    // int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
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

    // Assign random binary attributes to data points using Roaring Bitmaps
    // Start attribute assignment timing
    auto attr_start = std::chrono::high_resolution_clock::now();
    std::bernoulli_distribution distrib_bit(1.0 / 50.0); // Updated distribution
    for (int i = 0; i < max_elements; i++) {
        for (size_t j = 0; j < attribute_count; j++) {
            if (distrib_bit(rng)) { // Probability of 1/50
                label_attributes[i].add(j);
            }
        }
    }
    auto attr_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> attr_duration = attr_end - attr_start;
    std::cout << "Roaring Bitmap Attribute Assignment Time: " << attr_duration.count() << " seconds\n";

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Define query attributes using Roaring Bitmap
    roaring::Roaring query_attributes;
    // Set specific attributes
    query_attributes.add(0);
    query_attributes.add(5);

    // Specify the query point (e.g., the first element)
    int query_label = 0; // Change this to the desired query label
    float* query_point = data + query_label * dim;

    // Create the attribute filter for the query without excluding the query point
    AttributeFilter attributeFilter(query_attributes);

    int k = 2; // Number of nearest neighbors to retrieve
    std::vector<std::pair<float, hnswlib::labeltype>> result =
        alg_hnsw->searchKnnCloserFirst(query_point, k, &attributeFilter);

    // for (auto item : result) {
    //     roaring::Roaring intersection = label_attributes[item.second] & query_attributes;
    //     if (intersection == query_attributes) {
    //         // Print the label attributes for matching vectors
    //         std::cout << "Label ID: " << item.second << "\n";
    //     }
    //     else {
    //         std::cout << "Error: attributes do not match\n";
    //     }
    // }
    // std::cout << "####################\n";

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    // Estimate memory usage using croaring's portable size in bytes
    size_t roaring_memory_usage = 0;
    for (const auto& rb : label_attributes) {
        roaring_memory_usage += rb.getSizeInBytes();
    }
    std::cout << "Roaring Bitmap Approach Memory Usage: " << roaring_memory_usage << " bytes\n";
    std::cout << "Roaring Bitmap Approach Total Time: " << duration.count() << " seconds\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}
