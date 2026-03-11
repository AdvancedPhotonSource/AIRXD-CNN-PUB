#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Utility function to find the binning index for a value in theta_val
// Returns the index of the largest value in theta_val that is <= val
size_t find_bin_index(const std::vector<double>& theta_val, double val) {
    // Use binary search to find the appropriate bin
    auto it = std::upper_bound(theta_val.begin(), theta_val.end(), val);
    
    if (it == theta_val.begin()) {
        // Value is smaller than the smallest theta_val
        return 0;
    } else {
        // Return index of the previous element (largest not exceeding val)
        return std::distance(theta_val.begin(), it - 1);
    }
}

// Function to normalize and create a histogram in one pass (unchanged)
py::array_t<int> normalize_and_histogram(const std::vector<double>& values, int num_bins, double min_val, double max_val) {
    size_t n = values.size();
    
    // Calculate mean and std in one pass using Welford's algorithm for better numerical stability
    double mean = 0.0;
    double M2 = 0.0;
    double delta, delta2;
    
    for (size_t i = 0; i < n; i++) {
        delta = values[i] - mean;
        mean += delta / (i + 1);
        delta2 = values[i] - mean;
        M2 += delta * delta2;
    }
    
    double variance = n > 1 ? M2 / n : 0.0;
    double std_dev = std::sqrt(variance);
    
    // Create output histogram
    py::array_t<int> histogram(num_bins);
    py::buffer_info hist_buf = histogram.request();
    int *hist_ptr = static_cast<int *>(hist_buf.ptr);
    
    // Initialize histogram bins to zero
    std::fill(hist_ptr, hist_ptr + num_bins, 0);
    
    // Calculate bin width
    double bin_width = (max_val - min_val) / num_bins;
    
    // Fill histogram with normalized values
    if (std_dev > 1e-10) {  // Avoid division by very small numbers
        for (size_t i = 0; i < n; i++) {
            double norm_val = (values[i] - mean) / std_dev;
            if (norm_val >= min_val && norm_val < max_val) {
                int bin = static_cast<int>((norm_val - min_val) / bin_width);
                // Handle edge case for maximum value
                if (bin == num_bins) bin = num_bins - 1;
                hist_ptr[bin]++;
            }
        }
    } else {
        // If std_dev is effectively zero, all values are the same
        int bin = static_cast<int>((0.0 - min_val) / bin_width);
        if (bin >= 0 && bin < num_bins) {
            hist_ptr[bin] = n;
        }
    }
    
    return histogram;
}

// Updated function to discretize categorization values and process with mask
py::dict process_discretized_categorized_data(py::array_t<double> B, py::array_t<double> C, 
                                             py::array_t<int> D, py::array_t<double> theta_val, 
                                             int num_bins, double min_val, double max_val) {
    // Get buffer info for all arrays
    py::buffer_info b_buf = B.request();
    py::buffer_info c_buf = C.request();
    py::buffer_info d_buf = D.request();
    py::buffer_info theta_buf = theta_val.request();
    
    // Get pointers to data
    double *b_ptr = static_cast<double *>(b_buf.ptr);
    double *c_ptr = static_cast<double *>(c_buf.ptr);
    int *d_ptr = static_cast<int *>(d_buf.ptr);
    double *theta_ptr = static_cast<double *>(theta_buf.ptr);
    
    // Check that B, C, and D have the same size
    if (b_buf.size != c_buf.size || b_buf.size != d_buf.size) {
        throw std::runtime_error("Arrays B, C, and D must have the same size");
    }
    
    size_t data_size = b_buf.size;
    size_t num_theta_vals = theta_buf.size;
    
    // Copy theta_val values to a vector for easier manipulation
    std::vector<double> theta_values(theta_ptr, theta_ptr + num_theta_vals);
    
    // Sort theta_values to ensure binary search works correctly
    std::sort(theta_values.begin(), theta_values.end());
    
    // Pre-allocate vectors for each discrete theta value
    std::vector<std::vector<double>> categorized_data(num_theta_vals);
    
    // Group elements based on discretized theta values
    for (size_t i = 0; i < data_size; i++) {
        if (d_ptr[i] == 0) {  // Only process if mask value is 0 (i.e. not masked)
            double c_value = c_ptr[i];
            
            // Find the appropriate bin in theta_values
            size_t bin_idx = find_bin_index(theta_values, c_value);
            
            // Add the B value to the appropriate bin
            categorized_data[bin_idx].push_back(b_ptr[i]);
        }
    }
    
    // Create result dictionary
    py::dict result;
    
    // Process each category in parallel
    #pragma omp parallel for if(num_theta_vals > 4)
    for (size_t i = 0; i < num_theta_vals; i++) {
        // Only process categories that have enough data (at least 2 points for std calculation)
        if (categorized_data[i].size() >= 2) {
            py::array_t<int> histogram = normalize_and_histogram(
                categorized_data[i], num_bins, min_val, max_val);
            
            #pragma omp critical
            {
                // Use the theta value as the key
                result[py::float_(theta_values[i])] = histogram;
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(normalization_histogram, m) {
    m.doc() = "Fast C++ module for normalizing arrays and creating histograms with discretized categories";
    
    m.def("normalize_and_histogram", &normalize_and_histogram, 
          "Normalize an array and create a histogram in one pass",
          py::arg("values"), py::arg("num_bins"), py::arg("min_val"), py::arg("max_val"));
    
    m.def("process_discretized_categorized_data", &process_discretized_categorized_data, 
          "Process data with discretized category mapping",
          py::arg("B"), py::arg("C"), py::arg("D"), py::arg("theta_val"), 
          py::arg("num_bins"), py::arg("min_val"), py::arg("max_val"));
}