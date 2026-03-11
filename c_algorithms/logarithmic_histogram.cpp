#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

// Small value to avoid log(0)
const double epsilon = 5;

// Utility function to find the bin index for a category value in theta_val
size_t find_bin_index(const std::vector<double>& theta_val, double val) {
    auto it = std::upper_bound(theta_val.begin(), theta_val.end(), val);
    if (it == theta_val.begin()) {
        return 0;
    } else {
        return std::distance(theta_val.begin(), it - 1);
    }
}

// Function to return the bin edges for visualization/analysis
py::array_t<double> get_log_bin_edges(int num_bins, double min_val, double max_val) {
    py::array_t<double> bin_edges(num_bins + 1);
    py::buffer_info buf = bin_edges.request();
    double *ptr = static_cast<double *>(buf.ptr);
    
    min_val = std::max(min_val, epsilon);  // Ensure min_val is positive for log scale
    
    double min_log = std::log(min_val);
    double max_log = std::log(max_val);
    
    for (int i = 0; i <= num_bins; i++) {
        double log_val = min_log + (max_log - min_log) * (static_cast<double>(i) / num_bins);
        ptr[i] = std::exp(log_val);
    }
    
    return bin_edges;
}

// Function to create a normalized logarithmic histogram (returning doubles)
py::array_t<double> log_histogram_normalized(const std::vector<double>& values, int num_bins, double min_val, double max_val) {
    // Create output histogram (using doubles for normalized values)
    py::array_t<double> histogram(num_bins);
    py::buffer_info hist_buf = histogram.request();
    double *hist_ptr = static_cast<double *>(hist_buf.ptr);
    
    // Initialize histogram bins to zero
    std::fill(hist_ptr, hist_ptr + num_bins, 0.0);
    
    min_val = std::max(min_val, epsilon);  // Ensure min_val is positive for log scale
    
    // Create logarithmically spaced bin edges
    std::vector<double> bin_edges(num_bins + 1);
    
    // Define the logarithmic bin edges from min_val to max_val
    double min_log = std::log(min_val);
    double max_log = std::log(max_val);
    
    for (int i = 0; i <= num_bins; i++) {
        double log_val = min_log + (max_log - min_log) * (static_cast<double>(i) / num_bins);
        bin_edges[i] = std::exp(log_val);
    }
    
    // Count values in each logarithmic bin (using double for counts)
    for (double value : values) {
        if (value <= min_val) {
            // Values below min_val go in the first bin
            hist_ptr[0] += 1.0;
        } else if (value >= max_val) {
            // Values exceeding max_val go in the last bin
            hist_ptr[num_bins - 1] += 1.0;
        } else {
            // Find the appropriate bin using binary search
            auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
            int bin = std::distance(bin_edges.begin(), it) - 1;
            
            // Ensure valid bin index
            if (bin >= 0 && bin < num_bins) {
                hist_ptr[bin] += 1.0;
            } else {
                // Safety: use last bin if something goes wrong
                hist_ptr[num_bins - 1] += 1.0;
            }
        }
    }
    
    // Normalize the histogram
    double total_count = 0.0;
    for (int i = 0; i < num_bins; i++) {
        total_count += hist_ptr[i];
    }
    
    if (total_count > 0) {
        for (int i = 0; i < num_bins; i++) {
            hist_ptr[i] /= total_count;
        }
    }
    
    return histogram;
}

// Main function to process all data
py::dict process_logarithmic_histograms(py::array_t<double> B, py::array_t<double> C, 
                                       py::array_t<int> D, py::array_t<double> theta_val, 
                                       int num_bins, double min_std, double max_std) {
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
    
    // Check that arrays have the same size
    if (b_buf.size != c_buf.size || b_buf.size != d_buf.size) {
        throw std::runtime_error("Arrays B, C, and D must have the same size");
    }
    
    size_t data_size = b_buf.size;
    size_t num_theta_vals = theta_buf.size;
    
    // Copy theta_val values to a vector and ensure they're sorted
    std::vector<double> theta_values(theta_ptr, theta_ptr + num_theta_vals);
    std::sort(theta_values.begin(), theta_values.end());
    
    // Calculate mean and standard deviation using Welford's algorithm
    double mean = 0.0;
    double M2 = 0.0;  // Sum of squared differences from the current mean
    size_t count = 0;
    double delta, delta2;
    
    // Single pass through the data
    for (size_t i = 0; i < data_size; i++) {
        if (d_ptr[i] == 1) {  // Only consider unmasked values
            count++;
            delta = b_ptr[i] - mean;
            mean += delta / count;
            delta2 = b_ptr[i] - mean;
            M2 += delta * delta2;
        }
    }
    
    // Calculate variance and standard deviation
    double variance = (count > 1) ? M2 / count : 0.0;
    double std_dev = std::sqrt(variance);
    
    // Set min and max values based on mean and standard deviation
    // Custom rule: trying to minimize number of useless bins. Smallest val will be mean/2
    double min_val = std::max(mean - min_std * std_dev, mean/2);  // Ensure min_val is positive for log
    double max_val = mean + max_std * std_dev;
    
    // Bin elements into their respective groups
    std::vector<std::vector<double>> categorized_data(num_theta_vals);
    
    for (size_t i = 0; i < data_size; i++) {
        if (d_ptr[i] == 1) {  // Only process if mask value is 1
            size_t bin_idx = find_bin_index(theta_values, c_ptr[i]);
            categorized_data[bin_idx].push_back(b_ptr[i]);
        }
    }
    
    // Create logarithmic histograms for each group and return results
    py::dict result;
    py::array_t<double> bin_edges = get_log_bin_edges(num_bins, min_val, max_val);
    
    result["bin_edges"] = bin_edges;  // Add bin edges to the result
    result["mean"] = py::float_(mean);
    result["std_dev"] = py::float_(std_dev);
    result["min_val"] = py::float_(min_val);
    result["max_val"] = py::float_(max_val);
    
    // Also store the counts for each A_i
    py::list counts_list;
    
    #pragma omp parallel for if(num_theta_vals > 4)
    for (size_t i = 0; i < num_theta_vals; i++) {
        size_t group_size = categorized_data[i].size();
        
        #pragma omp critical
        {
            counts_list.append(py::int_(group_size));
        }
        
        if (group_size > 0) {
            py::array_t<double> histogram = log_histogram_normalized(
                categorized_data[i], num_bins, min_val, max_val);
            
            #pragma omp critical
            {
                result[py::float_(theta_values[i])] = histogram;
            }
        }
    }
    
    result["group_counts"] = counts_list;
    
    return result;
}

PYBIND11_MODULE(logarithmic_histogram, m) {
    m.doc() = "Fast C++ module for creating normalized logarithmic histograms";
    
    m.def("process_logarithmic_histograms", &process_logarithmic_histograms, 
          "Process data and create normalized logarithmic histograms",
          py::arg("B"), py::arg("C"), py::arg("D"), py::arg("theta_val"), 
          py::arg("num_bins"), py::arg("min_std"), py::arg("max_std"));
}