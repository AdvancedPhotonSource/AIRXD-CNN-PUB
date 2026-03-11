#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

namespace py = pybind11;

// Function to bin pixels azimuthally and preserve ring structure with optimal memory usage
std::vector<std::vector<float>> azimuthal_binning(
    const float* B,       // 2D image array flattened in row-major format
    const float* C,       // 2D categorization array flattened in row-major format
    const bool* D,        // Binary mask array (1 = exclude, 0 = include)
    int height,           // Height of the 2D arrays
    int width,            // Width of the 2D arrays
    const float* theta_val, // Array of bin boundaries
    int num_theta_vals    // Length of theta_val array
) {
    // Center of the image
    //float center_y = height / 2.0f;
    float center_x = width / 2.0f;
    
    int num_bins = num_theta_vals - 1;
    
    // Optimized approach: Just 2 sectors (left and right)
    std::vector<std::vector<std::vector<float>>> sector_bins(num_bins, 
                                                          std::vector<std::vector<float>>(2));
    
    // Pre-allocate memory for bins
    int avg_bin_size = (height * width) / (num_bins * 2);
    for (auto& bin : sector_bins) {
        for (auto& sector : bin) {
            sector.reserve(avg_bin_size);
        }
    }
    
    // Process the image
    #pragma omp parallel
    {
        // Thread-local storage to minimize critical sections
        std::vector<std::vector<std::vector<float>>> local_bins(num_bins, 
                                                             std::vector<std::vector<float>>(2));
                                                             
        // Initialize local storage
        for (auto& bin : local_bins) {
            for (auto& sector : bin) {
                sector.reserve(avg_bin_size/omp_get_num_threads());
            }
        }
        
        #pragma omp for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                
                // Skip masked pixels (mask value of 1)
                if (D[idx]) continue;
                
                float theta = C[idx];
                
                // Find the bin for this theta value
                // for (int bin = 0; bin < num_bins; ++bin) {
                //     if (theta >= theta_val[bin] && theta < theta_val[bin + 1]) {
                //         // Calculate position relative to center
                //         float dx = x - center_x;
                        
                //         // Simple check for left (0) or right (1) sector
                //         // Always use push_back for speed (we'll reverse the left side later)
                //         int sector = (dx < 0) ? 0 : 1;
                        
                //         // Always use fast push_back operation
                //         local_bins[bin][sector].push_back(B[idx]);
                //         break;
                //     }
                // }
                // Find the appropriate bin using binary search
                // Find the appropriate bin using binary search
                auto it = std::lower_bound(theta_val, theta_val + num_theta_vals, theta);
                int bin = std::distance(theta_val, it) - 1;
                
                // Check if the pixel falls within valid range
                if (bin >= 0 && bin < num_bins && theta >= theta_val[bin] && 
                    (bin == num_bins - 1 ? theta <= theta_val[bin + 1] : theta < theta_val[bin + 1])) {
                    
                    // Calculate position relative to center
                    float dx = x - center_x;
                    
                    // Simple check for left (0) or right (1) sector
                    int sector = (dx < 0) ? 0 : 1;
                    
                    // Always use fast push_back operation
                    local_bins[bin][sector].push_back(B[idx]);
                }
            }
        }
        
        // Merge thread-local results into global bins
        #pragma omp critical
        {
            for (int bin = 0; bin < num_bins; ++bin) {
                for (int sector = 0; sector < 2; ++sector) {
                    sector_bins[bin][sector].insert(
                        sector_bins[bin][sector].end(),
                        local_bins[bin][sector].begin(),
                        local_bins[bin][sector].end()
                    );
                }
            }
        }
    }
    
    // Reverse the left-side sectors to maintain clockwise ordering
    #pragma omp parallel for
    for (int bin = 0; bin < num_bins; ++bin) {
        // Sector 0 is left side - reverse it once (O(n) operation)
        std::reverse(sector_bins[bin][0].begin(), sector_bins[bin][0].end());
    }
    
    // Combine sectors into complete rings in proper azimuthal order
    std::vector<std::vector<float>> result(num_bins);
    
    for (int bin = 0; bin < num_bins; ++bin) {
        // Calculate total size
        size_t left_size = sector_bins[bin][0].size();
        size_t right_size = sector_bins[bin][1].size();
        size_t total_size = left_size + right_size;
        
        // Pre-allocate memory
        result[bin].reserve(total_size);
        
        // Right sector (1) followed by left sector (0) for clockwise ordering
        result[bin].insert(
            result[bin].end(),
            sector_bins[bin][1].begin(),
            sector_bins[bin][1].end()
        );
        
        result[bin].insert(
            result[bin].end(),
            sector_bins[bin][0].begin(),
            sector_bins[bin][0].end()
        );
    }
    
    return result;
}

// Function to calculate global mean of the image array with mask
float calculate_mean(const float* array, const bool* mask, int size) {
    double sum = 0.0;
    int count = 0;
    
    #pragma omp parallel
    {
        double local_sum = 0.0;
        int local_count = 0;
        
        #pragma omp for nowait
        for (int i = 0; i < size; ++i) {
            // Only include unmasked pixels (mask value of 0)
            if (!mask[i]) {
                local_sum += array[i];
                local_count++;
            }
        }
        
        #pragma omp critical
        {
            sum += local_sum;
            count += local_count;
        }
    }
    
    return (count > 0) ? static_cast<float>(sum / count) : 0.0f;
}

// Function to count spikes in a 1D array with rolling window
float count_spikes(const std::vector<float>& array, float threshold, int window_size) {
    if (array.empty()) return 0.0f;
    
    int spike_count = 0;
    bool in_signal = false;
    
    // Ensure window_size is valid
    window_size = std::max(1, std::min(window_size, static_cast<int>(array.size())));
    
    // Circular buffer to store recent values
    std::vector<float> recent_values(window_size, 0.0f);
    int buffer_pos = 0;
    
    // Initialize buffer with wrap-around if needed
    for (int i = 0; i < window_size; ++i) {
        // Use modulo to handle wraparound for initial values
        recent_values[i] = array[i % array.size()];
    }
    
    for (size_t i = 0; i < array.size(); ++i) {
        // Update circular buffer
        recent_values[buffer_pos] = array[i];
        buffer_pos = (buffer_pos + 1) % window_size;
        
        // Check if all values in window are above/below threshold
        bool all_above = true;
        bool all_below = true;
        
        for (float rv : recent_values) {
            if (rv <= threshold) all_above = false;
            if (rv > threshold) all_below = false;
        }
        
        if (!in_signal && all_above) {
            // Transition from background to signal
            spike_count++;
            in_signal = true;
        } else if (in_signal && all_below) {
            // Transition from signal to background
            in_signal = false;
        }
    }
    
    // Return spike density (count divided by array length)
    return static_cast<float>(spike_count) / array.size();
}

// Function to count spikes for all bins using global mean threshold
std::vector<float> count_all_spikes(
    const std::vector<std::vector<float>>& binned_arrays,
    float global_mean,
    int window_size = 5) {
    
    std::vector<float> result(binned_arrays.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < binned_arrays.size(); ++i) {
        result[i] = count_spikes(binned_arrays[i], global_mean, window_size);
    }
    
    return result;
}

// Wrapper function for azimuthal_binning that accepts numpy arrays
py::tuple py_azimuthal_binning(
    py::array_t<float> B,
    py::array_t<float> C,
    py::array_t<float> theta_val,
    py::array_t<bool> D = py::none()) {
    
    py::buffer_info B_buf = B.request();
    py::buffer_info C_buf = C.request();
    py::buffer_info theta_buf = theta_val.request();
    
    if (B_buf.ndim != 2 || C_buf.ndim != 2) {
        throw std::runtime_error("Input arrays B and C must be 2-dimensional");
    }
    
    if (B_buf.shape[0] != C_buf.shape[0] || B_buf.shape[1] != C_buf.shape[1]) {
        throw std::runtime_error("Input arrays B and C must have the same shape");
    }
    
    int height = B_buf.shape[0];
    int width = B_buf.shape[1];
    int num_theta_vals = theta_buf.shape[0];
    
    // Mask
    // Validate mask dimensions
    bool* mask_ptr = nullptr;
    py::buffer_info D_buf = D.request();
    if (D_buf.ndim != 2 || D_buf.shape[0] != height || D_buf.shape[1] != width) {
        throw std::runtime_error("Mask D must have the same dimensions as B and C");
    }
    mask_ptr = static_cast<bool*>(D_buf.ptr);

    
    // Calculate global mean of the image (using only unmasked pixels)
    float global_mean = calculate_mean(
        static_cast<float*>(B_buf.ptr),
        mask_ptr,
        height * width
    );
    
    auto binned = azimuthal_binning(
        static_cast<float*>(B_buf.ptr),
        static_cast<float*>(C_buf.ptr),
        mask_ptr,
        height,
        width,
        static_cast<float*>(theta_buf.ptr),
        num_theta_vals
    );
    
    // Convert the result to a list of numpy arrays
    py::list result;
    for (const auto& bin : binned) {
        py::array_t<float> arr(bin.size());
        auto buf = arr.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        std::memcpy(ptr, bin.data(), bin.size() * sizeof(float));
        result.append(arr);
    }
    
    // Return both the binned arrays and the global mean
    return py::make_tuple(result, global_mean);
}

// Wrapper function for count_all_spikes
py::array_t<float> py_count_all_spikes(
    py::list binned_arrays,
    float threshold,
    int window_size = 5) {
    
    std::vector<std::vector<float>> cpp_binned;
    cpp_binned.reserve(py::len(binned_arrays));
    
    for (size_t i = 0; i < py::len(binned_arrays); ++i) {
        py::array_t<float> arr = binned_arrays[i].cast<py::array_t<float>>();
        auto buf = arr.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        std::vector<float> bin(ptr, ptr + buf.size);
        cpp_binned.push_back(std::move(bin));
    }
    
    auto spike_counts = count_all_spikes(cpp_binned, threshold, window_size);
    
    // Convert the result to a numpy array
    py::array_t<float> result(spike_counts.size());
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    std::memcpy(ptr, spike_counts.data(), spike_counts.size() * sizeof(float));
    
    return result;
}

// Extracts pixels that fall within specified two-theta peak ranges and orders them azimuthally.
std::vector<std::vector<float>> extract_averaged_patterns(
    const float* B,       // 2D image intensity array
    const float* C,       // 2D azimuthal angle map (-pi to pi)
    const float* T,       // 2D two-theta map array
    const bool* D,        // Binary mask array (true = exclude)
    int height,
    int width,
    const std::vector<std::pair<float, float>>& peak_bounds,
    int num_azimuthal_bins
) {
    int num_peaks = peak_bounds.size();
    if (num_peaks == 0) return {};

    using BinType = std::pair<double, int>;
    std::vector<std::vector<BinType>> global_bins(num_peaks, std::vector<BinType>(num_azimuthal_bins, {0.0, 0}));

    #pragma omp parallel
    {
        std::vector<std::vector<BinType>> local_bins(num_peaks, std::vector<BinType>(num_azimuthal_bins, {0.0, 0}));

        #pragma omp for collapse(2) nowait
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                if (D[idx]) continue;

                float two_theta = T[idx];
                for (int i = 0; i < num_peaks; ++i) {
                    if (two_theta >= peak_bounds[i].first && two_theta <= peak_bounds[i].second) {
                        float azimuth = C[idx];

                        // --- THE FIX: Use 'double' for the normalization calculation ---
                        double normalized_azimuth = (static_cast<double>(azimuth) + M_PI) / (2.0 * M_PI);
                        // --- END OF FIX ---
                        
                        int az_bin_idx = static_cast<int>(normalized_azimuth * num_azimuthal_bins);
                        az_bin_idx = std::max(0, std::min(num_azimuthal_bins - 1, az_bin_idx));

                        local_bins[i][az_bin_idx].first += B[idx];
                        local_bins[i][az_bin_idx].second += 1;
                        
                        break;
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < num_peaks; ++i) {
                for (int j = 0; j < num_azimuthal_bins; ++j) {
                    global_bins[i][j].first += local_bins[i][j].first;
                    global_bins[i][j].second += local_bins[i][j].second;
                }
            }
        }
    }

    // Averaging phase remains the same
    std::vector<std::vector<float>> final_patterns(num_peaks, std::vector<float>(num_azimuthal_bins, 0.0f));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_peaks; ++i) {
        for (int j = 0; j < num_azimuthal_bins; ++j) {
            const auto& bin = global_bins[i][j];
            if (bin.second > 0) {
                final_patterns[i][j] = static_cast<float>(bin.first / bin.second);
            }
        }
    }

    return final_patterns;
}

// Wrapper for the new averaged extraction function
py::list py_extract_peak_patterns(
    py::array_t<float> B,
    py::array_t<float> C,
    py::array_t<float> T,
    py::array_t<bool> D,
    py::list py_peak_bounds,
    int num_azimuthal_bins = 3000) {

    // ... (Input validation is the same as before) ...
    py::buffer_info b_buf = B.request();
    py::buffer_info c_buf = C.request();
    py::buffer_info t_buf = T.request();
    py::buffer_info d_buf = D.request();

    if (b_buf.ndim != 2 || c_buf.ndim != 2 || t_buf.ndim != 2 || d_buf.ndim != 2) {
        throw std::runtime_error("All input arrays must be 2D");
    }
    
    int height = b_buf.shape[0];
    int width = b_buf.shape[1];

    std::vector<std::pair<float, float>> peak_bounds;
    peak_bounds.reserve(py::len(py_peak_bounds));
    for (const auto& item : py_peak_bounds) {
        py::list bound = item.cast<py::list>();
        peak_bounds.push_back({bound[0].cast<float>(), bound[1].cast<float>()});
    }

    // Call the core C++ function
    auto patterns = extract_averaged_patterns(
        static_cast<float*>(b_buf.ptr),
        static_cast<float*>(c_buf.ptr),
        static_cast<float*>(t_buf.ptr),
        static_cast<bool*>(d_buf.ptr),
        height,
        width,
        peak_bounds,
        num_azimuthal_bins
    );

    // Convert the result back to a Python list of NumPy arrays
    py::list result;
    for (const auto& pattern : patterns) {
        py::array_t<float> arr(pattern.size());
        auto buf = arr.request();
        std::memcpy(static_cast<float*>(buf.ptr), pattern.data(), pattern.size() * sizeof(float));
        result.append(arr);
    }

    return result;
}



PYBIND11_MODULE(azimuthal_processing, m) {
    m.doc() = "Fast azimuthal processing of X-ray images";
    
    m.def("azimuthal_binning", &py_azimuthal_binning,
        py::arg("B"),
        py::arg("C"),
        py::arg("theta_val"),
        py::arg("D") = py::none(),
        "Bin pixels azimuthally and preserve ring structure. Returns (binned_arrays, global_mean). "
        "Optional mask D (True=exclude, False=include)."
    );
    
    m.def("count_spikes", &py_count_all_spikes,
        py::arg("binned_arrays"),
        py::arg("threshold"),
        py::arg("window_size") = 5,
        "Count spikes in azimuthally binned arrays using the global mean as threshold and a rolling window"
    );

    m.def("extract_peak_patterns", &py_extract_peak_patterns,
        py::arg("B"),           // Intensity image
        py::arg("C"),           // Azimuthal map
        py::arg("T"),           // two-theta map
        py::arg("D"),           // mask
        py::arg("peak_bounds"), // list of [min, max]
        py::arg("num_azimuthal_bins") = 3000, // number of bins
        "Extracts thick ring patterns defined by two-theta bounds and sorts them azimuthally."
    );
}