import numpy as np
from numpy.fft import fft as np_fft
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf


# Numba configuration (optional, but can help debugging/performance)
# numba.config.NUMBA_NUM_THREADS = 4 # Example: Manually set thread count

def _calculate_spectral_entropy_single(series, fs=1.0):
    """
    Calculates spectral entropy for a single 1D time series using Numba.

    Internal helper function - assumes input is a valid 1D numpy array.

    Args:
        series (np.ndarray): The 1D time series (numeric dtype).
        fs (float): Sampling frequency (used for scaling, affects interpretation
                    but not the entropy value itself if consistent).

    Returns:
        float: The calculated spectral entropy (using natural log).
               Returns np.nan for invalid inputs (e.g., too short).
               Returns 0.0 for zero-variance series.
    """
    n = len(series)

    #Remove outliers from series
    corrected_series = series.copy()
    series_max = np.max(series)
    series_median = np.median(series)
    corrected_series[series > 0.95*series_max] = series_median

    # Basic check for sufficient length for FFT/PSD
    if n < 2:
        # Return 0 for empty series, NaN for single-point series (no frequency)
        return 0.0 if n == 0 else np.nan

    try:
        fft_coeffs = np.fft.fft(corrected_series)
        psd = np.abs(fft_coeffs[0:n // 2 + 1]**2)
    except ValueError:
        # Catch potential errors during PSD calculation (e.g., unexpected input)
        return np.nan

    # Normalize the PSD to form a probability distribution
    psd_sum = np.sum(psd)

    # Handle edge cases: constant signal (all power in DC or zero variance)
    if psd_sum <= 1e-12: # Use a small threshold for floating point precision
        return 0.0

    psd_norm = psd / psd_sum

    # Calculate Shannon Entropy
    # Filter out zero probabilities before taking the log

    non_zero_psd = psd_norm[psd_norm > 0]
    # Check if filtering resulted in an empty array (shouldn't if psd_sum > 0)
    if len(non_zero_psd) == 0:
        return 0.0 # Entropy is zero if there's effectively only one non-zero bin (delta function) after normalization
    
    spectral_entropy = -np.sum(non_zero_psd*np.log(non_zero_psd))

    # Optional: Normalize entropy (0 to 1 range)
    max_entropy = np.log(len(psd))
    if max_entropy > 0:
        spectral_entropy /= max_entropy

    #Calculate high frequency entropy
    freqs = np.fft.fftfreq(len(series), 1)
    freqs = freqs[0:len(freqs)//2 + 1]
    non_zero_freqs = freqs[psd_norm > 0]
    high_freqs_idx = np.abs(non_zero_freqs-0.2).argmin()

    high_spectral_entropy = -np.sum(non_zero_psd[high_freqs_idx:]*np.log(non_zero_psd[high_freqs_idx:]))
    max_high_entropy = np.log(len(non_zero_psd[high_freqs_idx:]))
    
    high_spectral_entropy /= max_high_entropy

    return spectral_entropy, high_spectral_entropy


def _calculate_spectral_hist(series, fs=1.0, bins=10, low_freq_cutoff=0.1):
    """
    Calculates normalized spectral histogram for a single 1D time series,
    excluding the lowest specified percentage of frequencies.
    
    Args:
        series (np.ndarray): The 1D time series (numeric dtype).
        fs (float): Sampling frequency.
        bins (int): Number of bins for the histogram.
        low_freq_cutoff (float): Fraction of lowest frequencies to exclude (0.0-1.0).
                                Default is 0.1 (10%).
        
    Returns:
        tuple: (bin_centers, histogram_values) where histogram_values sum to 1.0
               Returns (None, None) for invalid inputs.
    """
    n = len(series)

    #Remove outliers from series
    corrected_series = series.copy()
    series_max = np.max(series)
    series_median = np.median(series)
    corrected_series[series > 0.99*series_max] = series_median

    # Basic check for sufficient length for FFT/PSD
    if n < 2:
        # Return None for invalid series
        return None, None

    try:
        # Calculate FFT and PSD
        fft_coeffs = np.fft.rfft(corrected_series)
        psd = np.abs(fft_coeffs**2)
        
        # Get frequency values
        freqs = np.fft.rfftfreq(n, 1/fs)
        
        # Handle edge cases: constant signal
        psd_sum = np.sum(psd)
        if psd_sum <= 1e-12:
            hist_values = np.zeros(bins)
            hist_values[0] = 1.0
            bin_edges = np.linspace(0, np.max(freqs), bins+1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers, hist_values
        
        # Calculate the cutoff index based on low_freq_cutoff
        cutoff_idx = max(1, int(len(freqs) * low_freq_cutoff))
        
        # Exclude the lowest frequencies
        freqs_filtered = freqs[cutoff_idx:]
        psd_filtered = psd[cutoff_idx:]
        
        # Normalize the filtered PSD
        psd_filtered_sum = np.sum(psd_filtered)
        
        # If there's no power in the remaining frequencies, return zeros
        if psd_filtered_sum <= 1e-12:
            hist_values = np.zeros(bins)
            bin_edges = np.linspace(np.min(freqs_filtered), np.max(freqs_filtered), bins+1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers, hist_values
            
        psd_filtered_norm = psd_filtered / psd_filtered_sum
        
        # Create histogram by binning the filtered normalized PSD
        hist_values, bin_edges = np.histogram(
            freqs_filtered, 
            bins=bins, 
            range=(np.min(freqs_filtered), np.max(freqs_filtered)),
            weights=psd_filtered_norm
        )
        
        # Renormalize the histogram to ensure it sums to 1.0
        hist_sum = np.sum(hist_values)
        if hist_sum > 0:
            hist_values = hist_values / hist_sum
            
        # Calculate bin centers for return
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, hist_values
        
    except ValueError as e:
        # Catch potential errors during calculation
        print(f"Error in spectral histogram calculation: {e}")
        return None, None


def fast_spectral_entropy(list_of_series, fs=1.0):
    """
    Calculates spectral entropy for a list of 1D NumPy time series using
    Numba for parallel execution.

    Input series should be 1D NumPy arrays of numeric type.

    Args:
        list_of_series (list): A list where each element is a 1D NumPy array
                               representing a time series. Using a Numba typed
                               list is technically possible but standard lists
                               are often easier to work with initially.
        fs (float, optional): Sampling frequency of the time series. Defaults to 1.0.

    Returns:
        numpy.ndarray: A NumPy array containing the spectral entropy for each
                       input time series in the same order. Returns np.nan
                       for any series where calculation failed (e.g., too short
                       or not a valid NumPy array if checks were bypassed).
    """
    entropies = np.zeros(len(list_of_series), dtype=float) # Pre-allocate numpy array

    for i, series in enumerate(list_of_series):
        # Basic check: Ensure it's a numpy array and 1D
        if not isinstance(series, np.ndarray) or series.ndim != 1:
            print(f"Warning: Item at index {i} is not a 1D numpy array. Result will be NaN.")
            entropies[i] = np.nan
            continue

        entropies[i], _ = _calculate_spectral_entropy_single(series, fs=fs)

    return entropies

def fast_spectra_binning(list_of_series, fs=1.0, bins=10):
    """
    Calculates spectral entropy for a list of 1D NumPy time series using
    Numba for parallel execution.

    Input series should be 1D NumPy arrays of numeric type.

    Args:
        list_of_series (list): A list where each element is a 1D NumPy array
                               representing a time series. Using a Numba typed
                               list is technically possible but standard lists
                               are often easier to work with initially.
        fs (float, optional): Sampling frequency of the time series. Defaults to 1.0.
        bins (int, optiona): Number of bins for histogram


    Returns:
        numpy.ndarray: A NumPy array containing the spectral entropy for each
                       input time series in the same order. Returns np.nan
                       for any series where calculation failed (e.g., too short
                       or not a valid NumPy array if checks were bypassed).
    """
    hist_vals = np.zeros((len(list_of_series),bins)) # Pre-allocate numpy array
    hist_centers = np.zeros((len(list_of_series),bins))

    for i, series in enumerate(list_of_series):
        # Basic check: Ensure it's a numpy array and 1D
        if not isinstance(series, np.ndarray) or series.ndim != 1:
            print(f"Warning: Item at index {i} is not a 1D numpy array. Result will be NaN.")
            hist_vals[i] = np.nan
            continue

        hist_centers[i], hist_vals[i] = _calculate_spectral_hist(series, fs=fs,bins = bins)

    return hist_centers, hist_vals

#ACF calculations

def calculate_acf_entropy(series, max_lag, method='abs'):
    """
    Calculates the entropy of the Autocorrelation Function (ACF) up to a
    specified maximum lag.

    This measures how "concentrated" or "spread out" the correlations are
    across different lags. Low entropy suggests strong correlation concentrated
    at specific lags (more periodic/regular structure). High entropy suggests
    correlation is weak or spread across many lags (more noisy/irregular).

    Args:
        series (np.ndarray): The 1D time series (should be numeric).
        max_lag (int): The maximum lag to include in the ACF calculation
                       (must be >= 1 and < len(series)).
        method (str, optional): How to treat ACF values for distribution.
                                'abs': Use absolute values (default).
                                'squared': Use squared values.

    Returns:
        float: The calculated Shannon entropy of the ACF magnitudes (using
               natural log). Returns np.nan for invalid inputs or edge cases
               where calculation is not meaningful. Returns 0.0 for constant
               signals or if ACF magnitudes sum to zero after lag 0.
    """
    # --- Input Validation and Edge Cases ---
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        # print("Warning: Input series must be a 1D NumPy array.")
        return np.nan
    if not np.issubdtype(series.dtype, np.number):
        # print("Warning: Input series must have a numeric data type.")
        return np.nan
    if np.iscomplexobj(series):
        # print("Warning: Input series is complex. Taking magnitude.")
        series = np.abs(series) # Or handle as needed

    n = len(series)
    #Remove outliers from series
    corrected_series = series.copy()
    series_max = np.max(series)
    series_median = np.median(series)
    corrected_series[series > 0.95*series_max] = series_median

    if n < 2:
        # print("Warning: Series length must be at least 2.")
        return np.nan # Need at least 2 points for any correlation

    if not isinstance(max_lag, int) or max_lag < 1:
        # print(f"Warning: max_lag ({max_lag}) must be an integer >= 1.")
        return np.nan

    if max_lag >= n:
        # print(f"Warning: max_lag ({max_lag}) must be less than series length ({n}).")
        return np.nan

    # Check for constant series (zero variance) to avoid potential NaN from acf
    series_std = np.std(corrected_series)
    if np.isclose(series_std, 0):
        # print("Debug: Series is constant. ACF Entropy is 0.")
        return 0.0 # No structure beyond lag 0

    # --- Calculate ACF using statsmodels ---
    try:
        # nlags specifies the number of lags to return (0 to max_lag)
        # fft=True uses FFT for faster computation
        # missing='raise' ensures no NaNs in input (handle beforehand if needed)
        acf_values = acf(corrected_series, nlags=max_lag, fft=True, missing='raise')

        # Check if acf calculation returned NaNs (can happen with some inputs)
        if np.isnan(acf_values).any():
             # print("Warning: ACF calculation resulted in NaN values.")
             return np.nan

    except Exception as e:
        # Catch potential errors during ACF calculation
        # print(f"Error during ACF calculation: {e}")
        return np.nan

    # --- Process ACF for Entropy Calculation ---
    # Select lags from 1 to max_lag (exclude lag 0, which is always 1)
    selected_acf = acf_values[1 : max_lag + 1]

    # Calculate magnitudes based on the chosen method
    if method == 'abs':
        acf_mags = np.abs(selected_acf)
    elif method == 'squared':
        acf_mags = selected_acf**2
    else:
        # print(f"Warning: Invalid method '{method}'. Using 'abs'.")
        acf_mags = np.abs(selected_acf)

    # --- Normalize Magnitudes to form pseudo-probability distribution ---
    mags_sum = np.sum(acf_mags)

    # If sum is close to zero, correlations beyond lag 0 are negligible.
    # Entropy is effectively zero (perfect concentration at lag 0, which we ignore).
    if np.isclose(mags_sum, 0):
        # print("Debug: Sum of ACF magnitudes (lags > 0) is near zero. Entropy is 0.")
        return 0.0

    p_acf = acf_mags / mags_sum

    # --- Calculate Shannon Entropy ---
    # Filter out zero probabilities before taking log to avoid -inf * 0 = NaN
    p_non_zero = p_acf[p_acf > 1e-12] # Use tolerance for floating point precision

    # If filtering leaves nothing (shouldn't happen if mags_sum > 0, but safety check)
    if len(p_non_zero) == 0:
        return 0.0

    # Entropy H(X) = - sum(p(x) * log(p(x))) (using natural log)
    H_acf = -np.sum(p_non_zero * np.log(p_non_zero))

    return H_acf




