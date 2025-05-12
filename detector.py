# spt_analyzer/tracking/detector.py
"""
Particle detector module for SPT Analysis.

This module provides functions for detecting particles in images using various methods,
including subpixel refinement options.
"""

import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from skimage.feature import blob_dog, blob_log, blob_doh, peak_local_max
from skimage.filters import gaussian, laplace
import logging
from typing import Optional, Dict, Any, Tuple, Callable

# --- Optional Dependency: PyWavelets ---
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False
    logging.getLogger(__name__).warning(
        "PyWavelets (pywt) not found. 'direct' wavelet detection method will not be available."
    )
# --------------------------------------

# --- Optional Dependency: scikit-image restoration ---
# Keep this check as it's used by the 'residual' wavelet method
try:
    from skimage.restoration import denoise_wavelet, estimate_sigma
    _HAS_SKIMAGE_RESTORATION = True
except ImportError:
     _HAS_SKIMAGE_RESTORATION = False
     logging.getLogger(__name__).warning(
        "skimage.restoration not found. 'residual' wavelet detection method will not be available."
     )
# -----------------------------------------------------


logger = logging.getLogger(__name__)

# --- 2D Gaussian Function for Fitting ---
def _gaussian_2d(coords, amplitude, y0, x0, sigma_y, sigma_x, theta, offset):
    """2D Gaussian function for fitting."""
    y, x = coords
    y0 = float(y0)
    x0 = float(x0)
    a = (np.cos(theta)**2)/(2*sigma_y**2) + (np.sin(theta)**2)/(2*sigma_x**2)
    b = -(np.sin(2*theta))/(4*sigma_y**2) + (np.sin(2*theta))/(4*sigma_x**2)
    c = (np.sin(theta)**2)/(2*sigma_y**2) + (np.cos(theta)**2)/(2*sigma_x**2)
    g = offset + amplitude*np.exp( - (a*((y-y0)**2) + 2*b*(y-y0)*(x-x0) + c*((x-x0)**2)))
    return g.ravel()
# ----------------------------------------

# --- Subpixel Refinement Functions ---
def _refine_com(image_window: np.ndarray) -> Tuple[float, float]:
    """Refine position using Center of Mass on a window."""
    if image_window.size == 0 or np.all(image_window <= 0):
        # Window is empty or has no intensity, return center
        return (image_window.shape[0] - 1) / 2.0, (image_window.shape[1] - 1) / 2.0

    total = np.sum(image_window)
    if total <= 0: # Avoid division by zero if sum is not positive
        return (image_window.shape[0] - 1) / 2.0, (image_window.shape[1] - 1) / 2.0

    y_indices, x_indices = np.indices(image_window.shape)
    y_cm_rel = np.sum(y_indices * image_window) / total
    x_cm_rel = np.sum(x_indices * image_window) / total
    return y_cm_rel, x_cm_rel

def _refine_gaussian_2d(image_window: np.ndarray) -> Tuple[float, float]:
    """Refine position using 2D Gaussian fitting on a window."""
    y_center_init, x_center_init = (image_window.shape[0] - 1) / 2.0, (image_window.shape[1] - 1) / 2.0

    if image_window.size == 0 or image_window.max() <= 0:
        return y_center_init, x_center_init # Cannot fit empty or flat window

    y_indices, x_indices = np.indices(image_window.shape)

    # Initial parameter guesses
    amplitude_init = image_window.max() - image_window.min()
    sigma_init = max(image_window.shape) / 4.0 # Rough guess
    offset_init = image_window.min()
    initial_guess = (amplitude_init, y_center_init, x_center_init, sigma_init, sigma_init, 0, offset_init)

    # Define bounds to constrain the fit
    bounds = (
        [0, 0, 0, 0.5, 0.5, -np.pi/2, image_window.min()-1e-6], # Lower bounds (allow slightly negative offset)
        [amplitude_init*2, image_window.shape[0]-1, image_window.shape[1]-1, image_window.shape[0], image_window.shape[1], np.pi/2, image_window.max()+1e-6] # Upper bounds
    )

    try:
        popt, pcov = curve_fit(
            _gaussian_2d,
            (y_indices.ravel(), x_indices.ravel()), # Pass coordinates correctly
            image_window.ravel(),
            p0=initial_guess,
            bounds=bounds,
            maxfev=5000 # Increase iterations if needed
        )
        # Return fitted center coordinates relative to window origin
        y_fit, x_fit = popt[1], popt[2]
        # Basic check if fit is reasonable (within window bounds)
        if not (0 <= y_fit < image_window.shape[0] and 0 <= x_fit < image_window.shape[1]):
             logger.debug("Gaussian fit center outside window bounds, falling back to CoM.")
             return _refine_com(image_window)
        return y_fit, x_fit

    except (RuntimeError, ValueError, TypeError) as e:
        # If fit fails, fall back to Center of Mass
        logger.debug(f"Gaussian fit failed: {e}. Falling back to Center of Mass.")
        return _refine_com(image_window)

def refine_positions(image: np.ndarray, coordinates: np.ndarray, diameter: int,
                     subpixel_method: str = 'com') -> np.ndarray:
    """
    Refine particle positions to subpixel accuracy.

    Parameters
    ----------
    image : ndarray
        Input image used for refinement (should be original or minimally processed).
    coordinates : ndarray
        Initial integer or float coordinates (n, 2) from peak finding.
    diameter : int
        Approximate particle diameter, used to define refinement window size.
    subpixel_method : str, optional
        Method for subpixel refinement ('com' or 'gaussian'), by default 'com'.

    Returns
    -------
    ndarray
        Refined coordinates (n, 2) with subpixel accuracy.
    """
    if not subpixel_method or subpixel_method.lower() not in ['com', 'gaussian']:
        logger.warning(f"Invalid subpixel_method '{subpixel_method}', defaulting to 'com'.")
        subpixel_method = 'com'

    if subpixel_method == 'gaussian' and not _HAS_PYWT: # Gaussian fitting needs curve_fit from scipy
         logger.warning("Gaussian subpixel refinement requires SciPy. Falling back to 'com'.")
         subpixel_method = 'com'


    refined = np.zeros_like(coordinates, dtype=float)
    # Window size for refinement (use odd window size centered on initial coordinate)
    window_radius = int(diameter / 2) + 1
    if window_radius < 1: window_radius = 1
    window_size = window_radius * 2 + 1 # Ensure odd size

    # Pad image to handle refinement near borders
    pad_width = window_radius + 1 # Padding slightly larger than radius
    padded_image = np.pad(image, pad_width, mode='reflect') # Use reflect padding

    # Process each coordinate
    for i, (y, x) in enumerate(coordinates):
        # Use rounded integer coords as center for window extraction
        # Adjust coordinates for padding
        y_int_padded, x_int_padded = int(round(y)) + pad_width, int(round(x)) + pad_width

        # Define window boundaries in padded image
        y_min_pad = y_int_padded - window_radius
        y_max_pad = y_int_padded + window_radius + 1 # Exclusive
        x_min_pad = x_int_padded - window_radius
        x_max_pad = x_int_padded + window_radius + 1 # Exclusive

        # Ensure window boundaries are valid (should be due to padding)
        if not (0 <= y_min_pad < padded_image.shape[0] and 0 <= y_max_pad <= padded_image.shape[0] and
                0 <= x_min_pad < padded_image.shape[1] and 0 <= x_max_pad <= padded_image.shape[1]):
             logger.warning(f"Window calculation error for coord ({y},{x}). Using original coord.")
             refined[i] = [float(y), float(x)]
             continue

        window = padded_image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

        # Apply chosen refinement method
        if subpixel_method == 'gaussian':
            y_cm_rel, x_cm_rel = _refine_gaussian_2d(window)
        else: # Default to 'com'
            y_cm_rel, x_cm_rel = _refine_com(window)

        # Convert back to original image coordinates
        refined[i, 0] = (y_int_padded - window_radius + y_cm_rel) - pad_width
        refined[i, 1] = (x_int_padded - window_radius + x_cm_rel) - pad_width

    return refined
# -----------------------------------


class ParticleDetector:
    """
    Detector for particles in images.

    Provides methods for detecting particles using various algorithms.
    """

    def __init__(self, method='gaussian', threshold=0.5, threshold_is_relative=True,
                 min_distance=5, diameter=7, subpixel=True, subpixel_method='com', **kwargs):
        """
        Initialize the particle detector.

        Parameters
        ----------
        method : str, optional
            Detection method ('gaussian', 'laplacian', 'doh', 'log', 'wavelet').
            Default is 'gaussian'.
        threshold : float, optional
            Threshold for detection. Interpretation depends on `threshold_is_relative`.
            Default is 0.5.
        threshold_is_relative : bool, optional
            If True, `threshold` is relative to max intensity of filtered image (0-1).
            If False, `threshold` is an absolute value. Default is True.
        min_distance : int, optional
            Minimum distance between detected particle centers in pixels. Default is 5.
        diameter : int, optional
            Expected particle diameter in pixels (used for sigma estimation etc.). Default is 7.
        subpixel : bool, optional
            Whether to perform subpixel localization. Default is True.
        subpixel_method : str, optional
            Method for subpixel refinement ('com' or 'gaussian'). Default is 'com'.
        **kwargs
            Additional keyword arguments passed to the specific detection function
            (e.g., `wavelet_type`, `wavelet_levels`, `wavelet_enhancement_method`,
            `overlap` for blob detectors).
        """
        self.method = method
        self.threshold = threshold
        self.threshold_is_relative = threshold_is_relative
        self.min_distance = min_distance
        self.diameter = diameter
        self.subpixel = subpixel
        self.subpixel_method = subpixel_method
        self.kwargs = kwargs

        # Set up logger
        self.logger = logging.getLogger(__name__)

        # Validate method
        if method not in ['gaussian', 'laplacian', 'doh', 'log', 'wavelet']:
            raise ValueError(f"Unknown detection method: {method}")
        if method == 'wavelet' and (not _HAS_PYWT and kwargs.get('wavelet_enhancement_method','residual')=='direct'):
             raise ImportError("PyWavelets (pywt) is required for the 'direct' wavelet method.")
        if method == 'wavelet' and (not _HAS_SKIMAGE_RESTORATION and kwargs.get('wavelet_enhancement_method','residual')=='residual'):
             raise ImportError("skimage.restoration is required for the 'residual' wavelet method.")


    def detect(self, image):
        """
        Detect particles in an image.

        Parameters
        ----------
        image : ndarray
            Input image (2D).

        Returns
        -------
        ndarray
            Array of particle positions, shape (n, 2) for n particles.
            Each row contains (y, x) coordinates. Returns empty array if detection fails.
        """
        if image is None or not isinstance(image, np.ndarray) or image.ndim != 2:
            self.logger.error("Invalid input image provided. Must be a 2D NumPy array.")
            return np.empty((0, 2)) # Return empty array

        try:
            # Get detection function
            detect_func = get_detection_function(self.method)

            # Prepare parameters
            params = {
                'threshold': self.threshold,
                'threshold_is_relative': self.threshold_is_relative,
                'min_distance': self.min_distance,
                'diameter': self.diameter,
                'subpixel': self.subpixel,
                'subpixel_method': self.subpixel_method,
                **self.kwargs # Pass through other kwargs
            }

            # Detect particles
            positions = detect_func(image, **params)

            # Ensure output is a NumPy array
            if positions is None:
                 positions = np.empty((0, 2))
            elif not isinstance(positions, np.ndarray):
                 positions = np.array(positions)

            # Ensure correct shape
            if positions.ndim == 1 and positions.shape[0] == 2:
                 positions = positions.reshape(1, 2) # Handle single detection case
            elif positions.ndim != 2 or positions.shape[1] != 2:
                 if positions.size == 0: # Handle empty case
                      positions = np.empty((0, 2))
                 else:
                      self.logger.error(f"Detection function '{self.method}' returned unexpected shape: {positions.shape}")
                      return np.empty((0, 2))


            self.logger.info(f"Detected {len(positions)} particles using {self.method} method")
            return positions

        except Exception as e:
            self.logger.error(f"Error during particle detection using method '{self.method}': {e}", exc_info=True)
            return np.empty((0, 2)) # Return empty array on error


# --- Detection Functions ---

def detect_gaussian(image, threshold=0.5, threshold_is_relative=True, min_distance=5, diameter=7, subpixel=True, subpixel_method='com', **kwargs):
    """Detect particles using Gaussian peak finding."""
    try:
        sigma = max(0.5, diameter / 6.0) # Ensure sigma is positive
        smoothed = gaussian(image, sigma=sigma)

        if smoothed.max() <= smoothed.min(): # Handle flat image
             logger.debug("Gaussian detection: Image is flat after smoothing.")
             return np.empty((0, 2))

        if threshold_is_relative:
            threshold_value = threshold * (smoothed.max() - smoothed.min()) + smoothed.min()
        else:
            threshold_value = threshold # Absolute threshold

        coordinates = peak_local_max(
            smoothed,
            min_distance=min_distance,
            threshold_abs=threshold_value,
            exclude_border=False # Allow peaks near border
        )

        if subpixel and len(coordinates) > 0:
            coordinates = refine_positions(image, coordinates, diameter, subpixel_method)

        return coordinates
    except Exception as e:
        logger.error(f"Error in detect_gaussian: {e}", exc_info=True)
        return np.empty((0, 2))


def detect_laplacian(image, threshold=0.5, threshold_is_relative=True, min_distance=5, diameter=7, subpixel=True, subpixel_method='com', **kwargs):
    """Detect particles using Laplacian filtering."""
    try:
        sigma = max(0.5, diameter / 6.0)
        # ksize affects normalization, using sigma directly with filters.laplace is often better
        # ksize = int(max(3, sigma * 3) * 2 + 1)
        # filtered = -laplace(image, ksize=ksize) # Negative Laplacian for peak finding
        # Using LoG implicitly by providing sigma
        filtered = -filters.laplace(image, sigma=sigma) # Negative LoG

        if filtered.max() <= filtered.min(): # Handle flat image
             logger.debug("Laplacian detection: Image is flat after filtering.")
             return np.empty((0, 2))

        if threshold_is_relative:
            threshold_value = threshold * (filtered.max() - filtered.min()) + filtered.min()
        else:
            threshold_value = threshold # Absolute threshold

        coordinates = peak_local_max(
            filtered,
            min_distance=min_distance,
            threshold_abs=threshold_value,
            exclude_border=False
        )

        if subpixel and len(coordinates) > 0:
            coordinates = refine_positions(image, coordinates, diameter, subpixel_method)

        return coordinates
    except Exception as e:
        logger.error(f"Error in detect_laplacian: {e}", exc_info=True)
        return np.empty((0, 2))


def detect_doh(image, threshold=0.01, threshold_is_relative=False, min_distance=5, diameter=7, subpixel=True, subpixel_method='com', **kwargs):
    """Detect particles using Determinant of Hessian."""
    # Note: threshold for blob detectors is often absolute and lower than peak finders
    if threshold_is_relative:
         logger.warning("Relative threshold is not standard for blob_doh, using threshold as absolute value.")
    try:
        min_sigma = max(0.5, diameter / 6.0)
        max_sigma = max(min_sigma + 0.5, diameter / 3.0) # Ensure max > min

        blobs = blob_doh(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold, # Absolute threshold for DoH
            overlap=kwargs.get('overlap', 0.5),
            log_scale=kwargs.get('log_scale', False) # Add log_scale option
        )

        coordinates = blobs[:, :2] if len(blobs) > 0 else np.empty((0, 2))

        # Optional: Filter based on min_distance after blob detection
        # DOH doesn't have min_distance, but we can apply peak_local_max on the result
        # if min_distance > 0 and len(coordinates) > 1:
        #     # Create an image with blob responses (approximate)
        #     response_image = np.zeros_like(image, dtype=float)
        #     # Use blob sigma for response strength? Needs blob[:, 2]
        #     for y, x, sigma_blob in blobs:
        #          response_image[int(y), int(x)] = sigma_blob # Example: use sigma as response
        #     coordinates = peak_local_max(response_image, min_distance=min_distance, indices=True)


        if subpixel and len(coordinates) > 0:
            coordinates = refine_positions(image, coordinates, diameter, subpixel_method)

        return coordinates
    except Exception as e:
        logger.error(f"Error in detect_doh: {e}", exc_info=True)
        return np.empty((0, 2))


def detect_log(image, threshold=0.1, threshold_is_relative=False, min_distance=5, diameter=7, subpixel=True, subpixel_method='com', **kwargs):
    """Detect particles using Laplacian of Gaussian."""
    # Note: threshold for blob detectors is often absolute and lower than peak finders
    if threshold_is_relative:
         logger.warning("Relative threshold is not standard for blob_log, using threshold as absolute value.")
    try:
        min_sigma = max(0.5, diameter / 6.0)
        max_sigma = max(min_sigma + 0.5, diameter / 3.0) # Ensure max > min

        blobs = blob_log(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold, # Absolute threshold for LoG
            overlap=kwargs.get('overlap', 0.5),
            log_scale=kwargs.get('log_scale', False) # Add log_scale option
        )

        coordinates = blobs[:, :2] if len(blobs) > 0 else np.empty((0, 2))

        # Optional: Filter based on min_distance after blob detection
        # Similar logic as in detect_doh could be applied if needed

        if subpixel and len(coordinates) > 0:
            coordinates = refine_positions(image, coordinates, diameter, subpixel_method)

        return coordinates
    except Exception as e:
        logger.error(f"Error in detect_log: {e}", exc_info=True)
        return np.empty((0, 2))


def detect_wavelet(image, threshold=0.5, threshold_is_relative=True, min_distance=5, diameter=7,
                   subpixel=True, subpixel_method='com', **kwargs):
    """
    Detect particles using Wavelet filtering.

    Supports two enhancement methods:
    - 'residual': Uses the residual after wavelet denoising (requires skimage.restoration).
    - 'direct': Uses direct wavelet decomposition and reconstruction (requires pywt).

    Parameters are passed via kwargs:
    - wavelet_enhancement_method: 'residual' or 'direct' (default: 'residual')
    - wavelet_type: e.g., 'db4', 'sym5' (default: 'db4')
    - wavelet_levels: Number of decomposition levels (default: auto-calculated)
    - wavelet_detail_level: For 'direct' method, which detail level(s) to enhance (int or list, default: [1, 2])
    - denoise_sigma_factor: For 'residual' method, multiplier for estimated sigma (default: 3)
    """
    enhancement_method = kwargs.get('wavelet_enhancement_method', 'residual')
    wavelet_type = kwargs.get('wavelet_type', 'db4')

    try:
        enhanced_image = None
        if enhancement_method == 'residual':
            if not _HAS_SKIMAGE_RESTORATION:
                 raise ImportError("skimage.restoration is required for 'residual' wavelet method.")

            # Estimate levels if not provided
            wavelet_levels = kwargs.get('wavelet_levels')
            if wavelet_levels is None:
                 # Heuristic: levels capturing features around sigma=diameter/6
                 sigma_approx = max(0.5, diameter / 6.0)
                 # Level k roughly corresponds to 2^k scale. Find k such that 2^k is near sigma.
                 # This is a very rough heuristic.
                 level_guess = int(np.ceil(np.log2(max(1, sigma_approx))))
                 # Calculate max possible level
                 max_level = pywt.dwtn_max_level(image.shape, wavelet_type) if _HAS_PYWT else 3 # Default max if pywt missing
                 wavelet_levels = min(level_guess, max_level) if max_level > 0 else 3 # Use guess up to max
                 wavelet_levels = max(1, wavelet_levels) # Ensure at least 1 level
                 logger.debug(f"Auto-selected wavelet_levels={wavelet_levels} for residual method.")


            sigma_est = estimate_sigma(image, average_sigmas=True)
            if sigma_est is None: sigma_est = 0.01
            denoise_sigma_factor = kwargs.get('denoise_sigma_factor', 3)
            denoised_background = denoise_wavelet(image, wavelet=wavelet_type, method='BayesShrink',
                                                  mode='soft', sigma=sigma_est * denoise_sigma_factor,
                                                  rescale_sigma=True, wavelet_levels=wavelet_levels)
            enhanced_image = image - denoised_background

        elif enhancement_method == 'direct':
            if not _HAS_PYWT:
                raise ImportError("PyWavelets (pywt) is required for 'direct' wavelet method.")

            # Estimate levels if not provided
            wavelet_levels = kwargs.get('wavelet_levels')
            if wavelet_levels is None:
                 max_level = pywt.dwtn_max_level(image.shape, wavelet_type)
                 # Heuristic based on diameter
                 sigma_approx = max(0.5, diameter / 6.0)
                 level_guess = int(np.round(np.log2(max(1, sigma_approx)))) + 1 # Target levels around particle size
                 wavelet_levels = min(level_guess, max_level) if max_level > 0 else 3
                 wavelet_levels = max(1, wavelet_levels)
                 logger.debug(f"Auto-selected wavelet_levels={wavelet_levels} for direct method.")

            # Which detail levels to keep/enhance (e.g., corresponding to particle size)
            detail_levels_to_use = kwargs.get('wavelet_detail_level', [1, 2]) # Default: use first few detail levels
            if isinstance(detail_levels_to_use, int): detail_levels_to_use = [detail_levels_to_use]

            # Decompose
            coeffs = pywt.wavedec2(image, wavelet=wavelet_type, level=wavelet_levels)
            # Create new coefficient structure, zeroing out unwanted levels
            coeffs_enhanced = [coeffs[0]] # Keep approximation coefficients
            for i in range(1, len(coeffs)):
                 level_details = coeffs[i] # Tuple of (cH, cV, cD)
                 if i in detail_levels_to_use:
                      # Keep or enhance these levels (simple keep for now)
                      coeffs_enhanced.append(level_details)
                 else:
                      # Zero out other detail levels
                      zero_details = tuple(np.zeros_like(d) for d in level_details)
                      coeffs_enhanced.append(zero_details)

            # Reconstruct
            enhanced_image = pywt.waverec2(coeffs_enhanced, wavelet=wavelet_type)
            # Ensure shape matches original (waverec2 might pad slightly)
            enhanced_image = enhanced_image[:image.shape[0], :image.shape[1]]

        else:
            raise ValueError(f"Unknown wavelet_enhancement_method: {enhancement_method}")

        # Ensure non-negative for peak finding
        enhanced_image = np.clip(enhanced_image, 0, None)

        # --- Peak Finding ---
        if enhanced_image.max() <= enhanced_image.min(): # Handle flat image
             logger.debug("Wavelet detection: Image is flat after enhancement.")
             return np.empty((0, 2))

        if threshold_is_relative:
            threshold_value = threshold * (enhanced_image.max() - enhanced_image.min()) + enhanced_image.min()
        else:
            threshold_value = threshold # Absolute threshold

        coordinates = peak_local_max(
            enhanced_image,
            min_distance=min_distance,
            threshold_abs=threshold_value,
            exclude_border=False
        )

        # --- Subpixel Refinement ---
        if subpixel and len(coordinates) > 0:
            # Refine using the *original* image for better accuracy
            coordinates = refine_positions(image, coordinates, diameter, subpixel_method)

        return coordinates

    except ImportError as e:
         logger.error(f"Import error during wavelet detection: {e}. Ensure required libraries are installed.")
         return np.empty((0, 2))
    except Exception as e:
        logger.error(f"Error during wavelet detection: {e}", exc_info=True)
        return np.empty((0, 2))


# --- Factory Function ---
def get_detection_function(method: str) -> Callable:
    """
    Get the detection function for the specified method.

    Parameters
    ----------
    method : str
        Detection method name.

    Returns
    -------
    Callable
        The corresponding detection function.

    Raises
    ------
    ValueError
        If the method name is unknown.
    """
    methods = {
        'gaussian': detect_gaussian,
        'laplacian': detect_laplacian,
        'doh': detect_doh,
        'log': detect_log,
        'wavelet': detect_wavelet
    }

    if method not in methods:
        raise ValueError(f"Unknown detection method: {method}. Available methods: {list(methods.keys())}")

    return methods[method]
