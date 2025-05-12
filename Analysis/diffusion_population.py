# spt_analyzer/analysis/diffusion_population.py
"""
Diffusion population analysis module for SPT Analysis.

This module provides tools for analyzing diffusion populations, deconvoluting
jump size distributions, and segmenting trajectories into different diffusion states.

Assumes input track coordinates ('x', 'y') are in microns (μm) and
time intervals ('dt' initialization parameter) are in seconds (s).
Diffusion coefficients are calculated and reported in μm²/s.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats, optimize, linalg # Added linalg

# Optional dependencies - handled with try/except
try:
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False
try:
    from hmmlearn import hmm
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False
try:
    from sklearn.cluster import KMeans
    # KMeans usually included with sklearn, check _HAS_SKLEARN instead?
    # Check explicitly for robustness.
    _HAS_KMEANS = True
except ImportError:
     _HAS_KMEANS = False


logger = logging.getLogger(__name__)


class DiffusionPopulationAnalyzer:
    """
    Analyzer for identifying and characterizing diffusion populations.

    Provides methods for deconvoluting jump size distributions, segmenting
    trajectories, classifying diffusion types, and identifying diffusion states.

    Assumes input coordinates are in microns (μm).

    Parameters
    ----------
    dt : float, optional
        Time interval between frames in seconds, by default 0.014.
    max_populations : int, optional
        Maximum number of diffusion populations for mixture models, by default 5.
    min_segment_length : int, optional
        Minimum number of points for a valid trajectory segment, by default 10.
    """

    def __init__(self, dt: float = 0.014, max_populations: int = 5, min_segment_length: int = 10):
        if dt <= 0:
             raise ValueError("dt (frame interval) must be positive.")
        if max_populations < 1:
             raise ValueError("max_populations must be at least 1.")
        if min_segment_length < 3: # Need at least 3 points for MSD calculation
             raise ValueError("min_segment_length must be at least 3.")

        self.dt = dt # Time interval in seconds
        self.max_populations = max_populations
        self.min_segment_length = min_segment_length

        # Results storage
        self.jump_mixture_results: Dict[str, Any] = {}
        self.msd_mixture_results: Dict[str, Any] = {} # Placeholder if MSD mixture is added later
        self.segmented_trajectories: Dict[Any, List[Dict[str, Any]]] = {} # Key is track_id
        self.diffusion_state_results: Dict[str, Any] = {} # For HMM/KMeans results

    def deconvolute_jump_distributions(self,
                                     tracks_df: pd.DataFrame,
                                     compartment_masks: Optional[Dict[str, np.ndarray]] = None,
                                     method: str = 'gmm') -> Dict[str, Any]:
        """
        Deconvolute jump size distributions into distinct populations using GMM.

        Assumes input 'x', 'y' columns in tracks_df are in microns (μm).
        Calculates diffusion coefficients in μm²/s.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data ('track_id', 'frame', 'x', 'y').
        compartment_masks : dict, optional
            Dictionary of {name: mask_array} for compartments.
        method : str, optional
            Deconvolution method ('gmm' or 'bayesian'), by default 'gmm'.
            Requires scikit-learn.

        Returns
        -------
        dict
            Dictionary of deconvolution results, keys are 'all' and compartment names.
            Each value is a dict with model parameters and diffusion coefficients (μm²/s).
        """
        if not _HAS_SKLEARN:
             logger.error("scikit-learn is required for jump distribution deconvolution.")
             # Consider raising ImportError instead of returning dict
             raise ImportError("scikit-learn is required for this function.")

        if tracks_df is None or tracks_df.empty:
             logger.warning("Input tracks_df is empty or None.")
             return {'status': 'No track data provided'}

        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in tracks_df.columns]
             logger.error(f"Input tracks_df is missing required columns: {missing}")
             raise ValueError(f"tracks_df missing required columns: {missing}")

        logger.info(f"Deconvoluting jump distributions using method: {method}")
        try:
            # Calculate all jumps (squared displacements)
            all_squared_jumps = []
            for track_id, track_df in tracks_df.groupby('track_id'):
                if len(track_df) < 2: continue
                track_df = track_df.sort_values('frame')
                positions = track_df[['x', 'y']].values # Assumed units: μm
                squared_jumps = np.sum(np.diff(positions, axis=0)**2, axis=1) # Units: μm²
                all_squared_jumps.extend(squared_jumps)

            if not all_squared_jumps:
                logger.warning("No valid jumps found to analyze.")
                return {'status': 'No valid jumps found'}

            all_squared_jumps = np.array(all_squared_jumps)

            # --- Compartment-specific jumps ---
            jumps_by_compartment = {}
            if compartment_masks:
                logger.debug("Assigning jumps to compartments...")
                # Pre-build labeled map for efficiency
                first_mask = next(iter(compartment_masks.values()))
                labeled_map = np.zeros_like(first_mask, dtype=np.int32)
                comp_names = list(compartment_masks.keys())
                for idx, name in enumerate(comp_names, start=1):
                    labeled_map[compartment_masks[name]] = idx # Assumes masks are boolean

                jumps_by_compartment = {name: [] for name in comp_names}
                for track_id, track_df in tracks_df.groupby('track_id'):
                    if len(track_df) < 2: continue
                    track_df = track_df.sort_values('frame')
                    positions = track_df[['x', 'y']].values # μm
                    # Calculate midpoints for jump assignment
                    midpoints_x = (positions[:-1, 0] + positions[1:, 0]) / 2
                    midpoints_y = (positions[:-1, 1] + positions[1:, 1]) / 2
                    jump_sq_values = np.sum(np.diff(positions, axis=0)**2, axis=1) # μm²

                    # Convert midpoints to integer indices
                    mid_x_int = np.round(midpoints_x).astype(int)
                    mid_y_int = np.round(midpoints_y).astype(int)

                    # Clip indices to be within bounds
                    h, w = labeled_map.shape
                    valid_idx = (mid_y_int >= 0) & (mid_y_int < h) & (mid_x_int >= 0) & (mid_x_int < w)

                    labels_at_midpoints = np.zeros_like(midpoints_x, dtype=np.int32)
                    labels_at_midpoints[valid_idx] = labeled_map[mid_y_int[valid_idx], mid_x_int[valid_idx]]

                    for i, label_val in enumerate(labels_at_midpoints):
                         if label_val > 0:
                              comp_name = comp_names[label_val - 1]
                              jumps_by_compartment[comp_name].append(jump_sq_values[i])
                logger.debug("Finished assigning jumps to compartments.")
            # --- End Compartment Jumps ---

            # --- Fit Mixture Models ---
            mixture_results = {}
            min_jumps_for_fit = max(50, self.max_populations * 10) # Heuristic minimum points

            # Analyze overall distribution
            if len(all_squared_jumps) >= min_jumps_for_fit:
                logger.debug(f"Fitting mixture model to all {len(all_squared_jumps)} squared jumps.")
                mixture_results['all'] = self._fit_jump_mixture_model(all_squared_jumps, method)
            else:
                logger.warning(f"Insufficient jumps ({len(all_squared_jumps)}) for overall mixture model fitting (minimum {min_jumps_for_fit} required).")
                mixture_results['all'] = {'status': 'Insufficient data'}

            # Analyze compartment-specific distributions
            if jumps_by_compartment:
                mixture_results['by_compartment'] = {}
                for compartment, jumps_sq in jumps_by_compartment.items():
                    if len(jumps_sq) >= min_jumps_for_fit:
                        logger.debug(f"Fitting mixture model to {len(jumps_sq)} squared jumps in compartment '{compartment}'.")
                        mixture_results['by_compartment'][compartment] = self._fit_jump_mixture_model(np.array(jumps_sq), method)
                    else:
                         logger.warning(f"Insufficient jumps ({len(jumps_sq)}) for mixture model fitting in compartment '{compartment}' (minimum {min_jumps_for_fit} required).")
                         mixture_results['by_compartment'][compartment] = {'status': 'Insufficient data'}
            # --- End Fitting ---

            self.jump_mixture_results = mixture_results
            logger.info("Jump distribution deconvolution complete.")
            return mixture_results

        except ImportError: # Catch specific import error for sklearn
             logger.error("scikit-learn is required for jump distribution deconvolution.")
             raise # Re-raise, caller should handle this if critical
        except Exception as e:
            logger.error(f"Error in jump distribution deconvolution: {e}", exc_info=True)
            raise

    def _fit_jump_mixture_model(self, squared_jumps: np.ndarray, method: str) -> Dict[str, Any]:
        """Internal: Fit mixture model to squared jump data (μm²)."""
        if not _HAS_SKLEARN:
             # This should not be reached if called from deconvolute_jump_distributions
             logger.error("Scikit-learn called internally but not available.")
             raise ImportError("scikit-learn is required for mixture model fitting.")

        try:
            X = squared_jumps.reshape(-1, 1) # Ensure column vector
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                 logger.warning("NaN or Inf values found in squared jumps, removing them before fitting.")
                 X = X[~np.isnan(X) & ~np.isinf(X)].reshape(-1, 1)

            if len(X) < 10: # Absolute minimum points
                 logger.warning("Very few data points (<10) for mixture model fitting.")
                 return {'status': 'Insufficient data points for fitting'}

            best_model = None
            best_bic = np.inf
            best_n_components = 0

            # Try different numbers of components
            max_k = min(self.max_populations, len(X) // 10, 10) # Limit max components dynamically
            if max_k < 1:
                 logger.warning(f"Not enough data points ({len(X)}) to fit even one component.")
                 return {'status': 'Insufficient data points for specified max_populations'}

            logger.debug(f"Testing GMM components from 1 to {max_k}")
            for n_components in range(1, max_k + 1):
                 try:
                     # Use more robust fitting parameters
                     n_init = 10 # Number of initializations
                     max_iter = 200
                     tol = 1e-3
                     reg_covar = 1e-6 # Regularization added

                     if method == 'gmm':
                         model = GaussianMixture(
                             n_components=n_components, covariance_type='full', # Full allows different variances
                             random_state=42, max_iter=max_iter, n_init=n_init, tol=tol, reg_covar=reg_covar
                         )
                     elif method == 'bayesian':
                         model = BayesianGaussianMixture(
                             n_components=n_components, covariance_type='full',
                             random_state=42, max_iter=max_iter, n_init=n_init, tol=tol, reg_covar=reg_covar,
                             weight_concentration_prior_type='dirichlet_process'
                             # Consider adjusting weight_concentration_prior if needed
                         )
                     else:
                          # This should ideally be caught earlier, but defensive check
                          raise ValueError(f"Unknown mixture model method: {method}")

                     model.fit(X)
                     # Check for convergence
                     if not model.converged_:
                          logger.warning(f"GMM fitting did not converge for {n_components} components.")
                          # Calculate BIC anyway, but maybe add a warning flag to results?

                     bic = model.bic(X)

                     if not np.isfinite(bic):
                          logger.warning(f"BIC calculation resulted in non-finite value for {n_components} components.")
                          continue # Skip this number of components if BIC is invalid

                     # Update best model based on BIC
                     if bic < best_bic:
                         best_bic = bic
                         best_model = model
                         best_n_components = n_components

                 except (ValueError, linalg.LinAlgError) as fit_error: # Catch specific errors
                      logger.warning(f"GMM fitting failed for {n_components} components: {fit_error}")
                      continue # Try next number of components

            if best_model is None:
                 logger.error("Mixture model fitting failed for all component numbers tried.")
                 return {'error': 'Mixture model fitting failed'}

            logger.info(f"Best GMM fit found with {best_n_components} components (BIC={best_bic:.2f}).")
            # Extract components information
            means = best_model.means_.flatten() # <Δr²> in μm²
            # Handle different covariance types robustly
            if best_model.covariance_type == 'full':
                 # Variance is the diagonal element of the covariance matrix for 1D data
                 variances = np.array([cov[0, 0] for cov in best_model.covariances_])
            elif best_model.covariance_type == 'diag':
                 variances = best_model.covariances_.flatten()
            elif best_model.covariance_type == 'tied':
                  # Tied covariance is shared, get the diagonal element
                  variances = np.full(best_n_components, best_model.covariances_[0, 0])
            elif best_model.covariance_type == 'spherical':
                  variances = best_model.covariances_ # Already scalar variances
            else:
                  logger.warning(f"Unexpected covariance type: {best_model.covariance_type}. Attempting to extract variances.")
                  # Attempt to extract variance assuming it might be scalar or first diagonal
                  try:
                       variances = np.array([np.diag(c).mean() for c in best_model.covariances_]) # Example fallback
                  except:
                       variances = np.full(best_n_components, np.nan) # Fallback if extraction fails

            stds = np.sqrt(np.maximum(variances, 0)) # Ensure variance is non-negative before sqrt
            weights = best_model.weights_

            # Sort components by mean squared displacement (<Δr²>)
            sort_idx = np.argsort(means)
            means_sorted = means[sort_idx]
            stds_sorted = stds[sort_idx]
            weights_sorted = weights[sort_idx]

            # Calculate diffusion coefficients (D = <Δr²> / 4Δt for 2D)
            diffusion_coefficients = means_sorted / (4 * self.dt) # Units: μm²/s

            # Predict component labels for data points
            labels = best_model.predict(X)
            # Map labels according to sorted means
            label_map = {old_label: new_label for new_label, old_label in enumerate(sort_idx)}
            mapped_labels = np.array([label_map[l] for l in labels])

            # Determine effective components for Bayesian GMM
            effective_components = best_n_components
            if method == 'bayesian':
                effective_components = np.sum(weights_sorted > 0.01) # Example threshold

            results = {
                'method': method,
                'n_components_tested': max_k,
                'best_n_components': best_n_components,
                'effective_components': effective_components,
                'bic': best_bic,
                'means_sq_jump': means_sorted.tolist(), # <Δr²> in μm²
                'stds_sq_jump': stds_sorted.tolist(), # Std dev of <Δr²>
                'weights': weights_sorted.tolist(),
                'diffusion_coefficients': diffusion_coefficients.tolist(), # D in μm²/s
                'labels': mapped_labels.tolist() # Labels corresponding to sorted components
            }
            return results

        except Exception as e:
            logger.error(f"Error fitting mixture model: {e}", exc_info=True)
            # Return error status instead of raising if desired within the class workflow
            return {'error': f'Internal error during fitting: {e}'}


    def segment_trajectories(self, tracks_df: pd.DataFrame, method: str = 'changepoint') -> Dict[Any, List[Dict[str, Any]]]:
        """
        Segment trajectories into portions with different diffusion properties.

        Assumes input 'x', 'y' columns in tracks_df are in microns (μm).
        Calculates diffusion coefficients in μm²/s.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data ('track_id', 'frame', 'x', 'y').
        method : str, optional
            Segmentation method ('changepoint' or 'sliding_window'), by default 'changepoint'.
            'changepoint' requires the 'ruptures' package.

        Returns
        -------
        dict
            Dictionary mapping track_id to a list of segment dictionaries.
            Each segment dict contains properties like start/end frames, alpha, D (μm²/s), etc.
        """
        if tracks_df is None or tracks_df.empty:
             logger.warning("Input tracks_df is empty or None for segmentation.")
             return {}

        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in tracks_df.columns]
             logger.error(f"Input tracks_df is missing required columns: {missing}")
             raise ValueError(f"tracks_df missing required columns: {missing}")

        logger.info(f"Segmenting trajectories using method: {method}")
        segmented_trajectories = {}
        min_points_for_segment = max(self.min_segment_length, 3) # Need >= 3 points for MSD fit

        for track_id, track_df in tracks_df.groupby('track_id'):
            if len(track_df) < min_points_for_segment: # Need enough points for at least one segment
                 logger.debug(f"Skipping track {track_id}: too short ({len(track_df)} points).")
                 continue

            track_df = track_df.sort_values('frame')
            positions = track_df[['x', 'y']].values # Assumed units: μm
            frames = track_df['frame'].values

            if len(positions) < 2: continue # Need at least 2 points for jumps

            # Calculate squared jump sizes (signal for changepoint detection)
            jumps = np.diff(positions, axis=0)
            squared_jumps = np.sum(jumps**2, axis=1) # Units: μm²
            if len(squared_jumps) < 2: continue # Need at least 2 jumps for segmentation

            segment_boundaries = [] # Stores (start_index, end_index) for the squared_jumps array
            current_method = method

            # --- Apply Segmentation Method ---
            if current_method == 'changepoint':
                if not _HAS_RUPTURES:
                    logger.warning("ruptures package not available, falling back to sliding_window method")
                    current_method = 'sliding_window'
                else:
                    try:
                        model = "l2" # Variance change detection
                        # Robust penalty estimation (adjust multiplier if needed)
                        sigma_sq = np.var(squared_jumps)
                        if sigma_sq < 1e-9: sigma_sq = 1e-9 # Avoid zero variance
                        penalty = 2 * np.log(len(squared_jumps)) * sigma_sq # BIC-like penalty

                        algo = rpt.Pelt(model=model).fit(squared_jumps)
                        result = algo.predict(pen=penalty) # result contains end indices (exclusive)

                        # Convert result to (start, end) indices relative to squared_jumps array
                        start_idx = 0
                        for end_idx in result:
                            # Ensure end_idx doesn't exceed array bounds (can happen with Pelt)
                            end_idx = min(end_idx, len(squared_jumps))
                            if end_idx > start_idx: # Add segment only if it has non-zero length
                                 segment_boundaries.append((start_idx, end_idx))
                            start_idx = end_idx
                        # The last segment goes from the last breakpoint to the end
                        if start_idx < len(squared_jumps):
                             segment_boundaries.append((start_idx, len(squared_jumps)))

                    except Exception as rpt_err:
                         logger.warning(f"Ruptures changepoint detection failed for track {track_id} ({rpt_err}), falling back to sliding_window method.")
                         current_method = 'sliding_window'

            if current_method == 'sliding_window':
                window_size = self.min_segment_length # Use min_segment_length as window size
                step = max(1, window_size // 2)
                segment_boundaries = []
                current_segment_start = 0

                # Sliding window compare variances
                for i in range(window_size, len(squared_jumps)): # Iterate potential split points
                     # Define windows before and after point i
                     win1_start = max(0, i - window_size)
                     win1_end = i
                     win2_start = i
                     win2_end = min(len(squared_jumps), i + window_size)

                     # Ensure windows have enough points
                     if (win1_end - win1_start) < 3 or (win2_end - win2_start) < 3: continue

                     var1 = np.var(squared_jumps[win1_start:win1_end])
                     var2 = np.var(squared_jumps[win2_start:win2_end])

                     # Use a relative change threshold (adjust sensitivity as needed)
                     threshold = 1.5 # e.g., variance changes by > 150%
                     if max(var1, var2) > 1e-9: # Avoid division by zero / trivial changes
                         if abs(var2 - var1) / max(var1, var2) > threshold:
                              if i > current_segment_start + min_points_for_segment: # Ensure min segment length
                                   segment_boundaries.append((current_segment_start, i))
                                   current_segment_start = i

                # Add the final segment
                segment_boundaries.append((current_segment_start, len(squared_jumps)))
            # --- End Segmentation Method ---

            # --- Analyze Segments ---
            segments = []
            for start_idx_jump, end_idx_jump in segment_boundaries:
                # Segment length in terms of jumps
                num_jumps = end_idx_jump - start_idx_jump
                # Segment length in terms of points (+1)
                num_points = num_jumps + 1

                if num_points < min_points_for_segment:
                    continue # Skip segment if too short

                # Indices for the positions array
                start_idx_pos = start_idx_jump
                end_idx_pos = end_idx_jump # End index for positions is inclusive

                segment_positions = positions[start_idx_pos : end_idx_pos + 1]
                segment_frames = frames[start_idx_pos : end_idx_pos + 1]

                # Calculate MSD for this segment
                msd_values = []
                lag_times_sec = [] # Store time in seconds
                max_msd_lag = min(10, num_points // 2) # Limit MSD lag relative to segment length
                if max_msd_lag < 1: continue

                for lag in range(1, max_msd_lag + 1):
                    disp_sq = np.sum((segment_positions[lag:] - segment_positions[:-lag])**2, axis=1) # μm²
                    if len(disp_sq) > 0:
                        msd_values.append(np.mean(disp_sq))
                        lag_times_sec.append(lag * self.dt) # Seconds

                # Fit MSD to get D and alpha
                alpha = None
                D = None
                if len(lag_times_sec) >= 2: # Need at least 2 points for fit
                    try:
                        log_tau = np.log(lag_times_sec)
                        log_msd = np.log(np.maximum(msd_values, 1e-12)) # Avoid log(0)
                        valid_fit_indices = np.isfinite(log_tau) & np.isfinite(log_msd)

                        if np.sum(valid_fit_indices) >= 2:
                              slope, intercept = np.polyfit(log_tau[valid_fit_indices], log_msd[valid_fit_indices], 1)
                              alpha = slope
                              D = np.exp(intercept) / 4 # Units: μm²/s
                        else:
                              logger.debug(f"Track {track_id}, segment {start_idx_pos}-{end_idx_pos}: Not enough valid points for MSD fit.")

                    except (ValueError, linalg.LinAlgError, TypeError) as fit_err:
                         logger.warning(f"Track {track_id}, segment {start_idx_pos}-{end_idx_pos}: MSD fit failed: {fit_err}")
                         alpha, D = None, None

                # Calculate average jump size for the segment
                segment_jumps_sq = squared_jumps[start_idx_jump:end_idx_jump]
                mean_jump = np.sqrt(np.mean(segment_jumps_sq)) if len(segment_jumps_sq) > 0 else 0.0 # Units: μm

                # Classify diffusion mode based on alpha
                # Thresholds can be adjusted based on expected behavior/noise
                alpha_thresh_sub = 0.8
                alpha_thresh_super = 1.2
                if alpha is not None:
                    if alpha < alpha_thresh_sub: diffusion_mode = "Subdiffusion"
                    elif alpha > alpha_thresh_super: diffusion_mode = "Superdiffusion"
                    else: diffusion_mode = "Normal" # Changed from "Normal diffusion"
                else:
                    diffusion_mode = "Unknown"

                segments.append({
                    'start_frame': int(segment_frames[0]),
                    'end_frame': int(segment_frames[-1]),
                    'n_frames': len(segment_frames),
                    'alpha': float(alpha) if alpha is not None else None,
                    'diffusion_coefficient': float(D) if D is not None else None, # Units: μm²/s
                    'mean_jump': float(mean_jump), # Units: μm
                    'diffusion_mode': diffusion_mode,
                })
            # --- End Analyze Segments ---

            if segments: # Only add if segments were found
                segmented_trajectories[track_id] = segments
        # --- End Track Loop ---

        self.segmented_trajectories = segmented_trajectories
        logger.info(f"Trajectory segmentation complete for {len(segmented_trajectories)} tracks.")
        return segmented_trajectories

    def calculate_population_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics on identified diffusion populations/segments.

        Returns
        -------
        dict
            Dictionary containing statistics from 'jump_mixture' results and
            'trajectory_segments' results, if available.
            Diffusion coefficients are in μm²/s.
        """
        logger.info("Calculating population statistics...")
        population_stats = {}

        # --- Jump Mixture Statistics ---
        if self.jump_mixture_results:
            jump_stats = {}
            for source, results in self.jump_mixture_results.items(): # source is 'all' or compartment name
                if isinstance(results, dict) and 'diffusion_coefficients' in results:
                    try:
                        D_values = results['diffusion_coefficients'] # Units: μm²/s
                        weights = results['weights']
                        if D_values and weights and len(D_values) == len(weights): # Check validity
                            jump_stats[source] = {
                                'n_populations': len(D_values),
                                'diffusion_coefficients': D_values, # Units: μm²/s
                                'population_weights': weights,
                                'dominant_population_index': int(np.argmax(weights)),
                                'mean_D_weighted': float(np.sum(np.array(D_values) * np.array(weights)))
                            }
                        else:
                            jump_stats[source] = {'status': 'No valid populations found in results'}
                    except (KeyError, TypeError, IndexError, ValueError) as e: # Added ValueError
                         logger.warning(f"Could not process jump mixture results for '{source}': {e}")
                         jump_stats[source] = {'status': f'Error processing results: {e}'}
                else:
                     jump_stats[source] = {'status': results.get('status', 'No diffusion data')}
            if jump_stats:
                 population_stats['jump_mixture'] = jump_stats
        # --- End Jump Mixture ---

        # --- Trajectory Segment Statistics ---
        if self.segmented_trajectories:
            all_segments = [seg for segments in self.segmented_trajectories.values() for seg in segments]
            n_total_segments = len(all_segments)

            if n_total_segments == 0:
                segment_stats = {'status': 'No segments found'}
            else:
                segment_stats = {
                    'n_tracks_segmented': len(self.segmented_trajectories),
                    'n_total_segments': n_total_segments,
                    'avg_segments_per_track': n_total_segments / len(self.segmented_trajectories),
                    'diffusion_modes': {}
                }
                # Aggregate stats using DataFrame for easier handling
                seg_df = pd.DataFrame(all_segments)
                mode_groups = seg_df.groupby('diffusion_mode')

                for mode, group in mode_groups:
                    count = len(group)
                    # Use .dropna() before mean/std to handle potential None values
                    alphas = group['alpha'].dropna()
                    diff_coeffs = group['diffusion_coefficient'].dropna()

                    segment_stats['diffusion_modes'][mode] = {
                        'count': count,
                        'fraction': count / n_total_segments,
                        'mean_alpha': alphas.mean() if not alphas.empty else None,
                        'std_alpha': alphas.std() if len(alphas) > 1 else None,
                        'mean_D': diff_coeffs.mean() if not diff_coeffs.empty else None, # Units: μm²/s
                        'std_D': diff_coeffs.std() if len(diff_coeffs) > 1 else None, # Units: μm²/s
                    }
            population_stats['trajectory_segments'] = segment_stats
        # --- End Trajectory Segments ---

        # --- Diffusion State Statistics (from HMM/KMeans) ---
        if self.diffusion_state_results:
            state_stats = {'method': self.diffusion_state_results.get('method', 'Unknown')}
            if 'states' in self.diffusion_state_results and isinstance(self.diffusion_state_results['states'], dict):
                 state_info = self.diffusion_state_results['states']
                 state_stats['n_states'] = len(state_info)
                 state_stats['states_summary'] = {}
                 for state_id, state_data in state_info.items():
                      if isinstance(state_data, dict): # Ensure state_data is a dict
                           state_stats['states_summary'][state_id] = {
                                'mean_D': state_data.get('mean_D'), # Units: μm²/s
                                'mean_alpha': state_data.get('mean_alpha'), # If available
                                'occupancy': state_data.get('occupancy'),
                                'transition_probabilities': state_data.get('transition_probabilities')
                           }
                      else:
                           logger.warning(f"Unexpected data format for state {state_id}: {type(state_data)}")
            else:
                 state_stats['status'] = 'No state information found in results.'
            population_stats['diffusion_states'] = state_stats
        # --- End Diffusion States ---

        logger.info("Population statistics calculation complete.")
        return population_stats


    def analyze_state_transitions(self, min_track_length: int = 20) -> Dict[str, Any]:
        """
        Analyze transitions between diffusion modes identified by segmentation.

        Parameters
        ----------
        min_track_length : int, optional
            Minimum total track length (in frames) to include in transition analysis,
            by default 20.

        Returns
        -------
        dict
            Dictionary containing diffusion modes, transition count matrix,
            transition probability matrix, and mean dwell times per mode (in seconds).
        """
        if not self.segmented_trajectories:
            logger.warning("Cannot analyze transitions: No segmented trajectories available.")
            return {'status': 'No segmented trajectories available'}

        logger.info("Analyzing state transitions...")
        all_modes = set()
        valid_transitions = [] # List of (from_mode, to_mode) tuples
        dwell_frames_by_mode = {} # {mode: [list of dwell durations in frames]}

        # --- Collect transitions and dwell times ---
        n_tracks_processed = 0
        for track_id, segments in self.segmented_trajectories.items():
            if not segments: continue # Skip tracks with no segments

            total_frames_in_segments = sum(seg.get('n_frames', 0) for seg in segments)
            if total_frames_in_segments < min_track_length: continue
            n_tracks_processed += 1

            for i, segment in enumerate(segments):
                mode = segment.get('diffusion_mode', 'Unknown')
                if not isinstance(mode, str): mode = 'Unknown' # Ensure mode is hashable
                all_modes.add(mode)
                if mode not in dwell_frames_by_mode: dwell_frames_by_mode[mode] = []

                # Record dwell time (number of frames in the segment)
                dwell_frames = segment.get('n_frames', 0)
                if dwell_frames > 0: # Only record valid dwell times
                     dwell_frames_by_mode[mode].append(dwell_frames)

                # Record transition *from* the previous segment *to* this one
                if i > 0:
                    from_mode = segments[i-1].get('diffusion_mode', 'Unknown')
                    if not isinstance(from_mode, str): from_mode = 'Unknown'
                    to_mode = mode
                    valid_transitions.append((from_mode, to_mode))
        # --- End Collection ---

        if n_tracks_processed == 0:
             logger.warning(f"No tracks met the minimum length requirement ({min_track_length} frames) for transition analysis.")
             return {'status': f'No tracks >= {min_track_length} frames'}
        if not all_modes:
             logger.warning("No diffusion modes found in segmented tracks.")
             return {'status': 'No diffusion modes found'}

        all_modes_list = sorted(list(all_modes))
        n_modes = len(all_modes_list)
        mode_to_idx = {mode: i for i, mode in enumerate(all_modes_list)}

        # --- Calculate Transition Matrix ---
        transition_counts = np.zeros((n_modes, n_modes), dtype=int)
        for from_mode, to_mode in valid_transitions:
             # Check if modes are in our list (handles potential 'Unknown' modes properly)
             if from_mode in mode_to_idx and to_mode in mode_to_idx:
                 from_idx = mode_to_idx[from_mode]
                 to_idx = mode_to_idx[to_mode]
                 transition_counts[from_idx, to_idx] += 1

        transition_probs = np.zeros_like(transition_counts, dtype=float)
        row_sums = transition_counts.sum(axis=1)
        valid_rows = row_sums > 0 # Rows with at least one outgoing transition
        # Calculate probabilities only for rows with transitions
        transition_probs[valid_rows] = transition_counts[valid_rows] / row_sums[valid_rows, np.newaxis]
        # --- End Transition Matrix ---

        # --- Calculate Dwell Times ---
        dwell_times_stats = {}
        for mode in all_modes_list: # Iterate through known modes
            dwell_list = dwell_frames_by_mode.get(mode, []) # Get dwell times for this mode
            if dwell_list: # Check if list is not empty
                mean_frames = np.mean(dwell_list)
                std_frames = np.std(dwell_list) # np.std handles len=1 correctly (returns 0)
                dwell_times_stats[mode] = {
                    'count': len(dwell_list), # Number of segments in this mode
                    'mean_frames': float(mean_frames),
                    'std_frames': float(std_frames),
                    'mean_time_s': float(mean_frames * self.dt), # Dwell time in seconds
                    'std_time_s': float(std_frames * self.dt)   # Std dev in seconds
                }
            else:
                 dwell_times_stats[mode] = {'count': 0, 'mean_frames': 0, 'std_frames': 0, 'mean_time_s': 0, 'std_time_s': 0}
        # --- End Dwell Times ---

        logger.info(f"State transition analysis complete for {n_tracks_processed} tracks.")
        return {
            'diffusion_modes': all_modes_list,
            'transition_counts': transition_counts.tolist(),
            'transition_probabilities': transition_probs.tolist(),
            'dwell_times': dwell_times_stats # Now includes time in seconds
        }

    def classify_tracks_by_diffusion_type(self, tracks_df: pd.DataFrame, method: str = 'msd') -> Dict[str, Any]:
        """
        Classify entire tracks by their predominant diffusion type (MSD alpha or jump CV).

        Assumes input 'x', 'y' columns in tracks_df are in microns (μm).
        Calculates diffusion coefficients in μm²/s.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data ('track_id', 'frame', 'x', 'y').
        method : str, optional
            Classification method ('msd' or 'jump'), by default 'msd'.
            'msd' uses the anomalous exponent alpha.
            'jump' uses the coefficient of variation of jump distances.

        Returns
        -------
        dict
            Dictionary containing:
            - 'track_classifications': Dict mapping track_id to classification details.
            - 'diffusion_types': Counts of tracks per type.
            - 'diffusion_fractions': Fraction of tracks per type.
            - 'total_tracks_classified': Number of tracks meeting length requirement.
        """
        if tracks_df is None or tracks_df.empty:
             logger.warning("Input tracks_df is empty or None for classification.")
             return {'status': 'No track data provided'}

        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in tracks_df.columns]
             logger.error(f"Input tracks_df is missing required columns: {missing}")
             raise ValueError(f"tracks_df missing required columns: {missing}")

        logger.info(f"Classifying tracks by diffusion type using method: {method}")
        track_classifications = {}
        # Define min length required based on method
        min_points_required = 5 if method == 'msd' else 3 # Need more points for reliable MSD fit

        for track_id, track_df in tracks_df.groupby('track_id'):
            if len(track_df) < min_points_required: continue

            track_df = track_df.sort_values('frame')
            positions = track_df[['x', 'y']].values # Assumed units: μm

            if method == 'msd':
                # Calculate MSD
                msd_values = []
                lag_times_sec = []
                max_msd_lag = min(10, len(positions) // 2) # Limit lag for robustness
                if max_msd_lag < 2: continue # Need at least 2 lags

                for lag in range(1, max_msd_lag + 1):
                    disp_sq = np.sum((positions[lag:] - positions[:-lag])**2, axis=1) # μm²
                    if len(disp_sq) > 0:
                        msd_values.append(np.mean(disp_sq))
                        lag_times_sec.append(lag * self.dt) # seconds

                if len(lag_times_sec) < 2: continue # Still not enough

                # Fit MSD to power law MSD = 4*D*t^alpha
                try:
                    log_tau = np.log(lag_times_sec)
                    log_msd = np.log(np.maximum(msd_values, 1e-12)) # Avoid log(0)
                    valid_fit = np.isfinite(log_tau) & np.isfinite(log_msd)

                    if np.sum(valid_fit) >= 2:
                        slope, intercept, r_value, _, _ = stats.linregress(log_tau[valid_fit], log_msd[valid_fit])
                        alpha = slope
                        D = np.exp(intercept) / 4 # μm²/s

                        # Classify based on alpha thresholds (adjustable)
                        alpha_thresh_sub = 0.8
                        alpha_thresh_super = 1.2
                        if alpha < alpha_thresh_sub: diffusion_type = "Subdiffusion"
                        elif alpha > alpha_thresh_super: diffusion_type = "Superdiffusion"
                        else: diffusion_type = "Normal"

                        # Calculate R² for the log-log fit
                        r_squared = r_value**2

                        track_classifications[track_id] = {
                            'diffusion_type': diffusion_type,
                            'alpha': float(alpha),
                            'diffusion_coefficient': float(D), # μm²/s
                            'loglog_r_squared': float(r_squared)
                        }
                    else:
                         logger.debug(f"Track {track_id}: Not enough valid points for MSD fit during classification.")

                except (ValueError, linalg.LinAlgError, TypeError) as fit_err:
                     logger.warning(f"Track {track_id}: MSD fit failed during classification: {fit_err}")
                     # Optionally classify as 'Unknown' or skip
                     track_classifications[track_id] = {'diffusion_type': 'FitError', 'alpha': None, 'diffusion_coefficient': None, 'loglog_r_squared': None}

            elif method == 'jump':
                if len(positions) < 2: continue # Need jumps
                # Calculate jumps
                jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)) # Units: μm
                if len(jumps) == 0: continue

                # Calculate jump statistics
                mean_jump = np.mean(jumps) # μm
                std_jump = np.std(jumps) # μm
                # Coefficient of Variation
                cv = std_jump / mean_jump if mean_jump > 1e-9 else 0.0 # Avoid division by zero

                # Calculate diffusion coefficient from mean squared jump
                # D = <jump²> / 4*dt approx mean_jump^2 / (4*dt) for small dt/low variance
                # More accurately: D = (<jump²> + var(jump²)) / 4*dt ?? No, <r^2> = <jump^2>
                # Use Mean Squared Jump: D = mean(jump^2) / 4*dt
                mean_sq_jump = np.mean(jumps**2) # μm²
                D = mean_sq_jump / (4 * self.dt) # μm²/s


                # Classify based on coefficient of variation threshold (adjustable)
                cv_threshold = 0.6 # Example threshold for heterogeneity
                if cv > cv_threshold:
                    diffusion_type = "Heterogeneous" # High variation in step size
                else:
                    diffusion_type = "Homogeneous" # Low variation

                track_classifications[track_id] = {
                    'diffusion_type': diffusion_type,
                    'mean_jump': float(mean_jump), # μm
                    'std_jump': float(std_jump), # μm
                    'jump_coefficient_of_variation': float(cv),
                    'diffusion_coefficient': float(D) # μm²/s
                }
            else:
                 logger.warning(f"Unknown classification method: {method}. Skipping track {track_id}.")
                 continue

        # --- Calculate Summary Statistics ---
        diffusion_types = {}
        n_classified = 0
        for track_id, classification in track_classifications.items():
            # Only count valid classifications
            if classification.get('diffusion_type') and classification['diffusion_type'] != 'FitError':
                 dtype = classification['diffusion_type']
                 diffusion_types[dtype] = diffusion_types.get(dtype, 0) + 1
                 n_classified += 1

        diffusion_fractions = {dtype: count / n_classified for dtype, count in diffusion_types.items()} if n_classified > 0 else {}

        logger.info(f"Track classification complete. Classified {n_classified} tracks.")
        return {
            'track_classifications': track_classifications,
            'diffusion_types': diffusion_types,
            'diffusion_fractions': diffusion_fractions,
            'total_tracks_classified': n_classified
        }

    # --- Added identify_diffusion_states method ---
    def identify_diffusion_states(self, tracks_df: pd.DataFrame, n_states: int = 3, method: str = 'hmm') -> Dict[str, Any]:
        """
        Identify diffusion states using Hidden Markov Models (HMM) or KMeans.

        Requires hmmlearn (for HMM) or scikit-learn (for KMeans).
        Uses log of squared displacements as features.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data ('track_id', 'frame', 'x', 'y').
        n_states : int, optional
            Number of diffusion states to identify, by default 3.
        method : str, optional
            Method ('hmm' or 'kmeans'), by default 'hmm'.

        Returns
        -------
        dict
             Dictionary containing state information ('states'), track state assignments
             ('track_states'), method used, and number of states.
             Diffusion coefficients (mean_D) are in μm²/s.
        """
        logger.info(f"Identifying diffusion states using method: {method}")
        if tracks_df is None or tracks_df.empty:
             logger.warning("Input tracks_df is empty or None for state identification.")
             return {'status': 'No track data provided'}

        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
             missing = [col for col in required_cols if col not in tracks_df.columns]
             logger.error(f"Input tracks_df is missing required columns: {missing}")
             raise ValueError(f"tracks_df missing required columns: {missing}")

        # Validate method and check dependencies
        if method == 'hmm' and not _HAS_HMMLEARN:
             logger.warning("hmmlearn not available, falling back to kmeans")
             method = 'kmeans'
        if method == 'kmeans' and not _HAS_KMEANS:
             logger.error("scikit-learn (for KMeans) not available. Cannot identify states.")
             raise ImportError("scikit-learn is required for KMeans state identification.")

        # --- Process tracks to extract features ---
        # Using log of squared displacements (related to instantaneous D)
        log_sq_displacements = []
        track_lengths = [] # Store lengths of feature sequences per track
        track_ids_processed = [] # Store IDs of tracks used

        for track_id, track_df in tracks_df.groupby('track_id'):
            if len(track_df) < 3: continue # Need at least 3 points for 2 displacements

            track_df = track_df.sort_values('frame')
            positions = track_df[['x', 'y']].values # μm
            sq_disp = np.sum(np.diff(positions, axis=0)**2, axis=1) # μm²

            # Filter out zero displacements before log
            valid_disp = sq_disp[sq_disp > 1e-12] # Avoid log(0)
            if len(valid_disp) < 2: continue # Need at least 2 valid log features

            log_features = np.log(valid_disp)
            log_sq_displacements.append(log_features)
            track_lengths.append(len(log_features))
            track_ids_processed.append(track_id)

        if not log_sq_displacements:
            logger.warning("No suitable tracks found for diffusion state analysis after feature extraction.")
            return {'status': 'No suitable tracks found'}

        # Concatenate features for model training
        X = np.concatenate(log_sq_displacements).reshape(-1, 1)
        if np.any(~np.isfinite(X)):
             n_nonfinite = np.sum(~np.isfinite(X))
             logger.warning(f"Removing {n_nonfinite} non-finite values from features before fitting.")
             X = X[np.isfinite(X)].reshape(-1, 1)
             # Note: This might disrupt sequence lengths for HMM if not handled carefully,
             # but hmmlearn's fit method handles the lengths argument separately.

        if len(X) < n_states * 5: # Heuristic check for enough data
             logger.warning(f"Very few data points ({len(X)}) for fitting {n_states} states.")
             return {'status': f'Insufficient data points ({len(X)}) for {n_states} states.'}
        # --- End Feature Extraction ---

        # --- Identify states using specified method ---
        model_fit = None
        predicted_states_flat = None

        try:
            if method == 'hmm':
                # Train HMM
                logger.debug(f"Training GaussianHMM with {n_states} states.")
                # Use more iterations and initializations for potentially better fit
                model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42, n_iter=100, init_params='mc', params='mc') # Fit means and covariances
                # Pass sequence lengths to fit method for HMM
                model.fit(X, lengths=track_lengths)
                model_fit = model
                # Predict states for the concatenated sequence
                predicted_states_flat = model.predict(X, lengths=track_lengths)

            elif method == 'kmeans':
                # Apply K-means clustering
                logger.debug(f"Applying KMeans clustering with {n_states} states.")
                kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10) # Added n_init
                predicted_states_flat = kmeans.fit_predict(X)
                model_fit = kmeans # Store fitted model

            else:
                # Should not happen due to earlier checks
                raise ValueError(f"Unsupported state identification method: {method}")

        except Exception as fit_err:
             logger.error(f"Failed to fit model ({method}) for state identification: {fit_err}", exc_info=True)
             return {'error': f'Model fitting failed: {fit_err}'}
        # --- End State Identification ---


        # --- Process Model Results ---
        diffusion_states = {}
        track_states = {}
        state_centers_log_msd = [] # Log(<r^2>) value representing each state center

        if method == 'hmm':
            # Get HMM parameters (means are means of log features)
            state_centers_log_msd = model_fit.means_.flatten()
            variances_log_msd = model_fit.covars_.flatten() # Variance of log features
            transition_matrix = model_fit.transmat_
            # Sort states by mean log displacement (ascending)
            sort_idx = np.argsort(state_centers_log_msd)
            state_centers_log_msd = state_centers_log_msd[sort_idx]
            variances_log_msd = variances_log_msd[sort_idx]
            transition_matrix = transition_matrix[sort_idx][:, sort_idx]
            # Map original predicted states to sorted states
            state_map = {old: new for new, old in enumerate(sort_idx)}
            predicted_states_flat = np.array([state_map[s] for s in predicted_states_flat])

        elif method == 'kmeans':
            # Get KMeans cluster centers (means of log features)
            state_centers_log_msd = model_fit.cluster_centers_.flatten()
            variances_log_msd = np.zeros_like(state_centers_log_msd) # KMeans doesn't provide variance directly
            # Calculate variance within each cluster
            for k in range(n_states):
                cluster_features = X[predicted_states_flat == k]
                if len(cluster_features) > 0:
                     variances_log_msd[k] = np.var(cluster_features)

            # Sort states by center (ascending)
            sort_idx = np.argsort(state_centers_log_msd)
            state_centers_log_msd = state_centers_log_msd[sort_idx]
            variances_log_msd = variances_log_msd[sort_idx]
            # Map original predicted states to sorted states
            state_map = {old: new for new, old in enumerate(sort_idx)}
            predicted_states_flat = np.array([state_map[s] for s in predicted_states_flat])
            # Calculate transition matrix for KMeans (post-hoc)
            transition_counts = np.zeros((n_states, n_states), dtype=int)
            start = 0
            for length in track_lengths:
                 track_seq_states = predicted_states_flat[start:start+length]
                 for i in range(len(track_seq_states) - 1):
                      from_s, to_s = track_seq_states[i], track_seq_states[i+1]
                      transition_counts[from_s, to_s] += 1
                 start += length
            transition_matrix = np.zeros((n_states, n_states))
            row_sums = transition_counts.sum(axis=1)
            valid_rows = row_sums > 0
            transition_matrix[valid_rows] = transition_counts[valid_rows] / row_sums[valid_rows, np.newaxis]


        # Calculate diffusion coefficients from state centers
        # Feature is log(sq_disp), so center is log(<sq_disp>)_state
        # <sq_disp>_state = exp(center)
        # D_state = <sq_disp>_state / (4 * dt)
        mean_sq_disp_state = np.exp(state_centers_log_msd) # μm²
        diffusion_coefficients = mean_sq_disp_state / (4 * self.dt) # μm²/s

        # Assign states back to individual tracks
        start_idx = 0
        for i, track_id in enumerate(track_ids_processed):
            length = track_lengths[i]
            track_states[track_id] = predicted_states_flat[start_idx : start_idx + length].tolist()
            start_idx += length

        # Calculate overall state occupancy
        occupancy = np.zeros(n_states)
        if len(predicted_states_flat) > 0:
             state_counts = np.bincount(predicted_states_flat, minlength=n_states)
             occupancy = state_counts / len(predicted_states_flat)

        # Store final state information
        for i in range(n_states):
            diffusion_states[i] = {
                'state_id': i, # Corresponds to sorted state
                'mean_log_sq_disp': state_centers_log_msd[i],
                'variance_log_sq_disp': variances_log_msd[i],
                'mean_D': diffusion_coefficients[i], # μm²/s
                'occupancy': occupancy[i],
                'transition_probabilities': transition_matrix[i].tolist() # Row i = transitions FROM state i
            }
        # --- End Process Model Results ---

        self.diffusion_state_results = {
            'states': diffusion_states,
            'track_states': track_states, # Maps track_id to list of state indices
            'method': method,
            'n_states': n_states
        }
        logger.info(f"Diffusion state identification ({method}) complete.")
        return self.diffusion_state_results