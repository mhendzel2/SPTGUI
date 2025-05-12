"""
Diffusion analysis module for SPT Pro.

This module provides tools for diffusion coefficient calculation, MSD analysis,
and motion classification for single-particle tracking data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
import json
from functools import lru_cache
from numba import jit

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # MSD calculation parameters
    'msd_max_lag': 20,
    'msd_min_track_length': 10,
    
    # Diffusion model fitting parameters
    'max_fit_points': 10,
    
    # Track classification parameters
    'classification_min_track_length': 20,
    'classification_n_clusters': 3,
    
    # Performance parameters
    'use_numba': True,
    'cache_size': 128,
    'parallel_processing': True,
    'chunk_size': 1000
}


class DiffusionAnalyzer:
    """
    Analyzer for diffusion behavior and motion characteristics.
    
    Parameters
    ----------
    pixel_size : float, optional
        Pixel size in μm, by default 0.1
    frame_interval : float, optional
        Time between frames in seconds, by default 0.1
    config : dict, optional
        Configuration dictionary, by default None
    """
    
    def __init__(self, pixel_size=0.1, frame_interval=0.1, config=None):
        # Validate input parameters
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if frame_interval <= 0:
            raise ValueError("frame_interval must be positive")
            
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
        
        # Initialize configuration with defaults
        self.config = DEFAULT_CONFIG.copy()
        
        # Update with user-provided configuration
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("config must be a dictionary")
            self.config.update(config)
    
    @staticmethod
    @jit(nopython=True)
    def _fast_msd_calculation(positions, max_lag):
        """
        Numba-accelerated MSD calculation.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (n_points, 2)
        max_lag : int
            Maximum lag time
            
        Returns
        -------
        numpy.ndarray
            Array of MSD values for each lag
        """
        n_points = len(positions)
        msd_values = np.zeros(max_lag)
        counts = np.zeros(max_lag, dtype=np.int32)
        
        for lag in range(1, max_lag + 1):
            squared_displacements = np.zeros(n_points - lag)
            
            for i in range(n_points - lag):
                # Calculate squared displacement
                dx = positions[i + lag, 0] - positions[i, 0]
                dy = positions[i + lag, 1] - positions[i, 1]
                squared_displacements[i] = dx*dx + dy*dy
            
            msd_values[lag-1] = np.mean(squared_displacements)
            counts[lag-1] = len(squared_displacements)
        
        return msd_values, counts
    
    def compute_msd(self, tracks_df, max_lag=None, min_track_length=None):
        """
        Compute mean squared displacement for each track.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        max_lag : int, optional
            Maximum time lag to consider, by default None (uses config value)
        min_track_length : int, optional
            Minimum track length to analyze, by default None (uses config value)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with MSD values for each track and lag time
        pandas.DataFrame
            DataFrame with ensemble MSD values
        """
        try:
            # Validate input parameters
            if tracks_df is None or len(tracks_df) == 0:
                raise ValueError("Empty or None tracks_df provided")
            
            required_columns = ['track_id', 'frame', 'x', 'y']
            missing_columns = [col for col in required_columns if col not in tracks_df.columns]
            if missing_columns:
                raise ValueError(f"Required columns missing from tracks_df: {missing_columns}")
            
            # Use config values if parameters not provided
            max_lag = max_lag if max_lag is not None else self.config['msd_max_lag']
            min_track_length = min_track_length if min_track_length is not None else self.config['msd_min_track_length']
            
            # Validate parameters
            if not isinstance(max_lag, int) or max_lag < 1:
                raise ValueError("max_lag must be a positive integer")
                
            if not isinstance(min_track_length, int) or min_track_length < 2:
                raise ValueError("min_track_length must be an integer greater than 1")
            
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Get unique track IDs
            track_ids = tracks_df['track_id'].unique()
            
            # Prepare for parallel processing if enabled
            if self.config['parallel_processing'] and len(track_ids) > 1:
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing
                
                # Define worker function for parallel processing
                def process_track(track_id):
                    # Get track data
                    track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                    
                    # Skip tracks that are too short
                    if len(track) < min_track_length:
                        return []
                    
                    # Get positions
                    positions = track[['x_um', 'y_um']].values
                    frames = track['frame'].values
                    
                    # Compute MSD for different lag times
                    track_msd_data = []
                    
                    # Use Numba-accelerated calculation if enabled
                    if self.config['use_numba']:
                        actual_max_lag = min(max_lag, len(track) - 1)
                        msd_values, counts = self._fast_msd_calculation(positions, actual_max_lag)
                        
                        for lag in range(1, actual_max_lag + 1):
                            time_lag = lag * self.frame_interval
                            
                            # Add to MSD data
                            track_msd_data.append({
                                'track_id': track_id,
                                'lag': lag,
                                'time_lag': time_lag,
                                'msd': msd_values[lag-1],
                                'n_points': counts[lag-1]
                            })
                    else:
                        for lag in range(1, min(max_lag + 1, len(track))):
                            # Get positions separated by lag
                            lagged_positions = positions[lag:]
                            original_positions = positions[:-lag]
                            
                            # Compute squared displacements
                            squared_displacements = np.sum(
                                (lagged_positions - original_positions)**2,
                                axis=1
                            )
                            
                            # Compute time lag in seconds
                            time_lag = lag * self.frame_interval
                            
                            # Add to MSD data
                            track_msd_data.append({
                                'track_id': track_id,
                                'lag': lag,
                                'time_lag': time_lag,
                                'msd': np.mean(squared_displacements),
                                'n_points': len(squared_displacements)
                            })
                    
                    return track_msd_data
                
                # Process tracks in parallel
                max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers
                
                msd_data = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_track, track_ids))
                    
                    for result in results:
                        msd_data.extend(result)
            else:
                # Sequential processing
                msd_data = []
                
                for track_id in track_ids:
                    # Get track data
                    track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                    
                    # Skip tracks that are too short
                    if len(track) < min_track_length:
                        continue
                    
                    # Get positions
                    positions = track[['x_um', 'y_um']].values
                    frames = track['frame'].values
                    
                    # Compute MSD for different lag times
                    
                    # Use Numba-accelerated calculation if enabled
                    if self.config['use_numba']:
                        actual_max_lag = min(max_lag, len(track) - 1)
                        msd_values, counts = self._fast_msd_calculation(positions, actual_max_lag)
                        
                        for lag in range(1, actual_max_lag + 1):
                            time_lag = lag * self.frame_interval
                            
                            # Add to MSD data
                            msd_data.append({
                                'track_id': track_id,
                                'lag': lag,
                                'time_lag': time_lag,
                                'msd': msd_values[lag-1],
                                'n_points': counts[lag-1]
                            })
                    else:
                        for lag in range(1, min(max_lag + 1, len(track))):
                            # Get positions separated by lag
                            lagged_positions = positions[lag:]
                            original_positions = positions[:-lag]
                            
                            # Compute squared displacements
                            squared_displacements = np.sum(
                                (lagged_positions - original_positions)**2,
                                axis=1
                            )
                            
                            # Compute time lag in seconds
                            time_lag = lag * self.frame_interval
                            
                            # Add to MSD data
                            msd_data.append({
                                'track_id': track_id,
                                'lag': lag,
                                'time_lag': time_lag,
                                'msd': np.mean(squared_displacements),
                                'n_points': len(squared_displacements)
                            })
            
            msd_df = pd.DataFrame(msd_data)
            
            # Check if we have any data
            if len(msd_df) == 0:
                logger.warning("No tracks met the minimum length requirement")
                return pd.DataFrame(), pd.DataFrame()
            
            # Compute ensemble average MSD per lag time
            ensemble_msd = msd_df.groupby('lag').agg({
                'msd': 'mean',
                'n_points': 'sum'
            }).reset_index()
            
            ensemble_msd['time_lag'] = ensemble_msd['lag'] * self.frame_interval
            
            return msd_df, ensemble_msd
        
        except Exception as e:
            logger.error(f"Error computing MSD: {str(e)}")
            raise
    
    @lru_cache(maxsize=128)
    def _cached_diffusion_model(self, model_type, t, *params):
        """
        Cached diffusion model calculation.
        
        Parameters
        ----------
        model_type : str
            Type of diffusion model
        t : float
            Time lag
        *params : tuple
            Model parameters
            
        Returns
        -------
        float
            Model prediction
        """
        if model_type == 'simple':
            D = params[0]
            return 4 * D * t
        elif model_type == 'anomalous':
            D, alpha = params
            return 4 * D * t**alpha
        elif model_type == 'confined':
            D, L = params
            return L**2 * (1 - np.exp(-12 * D * t / L**2))
        elif model_type == 'directed':
            D, v = params
            return 4 * D * t + (v * t)**2
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit_diffusion_models(self, ensemble_msd, max_fit_points=None):
        """
        Fit different diffusion models to MSD curve.
        
        Parameters
        ----------
        ensemble_msd : pandas.DataFrame
            DataFrame with ensemble MSD values
        max_fit_points : int, optional
            Maximum number of points to use for fitting, by default None (uses config value)
            
        Returns
        -------
        dict
            Dictionary with fitted model parameters
        """
        try:
            # Validate input parameters
            if ensemble_msd is None or len(ensemble_msd) == 0:
                raise ValueError("Empty or None ensemble_msd provided")
            
            required_columns = ['time_lag', 'msd']
            missing_columns = [col for col in required_columns if col not in ensemble_msd.columns]
            if missing_columns:
                raise ValueError(f"Required columns missing from ensemble_msd: {missing_columns}")
            
            # Use config value if parameter not provided
            max_fit_points = max_fit_points if max_fit_points is not None else self.config['max_fit_points']
            
            # Validate parameter
            if not isinstance(max_fit_points, int) or max_fit_points < 2:
                raise ValueError("max_fit_points must be an integer greater than 1")
            
            # Limit number of points for fitting
            fit_data = ensemble_msd.head(max_fit_points)
            
            # Extract time lags and MSD values
            time_lags = fit_data['time_lag'].values
            msd_values = fit_data['msd'].values
            
            # Define model functions
            # Using vectorized operations for better performance
            
            # Model 1: Simple diffusion (MSD = 4Dt)
            def simple_diffusion(t, D):
                return np.vectorize(lambda x: self._cached_diffusion_model('simple', x, D))(t)
            
            # Model 2: Anomalous diffusion (MSD = 4Dt^α)
            def anomalous_diffusion(t, D, alpha):
                return np.vectorize(lambda x: self._cached_diffusion_model('anomalous', x, D, alpha))(t)
            
            # Model 3: Confined diffusion (MSD = L²[1-exp(-12Dt/L²)])
            def confined_diffusion(t, D, L):
                return np.vectorize(lambda x: self._cached_diffusion_model('confined', x, D, L))(t)
            
            # Model 4: Directed diffusion (MSD = 4Dt + (vt)²)
            def directed_diffusion(t, D, v):
                return np.vectorize(lambda x: self._cached_diffusion_model('directed', x, D, v))(t)
            
            # Fit models
            from scipy.optimize import curve_fit
            
            results = {}
            
            # Simple diffusion fit
            try:
                popt, pcov = curve_fit(simple_diffusion, time_lags, msd_values)
                D_simple = popt[0]
                D_simple_err = np.sqrt(np.diag(pcov))[0]
                
                # Calculate R²
                residuals = msd_values - simple_diffusion(time_lags, D_simple)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_simple = 1 - (ss_res / ss_tot)
                
                results['simple_diffusion'] = {
                    'D': D_simple,
                    'D_err': D_simple_err,
                    'r_squared': r_squared_simple
                }
            except Exception as e:
                logger.warning(f"Error fitting simple diffusion model: {str(e)}")
            
            # Anomalous diffusion fit
            try:
                popt, pcov = curve_fit(anomalous_diffusion, time_lags, msd_values, bounds=([0, 0], [np.inf, 2]))
                D_anom, alpha = popt
                D_anom_err, alpha_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - anomalous_diffusion(time_lags, D_anom, alpha)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_anom = 1 - (ss_res / ss_tot)
                
                results['anomalous_diffusion'] = {
                    'D': D_anom,
                    'D_err': D_anom_err,
                    'alpha': alpha,
                    'alpha_err': alpha_err,
                    'r_squared': r_squared_anom
                }
            except Exception as e:
                logger.warning(f"Error fitting anomalous diffusion model: {str(e)}")
            
            # Confined diffusion fit
            try:
                popt, pcov = curve_fit(confined_diffusion, time_lags, msd_values, bounds=([0, 0], [np.inf, np.inf]))
                D_conf, L = popt
                D_conf_err, L_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - confined_diffusion(time_lags, D_conf, L)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_conf = 1 - (ss_res / ss_tot)
                
                results['confined_diffusion'] = {
                    'D': D_conf,
                    'D_err': D_conf_err,
                    'L': L,
                    'L_err': L_err,
                    'r_squared': r_squared_conf
                }
            except Exception as e:
                logger.warning(f"Error fitting confined diffusion model: {str(e)}")
            
            # Directed diffusion fit
            try:
                popt, pcov = curve_fit(directed_diffusion, time_lags, msd_values)
                D_dir, v = popt
                D_dir_err, v_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - directed_diffusion(time_lags, D_dir, v)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_dir = 1 - (ss_res / ss_tot)
                
                results['directed_diffusion'] = {
                    'D': D_dir,
                    'D_err': D_dir_err,
                    'v': v,
                    'v_err': v_err,
                    'r_squared': r_squared_dir
                }
            except Exception as e:
                logger.warning(f"Error fitting directed diffusion model: {str(e)}")
            
            # Find best model based on R²
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['r_squared'])[0]
                results['best_model'] = best_model
            else:
                logger.warning("No diffusion models could be fitted successfully")
            
            return results
        
        except Exception as e:
            logger.error(f"Error fitting diffusion models: {str(e)}")
            raise
    
    @staticmethod
    @jit(nopython=True)
    def _fast_track_features(positions, frame_interval):
        """
        Numba-accelerated track feature calculation.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (n_points, 2)
        frame_interval : float
            Time between frames in seconds
            
        Returns
        -------
        tuple
            Tuple of track features (D_inst, asymmetry, straightness, alpha)
        """
        n_points = len(positions)
        
        # Compute displacements
        displacements = np.zeros(n_points - 1)
        for i in range(n_points - 1):
            dx = positions[i+1, 0] - positions[i, 0]
            dy = positions[i+1, 1] - positions[i, 1]
            displacements[i] = np.sqrt(dx*dx + dy*dy)
        
        # Compute track length
        track_length = np.sum(displacements)
        
        # Compute MSD values for first 4 lags
        max_lag = min(5, n_points - 1)
        msd_values = np.zeros(max_lag - 1)
        dt_values = np.zeros(max_lag - 1)
        
        for lag in range(1, max_lag):
            squared_displacements = np.zeros(n_points - lag)
            
            for i in range(n_points - lag):
                dx = positions[i + lag, 0] - positions[i, 0]
                dy = positions[i + lag, 1] - positions[i, 1]
                squared_displacements[i] = dx*dx + dy*dy
            
            msd_values[lag-1] = np.mean(squared_displacements)
            dt_values[lag-1] = lag * frame_interval
        
        # Compute instantaneous diffusion coefficient
        # Use linear regression to find slope
        n = len(dt_values)
        sum_x = np.sum(dt_values)
        sum_y = np.sum(msd_values)
        sum_xy = np.sum(dt_values * msd_values)
        sum_xx = np.sum(dt_values * dt_values)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        D_inst = slope / 4.0
        
        # Compute asymmetry
        # Calculate gyration tensor
        mean_x = np.mean(positions[:, 0])
        mean_y = np.mean(positions[:, 1])
        
        cov_xx = 0.0
        cov_xy = 0.0
        cov_yy = 0.0
        
        for i in range(n_points):
            dx = positions[i, 0] - mean_x
            dy = positions[i, 1] - mean_y
            
            cov_xx += dx * dx
            cov_xy += dx * dy
            cov_yy += dy * dy
        
        cov_xx /= n_points
        cov_xy /= n_points
        cov_yy /= n_points
        
        # Calculate eigenvalues
        trace = cov_xx + cov_yy
        det = cov_xx * cov_yy - cov_xy * cov_xy
        
        # Solve characteristic equation: lambda^2 - trace*lambda + det = 0
        discriminant = trace * trace - 4 * det
        
        if discriminant > 0:
            sqrt_discriminant = np.sqrt(discriminant)
            eig_val1 = (trace + sqrt_discriminant) / 2
            eig_val2 = (trace - sqrt_discriminant) / 2
            
            # Ensure eig_val1 >= eig_val2
            if eig_val1 < eig_val2:
                eig_val1, eig_val2 = eig_val2, eig_val1
                
            asymmetry = eig_val2 / eig_val1 if eig_val1 > 0 else 1.0
        else:
            # Equal eigenvalues or complex eigenvalues (shouldn't happen for real symmetric matrix)
            asymmetry = 1.0
        
        # Compute straightness
        if track_length > 0:
            dx = positions[-1, 0] - positions[0, 0]
            dy = positions[-1, 1] - positions[0, 1]
            end_to_end = np.sqrt(dx*dx + dy*dy)
            straightness = end_to_end / track_length
        else:
            straightness = 0.0
        
        # Compute anomalous exponent (α) from MSD curve
        if len(msd_values) >= 3:
            # Use linear regression in log-log space
            log_dt = np.log(dt_values)
            log_msd = np.log(msd_values)
            
            n = len(log_dt)
            sum_x = np.sum(log_dt)
            sum_y = np.sum(log_msd)
            sum_xy = np.sum(log_dt * log_msd)
            sum_xx = np.sum(log_dt * log_dt)
            
            # Calculate slope
            alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        else:
            alpha = 1.0  # Default to normal diffusion
        
        return D_inst, asymmetry, straightness, alpha
    
    def classify_tracks(self, tracks_df, min_track_length=None, n_clusters=None):
        """
        Classify tracks based on diffusion properties.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        min_track_length : int, optional
            Minimum track length to consider, by default None (uses config value)
        n_clusters : int, optional
            Number of clusters for classification, by default None (uses config value)
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with track classifications
        """
        try:
            # Validate input parameters
            if tracks_df is None or len(tracks_df) == 0:
                raise ValueError("Empty or None tracks_df provided")
            
            required_columns = ['track_id', 'frame', 'x', 'y']
            missing_columns = [col for col in required_columns if col not in tracks_df.columns]
            if missing_columns:
                raise ValueError(f"Required columns missing from tracks_df: {missing_columns}")
            
            # Use config values if parameters not provided
            min_track_length = min_track_length if min_track_length is not None else self.config['classification_min_track_length']
            n_clusters = n_clusters if n_clusters is not None else self.config['classification_n_clusters']
            
            # Validate parameters
            if not isinstance(min_track_length, int) or min_track_length < 4:
                raise ValueError("min_track_length must be an integer greater than 3")
                
            if not isinstance(n_clusters, int) or n_clusters < 2:
                raise ValueError("n_clusters must be an integer greater than 1")
            
            # Compute per-track diffusion parameters
            track_ids = tracks_df['track_id'].unique()
            
            # Prepare for parallel processing if enabled
            if self.config['parallel_processing'] and len(track_ids) > 1:
                from concurrent.futures import ProcessPoolExecutor
                import multiprocessing
                
                # Define worker function for parallel processing
                def process_track(track_id):
                    # Get track data
                    track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                    
                    # Skip tracks that are too short
                    if len(track) < min_track_length:
                        return None
                    
                    # Convert coordinates to μm
                    x = track['x'] * self.pixel_size
                    y = track['y'] * self.pixel_size
                    
                    # Get positions
                    positions = np.vstack([x, y]).T
                    
                    # Compute track features
                    if self.config['use_numba']:
                        D_inst, asymmetry, straightness, alpha = self._fast_track_features(positions, self.frame_interval)
                    else:
                        # Compute track length
                        displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                        track_length = np.sum(displacements)
                        
                        # Compute features for classification
                        
                        # 1. Instantaneous diffusion coefficient
                        # Use first 4 points of MSD curve for linear fit
                        msd_values = []
                        
                        for lag in range(1, min(5, len(track) - 1)):
                            dt = lag * self.frame_interval
                            dx = x.iloc[lag:].values - x.iloc[:-lag].values
                            dy = y.iloc[lag:].values - y.iloc[:-lag].values
                            msd = np.mean(dx**2 + dy**2)
                            msd_values.append((dt, msd))
                        
                        dt_values, msd_values = zip(*msd_values)
                        slope, _, _, _, _ = stats.linregress(dt_values, msd_values)
                        D_inst = slope / 4.0
                        
                        # 2. Asymmetry (ratio of eigenvalues of gyration tensor)
                        if len(track) >= 4:
                            positions_centered = positions - np.mean(positions, axis=0)
                            gyration_tensor = np.cov(positions_centered.T)
                            eig_vals = np.linalg.eigvals(gyration_tensor)
                            asymmetry = np.min(eig_vals) / np.max(eig_vals) if np.max(eig_vals) > 0 else 1.0
                        else:
                            asymmetry = 1.0
                        
                        # 3. Straightness (end-to-end distance / path length)
                        if track_length > 0:
                            end_to_end = np.sqrt(
                                (x.iloc[-1] - x.iloc[0])**2 + 
                                (y.iloc[-1] - y.iloc[0])**2
                            )
                            straightness = end_to_end / track_length
                        else:
                            straightness = 0.0
                        
                        # 4. Anomalous exponent (α) from MSD curve
                        if len(msd_values) >= 3:
                            log_dt = np.log(dt_values)
                            log_msd = np.log(msd_values)
                            slope, _, _, _, _ = stats.linregress(log_dt, log_msd)
                            alpha = slope
                        else:
                            alpha = 1.0  # Default to normal diffusion
                    
                    # Add features
                    return {
                        'track_id': track_id,
                        'D': D_inst,
                        'asymmetry': asymmetry,
                        'straightness': straightness,
                        'alpha': alpha,
                        'track_length': len(track)
                    }
                
                # Process tracks in parallel
                max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 workers
                
                track_features = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_track, track_ids))
                    
                    for result in results:
                        if result is not None:
                            track_features.append(result)
            else:
                # Sequential processing
                track_features = []
                
                for track_id in track_ids:
                    # Get track data
                    track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                    
                    # Skip tracks that are too short
                    if len(track) < min_track_length:
                        continue
                    
                    # Convert coordinates to μm
                    x = track['x'] * self.pixel_size
                    y = track['y'] * self.pixel_size
                    
                    # Get positions
                    positions = np.vstack([x, y]).T
                    
                    # Compute track features
                    if self.config['use_numba']:
                        D_inst, asymmetry, straightness, alpha = self._fast_track_features(positions, self.frame_interval)
                    else:
                        # Compute track length
                        displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                        track_length = np.sum(displacements)
                        
                        # Compute features for classification
                        
                        # 1. Instantaneous diffusion coefficient
                        # Use first 4 points of MSD curve for linear fit
                        msd_values = []
                        
                        for lag in range(1, min(5, len(track) - 1)):
                            dt = lag * self.frame_interval
                            dx = x.iloc[lag:].values - x.iloc[:-lag].values
                            dy = y.iloc[lag:].values - y.iloc[:-lag].values
                            msd = np.mean(dx**2 + dy**2)
                            msd_values.append((dt, msd))
                        
                        dt_values, msd_values = zip(*msd_values)
                        slope, _, _, _, _ = stats.linregress(dt_values, msd_values)
                        D_inst = slope / 4.0
                        
                        # 2. Asymmetry (ratio of eigenvalues of gyration tensor)
                        if len(track) >= 4:
                            positions_centered = positions - np.mean(positions, axis=0)
                            gyration_tensor = np.cov(positions_centered.T)
                            eig_vals = np.linalg.eigvals(gyration_tensor)
                            asymmetry = np.min(eig_vals) / np.max(eig_vals) if np.max(eig_vals) > 0 else 1.0
                        else:
                            asymmetry = 1.0
                        
                        # 3. Straightness (end-to-end distance / path length)
                        if track_length > 0:
                            end_to_end = np.sqrt(
                                (x.iloc[-1] - x.iloc[0])**2 + 
                                (y.iloc[-1] - y.iloc[0])**2
                            )
                            straightness = end_to_end / track_length
                        else:
                            straightness = 0.0
                        
                        # 4. Anomalous exponent (α) from MSD curve
                        if len(msd_values) >= 3:
                            log_dt = np.log(dt_values)
                            log_msd = np.log(msd_values)
                            slope, _, _, _, _ = stats.linregress(log_dt, log_msd)
                            alpha = slope
                        else:
                            alpha = 1.0  # Default to normal diffusion
                    
                    # Add features
                    track_features.append({
                        'track_id': track_id,
                        'D': D_inst,
                        'asymmetry': asymmetry,
                        'straightness': straightness,
                        'alpha': alpha,
                        'track_length': len(track)
                    })
            
            # Create DataFrame with features
            features_df = pd.DataFrame(track_features)
            
            if len(features_df) < n_clusters:
                logger.warning(f"Not enough tracks ({len(features_df)}) for {n_clusters} clusters")
                return None
            
            # Normalize features for clustering
            from sklearn.preprocessing import StandardScaler
            
            # Select features for clustering
            X = features_df[['D', 'asymmetry', 'straightness', 'alpha']].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to features DataFrame
            features_df['cluster'] = cluster_labels
            
            # Compute cluster statistics
            cluster_stats = []
            
            for cluster_id in range(n_clusters):
                cluster_tracks = features_df[features_df['cluster'] == cluster_id]
                
                # Compute mean and std of features
                D_mean = cluster_tracks['D'].mean()
                D_std = cluster_tracks['D'].std()
                asymmetry_mean = cluster_tracks['asymmetry'].mean()
                straightness_mean = cluster_tracks['straightness'].mean()
                alpha_mean = cluster_tracks['alpha'].mean()
                
                # Determine diffusion type based on alpha
                if alpha_mean > 1.1:
                    diffusion_type = "Superdiffusive"
                elif alpha_mean < 0.9:
                    diffusion_type = "Subdiffusive"
                else:
                    diffusion_type = "Normal diffusion"
                
                # Add to cluster statistics
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'n_tracks': len(cluster_tracks),
                    'D_mean': D_mean,
                    'D_std': D_std,
                    'asymmetry_mean': asymmetry_mean,
                    'straightness_mean': straightness_mean,
                    'alpha_mean': alpha_mean,
                    'diffusion_type': diffusion_type
                })
            
            # Create cluster stats DataFrame
            cluster_df = pd.DataFrame(cluster_stats)
            
            # Add diffusion type to features DataFrame
            features_df['diffusion_type'] = features_df['cluster'].map(
                {row['cluster_id']: row['diffusion_type'] for _, row in cluster_df.iterrows()}
            )
            
            return features_df, cluster_df
        
        except Exception as e:
            logger.error(f"Error classifying tracks: {str(e)}")
            raise
    
    @classmethod
    def load_config(cls, config_file):
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        config_file : str
            Path to configuration file
            
        Returns
        -------
        dict
            Configuration dictionary
        """
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")
                
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            return config
        
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_config(self, config_file):
        """
        Save current configuration to a JSON file.
        
        Parameters
        ----------
        config_file : str
            Path to save configuration file
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info(f"Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
