"""
Gel structure analysis module for SPT Analysis.

This module provides tools for analyzing jump size distributions, gel pore sizes,
and characterizing the physical properties of gel-like compartments.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats, optimize

logger = logging.getLogger(__name__)


class GelStructureAnalyzer:
    """
    Analyzer for jump size distributions and gel structure properties.
    
    This class provides methods for analyzing jump size distributions,
    determining gel pore sizes, and characterizing the physical properties
    of gel-like compartments.
    
    Parameters
    ----------
    dt : float, optional
        Time interval between frames in seconds, by default 0.014
    min_jumps : int, optional
        Minimum number of jumps for analysis, by default 100
    """
    
    def __init__(self, dt=0.014, min_jumps=100):
        self.dt = dt
        self.min_jumps = min_jumps
        
        # Results storage
        self.jump_distributions = {}
        self.pore_size_results = {}
        self.gel_properties = {}
    
    def analyze_jump_distributions(self, tracks_df, compartment_masks=None):
        """
        Analyze jump size distributions to characterize gel structure.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        compartment_masks : dict, optional
            Dictionary of binary masks for compartments, by default None
            
        Returns
        -------
        dict
            Dictionary of jump distribution analysis results
        """
        try:
            # Calculate jumps for all tracks
            all_jumps = []
            
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                positions = track_df[['x', 'y']].values
                
                # Calculate jumps
                jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                all_jumps.extend(jumps)
            
            # Check if we have enough jumps
            if len(all_jumps) < self.min_jumps:
                return {
                    'status': 'Insufficient jumps for analysis'
                }
            
            # Calculate jump distribution by compartment if masks provided
            compartment_jumps = {}
            
            if compartment_masks:
                for comp_name, mask in compartment_masks.items():
                    # Filter tracks by compartment
                    comp_jumps = []
                    
                    for track_id, track_df in tracks_df.groupby('track_id'):
                        # Sort by frame
                        track_df = track_df.sort_values('frame')
                        
                        positions = track_df[['x', 'y']].values
                        
                        # Check if track is in this compartment
                        in_compartment = False
                        for x, y in positions.astype(int):
                            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                                if mask[y, x]:
                                    in_compartment = True
                                    break
                        
                        if in_compartment:
                            # Calculate jumps
                            jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                            comp_jumps.extend(jumps)
                    
                    if len(comp_jumps) >= self.min_jumps:
                        compartment_jumps[comp_name] = comp_jumps
            
            # Analyze jump distributions
            jump_analysis = self._analyze_jump_distribution(all_jumps)
            
            # Analyze compartment-specific distributions
            compartment_analysis = {}
            
            for comp, jumps in compartment_jumps.items():
                compartment_analysis[comp] = self._analyze_jump_distribution(jumps)
            
            # Store results
            self.jump_distributions = {
                'all': jump_analysis,
                'by_compartment': compartment_analysis
            }
            
            return self.jump_distributions
        
        except Exception as e:
            logger.error(f"Error in jump distribution analysis: {str(e)}")
            raise
    
    def _analyze_jump_distribution(self, jumps):
        """
        Analyze a jump size distribution.
        
        Parameters
        ----------
        jumps : list
            List of jump sizes
            
        Returns
        -------
        dict
            Dictionary of jump distribution analysis results
        """
        # Calculate basic statistics
        mean_jump = np.mean(jumps)
        median_jump = np.median(jumps)
        std_jump = np.std(jumps)
        
        # Create histogram
        hist, bin_edges = np.histogram(jumps, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Fit different distributions
        # 1. Gaussian (normal diffusion)
        try:
            gaussian_params = stats.norm.fit(jumps)
            gaussian_pdf = stats.norm.pdf(bin_centers, *gaussian_params)
            _, gaussian_pval = stats.kstest(jumps, 'norm', args=gaussian_params)
        except:
            gaussian_params = None
            gaussian_pdf = None
            gaussian_pval = 0
        
        # 2. Exponential (typical for some gels)
        try:
            exp_params = stats.expon.fit(jumps)
            exp_pdf = stats.expon.pdf(bin_centers, *exp_params)
            _, exp_pval = stats.kstest(jumps, 'expon', args=exp_params)
        except:
            exp_params = None
            exp_pdf = None
            exp_pval = 0
        
        # 3. Gamma (often good for crowded/gel environments)
        try:
            gamma_params = stats.gamma.fit(jumps)
            gamma_pdf = stats.gamma.pdf(bin_centers, *gamma_params)
            _, gamma_pval = stats.kstest(jumps, 'gamma', args=gamma_params)
        except:
            gamma_params = None
            gamma_pdf = None
            gamma_pval = 0
        
        # Determine best fit
        pvals = [gaussian_pval, exp_pval, gamma_pval]
        distributions = ['gaussian', 'exponential', 'gamma']
        
        if max(pvals) > 0.05:
            best_fit = distributions[np.argmax(pvals)]
            good_fit = True
        else:
            best_fit = distributions[np.argmax(pvals)]
            good_fit = False
        
        # Return results
        return {
            'mean_jump': mean_jump,
            'median_jump': median_jump,
            'std_jump': std_jump,
            'histogram': {
                'bin_centers': bin_centers.tolist(),
                'counts': hist.tolist()
            },
            'fits': {
                'gaussian': {
                    'params': gaussian_params,
                    'pdf': gaussian_pdf.tolist() if gaussian_pdf is not None else None,
                    'p_value': gaussian_pval
                },
                'exponential': {
                    'params': exp_params,
                    'pdf': exp_pdf.tolist() if exp_pdf is not None else None,
                    'p_value': exp_pval
                },
                'gamma': {
                    'params': gamma_params,
                    'pdf': gamma_pdf.tolist() if gamma_pdf is not None else None,
                    'p_value': gamma_pval
                }
            },
            'best_fit': best_fit,
            'good_fit': good_fit
        }
    
    def estimate_pore_size(self, tracks_df, compartment_masks=None, method='jump_distribution'):
        """
        Estimate gel pore size from tracking data.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        compartment_masks : dict, optional
            Dictionary of binary masks for compartments, by default None
        method : str, optional
            Method for pore size estimation, by default 'jump_distribution'
            
        Returns
        -------
        dict
            Dictionary of pore size estimation results
        """
        try:
            # Check if we have jump distribution results
            if not self.jump_distributions:
                self.analyze_jump_distributions(tracks_df, compartment_masks)
            
            # Estimate pore size for all tracks
            all_pore_size = self._estimate_pore_size_from_jumps(
                self.jump_distributions.get('all', {}),
                method=method
            )
            
            # Estimate pore size by compartment
            compartment_pore_sizes = {}
            
            for comp, comp_analysis in self.jump_distributions.get('by_compartment', {}).items():
                compartment_pore_sizes[comp] = self._estimate_pore_size_from_jumps(
                    comp_analysis,
                    method=method
                )
            
            # Store results
            self.pore_size_results = {
                'all': all_pore_size,
                'by_compartment': compartment_pore_sizes
            }
            
            return self.pore_size_results
        
        except Exception as e:
            logger.error(f"Error in pore size estimation: {str(e)}")
            raise
    
    def _estimate_pore_size_from_jumps(self, jump_analysis, method='jump_distribution'):
        """
        Estimate pore size from jump distribution analysis.
        
        Parameters
        ----------
        jump_analysis : dict
            Dictionary of jump distribution analysis results
        method : str, optional
            Method for pore size estimation, by default 'jump_distribution'
            
        Returns
        -------
        dict
            Dictionary of pore size estimation results
        """
        if not jump_analysis:
            return {
                'status': 'No jump analysis results available'
            }
        
        # Different methods for pore size estimation
        if method == 'jump_distribution':
            # Use characteristic jump size as proxy for pore size
            # For gels, the mode of the jump distribution often corresponds to pore size
            
            # Get histogram data
            hist_data = jump_analysis.get('histogram', {})
            bin_centers = hist_data.get('bin_centers', [])
            counts = hist_data.get('counts', [])
            
            if not bin_centers or not counts:
                return {
                    'status': 'No histogram data available'
                }
            
            # Find mode (peak) of distribution
            mode_idx = np.argmax(counts)
            mode_jump = bin_centers[mode_idx]
            
            # Pore size is approximately the mode of jump distribution
            pore_size = mode_jump
            
            # Get additional statistics
            mean_jump = jump_analysis.get('mean_jump')
            median_jump = jump_analysis.get('median_jump')
            
            # Estimate uncertainty
            uncertainty = abs(mean_jump - mode_jump) / mode_jump if mode_jump > 0 else np.nan
            
            return {
                'pore_size': pore_size,
                'uncertainty': uncertainty,
                'method': 'jump_distribution_mode',
                'mean_jump': mean_jump,
                'median_jump': median_jump,
                'mode_jump': mode_jump
            }
            
        elif method == 'confinement':
            # Use best-fit distribution parameters
            best_fit = jump_analysis.get('best_fit')
            fits = jump_analysis.get('fits', {})
            
            if best_fit == 'gamma':
                # Gamma distribution parameters can be related to pore size
                gamma_params = fits.get('gamma', {}).get('params')
                
                if gamma_params and len(gamma_params) >= 2:
                    # Shape parameter relates to confinement
                    shape = gamma_params[0]
                    scale = gamma_params[2]  # Location is at index 1
                    
                    # For confined diffusion, shape parameter < 2 indicates strong confinement
                    confinement_strength = 2.0 / shape if shape > 0 else np.nan
                    
                    # Pore size estimate from gamma parameters
                    pore_size = scale * shape  # Mode of gamma distribution
                    
                    return {
                        'pore_size': pore_size,
                        'uncertainty': 1.0 / np.sqrt(shape) if shape > 0 else np.nan,
                        'method': 'gamma_distribution',
                        'confinement_strength': confinement_strength,
                        'gamma_shape': shape,
                        'gamma_scale': scale
                    }
            
            # Fallback to mean jump
            mean_jump = jump_analysis.get('mean_jump')
            return {
                'pore_size': mean_jump,
                'uncertainty': jump_analysis.get('std_jump') / mean_jump if mean_jump > 0 else np.nan,
                'method': 'mean_jump',
                'note': 'Fallback method, less reliable'
            }
            
        else:
            return {
                'status': f'Unknown method: {method}'
            }
    
    def calculate_gel_properties(self, tracks_df, particle_radius=5.0, temperature=298):
        """
        Calculate physical properties of gel-like environments.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        particle_radius : float, optional
            Radius of tracked particle in nm, by default 5.0
        temperature : float, optional
            Temperature in Kelvin, by default 298
            
        Returns
        -------
        dict
            Dictionary of gel property results
        """
        try:
            # Convert particle radius to meters
            particle_radius_m = particle_radius * 1e-9
            
            # Boltzmann constant
            kB = 1.38e-23  # J/K
            
            # Calculate MSD for all tracks
            msd_results = self._calculate_msd(tracks_df)
            
            # Extract subdiffusive tracks (alpha < 0.8)
            subdiffusive_tracks = []
            
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                positions = track_df[['x', 'y']].values
                
                # Skip short tracks
                if len(positions) < 10:
                    continue
                
                # Calculate MSD
                lag_times = []
                msd_values = []
                
                for lag in range(1, min(10, len(positions) // 2)):
                    disp = positions[lag:] - positions[:-lag]
                    sq_disp = np.sum(disp**2, axis=1)
                    lag_times.append(lag * self.dt)
                    msd_values.append(np.mean(sq_disp))
                
                # Fit power law
                log_tau = np.log(lag_times)
                log_msd = np.log(msd_values)
                slope, intercept = np.polyfit(log_tau, log_msd, 1)
                
                alpha = slope
                
                # Check if subdiffusive
                if alpha < 0.8:
                    subdiffusive_tracks.append({
                        'track_id': track_id,
                        'alpha': alpha,
                        'positions': positions,
                        'msd_values': msd_values,
                        'lag_times': lag_times
                    })
            
            if not subdiffusive_tracks:
                return {
                    'status': 'No subdiffusive tracks found'
                }
            
            # Calculate gel properties
            
            # 1. Mesh size from confinement
            mesh_sizes = []
            
            for track in subdiffusive_tracks:
                positions = track['positions']
                
                # Calculate radius of gyration as estimate of confinement
                center = np.mean(positions, axis=0)
                rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
                
                mesh_sizes.append(rg)
            
            mean_mesh_size = np.mean(mesh_sizes)
            
            # 2. Elastic modulus from confinement
            # G ~ kT/ξ³, where ξ is mesh size
            elastic_modulus = kB * temperature / (mean_mesh_size**3 * 1e-18)  # Convert to Pa
            
            # 3. Viscosity from MSD
            # For subdiffusive motion, η ~ kT/(6πa*D*t^(1-α))
            # We'll evaluate at t = 1s
            
            viscosities = []
            
            for track in subdiffusive_tracks:
                alpha = track['alpha']
                msd_values = track['msd_values']
                lag_times = track['lag_times']
                
                # Find MSD at or near t = 1s
                target_time = 1.0  # seconds
                idx = np.argmin(np.abs(np.array(lag_times) - target_time))
                
                msd_at_target = msd_values[idx]
                time_at_target = lag_times[idx]
                
                # Calculate apparent diffusion coefficient
                D_app = msd_at_target / (4 * time_at_target**alpha)
                
                # Calculate viscosity
                viscosity = kB * temperature / (6 * np.pi * particle_radius_m * D_app * time_at_target**(1-alpha))
                
                viscosities.append(viscosity)
            
            mean_viscosity = np.mean(viscosities)
            
            # 4. Calculate storage and loss moduli
            # For subdiffusive motion:
            # G'(ω) ~ ω^α * cos(πα/2)
            # G''(ω) ~ ω^α * sin(πα/2)
            
            # We'll evaluate at ω = 1 rad/s
            alphas = [track['alpha'] for track in subdiffusive_tracks]
            mean_alpha = np.mean(alphas)
            
            # Calculate moduli prefactor
            # G* ~ kT/(πa*<Δr²(1/ω)>*Γ(1+α))
            from scipy.special import gamma
            
            # Find average MSD at t = 1s
            msd_at_1s = []
            
            for track in subdiffusive_tracks:
                msd_values = track['msd_values']
                lag_times = track['lag_times']
                
                # Interpolate to t = 1s
                if min(lag_times) <= 1.0 <= max(lag_times):
                    from scipy.interpolate import interp1d
                    msd_interp = interp1d(lag_times, msd_values)
                    msd_at_1s.append(float(msd_interp(1.0)))
                elif len(lag_times) > 0:
                    # Extrapolate using power law
                    alpha = track['alpha']
                    msd_0 = msd_values[0]
                    t_0 = lag_times[0]
                    msd_at_1s.append(msd_0 * (1.0 / t_0)**alpha)
            
            if msd_at_1s:
                mean_msd_1s = np.mean(msd_at_1s)
                
                # Calculate moduli prefactor
                G_prefactor = kB * temperature / (np.pi * particle_radius_m * mean_msd_1s * gamma(1 + mean_alpha))
                
                # Calculate storage and loss moduli at ω = 1 rad/s
                G_prime = G_prefactor * np.cos(np.pi * mean_alpha / 2)
                G_loss = G_prefactor * np.sin(np.pi * mean_alpha / 2)
            else:
                G_prime = np.nan
                G_loss = np.nan
            
            # Store results
            gel_properties = {
                'n_subdiffusive_tracks': len(subdiffusive_tracks),
                'mean_alpha': mean_alpha,
                'mesh_size': mean_mesh_size,
                'elastic_modulus': elastic_modulus,
                'viscosity': mean_viscosity,
                'storage_modulus': G_prime,
                'loss_modulus': G_loss,
                'temperature': temperature,
                'particle_radius': particle_radius
            }
            
            self.gel_properties = gel_properties
            
            return gel_properties
        
        except Exception as e:
            logger.error(f"Error in gel property calculation: {str(e)}")
            raise
    
    def _calculate_msd(self, tracks_df):
        """
        Calculate MSD for all tracks.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
            
        Returns
        -------
        dict
            Dictionary of MSD results
        """
        # Maximum lag time to consider
        max_lag = 30
        
        # Initialize arrays for MSD values
        all_displacements = [[] for _ in range(max_lag)]
        
        # Process each track
        for track_id, track_df in tracks_df.groupby('track_id'):
            # Sort by frame
            track_df = track_df.sort_values('frame')
            
            positions = track_df[['x', 'y']].values
            
            # Calculate displacements for different lag times
            for lag in range(1, min(max_lag + 1, len(positions))):
                # Get positions separated by lag frames
                pos1 = positions[:-lag]
                pos2 = positions[lag:]
                
                # Calculate squared displacements
                squared_displacements = np.sum((pos2 - pos1)**2, axis=1)
                
                # Add to collection for this lag time
                all_displacements[lag-1].extend(squared_displacements)
        
        # Calculate MSD for each lag time
        tau = np.arange(1, max_lag + 1) * self.dt
        msd = np.array([np.mean(disps) if disps else np.nan for disps in all_displacements])
        msd_stderr = np.array([np.std(disps) / np.sqrt(len(disps)) if len(disps) > 1 else np.nan 
                              for disps in all_displacements])
        
        return {
            'tau': tau,
            'msd': msd,
            'msd_stderr': msd_stderr
        }


class DiffusionGelPopulationAnalyzer:
    """
    Analyzer for identifying and characterizing diffusion populations.
    
    This class provides methods for segmenting trajectories into different
    diffusion states and analyzing the properties of diffusion populations.
    
    Parameters
    ----------
    dt : float, optional
        Time interval between frames in seconds, by default 0.014
    min_segment_length : int, optional
        Minimum length of trajectory segments, by default 5
    """
    
    def __init__(self, dt=0.014, min_segment_length=5):
        self.dt = dt
        self.min_segment_length = min_segment_length
        
        # Results storage
        self.segmented_trajectories = {}
        self.jump_mixture_results = {}
        self.diffusion_states = {}
    
    def segment_trajectories(self, tracks_df, method='change_point', window_size=10):
        """
        Segment trajectories into different diffusion states.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        method : str, optional
            Segmentation method, by default 'change_point'
        window_size : int, optional
            Window size for local analysis, by default 10
            
        Returns
        -------
        dict
            Dictionary of segmented trajectories
        """
        try:
            # Initialize results
            segmented_trajectories = {}
            
            # Process each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                positions = track_df[['x', 'y']].values
                frames = track_df['frame'].values
                
                # Skip short tracks
                if len(positions) < 2 * self.min_segment_length:
                    continue
                
                # Calculate squared jumps
                squared_jumps = np.sum(np.diff(positions, axis=0)**2, axis=1)
                
                # Segment trajectory based on specified method
                segment_boundaries = []
                
                if method == 'change_point':
                    # Use change point detection
                    try:
                        import ruptures as rpt
                        
                        # Apply change point detection to squared jumps
                        model = "l2"  # L2 cost for change point detection
                        algo = rpt.Pelt(model=model).fit(squared_jumps[:, np.newaxis])
                        result = algo.predict(pen=np.log(len(squared_jumps)) * 0.5)
                        
                        # Convert to segment boundaries
                        segment_boundaries = [(0, result[0])]
                        for i in range(len(result)-1):
                            segment_boundaries.append((result[i], result[i+1]))
                        
                        # Add final segment
                        if result:
                            segment_boundaries.append((result[-1], len(squared_jumps)))
                        else:
                            segment_boundaries = [(0, len(squared_jumps))]
                        
                    except ImportError:
                        # Fallback to sliding window if ruptures not available
                        method = 'sliding_window'
                
                if method == 'sliding_window':
                    # Use sliding window to detect changes in jump statistics
                    current_segment_start = 0
                    
                    for i in range(window_size, len(squared_jumps) - window_size):
                        # Compare statistics in adjacent windows
                        window1 = squared_jumps[i-window_size:i]
                        window2 = squared_jumps[i:i+window_size]
                        
                        # Calculate mean squared jump in each window
                        msj1 = np.mean(window1)
                        msj2 = np.mean(window2)
                        
                        # Check if there's a significant change
                        if abs(msj2 - msj1) > 0.5 * max(msj1, msj2):
                            # End current segment
                            segment_boundaries.append((current_segment_start, i))
                            current_segment_start = i
                    
                    # Add final segment
                    segment_boundaries.append((current_segment_start, len(squared_jumps)))
                
                else:
                    raise ValueError(f"Unknown segmentation method: {method}")
                
                # Filter segments that are too short
                segment_boundaries = [seg for seg in segment_boundaries 
                                    if seg[1] - seg[0] >= self.min_segment_length]
                
                # Analyze properties of each segment
                segments = []
                
                for start_idx, end_idx in segment_boundaries:
                    segment_positions = positions[start_idx:end_idx+1]
                    segment_frames = frames[start_idx:end_idx+1]
                    
                    # Skip segments that are too short after boundary adjustment
                    if len(segment_positions) < self.min_segment_length:
                        continue
                    
                    # Calculate MSD for this segment
                    lag_times = []
                    msd_values = []
                    
                    for lag in range(1, min(11, len(segment_positions) // 2)):
                        disp = segment_positions[lag:] - segment_positions[:-lag]
                        sq_disp = np.sum(disp**2, axis=1)
                        lag_times.append(lag * self.dt)
                        msd_values.append(np.mean(sq_disp))
                    
                    if not msd_values:
                        continue
                    
                    # Fit MSD to power law MSD = 4*D*t^alpha
                    try:
                        # Log-log fit
                        log_tau = np.log(lag_times)
                        log_msd = np.log(msd_values)
                        slope, intercept = np.polyfit(log_tau, log_msd, 1)
                        alpha = slope
                        D = np.exp(intercept - np.log(4))
                    except:
                        alpha = None
                        D = None
                    
                    # Calculate average jump size
                    segment_jumps = np.sqrt(np.sum(np.diff(segment_positions, axis=0)**2, axis=1))
                    mean_jump = np.mean(segment_jumps)
                    
                    # Classify diffusion mode based on alpha
                    if alpha is not None:
                        if alpha < 0.7:
                            diffusion_mode = "Subdiffusion"
                        elif alpha > 1.3:
                            diffusion_mode = "Superdiffusion"
                        else:
                            diffusion_mode = "Normal diffusion"
                    else:
                        diffusion_mode = "Unknown"
                    
                    # Store segment information
                    segment = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_frame': segment_frames[0],
                        'end_frame': segment_frames[-1],
                        'n_frames': len(segment_frames),
                        'positions': segment_positions,
                        'alpha': alpha,
                        'diffusion_coefficient': D,
                        'mean_jump': mean_jump,
                        'diffusion_mode': diffusion_mode
                    }
                    
                    segments.append(segment)
                
                segmented_trajectories[track_id] = segments
            
            # Store results
            self.segmented_trajectories = segmented_trajectories
            
            return segmented_trajectories
        
        except Exception as e:
            logger.error(f"Error in trajectory segmentation: {str(e)}")
            raise
    
    def calculate_population_statistics(self):
        """
        Calculate statistics on identified diffusion populations.
        
        Returns
        -------
        dict
            Dictionary of population statistics
        """
        try:
            population_stats = {}
            
            # Jump mixture statistics
            if self.jump_mixture_results:
                jump_stats = {}
                
                for compartment, results in self.jump_mixture_results.items():
                    if 'diffusion_coefficients' in results:
                        D_values = results['diffusion_coefficients']
                        weights = results['weights']
                        
                        jump_stats[compartment] = {
                            'n_populations': len(D_values),
                            'diffusion_coefficients': D_values,
                            'population_weights': weights,
                            'dominant_population': np.argmax(weights)
                        }
                
                population_stats['jump_mixture'] = jump_stats
            
            # Trajectory segment statistics
            if self.segmented_trajectories:
                segment_stats = {
                    'n_tracks': len(self.segmented_trajectories),
                    'n_segments': sum(len(segments) for segments in self.segmented_trajectories.values()),
                    'diffusion_modes': {}
                }
                
                # Count segments by diffusion mode
                mode_counts = {}
                alphas_by_mode = {}
                D_by_mode = {}
                
                for track_id, segments in self.segmented_trajectories.items():
                    for segment in segments:
                        mode = segment.get('diffusion_mode', 'Unknown')
                        
                        if mode not in mode_counts:
                            mode_counts[mode] = 0
                            alphas_by_mode[mode] = []
                            D_by_mode[mode] = []
                        
                        mode_counts[mode] += 1
                        
                        if segment.get('alpha') is not None:
                            alphas_by_mode[mode].append(segment['alpha'])
                        
                        if segment.get('diffusion_coefficient') is not None:
                            D_by_mode[mode].append(segment['diffusion_coefficient'])
                
                # Calculate statistics for each mode
                for mode in mode_counts:
                    segment_stats['diffusion_modes'][mode] = {
                        'count': mode_counts[mode],
                        'fraction': mode_counts[mode] / segment_stats['n_segments'],
                        'mean_alpha': np.mean(alphas_by_mode[mode]) if alphas_by_mode[mode] else None,
                        'std_alpha': np.std(alphas_by_mode[mode]) if len(alphas_by_mode[mode]) > 1 else None,
                        'mean_D': np.mean(D_by_mode[mode]) if D_by_mode[mode] else None,
                        'std_D': np.std(D_by_mode[mode]) if len(D_by_mode[mode]) > 1 else None
                    }
                
                population_stats['trajectory_segments'] = segment_stats
            
            # Diffusion state statistics
            if self.diffusion_states:
                state_stats = {
                    'n_states': len(self.diffusion_states),
                    'states': {}
                }
                
                for state_id, state in self.diffusion_states.items():
                    state_stats['states'][state_id] = {
                        'mean_D': state.get('mean_D'),
                        'mean_alpha': state.get('mean_alpha'),
                        'occupancy': state.get('occupancy'),
                        'transition_probabilities': state.get('transition_probabilities')
                    }
                
                population_stats['diffusion_states'] = state_stats
            
            return population_stats
        
        except Exception as e:
            logger.error(f"Error in population statistics calculation: {str(e)}")
            raise
    
    def analyze_jump_size_distribution(self, tracks_df, n_components=2, compartment_masks=None):
        """
        Analyze jump size distribution using Gaussian mixture model.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        n_components : int, optional
            Number of mixture components, by default 2
        compartment_masks : dict, optional
            Dictionary of binary masks for compartments, by default None
            
        Returns
        -------
        dict
            Dictionary of jump mixture analysis results
        """
        try:
            from sklearn.mixture import GaussianMixture
            
            # Calculate jumps for all tracks
            all_jumps = []
            
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                positions = track_df[['x', 'y']].values
                
                # Calculate jumps
                jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                all_jumps.extend(jumps)
            
            # Convert to squared jumps (more directly related to diffusion coefficient)
            all_squared_jumps = np.array(all_jumps)**2
            
            # Fit Gaussian mixture model to squared jumps
            gmm = GaussianMixture(n_components=n_components, random_state=0)
            gmm.fit(all_squared_jumps.reshape(-1, 1))
            
            # Extract diffusion coefficients from mixture components
            # D = <r²>/4dt
            means = gmm.means_.flatten()
            weights = gmm.weights_
            
            # Sort by diffusion coefficient (ascending)
            sort_idx = np.argsort(means)
            means = means[sort_idx]
            weights = weights[sort_idx]
            
            # Calculate diffusion coefficients
            diffusion_coefficients = means / (4 * self.dt)
            
            # Store results for all tracks
            all_results = {
                'diffusion_coefficients': diffusion_coefficients.tolist(),
                'weights': weights.tolist(),
                'n_components': n_components,
                'bic': gmm.bic(all_squared_jumps.reshape(-1, 1)),
                'aic': gmm.aic(all_squared_jumps.reshape(-1, 1))
            }
            
            # Analyze by compartment if masks provided
            compartment_results = {}
            
            if compartment_masks:
                for comp_name, mask in compartment_masks.items():
                    # Filter tracks by compartment
                    comp_jumps = []
                    
                    for track_id, track_df in tracks_df.groupby('track_id'):
                        # Sort by frame
                        track_df = track_df.sort_values('frame')
                        
                        positions = track_df[['x', 'y']].values
                        
                        # Check if track is in this compartment
                        in_compartment = False
                        for x, y in positions.astype(int):
                            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                                if mask[y, x]:
                                    in_compartment = True
                                    break
                        
                        if in_compartment:
                            # Calculate jumps
                            jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                            comp_jumps.extend(jumps)
                    
                    if len(comp_jumps) > n_components * 10:  # Need sufficient data
                        # Convert to squared jumps
                        comp_squared_jumps = np.array(comp_jumps)**2
                        
                        # Fit Gaussian mixture model
                        gmm = GaussianMixture(n_components=n_components, random_state=0)
                        gmm.fit(comp_squared_jumps.reshape(-1, 1))
                        
                        # Extract parameters
                        means = gmm.means_.flatten()
                        weights = gmm.weights_
                        
                        # Sort by diffusion coefficient
                        sort_idx = np.argsort(means)
                        means = means[sort_idx]
                        weights = weights[sort_idx]
                        
                        # Calculate diffusion coefficients
                        diffusion_coefficients = means / (4 * self.dt)
                        
                        compartment_results[comp_name] = {
                            'diffusion_coefficients': diffusion_coefficients.tolist(),
                            'weights': weights.tolist(),
                            'n_components': n_components,
                            'bic': gmm.bic(comp_squared_jumps.reshape(-1, 1)),
                            'aic': gmm.aic(comp_squared_jumps.reshape(-1, 1))
                        }
            
            # Store results
            self.jump_mixture_results = {
                'all': all_results,
                'by_compartment': compartment_results
            }
            
            return self.jump_mixture_results
        
        except Exception as e:
            logger.error(f"Error in jump size distribution analysis: {str(e)}")
            raise
    
    def identify_diffusion_states(self, tracks_df, n_states=3, method='hmm'):
        """
        Identify diffusion states using hidden Markov model or other methods.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing tracking data
        n_states : int, optional
            Number of diffusion states, by default 3
        method : str, optional
            Method for state identification, by default 'hmm'
            
        Returns
        -------
        dict
            Dictionary of diffusion state results
        """
        try:
            if method == 'hmm':
                try:
                    from hmmlearn import hmm
                except ImportError:
                    logger.warning("hmmlearn not available, falling back to kmeans")
                    method = 'kmeans'
            
            # Process tracks to extract features
            track_features = []
            track_ids = []
            
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                positions = track_df[['x', 'y']].values
                
                # Skip short tracks
                if len(positions) < 10:
                    continue
                
                # Calculate jumps
                jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                
                # Calculate local MSD (using sliding window)
                window_size = 5
                local_msds = []
                
                for i in range(len(positions) - window_size):
                    window_positions = positions[i:i+window_size]
                    
                    # Calculate MSD for lag=1
                    disp = window_positions[1:] - window_positions[:-1]
                    sq_disp = np.sum(disp**2, axis=1)
                    local_msds.append(np.mean(sq_disp))
                
                # Skip if not enough local MSDs
                if len(local_msds) < 5:
                    continue
                
                # Use log of local MSD as feature
                log_local_msds = np.log(local_msds)
                
                track_features.append(log_local_msds)
                track_ids.append(track_id)
            
            if not track_features:
                return {
                    'status': 'No suitable tracks for diffusion state analysis'
                }
            
            # Identify states using specified method
            if method == 'hmm':
                # Prepare data for HMM
                X = np.concatenate(track_features).reshape(-1, 1)
                
                # Train HMM
                model = hmm.GaussianHMM(n_components=n_states, random_state=0)
                model.fit(X)
                
                # Get state parameters
                means = model.means_.flatten()
                variances = model.covars_.flatten()
                transition_matrix = model.transmat_
                
                # Sort states by diffusion coefficient (ascending)
                sort_idx = np.argsort(means)
                means = means[sort_idx]
                variances = variances[sort_idx]
                
                # Reorder transition matrix
                transition_matrix = transition_matrix[sort_idx][:, sort_idx]
                
                # Calculate diffusion coefficients from means
                # D = exp(log_msd) / 4dt
                diffusion_coefficients = np.exp(means) / (4 * self.dt)
                
                # Decode states for each track
                track_states = {}
                
                for i, (track_id, features) in enumerate(zip(track_ids, track_features)):
                    # Decode states
                    states = model.predict(features.reshape(-1, 1))
                    
                    # Remap states according to sorting
                    remapped_states = np.zeros_like(states)
                    for j, idx in enumerate(sort_idx):
                        remapped_states[states == idx] = j
                    
                    track_states[track_id] = remapped_states.tolist()
                
                # Calculate state occupancy
                all_states = np.concatenate([np.array(states) for states in track_states.values()])
                occupancy = np.zeros(n_states)
                
                for i in range(n_states):
                    occupancy[i] = np.sum(all_states == i) / len(all_states)
                
                # Store state information
                diffusion_states = {}
                
                for i in range(n_states):
                    diffusion_states[i] = {
                        'mean_D': diffusion_coefficients[i],
                        'variance': variances[i],
                        'occupancy': occupancy[i],
                        'transition_probabilities': transition_matrix[i].tolist()
                    }
                
                # Store results
                self.diffusion_states = {
                    'states': diffusion_states,
                    'track_states': track_states,
                    'method': 'hmm',
                    'n_states': n_states
                }
                
                return self.diffusion_states
                
            elif method == 'kmeans':
                from sklearn.cluster import KMeans
                
                # Prepare data
                X = np.concatenate(track_features).reshape(-1, 1)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=n_states, random_state=0)
                labels = kmeans.fit_predict(X)
                
                # Get cluster centers
                centers = kmeans.cluster_centers_.flatten()
                
                # Sort clusters by diffusion coefficient (ascending)
                sort_idx = np.argsort(centers)
                centers = centers[sort_idx]
                
                # Calculate diffusion coefficients from centers
                diffusion_coefficients = np.exp(centers) / (4 * self.dt)
                
                # Assign states to each track
                track_states = {}
                start_idx = 0
                
                for i, (track_id, features) in enumerate(zip(track_ids, track_features)):
                    # Get states for this track
                    end_idx = start_idx + len(features)
                    states = labels[start_idx:end_idx]
                    
                    # Remap states according to sorting
                    remapped_states = np.zeros_like(states)
                    for j, idx in enumerate(sort_idx):
                        remapped_states[states == idx] = j
                    
                    track_states[track_id] = remapped_states.tolist()
                    start_idx = end_idx
                
                # Calculate state occupancy
                all_states = np.concatenate([np.array(states) for states in track_states.values()])
                occupancy = np.zeros(n_states)
                
                for i in range(n_states):
                    occupancy[i] = np.sum(all_states == i) / len(all_states)
                
                # Calculate transition probabilities
                transition_counts = np.zeros((n_states, n_states))
                
                for states in track_states.values():
                    states_array = np.array(states)
                    for i in range(len(states_array) - 1):
                        from_state = states_array[i]
                        to_state = states_array[i + 1]
                        transition_counts[int(from_state), int(to_state)] += 1
                
                # Normalize to get probabilities
                transition_probs = np.zeros((n_states, n_states))
                
                for i in range(n_states):
                    row_sum = np.sum(transition_counts[i])
                    if row_sum > 0:
                        transition_probs[i] = transition_counts[i] / row_sum
                
                # Store state information
                diffusion_states = {}
                
                for i in range(n_states):
                    diffusion_states[i] = {
                        'mean_D': diffusion_coefficients[i],
                        'occupancy': occupancy[i],
                        'transition_probabilities': transition_probs[i].tolist()
                    }
                
                # Store results
                self.diffusion_states = {
                    'states': diffusion_states,
                    'track_states': track_states,
                    'method': 'kmeans',
                    'n_states': n_states
                }
                
                return self.diffusion_states
                
            else:
                return {
                    'status': f'Unknown method: {method}'
                }
        
        except Exception as e:
            logger.error(f"Error in diffusion state identification: {str(e)}")
            raise
