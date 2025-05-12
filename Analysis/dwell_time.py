"""
Dwell time analysis module for SPT Analysis.

This module provides functions for analyzing dwell times, 
binding events, and cage escape dynamics in particle trajectories.
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class DwellTimeAnalyzer:
    """
    Analyzer for dwell times, binding events, and cage dynamics.
    
    Parameters
    ----------
    dt : float, optional
        Time between frames in seconds, by default 0.014
    immobility_threshold : float, optional
        Displacement threshold for considering a particle immobile, by default 1.0
    min_binding_frames : int, optional
        Minimum number of frames for a binding event, by default 10
    cage_detection_window : int, optional
        Window size for cage detection, by default 20
    """
    
    def __init__(self, dt=0.014, immobility_threshold=1.0, min_binding_frames=10, cage_detection_window=20):
        self.dt = dt
        self.immobility_threshold = immobility_threshold
        self.min_binding_frames = min_binding_frames
        self.cage_detection_window = cage_detection_window
        
        # Results storage
        self.dwell_times = {}
        self.binding_events = {}
        self.cage_escapes = {}
        self.kinetic_parameters = {}
    
    def analyze_compartment_dwell_times(self, tracks_df, compartment_masks, nucleus_mask=None):
        """
        Analyze dwell times of particles in different compartments.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict
            Dictionary mapping compartment names to binary masks
        nucleus_mask : numpy.ndarray, optional
            Binary mask for nucleus, by default None
            
        Returns
        -------
        dict
            Dictionary with dwell time results
        """
        try:
            # Create a labeled map from compartment masks
            first_mask = next(iter(compartment_masks.values()))
            labeled_map = np.zeros_like(first_mask, dtype=np.int32)
            
            compartment_names = list(compartment_masks.keys())
            for i, name in enumerate(compartment_names, start=1):
                labeled_map[compartment_masks[name] > 0] = i
            
            # Initialize dwell time storage
            dwell_times = {name: [] for name in compartment_names}
            dwell_events = {name: [] for name in compartment_names}
            
            # Analyze each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                # Get positions and frames
                positions = track_df[['x', 'y']].values.astype(int)
                frames = track_df['frame'].values
                
                # Track compartment changes and record entry positions explicitly
                current_compartment = None
                compartment_entry_frame = None
                compartment_entry_position = None
                
                for i, (pos, frame) in enumerate(zip(positions, frames)):
                    x, y = pos
                    
                    # Check if position is within image bounds
                    if 0 <= y < labeled_map.shape[0] and 0 <= x < labeled_map.shape[1]:
                        compartment_label = labeled_map[y, x]
                        compartment_name = compartment_names[compartment_label - 1] if compartment_label > 0 else None
                        
                        # Detect compartment change
                        if compartment_name != current_compartment:
                            # Record dwell time for previous compartment
                            if current_compartment is not None and compartment_entry_frame is not None:
                                dwell_time = (frame - compartment_entry_frame) * self.dt
                                dwell_times[current_compartment].append(dwell_time)
                                dwell_events[current_compartment].append({
                                    'track_id': track_id,
                                    'start_frame': compartment_entry_frame,
                                    'end_frame': frame,
                                    'dwell_time': dwell_time,
                                    'start_position': compartment_entry_position,
                                    'end_position': pos
                                })
                            
                            # Update current compartment and record entry details
                            current_compartment = compartment_name
                            compartment_entry_frame = frame
                            compartment_entry_position = pos
                
                # Handle the last compartment
                if current_compartment is not None and compartment_entry_frame is not None:
                    dwell_time = (frames[-1] - compartment_entry_frame) * self.dt
                    dwell_times[current_compartment].append(dwell_time)
                    dwell_events[current_compartment].append({
                        'track_id': track_id,
                        'start_frame': compartment_entry_frame,
                        'end_frame': frames[-1],
                        'dwell_time': dwell_time,
                        'start_position': compartment_entry_position,
                        'end_position': positions[-1],
                        'censored': True  # Event is censored (track ends while still in compartment)
                    })
            
            # Compute statistics
            dwell_statistics = {}
            
            for comp, times in dwell_times.items():
                if times:
                    mean_dwell = np.mean(times)
                    median_dwell = np.median(times)
                    std_dwell = np.std(times)
                    n_events = len(times)
                    
                    # Test for exponential distribution
                    try:
                        # Fit exponential distribution
                        rate_parameter = 1.0 / mean_dwell
                        
                        # Kolmogorov-Smirnov test
                        _, pval = stats.kstest(times, 'expon', args=(0, 1.0/rate_parameter))
                        
                        exponential_fit = {
                            'rate': rate_parameter,
                            'mean_dwell': 1.0 / rate_parameter,
                            'p_value': pval,
                            'good_fit': pval > 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Error fitting exponential distribution: {str(e)}")
                        exponential_fit = None
                    
                    dwell_statistics[comp] = {
                        'mean': mean_dwell,
                        'median': median_dwell,
                        'std': std_dwell,
                        'n_events': n_events,
                        'exponential_fit': exponential_fit
                    }
            
            # Store results
            self.dwell_times = {
                'events': dwell_events,
                'statistics': dwell_statistics
            }
            
            return self.dwell_times
        
        except Exception as e:
            logger.error(f"Error in compartment dwell time analysis: {str(e)}")
            raise
    
    def detect_binding_events(self, tracks_df, compartment_masks=None):
        """
        Detect particle binding events.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict, optional
            Dictionary mapping compartment names to binary masks, by default None
            
        Returns
        -------
        dict
            Dictionary with binding event results
        """
        try:
            binding_events = []
            
            # Analyze each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                # Get positions and frames
                positions = track_df[['x', 'y']].values
                frames = track_df['frame'].values
                
                # Calculate displacements
                displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                
                # Identify immobile segments
                immobile = displacements < self.immobility_threshold
                
                # Find binding segments
                binding_segments = []
                current_segment = None
                
                for i, (is_immobile, frame) in enumerate(zip(immobile, frames[1:])):
                    if is_immobile:
                        # Start new segment or continue current one
                        if current_segment is None:
                            current_segment = {
                                'start_idx': i,
                                'start_frame': frames[i],
                                'frames': [frames[i], frame],
                                'positions': [positions[i], positions[i+1]]
                            }
                        else:
                            current_segment['frames'].append(frame)
                            current_segment['positions'].append(positions[i+1])
                    else:
                        # End current segment if it exists
                        if current_segment is not None:
                            current_segment['end_idx'] = i
                            current_segment['end_frame'] = frames[i]
                            current_segment['duration'] = (current_segment['end_frame'] - current_segment['start_frame']) * self.dt
                            current_segment['n_frames'] = len(current_segment['frames'])
                            
                            # Only keep segments that meet minimum length
                            if current_segment['n_frames'] >= self.min_binding_frames:
                                binding_segments.append(current_segment)
                            
                            current_segment = None
                
                # Handle last segment
                if current_segment is not None:
                    current_segment['end_idx'] = len(positions) - 1
                    current_segment['end_frame'] = frames[-1]
                    current_segment['duration'] = (current_segment['end_frame'] - current_segment['start_frame']) * self.dt
                    current_segment['n_frames'] = len(current_segment['frames'])
                    
                    if current_segment['n_frames'] >= self.min_binding_frames:
                        binding_segments.append(current_segment)
                
                # Process each binding segment
                for segment in binding_segments:
                    # Calculate mean binding position
                    binding_pos = np.mean(segment['positions'], axis=0)
                    
                    # Determine compartment if masks provided
                    compartment = None
                    if compartment_masks is not None:
                        x, y = int(binding_pos[0]), int(binding_pos[1])
                        
                        for name, mask in compartment_masks.items():
                            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                                compartment = name
                                break
                    
                    # Calculate binding MSD
                    binding_positions = np.array(segment['positions'])
                    binding_msd = np.mean(np.sum((binding_positions - binding_pos)**2, axis=1))
                    
                    # Classify binding type
                    binding_type = "Stable" if segment['duration'] > 2.0 and binding_msd < 0.5 else "Transient"
                    
                    # Create event record
                    event = {
                        'track_id': track_id,
                        'start_frame': segment['start_frame'],
                        'end_frame': segment['end_frame'],
                        'duration': segment['duration'],
                        'n_frames': segment['n_frames'],
                        'position': binding_pos,
                        'compartment': compartment,
                        'msd_during_binding': binding_msd,
                        'binding_type': binding_type,
                        'censored': segment['end_frame'] == frames[-1]  # Track ends during binding
                    }
                    
                    binding_events.append(event)
            
            # Store results
            self.binding_events = {'events': binding_events}
            
            return self.binding_events
        
        except Exception as e:
            logger.error(f"Error in binding event detection: {str(e)}")
            raise
    
    def analyze_cage_dynamics(self, tracks_df, compartment_masks=None):
        """
        Analyze cage dynamics in particle trajectories.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict, optional
            Dictionary mapping compartment names to binary masks, by default None
            
        Returns
        -------
        dict
            Dictionary with cage dynamics results
        """
        try:
            cage_events = []
            
            # Analyze each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                # Get positions and frames
                positions = track_df[['x', 'y']].values
                frames = track_df['frame'].values
                
                # Skip short tracks
                if len(positions) < 2 * self.cage_detection_window:
                    continue
                
                # Detect cages using sliding window
                cages = []
                
                for i in range(len(positions) - self.cage_detection_window):
                    # Get positions within window
                    window_positions = positions[i:i+self.cage_detection_window]
                    window_frames = frames[i:i+self.cage_detection_window]
                    
                    # Calculate cage center and radius
                    cage_center = np.mean(window_positions, axis=0)
                    distances = np.sqrt(np.sum((window_positions - cage_center)**2, axis=1))
                    cage_radius = np.max(distances)
                    
                    # Calculate MSD scaling (alpha) to detect subdiffusion
                    tau = np.arange(1, min(11, self.cage_detection_window))
                    msd_values = []
                    
                    for lag in tau:
                        disp = window_positions[lag:] - window_positions[:-lag]
                        msd_values.append(np.mean(np.sum(disp**2, axis=1)))
                    
                    msd_values = np.array(msd_values)
                    
                    # Fit power law to get alpha
                    try:
                        log_tau = np.log(tau * self.dt)
                        log_msd = np.log(msd_values)
                        slope, intercept = np.polyfit(log_tau, log_msd, 1)
                        
                        alpha = slope
                        D = np.exp(intercept - np.log(4))
                    except Exception:
                        alpha = None
                        D = None
                    
                    # Identify cages (subdiffusive regions with confined motion)
                    if alpha is not None and alpha < 0.7 and cage_radius < 10.0:
                        cage = {
                            'start_idx': i,
                            'end_idx': i + self.cage_detection_window - 1,
                            'start_frame': window_frames[0],
                            'end_frame': window_frames[-1],
                            'center': cage_center,
                            'radius': cage_radius,
                            'alpha': alpha,
                            'diffusion_coef': D,
                            'positions': window_positions
                        }
                        cages.append(cage)
                
                # Add cage events from this track
                for cage in cages:
                    cage_events.append({**cage, 'track_id': track_id})
            
            # Store results
            self.cage_escapes = {'events': cage_events}
            
            return self.cage_escapes
        
        except Exception as e:
            logger.error(f"Error in cage dynamics analysis: {str(e)}")
            raise
    
    def extract_kinetic_parameters(self, particle_type=None, compartment=None):
        """
        Extract kinetic parameters from dwell time and binding analyses.
        
        Parameters
        ----------
        particle_type : str, optional
            Particle type to analyze, by default None (all particle types)
        compartment : str, optional
            Compartment to analyze, by default None (all compartments)
            
        Returns
        -------
        dict
            Dictionary with kinetic parameters
        """
        try:
            kinetic_params = {}
            
            # Extract from dwell times
            if self.dwell_times and 'statistics' in self.dwell_times:
                dwell_stats = self.dwell_times['statistics']
                
                if compartment and compartment in dwell_stats:
                    stats = dwell_stats[compartment]
                    
                    if 'exponential_fit' in stats and stats['exponential_fit']:
                        kinetic_params['koff'] = stats['exponential_fit']['rate']
                        kinetic_params['residence_time'] = stats['exponential_fit']['mean_dwell']
                elif not compartment:
                    # Compute average across compartments
                    koff_values = []
                    residence_times = []
                    
                    for comp, stats in dwell_stats.items():
                        if 'exponential_fit' in stats and stats['exponential_fit']:
                            koff_values.append(stats['exponential_fit']['rate'])
                            residence_times.append(stats['exponential_fit']['mean_dwell'])
                    
                    if koff_values:
                        kinetic_params['koff'] = np.mean(koff_values)
                        kinetic_params['residence_time'] = np.mean(residence_times)
            
            # Extract from binding events
            if self.binding_events and 'events' in self.binding_events:
                binding_events = self.binding_events['events']
                
                # Filter events by compartment if specified
                if compartment:
                    binding_events = [e for e in binding_events if e.get('compartment') == compartment]
                
                # Calculate dissociation rate from binding durations
                if binding_events:
                    durations = [e['duration'] for e in binding_events]
                    mean_duration = np.mean(durations)
                    
                    if 'koff' not in kinetic_params:
                        kinetic_params['koff'] = 1.0 / mean_duration
                        kinetic_params['residence_time'] = mean_duration
                    
                    # Calculate binding frequency (very simplified)
                    n_events = len(binding_events)
                    total_time = max([e['end_frame'] for e in binding_events]) * self.dt
                    
                    if total_time > 0:
                        kinetic_params['binding_frequency'] = n_events / total_time
            
            # Extract from cage dynamics
            if self.cage_escapes and 'events' in self.cage_escapes:
                cage_events = self.cage_escapes['events']
                
                if cage_events:
                    # Calculate cage sizes
                    cage_radii = [e['radius'] for e in cage_events]
                    kinetic_params['mean_cage_radius'] = np.mean(cage_radii)
                    
                    # Calculate cage lifetime and escape rate (simplified)
                    cage_durations = [(e['end_frame'] - e['start_frame']) * self.dt for e in cage_events]
                    mean_duration = np.mean(cage_durations)
                    
                    kinetic_params['mean_cage_lifetime'] = mean_duration
                    kinetic_params['cage_escape_rate'] = 1.0 / mean_duration
            
            # Store results
            self.kinetic_parameters = kinetic_params
            
            return kinetic_params
        
        except Exception as e:
            logger.error(f"Error extracting kinetic parameters: {str(e)}")
            raise
    
    def visualize_dwell_time_distribution(self, compartment=None, ax=None):
        """
        Visualize dwell time distributions.
        
        Parameters
        ----------
        compartment : str, optional
            Compartment to visualize, by default None (all compartments)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        import matplotlib.pyplot as plt
        
        # Check if data exists
        if not self.dwell_times or 'events' not in self.dwell_times:
            if ax is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No dwell time data", ha='center', va='center')
                return fig
            else:
                ax.text(0.5, 0.5, "No dwell time data", ha='center', va='center')
                return ax.figure
        
        # Prepare data
        data = []
        labels = []
        
        if compartment:
            # Single compartment
            if compartment in self.dwell_times['events']:
                data = [[e['dwell_time'] for e in self.dwell_times['events'][compartment]]]
                labels = [compartment]
        else:
            # All compartments
            for comp, events in self.dwell_times['events'].items():
                data.append([e['dwell_time'] for e in events])
                labels.append(comp)
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        # Create boxplot
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Dwell Time (s)")
        ax.set_title("Dwell Time Distribution")
        
        # Add individual points for better visualization
        for i, d in enumerate(data, 1):
            y = d
            # Adding small random noise to x coordinate for visualization clarity
            x = np.random.normal(i, 0.04, size=len(y))
            ax.plot(x, y, 'o', alpha=0.5, markersize=4)
        
        return fig
    
    def visualize_binding_events(self, ax=None):
        """
        Visualize binding events.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        import matplotlib.pyplot as plt
        
        # Check if data exists
        if not self.binding_events or 'events' not in self.binding_events:
            if ax is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No binding event data", ha='center', va='center')
                return fig
            else:
                ax.text(0.5, 0.5, "No binding event data", ha='center', va='center')
                return ax.figure
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        
        # Get binding events
        events = self.binding_events['events']
        
        # Sort events by start frame
        events = sorted(events, key=lambda e: e['start_frame'])
        
        # Group events by type
        stable_events = [e for e in events if e['binding_type'] == 'Stable']
        transient_events = [e for e in events if e['binding_type'] == 'Transient']
        
        # Plot events as horizontal lines
        y_stable = 1
        y_transient = 2
        
        for event in stable_events:
            start = event['start_frame'] * self.dt
            end = event['end_frame'] * self.dt
            ax.plot([start, end], [y_stable, y_stable], 'r-', linewidth=2)
        
        for event in transient_events:
            start = event['start_frame'] * self.dt
            end = event['end_frame'] * self.dt
            ax.plot([start, end], [y_transient, y_transient], 'b-', linewidth=2)
        
        # Add legend and labels
        ax.set_yticks([y_stable, y_transient])
        ax.set_yticklabels(['Stable', 'Transient'])
        ax.set_xlabel('Time (s)')
        ax.set_title('Binding Events')
        
        return fig
