
"""
Active transport analysis module for SPT Analysis.

This module provides tools for identifying and analyzing active transport
processes in particle trajectories, such as directed motion and motor-driven transport.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class ActiveTransportAnalyzer:
    """
    Analyzer for active transport phenomena.
    
    Parameters
    ----------
    dt : float, optional
        Time between frames in seconds, by default 0.014
    min_track_length : int, optional
        Minimum track length for analysis, by default 10
    min_superdiffusive_alpha : float, optional
        Minimum alpha value for superdiffusion classification, by default 1.3
    """
    
    def __init__(self, dt=0.014, min_track_length=10, min_superdiffusive_alpha=1.3):
        self.dt = dt
        self.min_track_length = min_track_length
        self.min_superdiffusive_alpha = min_superdiffusive_alpha
        
        # Results storage
        self.superdiffusive_tracks = {}
        self.directed_motion_results = {}
        self.transport_statistics = {}
    
    def analyze_superdiffusion(self, tracks_df, compartment_masks=None):
        """
        Identify and analyze superdiffusive track segments.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict, optional
            Dictionary mapping compartment names to binary masks, by default None
            
        Returns
        -------
        dict
            Dictionary with superdiffusion analysis results
        """
        try:
            results = {}
            
            # Process each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                # Get positions and frames
                positions = track_df[['x', 'y']].values
                frames = track_df['frame'].values
                
                # Skip if track is too short
                if len(positions) < self.min_track_length:
                    continue
                
                # Calculate MSD for different lag times
                lag_times = []
                msd_values = []
                
                for lag in range(1, min(11, len(positions) // 2)):
                    disp = positions[lag:] - positions[:-lag]
                    sq_disp = np.sum(disp**2, axis=1)
                    lag_times.append(lag * self.dt)
                    msd_values.append(np.mean(sq_disp))
                
                # Fit power law to MSD (MSD ~ t^alpha)
                log_tau = np.log(lag_times)
                log_msd = np.log(msd_values)
                slope, intercept = np.polyfit(log_tau, log_msd, 1)
                
                alpha = slope
                D = np.exp(intercept - np.log(4))
                
                # Determine if track is superdiffusive
                is_superdiffusive = alpha >= self.min_superdiffusive_alpha
                
                # Additional analysis for superdiffusive tracks
                if is_superdiffusive:
                    # Calculate average velocity
                    displacements = positions[1:] - positions[:-1]
                    velocities = displacements / self.dt
                    mean_velocity = np.mean(velocities, axis=0)
                    speed = np.linalg.norm(mean_velocity)
                    
                    # Calculate directional persistence
                    if len(displacements) > 1:
                        unit_vectors = displacements / np.linalg.norm(displacements, axis=1)[:, np.newaxis]
                        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
                        directional_persistence = np.mean(dot_products)
                    else:
                        directional_persistence = None
                    
                    # Determine compartment if masks provided
                    compartment = None
                    if compartment_masks:
                        # Use mean position to determine compartment
                        mean_pos = np.mean(positions, axis=0)
                        x, y = int(mean_pos[0]), int(mean_pos[1])
                        
                        for name, mask in compartment_masks.items():
                            if (0 <= y < mask.shape[0] and 
                                0 <= x < mask.shape[1] and 
                                mask[y, x]):
                                compartment = name
                                break
                    
                    # Store detailed results
                    results[track_id] = {
                        'alpha': alpha,
                        'diffusion_coefficient': D,
                        'superdiffusive': True,
                        'speed': speed,
                        'velocity_x': mean_velocity[0],
                        'velocity_y': mean_velocity[1],
                        'directional_persistence': directional_persistence,
                        'compartment': compartment,
                        'n_frames': len(positions),
                        'duration': (frames[-1] - frames[0]) * self.dt
                    }
                else:
                    # Store basic results for non-superdiffusive tracks
                    results[track_id] = {
                        'alpha': alpha,
                        'diffusion_coefficient': D,
                        'superdiffusive': False
                    }
            
            # Store results
            self.superdiffusive_tracks = results
            
            return results
        
        except Exception as e:
            logger.error(f"Error in superdiffusion analysis: {str(e)}")
            raise
    
    def analyze_directed_motion(self, tracks_df, min_duration=1.0, min_displacement=2.0):
        """
        Analyze directed motion segments in tracks.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        min_duration : float, optional
            Minimum duration for directed segments (seconds), by default 1.0
        min_displacement : float, optional
            Minimum displacement for directed segments (μm), by default 2.0
            
        Returns
        -------
        dict
            Dictionary with directed motion analysis results
        """
        try:
            directed_segments = []
            
            # Process each track
            for track_id, track_df in tracks_df.groupby('track_id'):
                # Sort by frame
                track_df = track_df.sort_values('frame')
                
                # Get positions and frames
                positions = track_df[['x', 'y']].values
                frames = track_df['frame'].values
                
                # Skip if track is too short
                if len(positions) < self.min_track_length:
                    continue
                
                # Try to detect segments with consistent direction
                try:
                    # Use change point detection for direction changes
                    import ruptures as rpt
                    
                    # Calculate angles of displacements
                    displacements = positions[1:] - positions[:-1]
                    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
                    
                    # Convert to change in angles (more meaningful for directional changes)
                    angle_changes = np.diff(angles)
                    
                    # Wrap angle changes to [-π, π]
                    angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
                    
                    # Detect change points in the angle sequence
                    model = "l2"  # L2 cost for change point detection
                    algo = rpt.Pelt(model=model).fit(angle_changes[:, np.newaxis])
                    result = algo.predict(pen=np.log(len(angle_changes)) * 0.5)
                    
                    # Convert to segment boundaries (account for the two differencing operations)
                    segment_boundaries = [(0, result[0]+2)]
                    for i in range(len(result)-1):
                        segment_boundaries.append((result[i]+2, result[i+1]+2))
                    
                    # Add final segment
                    if result:
                        segment_boundaries.append((result[-1]+2, len(positions)))
                    else:
                        segment_boundaries = [(0, len(positions))]
                    
                except:
                    # Fallback to simple analysis if change point detection fails
                    segment_boundaries = [(0, len(positions))]
                
                # Analyze each segment
                for start_idx, end_idx in segment_boundaries:
                    # Skip if segment is too short
                    if end_idx - start_idx < 5:
                        continue
                    
                    # Get segment positions and frames
                    segment_positions = positions[start_idx:end_idx]
                    segment_frames = frames[start_idx:end_idx]
                    
                    # Calculate duration
                    duration = (segment_frames[-1] - segment_frames[0]) * self.dt
                    
                    # Calculate net displacement
                    displacement = np.linalg.norm(segment_positions[-1] - segment_positions[0])
                    
                    # Calculate path length
                    path_length = np.sum(np.linalg.norm(np.diff(segment_positions, axis=0), axis=1))
                    
                    # Calculate straightness (displacement / path length)
                    straightness = displacement / path_length if path_length > 0 else 0
                    
                    # Identify directed segments
                    is_directed = (duration >= min_duration and 
                                  displacement >= min_displacement and 
                                  straightness >= 0.7)
                    
                    if is_directed:
                        # Calculate velocity
                        velocity_vector = (segment_positions[-1] - segment_positions[0]) / duration
                        speed = np.linalg.norm(velocity_vector)
                        
                        # Store directed segment
                        directed_segments.append({
                            'track_id': track_id,
                            'start_frame': segment_frames[0],
                            'end_frame': segment_frames[-1],
                            'duration': duration,
                            'displacement': displacement,
                            'path_length': path_length,
                            'straightness': straightness,
                            'speed': speed,
                            'velocity_x': velocity_vector[0],
                            'velocity_y': velocity_vector[1],
                            'start_position': segment_positions[0],
                            'end_position': segment_positions[-1],
                            'segment_positions': segment_positions
                        })
            
            # Store results
            self.directed_motion_results = {
                'segments': directed_segments,
                'n_segments': len(directed_segments)
            }
            
            # Calculate statistics
            if directed_segments:
                speeds = [s['speed'] for s in directed_segments]
                durations = [s['duration'] for s in directed_segments]
                straightness = [s['straightness'] for s in directed_segments]
                
                statistics = {
                    'n_segments': len(directed_segments),
                    'mean_speed': np.mean(speeds),
                    'std_speed': np.std(speeds),
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'mean_straightness': np.mean(straightness),
                    'std_straightness': np.std(straightness)
                }
                
                self.transport_statistics = statistics
            
            return self.directed_motion_results
        
        except Exception as e:
            logger.error(f"Error in directed motion analysis: {str(e)}")
            raise
    
    def calculate_transport_parameters(self, compartment=None):
        """
        Calculate transport parameters from directed motion analysis.
        
        Parameters
        ----------
        compartment : str, optional
            Compartment to analyze, by default None
            
        Returns
        -------
        dict
            Dictionary with transport parameters
        """
        try:
            if not self.directed_motion_results or 'segments' not in self.directed_motion_results:
                return {'status': 'No directed motion analysis results available'}
            
            # Get directed segments
            segments = self.directed_motion_results['segments']
            
            if not segments:
                return {'status': 'No directed motion segments detected'}
            
            # Filter by compartment if specified
            if compartment and 'compartment' in segments[0]:
                segments = [s for s in segments if s.get('compartment') == compartment]
                
                if not segments:
                    return {'status': f'No directed motion segments in {compartment}'}
            
            # Calculate parameters
            speeds = [s['speed'] for s in segments]
            durations = [s['duration'] for s in segments]
            displacements = [s['displacement'] for s in segments]
            
            # Calculate processivity (mean displacement per directed segment)
            mean_processivity = np.mean(displacements)
            
            # Calculate run length distribution parameters
            try:
                # Fit exponential distribution to run lengths
                from scipy import stats
                
                # Run length = displacement during directed motion
                rate_param = 1.0 / mean_processivity
                
                # KS test to see if exponential is a good fit
                _, pval = stats.kstest(displacements, 'expon', args=(0, 1.0/rate_param))
                
                run_length_fit = {
                    'distribution': 'exponential',
                    'rate': rate_param,
                    'mean': mean_processivity,
                    'p_value': pval,
                    'good_fit': pval > 0.05
                }
                
            except:
                run_length_fit = None
            
            # Calculate transport efficiency (ratio of directed to total motion)
            # Here we approximate by looking at mean speed and straightness
            mean_speed = np.mean(speeds)
            mean_straightness = np.mean([s.get('straightness', 0) for s in segments])
            
            transport_efficiency = mean_speed * mean_straightness
            
            # Calculate transport frequency (segments per track)
            n_unique_tracks = len(set(s['track_id'] for s in segments))
            segments_per_track = len(segments) / n_unique_tracks if n_unique_tracks > 0 else 0
            
            # Compile results
            parameters = {
                'mean_speed': mean_speed,
                'std_speed': np.std(speeds),
                'mean_processivity': mean_processivity,
                'std_processivity': np.std(displacements),
                'mean_duration': np.mean(durations),
                'std_duration': np.std(durations),
                'run_length_distribution': run_length_fit,
                'transport_efficiency': transport_efficiency,
                'segments_per_track': segments_per_track,
                'n_segments': len(segments),
                'n_tracks': n_unique_tracks
            }
            
            return parameters
        
        except Exception as e:
            logger.error(f"Error calculating transport parameters: {str(e)}")
            raise
    
    def visualize_directed_segments(self, tracks_df, ax=None):
        """
        Visualize directed motion segments.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, by default None
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with visualization
        """
        import matplotlib.pyplot as plt
        
        # Check if data exists
        if not self.directed_motion_results or 'segments' not in self.directed_motion_results:
            if ax is None:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No directed motion analysis results", 
                        ha='center', va='center')
                return fig
            else:
                ax.text(0.5, 0.5, "No directed motion analysis results", 
                        ha='center', va='center')
                return ax.figure
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # Plot all tracks in gray
        for track_id, track_df in tracks_df.groupby('track_id'):
            track_df = track_df.sort_values('frame')
            pos = track_df[['x', 'y']].values
            ax.plot(pos[:, 0], pos[:, 1], '-', color='lightgray', alpha=0.5, linewidth=1)
        
        # Plot directed segments with arrows
        segments = self.directed_motion_results['segments']
        
        colors = plt.cm.tab10.colors
        
        for i, segment in enumerate(segments):
            # Get segment positions
            pos = segment['segment_positions']
            
            # Plot segment
            color = colors[i % len(colors)]
            ax.plot(pos[:, 0], pos[:, 1], '-', color=color, linewidth=2, alpha=0.8)
            
            # Add arrow for direction
            arrow_start = pos[0]
            arrow_end = pos[-1]
            
            ax.arrow(arrow_start[0], arrow_start[1], 
                   (arrow_end[0] - arrow_start[0]) * 0.9, 
                   (arrow_end[1] - arrow_start[1]) * 0.9,
                   head_width=2, head_length=2, fc=color, ec=color)
            
            # Add text label with speed
            mid_point = (arrow_start + arrow_end) / 2
            ax.text(mid_point[0], mid_point[1], 
                   f"{segment['speed']:.1f} μm/s", 
                   color=color, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Directed Motion Segments')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', lw=1, alpha=0.5, label='All tracks'),
            Line2D([0], [0], color=colors[0], lw=2, alpha=0.8, label='Directed segments')
        ]
        ax.legend(handles=legend_elements)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        return fig
