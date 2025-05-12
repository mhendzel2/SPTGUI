
"""
Diffusion visualization module for SPT Analysis.

This module provides functions for visualizing diffusion analysis results,
including MSD curves, diffusion coefficients, and motion classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_diffusion_map(tracks_df, diffusion_df, background=None, cmap='viridis',
                     min_track_length=5, ax=None, figsize=(10, 8), 
                     pixel_size=1.0, title=None, colorbar_label='D (μm²/s)'):
    """
    Plot a map of tracks colored by their diffusion coefficients.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    diffusion_df : pandas.DataFrame
        DataFrame with diffusion coefficients
    background : numpy.ndarray, optional
        Background image, by default None
    cmap : str, optional
        Colormap to use, by default 'viridis'
    min_track_length : int, optional
        Minimum track length to plot, by default 5
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Plot title, by default None
    colorbar_label : str, optional
        Label for colorbar, by default 'D (μm²/s)'
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background image if provided
        if background is not None:
            ax.imshow(background, cmap='gray', alpha=0.5)
        
        # Get diffusion coefficient lookup
        diff_lookup = dict(zip(diffusion_df['track_id'], diffusion_df['D']))
        
        # Find track IDs with diffusion coefficients
        valid_track_ids = set(diffusion_df['track_id'])
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        vmin = diffusion_df['D'].min()
        vmax = diffusion_df['D'].max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot tracks
        for track_id, track in tracks_df.groupby('track_id'):
            # Skip if track is too short or doesn't have diffusion coefficient
            if len(track) < min_track_length or track_id not in valid_track_ids:
                continue
            
            # Sort track by frame
            track = track.sort_values('frame')
            
            # Get coordinates
            x = track['x'] * pixel_size
            y = track['y'] * pixel_size
            
            # Get diffusion coefficient
            D = diff_lookup[track_id]
            
            # Get color
            color = cmap_obj(norm(D))
            
            # Plot track
            ax.plot(x, y, '-', color=color, linewidth=1.5, alpha=0.7)
            ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=5, alpha=0.8)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', color=color, markersize=5, alpha=0.8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(colorbar_label)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting diffusion map: {str(e)}")
        raise


def plot_diffusion_histogram(diffusion_df, bins=30, ax=None, figsize=(10, 6), 
                           log_scale=True, kde=True, title=None):
    """
    Plot histogram of diffusion coefficients.
    
    Parameters
    ----------
    diffusion_df : pandas.DataFrame
        DataFrame with diffusion coefficients
    bins : int, optional
        Number of bins, by default 30
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    log_scale : bool, optional
        Whether to use log scale for x-axis, by default True
    kde : bool, optional
        Whether to overlay a KDE, by default True
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get diffusion coefficients
        D_values = diffusion_df['D'].values
        
        # Apply log transformation if requested
        if log_scale:
            # Filter out non-positive values
            D_values = D_values[D_values > 0]
            D_values = np.log10(D_values)
            xlabel = 'log₁₀ D (μm²/s)'
        else:
            xlabel = 'D (μm²/s)'
        
        # Plot histogram
        sns.histplot(D_values, bins=bins, kde=kde, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting diffusion histogram: {str(e)}")
        raise


def plot_msd_fits(time_lags, msd_values, fit_results, ax=None, figsize=(10, 6),
                 max_lag=None, title=None):
    """
    Plot MSD data with model fits.
    
    Parameters
    ----------
    time_lags : array_like
        Time lag values
    msd_values : array_like
        MSD values
    fit_results : dict
        Dictionary with fitted model parameters
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    max_lag : int, optional
        Maximum lag to plot, by default None
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Filter by max lag if provided
        if max_lag is not None:
            time_lags = time_lags[:max_lag]
            msd_values = msd_values[:max_lag]
        
        # Plot MSD data
        ax.plot(time_lags, msd_values, 'ko', label='Data')
        
        # Generate fine-grained time lags for smooth curves
        t_fine = np.linspace(time_lags[0], time_lags[-1], 100)
        
        # Define model functions
        def simple_diffusion(t, D):
            return 4 * D * t
        
        def anomalous_diffusion(t, D, alpha):
            return 4 * D * t**alpha
        
        def confined_diffusion(t, D, L):
            return L**2 * (1 - np.exp(-12 * D * t / L**2))
        
        def directed_diffusion(t, D, v):
            return 4 * D * t + (v * t)**2
        
        # Plot model fits
        if 'simple_diffusion' in fit_results:
            params = fit_results['simple_diffusion']
            D = params['D']
            r2 = params['r_squared']
            y_fit = simple_diffusion(t_fine, D)
            ax.plot(t_fine, y_fit, '-', label=f'Simple (D={D:.3f}, R²={r2:.3f})')
        
        if 'anomalous_diffusion' in fit_results:
            params = fit_results['anomalous_diffusion']
            D = params['D']
            alpha = params['alpha']
            r2 = params['r_squared']
            y_fit = anomalous_diffusion(t_fine, D, alpha)
            ax.plot(t_fine, y_fit, '--', label=f'Anomalous (D={D:.3f}, α={alpha:.3f}, R²={r2:.3f})')
        
        if 'confined_diffusion' in fit_results:
            params = fit_results['confined_diffusion']
            D = params['D']
            L = params['L']
            r2 = params['r_squared']
            y_fit = confined_diffusion(t_fine, D, L)
            ax.plot(t_fine, y_fit, '-.', label=f'Confined (D={D:.3f}, L={L:.3f}, R²={r2:.3f})')
        
        if 'directed_diffusion' in fit_results:
            params = fit_results['directed_diffusion']
            D = params['D']
            v = params['v']
            r2 = params['r_squared']
            y_fit = directed_diffusion(t_fine, D, v)
            ax.plot(t_fine, y_fit, ':', label=f'Directed (D={D:.3f}, v={v:.3f}, R²={r2:.3f})')
        
        # Highlight best model
        if 'best_model' in fit_results:
            best_model = fit_results['best_model']
            ax.text(0.05, 0.95, f'Best model: {best_model}',
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('MSD (μm²)')
        if title:
            ax.set_title(title)
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Set log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting MSD fits: {str(e)}")
        raise


def plot_motion_classification(tracks_df, classification_df, background=None,
                             ax=None, figsize=(10, 8), pixel_size=1.0, 
                             cmap='tab10', title=None):
    """
    Plot tracks colored by motion classification.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    classification_df : pandas.DataFrame
        DataFrame with motion classification
    background : numpy.ndarray, optional
        Background image, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    cmap : str, optional
        Colormap to use, by default 'tab10'
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background image if provided
        if background is not None:
            ax.imshow(background, cmap='gray', alpha=0.5)
        
        # Get motion type lookup
        motion_lookup = dict(zip(classification_df['track_id'], classification_df['motion_type']))
        
        # Get unique motion types
        motion_types = sorted(classification_df['motion_type'].unique())
        
        # Create color mapping
        cmap_obj = plt.get_cmap(cmap, len(motion_types))
        color_dict = {motion_type: cmap_obj(i) for i, motion_type in enumerate(motion_types)}
        
        # Plot tracks
        for track_id, track in tracks_df.groupby('track_id'):
            # Skip if track doesn't have motion classification
            if track_id not in motion_lookup:
                continue
            
            # Sort track by frame
            track = track.sort_values('frame')
            
            # Get coordinates
            x = track['x'] * pixel_size
            y = track['y'] * pixel_size
            
            # Get motion type
            motion_type = motion_lookup[track_id]
            
            # Get color
            color = color_dict[motion_type]
            
            # Plot track
            ax.plot(x, y, '-', color=color, linewidth=1.5, alpha=0.7, label=motion_type)
            ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=5, alpha=0.8)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', color=color, markersize=5, alpha=0.8)
        
        # Add legend with unique labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting motion classification: {str(e)}")
        raise


def plot_trajectory_features(classification_df, ax=None, figsize=(10, 8),
                           cmap='tab10', title=None):
    """
    Plot trajectory features by motion type.
    
    Parameters
    ----------
    classification_df : pandas.DataFrame
        DataFrame with trajectory features and classification
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap to use, by default 'tab10'
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique motion types
        motion_types = sorted(classification_df['motion_type'].unique())
        
        # Create color mapping
        cmap_obj = plt.get_cmap(cmap, len(motion_types))
        color_dict = {motion_type: cmap_obj(i) for i, motion_type in enumerate(motion_types)}
        
        # Plot features
        for motion_type in motion_types:
            df_subset = classification_df[classification_df['motion_type'] == motion_type]
            ax.scatter(df_subset['D'], df_subset['alpha'], 
                      c=[color_dict[motion_type]], label=motion_type,
                      alpha=0.7, s=50)
        
        # Add legend
        ax.legend()
        
        # Set labels and title
        ax.set_xlabel('Diffusion coefficient (μm²/s)')
        ax.set_ylabel('Anomalous exponent (α)')
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting trajectory features: {str(e)}")
        raise


def plot_diffusion_boxplots(classification_df, feature='D', ax=None, figsize=(10, 6),
                          title=None):
    """
    Plot boxplots of a feature by motion type.
    
    Parameters
    ----------
    classification_df : pandas.DataFrame
        DataFrame with classification and features
    feature : str, optional
        Feature to plot, by default 'D'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(x='motion_type', y=feature, data=classification_df, ax=ax)
        
        # Add individual points
        sns.stripplot(x='motion_type', y=feature, data=classification_df, 
                     ax=ax, color='black', alpha=0.5, size=4)
        
        # Set labels and title
        feature_labels = {
            'D': 'Diffusion coefficient (μm²/s)',
            'alpha': 'Anomalous exponent (α)',
            'asymmetry': 'Asymmetry',
            'straightness': 'Straightness'
        }
        
        ax.set_ylabel(feature_labels.get(feature, feature))
        ax.set_xlabel('Motion type')
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting diffusion boxplots: {str(e)}")
        raise


def plot_diffusion_time_evolution(tracks_df, window_size=10, overlap=5,
                                cmap='viridis', ax=None, figsize=(10, 8),
                                pixel_size=0.1, frame_interval=0.1, 
                                title=None):
    """
    Plot evolution of diffusion coefficients over time.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    window_size : int, optional
        Window size for calculating diffusion coefficient, by default 10
    overlap : int, optional
        Overlap between consecutive windows, by default 5
    cmap : str, optional
        Colormap to use, by default 'viridis'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    pixel_size : float, optional
        Pixel size for proper scaling, by default 0.1
    frame_interval : float, optional
        Time between frames in seconds, by default 0.1
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        from scipy import stats
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Filter for sufficiently long tracks
        long_tracks = []
        for track_id in track_ids:
            track = tracks_df[tracks_df['track_id'] == track_id]
            if len(track) >= window_size:
                long_tracks.append(track_id)
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=len(long_tracks) - 1)
        
        # Process each long track
        for i, track_id in enumerate(long_tracks):
            # Get track data
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Get time evolution of diffusion coefficient
            diffusion_time = []
            diffusion_values = []
            
            for start_idx in range(0, len(track) - window_size + 1, overlap):
                end_idx = start_idx + window_size
                window = track.iloc[start_idx:end_idx]
                
                # Compute MSD for this window
                msd_values = []
                time_lags = []
                
                for lag in range(1, min(5, window_size)):
                    positions_t0 = window[['x', 'y']].values[:-lag]
                    positions_t1 = window[['x', 'y']].values[lag:]
                    
                    # Convert to μm
                    positions_t0 = positions_t0 * pixel_size
                    positions_t1 = positions_t1 * pixel_size
                    
                    # Compute squared displacements
                    squared_disp = np.sum((positions_t1 - positions_t0)**2, axis=1)
                    
                    # Compute mean and add to MSD values
                    msd = np.mean(squared_disp)
                    msd_values.append(msd)
                    
                    # Add time lag in seconds
                    time_lags.append(lag * frame_interval)
                
                # Fit MSD curve to get diffusion coefficient
                slope, _, _, _, _ = stats.linregress(time_lags, msd_values)
                D = slope / 4.0
                
                # Add to time evolution
                time_point = window['frame'].mean() * frame_interval
                diffusion_time.append(time_point)
                diffusion_values.append(D)
            
            # Plot time evolution
            color = cmap_obj(norm(i))
            ax.plot(diffusion_time, diffusion_values, '-o', color=color, alpha=0.7, 
                   linewidth=1.5, markersize=4, label=f'Track {track_id}')
        
        # Add legend if not too many tracks
        if len(long_tracks) <= 10:
            ax.legend()
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Diffusion coefficient (μm²/s)')
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting diffusion time evolution: {str(e)}")
        raise
