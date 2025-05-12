# Modified: spt_analyzer/visualization/utils.py
"""
Utility functions for visualization in SPT Analysis.

Provides colormap generation, figure saving, animation, interactive plots (if mpld3 available),
and implementations for previously placeholder plotting functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PatchCollection # Added PatchCollection
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches # For detection overlay circles
import colorsys
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
import seaborn as sns # For correlation matrix

# --- Optional Dependency: mpld3 ---
try:
    import mpld3
    from mpld3 import plugins
    _HAS_MPLD3 = True
except ImportError:
    _HAS_MPLD3 = False
    logging.getLogger(__name__).warning(
        "mpld3 not found. Interactive plots ('create_interactive_plot') will fallback to static plots."
    )
# ---------------------------------

logger = logging.getLogger(__name__)


# --- Color and Figure Utilities (Keep as is) ---
def create_colormap(n_colors, cmap='viridis', start=0.0, stop=1.0, alpha=1.0):
    """ Create a list of RGBA colors from a matplotlib colormap. """
    try:
        if n_colors <= 0: return []
        base_cmap = plt.get_cmap(cmap)
        color_list = []
        denominator = max(1, n_colors - 1) if n_colors > 1 else 1
        for i in range(n_colors):
            color_val = start + (stop - start) * i / denominator
            rgba = list(base_cmap(color_val))
            rgba[3] = alpha
            color_list.append(tuple(rgba))
        return color_list
    except Exception as e: logger.error(f"Error creating colormap: {e}", exc_info=True); raise

def create_distinct_colors(n_colors, alpha=1.0):
    """ Create a list of visually distinct RGBA colors using HSV space. """
    try:
        colors = []
        if n_colors <= 0: return colors
        for i in range(n_colors):
            h = (i * 0.618033988749895) % 1.0 # Golden ratio hue distribution
            s = 0.7 + (i % 3) * 0.1 # Vary saturation
            v = 0.8 + (i % 2) * 0.1 # Vary value
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append((r, g, b, alpha))
        return colors
    except Exception as e: logger.error(f"Error creating distinct colors: {e}", exc_info=True); raise

def save_figure(fig: Figure, filename: str, dpi: int = 300, transparent: bool = False, bbox_inches: str = 'tight'):
    """ Save matplotlib figure to file. """
    try:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory): os.makedirs(directory)
        fig.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)
        logger.info(f"Figure saved to {filename}")
        return filename
    except Exception as e: logger.error(f"Error saving figure: {e}", exc_info=True); raise

# --- Animation and Interactive Plot (Keep as is) ---
def create_track_animation(tracks_df: pd.DataFrame, output_file: str, fps: int = 10, dpi: int = 100,
                         background: Optional[np.ndarray] = None, trail_length: int = 10, marker_size: int = 5,
                         figsize: Tuple[int, int] = (10, 8), cmap: str = 'viridis', pixel_size: float = 1.0,
                         title: Optional[str] = None) -> Optional[str]:
    """ Create an animation of particle tracks. """
    required_cols = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_cols): raise ValueError(f"tracks_df missing: {required_cols}")
    if tracks_df.empty: logger.warning("Empty tracks_df for animation."); return None
    try:
        fig, ax = plt.subplots(figsize=figsize)
        x_min, x_max = tracks_df['x'].min(), tracks_df['x'].max()
        y_min, y_max = tracks_df['y'].min(), tracks_df['y'].max()
        extent = [x_min * pixel_size, x_max * pixel_size, y_min * pixel_size, y_max * pixel_size]
        if background is not None:
             if background.ndim == 2: ax.imshow(background, cmap='gray', alpha=0.5, extent=extent)
             else: logger.warning(f"Background image not 2D. Skipping.")
        units = "μm" if pixel_size != 1.0 else "pixels"
        ax.set_xlabel(f'X ({units})'); ax.set_ylabel(f'Y ({units})')
        ax.set_title(title or "Track Animation")
        track_ids = tracks_df['track_id'].unique()
        frames = sorted(tracks_df['frame'].unique())
        if not frames: logger.warning("No frames found."); plt.close(fig); return None
        track_colors = create_distinct_colors(len(track_ids), alpha=0.8)
        track_id_to_color = {tid: track_colors[i % len(track_colors)] for i, tid in enumerate(track_ids)}
        scatter = ax.scatter([], [], s=marker_size**2, c=[], alpha=0.9)
        line_collection = LineCollection([], linewidths=1, alpha=0.5)
        ax.add_collection(line_collection)
        margin_x = 0.05 * (extent[1] - extent[0]) if (extent[1] - extent[0]) > 0 else 1
        margin_y = 0.05 * (extent[3] - extent[2]) if (extent[3] - extent[2]) > 0 else 1
        ax.set_xlim(extent[0] - margin_x, extent[1] + margin_x)
        ax.set_ylim(extent[3] + margin_y, extent[2] - margin_y)
        ax.set_aspect('equal')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        def init():
            scatter.set_offsets(np.empty((0, 2))); line_collection.set_segments([]); time_text.set_text('')
            return scatter, line_collection, time_text

        def update(frame_idx):
            frame = frames[frame_idx]; time_text.set_text(f'Frame: {frame}')
            current_points = tracks_df[tracks_df['frame'] == frame]
            if not current_points.empty:
                positions = current_points[['x', 'y']].values * pixel_size
                colors = [track_id_to_color[tid] for tid in current_points['track_id']]
                scatter.set_offsets(positions); scatter.set_facecolor(colors)
            else: scatter.set_offsets(np.empty((0, 2)))
            segments = []; segment_colors = []
            for i, track_id in enumerate(track_ids):
                track = tracks_df[(tracks_df['track_id'] == track_id) & (tracks_df['frame'] <= frame) & (tracks_df['frame'] > frame - trail_length)]
                if len(track) < 2: continue
                track = track.sort_values('frame')
                trail_points = track[['x', 'y']].values * pixel_size
                points_reshaped = np.c_[trail_points[:-1], trail_points[1:]].reshape(-1, 2, 2)
                segments.extend(points_reshaped)
                segment_colors.extend([track_id_to_color[track_id]] * len(points_reshaped))
            if segments: line_collection.set_segments(segments); line_collection.set_color(segment_colors)
            else: line_collection.set_segments([])
            return scatter, line_collection, time_text

        anim = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False, interval=1000/fps)
        logger.info(f"Saving animation to {output_file}...")
        try:
            if output_file.endswith('.mp4'): writer = animation.FFMpegWriter(fps=fps)
            elif output_file.endswith('.gif'): writer = animation.PillowWriter(fps=fps)
            else: writer = None # Rely on default based on extension
            anim.save(output_file, writer=writer, dpi=dpi)
        except Exception as save_err: logger.error(f"Failed to save animation (ensure writers installed): {save_err}", exc_info=True); plt.close(fig); raise RuntimeError(f"Animation saving failed: {save_err}") from save_err
        plt.close(fig)
        logger.info(f"Animation saved to {output_file}")
        return output_file
    except Exception as e: logger.error(f"Error creating track animation: {e}", exc_info=True); raise

def create_interactive_plot(tracks_df: pd.DataFrame, diffusion_df: Optional[pd.DataFrame] = None, cluster_df: Optional[pd.DataFrame] = None,
                           figsize: Tuple[int, int] = (10, 8), cmap: str = 'viridis', pixel_size: float = 1.0,
                           title: Optional[str] = None) -> Union[Figure, str, None]:
    """ Create an interactive plot of tracks with hover info (requires mpld3). """
    required_cols = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_cols): raise ValueError(f"tracks_df missing: {required_cols}")
    if not _HAS_MPLD3: logger.warning("mpld3 not installed. Returning static plot Figure."); interactive = False
    else: interactive = True
    try:
        fig, ax = plt.subplots(figsize=figsize)
        diff_lookup = {}; cluster_lookup = {}
        if diffusion_df is not None and not diffusion_df.empty and 'track_id' in diffusion_df.columns and 'D' in diffusion_df.columns:
            diff_lookup = dict(zip(diffusion_df.drop_duplicates(subset='track_id')['track_id'], diffusion_df.drop_duplicates(subset='track_id')['D']))
        if cluster_df is not None and not cluster_df.empty and 'track_id' in cluster_df.columns and 'cluster' in cluster_df.columns:
            cluster_lookup = cluster_df.groupby('track_id')['cluster'].agg(lambda x: x.mode()[0] if not x.mode().empty else -1).to_dict()
        unique_track_ids = tracks_df['track_id'].unique()
        track_colors = create_distinct_colors(len(unique_track_ids), alpha=0.8)
        track_id_to_color = {tid: track_colors[i % len(track_colors)] for i, tid in enumerate(unique_track_ids)}
        plot_elements = []
        for track_id, track in tracks_df.groupby('track_id'):
            track = track.sort_values('frame'); x = (track['x'] * pixel_size).tolist(); y = (track['y'] * pixel_size).tolist()
            if not x or not y: continue
            line_data = [[[x[i], y[i]], [x[i+1], y[i+1]]] for i in range(len(x) - 1)]
            tooltip_base = f"Track ID: {track_id}<br>Length: {len(track)} frames<br>"
            if track_id in diff_lookup: tooltip_base += f"D: {diff_lookup[track_id]:.3E} μm²/s<br>"
            if track_id in cluster_lookup: tooltip_base += f"Cluster: {cluster_lookup[track_id]}<br>"
            point_tooltips = [tooltip_base + f"Frame: {track['frame'].iloc[i]}" for i in range(len(track))]
            plot_elements.append({'lines': line_data, 'points_x': x, 'points_y': y, 'point_tooltips': point_tooltips, 'color': mcolors.to_hex(track_id_to_color[track_id], keep_alpha=True)})

        if interactive:
             all_points_collections = []
             for element in plot_elements:
                  if element['lines']: lc = LineCollection(element['lines'], colors=element['color'], linewidths=1, alpha=0.7); ax.add_collection(lc)
                  pts = ax.scatter(element['points_x'], element['points_y'], c=element['color'], s=20, alpha=0.7)
                  all_points_collections.append((pts, element['point_tooltips']))
             for pts_collection, tooltips in all_points_collections:
                  tooltip = plugins.PointHTMLTooltip(pts_collection, labels=tooltips, voffset=10, hoffset=10); plugins.connect(fig, tooltip)
             plugins.connect(fig, plugins.Zoom()); plugins.connect(fig, plugins.Reset())
        else: # Static fallback
             for element in plot_elements:
                  if element['lines']: lc = LineCollection(element['lines'], colors=element['color'], linewidths=1, alpha=0.7); ax.add_collection(lc)
                  ax.scatter(element['points_x'], element['points_y'], c=element['color'], s=20, alpha=0.7)

        units = "μm" if pixel_size != 1.0 else "pixels"
        ax.set_xlabel(f'X ({units})'); ax.set_ylabel(f'Y ({units})')
        ax.set_aspect('equal'); ax.autoscale_view(); ax.invert_yaxis()
        if title: ax.set_title(title)

        if interactive: html_output = mpld3.fig_to_html(fig); plt.close(fig); return html_output
        else: return fig # Return static figure
    # --- Corrected except block ---
    except Exception as e:
        logger.error(f"Error creating interactive plot: {e}", exc_info=True)
        # Close figure only if it was defined and still exists
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        raise # Re-raise the exception
    # ----------------------------

def plot_overlay_image(image: np.ndarray, tracks_df: pd.DataFrame, alpha: float = 0.5, figsize: Tuple[int, int] = (10, 8),
                     cmap_image: str = 'gray', cmap_tracks: str = 'viridis', pixel_size: float = 1.0,
                     title: Optional[str] = None, track_id_map: Optional[Dict] = None) -> Axes:
    """ Plot tracks overlaid on an image. """
    # --- Corrected if block ---
    if image.ndim != 2:
        logger.warning(f"Input image not 2D (shape: {image.shape}). Plotting first slice if possible.")
        if image.ndim > 2:
            image = image[0] # Take first frame/slice
        # Check again after potential slicing
        if image.ndim != 2:
            raise ValueError("Input image must be 2D or reducible to 2D.")
    # --------------------------
    required_cols = ['track_id', 'frame', 'x', 'y']
    if not all(col in tracks_df.columns for col in required_cols): raise ValueError(f"tracks_df missing: {required_cols}")
    try:
        fig, ax = plt.subplots(figsize=figsize)
        height, width = image.shape
        extent = [0, width * pixel_size, 0, height * pixel_size]
        ax.imshow(image, cmap=cmap_image, extent=extent)
        if track_id_map: unique_track_ids = sorted(track_id_map.keys()); id_to_index = {tid: i for i, tid in enumerate(unique_track_ids)}; num_colors = len(unique_track_ids)
        else: unique_track_ids = tracks_df['track_id'].unique(); id_to_index = {tid: i for i, tid in enumerate(unique_track_ids)}; num_colors = len(unique_track_ids)
        cmap_obj = plt.get_cmap(cmap_tracks); norm = mcolors.Normalize(vmin=0, vmax=max(1, num_colors - 1))
        plotted_ids = set()
        for track_id, track in tracks_df.groupby('track_id'):
            if track_id not in id_to_index: continue
            track = track.sort_values('frame'); x = track['x'] * pixel_size; y = track['y'] * pixel_size
            color_index = id_to_index[track_id]; color = cmap_obj(norm(color_index))
            label = f'Track {track_id}' if track_id not in plotted_ids else None
            ax.plot(x, y, '-', color=color, linewidth=1, alpha=alpha, label=label)
            ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=5, alpha=alpha)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', color=color, markersize=5, alpha=alpha)
            plotted_ids.add(track_id)
        units = "μm" if pixel_size != 1.0 else "pixels"
        ax.set_xlabel(f'X ({units})'); ax.set_ylabel(f'Y ({units})')
        if title: ax.set_title(title)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
        ax.set_aspect('equal')
        # if num_colors <= 15: ax.legend(loc='upper right', fontsize='small')
        return ax
    except Exception as e: logger.error(f"Error plotting overlay image: {e}", exc_info=True); raise

# --- Implemented Placeholder Functions ---

def plot_preprocessing_steps(original_image: np.ndarray, processed_images: List[np.ndarray], titles: List[str],
                            figsize_scale: float = 4.0, cmap: str = 'gray') -> Figure:
    """
    Visualize original image and sequential preprocessing steps side-by-side.

    Parameters
    ----------
    original_image : np.ndarray
        The starting 2D image.
    processed_images : List[np.ndarray]
        A list of 2D images representing the output after each preprocessing step.
    titles : List[str]
        A list of titles corresponding to each processed image. Should have the same length as processed_images.
    figsize_scale : float, optional
        Scaling factor for figure width based on number of images, by default 4.0.
    cmap : str, optional
        Colormap for displaying images, by default 'gray'.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    logger.info("Plotting preprocessing steps.")
    num_steps = len(processed_images)
    if len(titles) != num_steps:
        logger.warning("Number of titles does not match number of processed images. Using generic titles.")
        titles = [f"Step {i+1}" for i in range(num_steps)]

    num_plots = num_steps + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(figsize_scale * num_plots, figsize_scale), squeeze=False)
    axes = axes.flatten() # Ensure axes is 1D

    # Plot Original
    axes[0].imshow(original_image, cmap=cmap)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Plot Processed Steps
    for i, (img, title) in enumerate(zip(processed_images, titles)):
        if i < len(axes) - 1: # Check if axes exist
            ax = axes[i+1]
            if isinstance(img, np.ndarray):
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                # fig.colorbar(im, ax=ax, shrink=0.7) # Optional colorbar
            else:
                ax.text(0.5, 0.5, "Invalid Image Data", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    fig.tight_layout()
    return fig

def plot_detection_overlay(image: np.ndarray, detected_points: np.ndarray,
                           diameter: Optional[int] = None, ax: Optional[Axes] = None,
                           figsize: Tuple[int, int] = (8, 8), cmap: str = 'gray',
                           point_color: str = 'red', point_size: int = 5,
                           title: Optional[str] = None) -> Axes:
    """
    Visualize detected points overlaid on an image. Optionally draws circles around points.

    Parameters
    ----------
    image : np.ndarray
        The 2D image on which detection was performed.
    detected_points : np.ndarray
        Array of detected points, shape (n, 2) with (y, x) coordinates.
    diameter : int, optional
        If provided, draws circles with this approximate diameter around points, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None.
    figsize : tuple, optional
        Figure size if creating a new figure, by default (8, 8).
    cmap : str, optional
        Colormap for the background image, by default 'gray'.
    point_color : str, optional
        Color for the detected points/circles, by default 'red'.
    point_size : int, optional
        Marker size for detected points (if diameter is None), by default 5.
    title : str, optional
        Plot title, by default None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    """
    logger.info("Plotting detection overlay.")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(image, cmap=cmap)

    num_points = 0
    if detected_points is not None and detected_points.ndim == 2 and detected_points.shape[1] == 2:
        num_points = len(detected_points)
        y_coords, x_coords = detected_points[:, 0], detected_points[:, 1]

        if diameter is not None and diameter > 0:
            # Draw circles around points
            radius = diameter / 2.0
            # Use PatchCollection for efficiency if many points
            circle_patches = [patches.Circle((x, y), radius, color=point_color, fill=False, linewidth=1)
                              for y, x in detected_points]
            p = PatchCollection(circle_patches, match_original=True) # match_original uses patch properties
            ax.add_collection(p)
        else:
            # Plot markers
            ax.plot(x_coords, y_coords, 'o', color=point_color, markersize=point_size,
                    fillstyle='none', markeredgewidth=1)

    plot_title = title if title else f"Detections ({num_points} points)"
    ax.set_title(plot_title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.axis('image') # Use image limits and aspect ratio
    # ax.invert_yaxis() # Often desired for images, uncomment if needed

    return ax

def plot_tracking_summary(tracks_df: pd.DataFrame, figsize: Tuple[int, int] = (12, 5)) -> Figure:
    """
    Visualize tracking summary statistics, like track length distribution and tracks per frame.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with track data (requires 'track_id', 'frame').
    figsize : tuple, optional
        Figure size, by default (12, 5).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the summary plots.
    """
    logger.info("Plotting tracking summary.")
    if tracks_df is None or tracks_df.empty:
         logger.warning("No track data provided for summary plot.")
         fig, ax = plt.subplots(figsize=figsize)
         ax.text(0.5, 0.5, "No Track Data Available", ha='center', va='center', transform=ax.transAxes)
         ax.axis('off')
         return fig

    if not all(col in tracks_df.columns for col in ['track_id', 'frame']):
         logger.warning("Track data missing 'track_id' or 'frame' column for summary plot.")
         fig, ax = plt.subplots(figsize=figsize)
         ax.text(0.5, 0.5, "Missing Columns in Track Data", ha='center', va='center', transform=ax.transAxes)
         ax.axis('off')
         return fig

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Plot track length distribution
    track_lengths = tracks_df.groupby('track_id').size()
    if not track_lengths.empty:
         sns.histplot(track_lengths, bins=min(30, max(1,track_lengths.nunique())), kde=True, ax=axes[0])
         axes[0].set_title(f"Track Length Distribution (N={len(track_lengths)})")
         axes[0].set_xlabel("Length (frames)")
         axes[0].set_ylabel("Count / Density") # Updated label based on histplot default
         axes[0].text(0.95, 0.95, f"Mean: {track_lengths.mean():.1f}\nMedian: {track_lengths.median():.0f}\nMax: {track_lengths.max()}",
                      transform=axes[0].transAxes, ha='right', va='top', fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    else:
         axes[0].text(0.5, 0.5, "No Tracks Found", ha='center', va='center', transform=axes[0].transAxes)
         axes[0].set_title("Track Length Distribution")


    # 2. Plot number of tracks per frame
    tracks_per_frame = tracks_df.groupby('frame')['track_id'].nunique() # Count unique tracks per frame
    if not tracks_per_frame.empty:
         axes[1].plot(tracks_per_frame.index, tracks_per_frame.values)
         axes[1].set_title("Active Tracks per Frame")
         axes[1].set_xlabel("Frame")
         axes[1].set_ylabel("Number of Tracks")
         axes[1].grid(True, linestyle='--', alpha=0.6)
         axes[1].set_xlim(left=tracks_df['frame'].min(), right=tracks_df['frame'].max()) # Set limits
         axes[1].set_ylim(bottom=0)
    else:
         axes[1].text(0.5, 0.5, "No Frame Data", ha='center', va='center', transform=axes[1].transAxes)
         axes[1].set_title("Active Tracks per Frame")


    fig.tight_layout()
    return fig

def plot_correlation_matrix(data_df: pd.DataFrame, columns: Optional[List[str]] = None,
                            title: str = "Feature Correlation Matrix", figsize: Tuple[int, int] = (8, 7),
                            cmap: str = 'coolwarm', annot: bool = True) -> Optional[Figure]:
    """
    Compute and plot the correlation matrix for selected columns of a DataFrame.

    Parameters
    ----------
    data_df : pd.DataFrame
        Input DataFrame containing numerical features.
    columns : List[str], optional
        List of column names to include in the correlation matrix. If None, uses all numeric columns.
        By default None.
    title : str, optional
        Title for the plot, by default "Feature Correlation Matrix".
    figsize : tuple, optional
        Figure size, by default (8, 7).
    cmap : str, optional
        Colormap for the heatmap, by default 'coolwarm'.
    annot : bool, optional
        Whether to annotate the heatmap with correlation values, by default True.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure containing the heatmap, or None if plotting fails.
    """
    logger.info("Plotting correlation matrix.")
    if data_df is None or data_df.empty:
         logger.warning("No data provided for correlation matrix.")
         return None

    # Select columns for correlation
    if columns:
        # Check if specified columns exist and are numeric
        valid_cols = [col for col in columns if col in data_df.columns and pd.api.types.is_numeric_dtype(data_df[col])]
        if not valid_cols:
             logger.warning(f"None of the specified columns {columns} are numeric or found in the DataFrame.")
             return None
        data_to_correlate = data_df[valid_cols].copy()
    else:
        # Select only numeric columns for correlation
        data_to_correlate = data_df.select_dtypes(include=np.number).copy()

    if data_to_correlate.empty or data_to_correlate.shape[1] < 2:
         logger.warning("Not enough numeric columns found for correlation matrix.")
         return None

    try:
        # Compute correlation matrix
        corr = data_to_correlate.corr()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=annot, cmap=cmap, fmt=".2f", linewidths=.5, ax=ax,
                    cbar_kws={"shrink": .8}) # Adjust colorbar size
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right') # Rotate labels for readability
        plt.yticks(rotation=0)
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Could not compute or plot correlation matrix: {e}", exc_info=True)
        # Optionally create a figure with an error message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Correlation Plot Error:\n{e}", ha='center', va='center', transform=ax.transAxes, color='red')
        ax.axis('off')
        return fig # Return the figure with the error message
