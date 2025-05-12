# spt_analyzer/tracking/tracker.py
"""
Tracker module for SPT Analysis.

This module integrates detection and linking to create a complete tracking pipeline
for single-particle tracking, with capabilities for handling complex tracking scenarios.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
# Assuming detector and linker modules provide the get_detector/get_linker functions
# and the respective classes/functions they return.
from . import detector
from . import linker

logger = logging.getLogger(__name__)


class ParticleTracker:
    """
    Main tracker class that integrates detection and linking.

    The performance and accuracy of this tracker are highly dependent on the
    chosen detector and linker methods and their implementations.

    Parameters
    ----------
    detector_method : str, optional
        Detection method to use (passed to detector.get_detector), by default "wavelet".
    detector_params : dict, optional
        Parameters for the detector, by default None.
    linker_method : str, optional
        Linking method to use (passed to linker.get_linker), by default "graph".
    linker_params : dict, optional
        Parameters for the linker, by default None.
    """

    def __init__(self, detector_method="wavelet", detector_params=None,
                 linker_method="graph", linker_params=None):
        self.detector_method = detector_method
        self.detector_params = detector_params or {}
        self.linker_method = linker_method
        self.linker_params = linker_params or {}

        # Initialize detector and linker using factory functions.
        # The functionality relies entirely on these returned objects.
        try:
            self.detector = detector.get_detector(detector_method, **self.detector_params)
            logger.info(f"Initialized detector: {detector_method}")
        except Exception as e:
            logger.error(f"Failed to initialize detector '{detector_method}': {e}", exc_info=True)
            raise ValueError(f"Could not create detector '{detector_method}'") from e

        try:
            self.linker = linker.get_linker(linker_method, **self.linker_params)
            logger.info(f"Initialized linker: {linker_method}")
        except Exception as e:
            logger.error(f"Failed to initialize linker '{linker_method}': {e}", exc_info=True)
            raise ValueError(f"Could not create linker '{linker_method}'") from e

        # Storage for tracks
        self._raw_detections = None # Store raw detections if needed later
        self._raw_tracks = None # Store linker output if needed later
        self._track_df = None

    def track(self, frames, batch_size=None):
        """
        Perform tracking on a sequence of frames.

        Parameters
        ----------
        frames : numpy.ndarray
            Array of image frames, shape (n_frames, height, width) or (n_frames, channels, height, width).
            Assumes tracking is done on a single channel if 4D.
        batch_size : int, optional
            Batch size for processing frames during detection. If None, processes all at once.
            Default is None.

        Returns
        -------
        pandas.DataFrame
            DataFrame with track data ('track_id', 'frame', 'y', 'x'). Returns empty DataFrame if tracking fails.
        """
        try:
            if not isinstance(frames, np.ndarray):
                raise TypeError("Input 'frames' must be a NumPy array.")
            if frames.ndim < 3 or frames.ndim > 4:
                raise ValueError(f"Input 'frames' must be 3D (T,Y,X) or 4D (T,C,Y,X), got {frames.ndim}D.")

            n_frames = frames.shape[0]
            if n_frames < 2:
                 logger.warning("Need at least 2 frames for tracking. Returning empty DataFrame.")
                 self._track_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
                 return self._track_df

            frame_indices = list(range(n_frames)) # Keep track of original frame index

            logger.info(f"Starting tracking with {n_frames} frames using detector '{self.detector_method}' and linker '{self.linker_method}'")

            # --- Detection ---
            detections = [] # List to store detections for each frame
            self._raw_detections = [] # Store raw detections if needed
            logger.info("Detecting particles...")
            # Determine which image data to use for detection
            image_for_detection = frames
            if frames.ndim == 4:
                 # Assuming tracking on the first channel if 4D input
                 logger.warning("Input is 4D (T,C,Y,X), performing detection on channel 0.")
                 image_for_detection = frames[:, 0, :, :]

            # Process frames (batched or all at once)
            if batch_size is None or batch_size >= n_frames:
                for i, frame in enumerate(image_for_detection):
                    # logger.debug(f"Detecting particles in frame {i}/{n_frames}")
                    frame_detections = self.detector.detect(frame)
                    detections.append(frame_detections if frame_detections is not None else np.empty((0, 2))) # Ensure list contains arrays
            else:
                for batch_start in range(0, n_frames, batch_size):
                    batch_end = min(batch_start + batch_size, n_frames)
                    batch_frames = image_for_detection[batch_start:batch_end]
                    logger.debug(f"Processing detection batch frames {batch_start}-{batch_end-1}")
                    for i, frame in enumerate(batch_frames):
                        # logger.debug(f"Detecting particles in frame {batch_start + i}/{n_frames}")
                        frame_detections = self.detector.detect(frame)
                        detections.append(frame_detections if frame_detections is not None else np.empty((0, 2)))

            # Store raw detections
            self._raw_detections = detections
            total_detections = sum(len(d) for d in detections)
            logger.info(f"Detection complete. Found {total_detections} total detections across {n_frames} frames.")

            # --- Linking ---
            logger.info("Linking particles into tracks...")
            # The linker is expected to return a list of tracks.
            # Each track is a list of tuples: (frame_number, frame_index, detection_index)
            # This relies heavily on the specific linker implementation.
            raw_tracks = self.linker.link(detections, frame_indices)
            self._raw_tracks = raw_tracks # Store linker output
            logger.info(f"Linking complete. Found {len(raw_tracks)} raw tracks.")

            # --- DataFrame Conversion (Optimized) ---
            logger.info("Converting tracks to DataFrame...")
            if not raw_tracks:
                logger.warning("No tracks were generated by the linker.")
                self._track_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
                return self._track_df

            # Pre-allocate lists for columns
            track_ids = []
            frames_col = []
            y_coords = []
            x_coords = []

            for track_id, track in enumerate(raw_tracks):
                if not track: continue # Skip empty tracks from linker
                # Unpack track points efficiently
                try:
                    # Assumes track is list of (frame_num, frame_idx, det_idx)
                    t_frames, t_frame_indices, t_det_indices = zip(*track)
                except (ValueError, TypeError) as e:
                     logger.warning(f"Skipping track {track_id} due to unexpected format from linker: {track}. Error: {e}")
                     continue

                # Retrieve coordinates using indices
                # This assumes detections list structure is consistent
                try:
                    coords = [detections[f_idx][d_idx] for f_idx, d_idx in zip(t_frame_indices, t_det_indices)]
                    coords_arr = np.array(coords) # Shape (n_points, 2)
                except (IndexError, TypeError) as e:
                     logger.warning(f"Skipping track {track_id} due to issue retrieving coordinates from detections using indices. Error: {e}")
                     continue

                n_points = len(t_frames)
                track_ids.extend([track_id] * n_points)
                frames_col.extend(t_frames)
                y_coords.extend(coords_arr[:, 0]) # y is first column
                x_coords.extend(coords_arr[:, 1]) # x is second column

            # Create DataFrame from lists
            self._track_df = pd.DataFrame({
                'track_id': track_ids,
                'frame': frames_col,
                'y': y_coords,
                'x': x_coords
            })
            logger.info(f"DataFrame created with {len(self._track_df)} points.")

            # --- Final Filtering (Optional - Basic length filter) ---
            # More complex filtering is in filter_tracks method
            min_len_param = self.linker_params.get('min_track_length', 1) # Get min length used by linker if available
            if min_len_param > 1:
                 logger.info(f"Applying minimum track length filter ({min_len_param}) post-linking...")
                 track_lengths = self._track_df.groupby('track_id').size()
                 valid_tracks = track_lengths[track_lengths >= min_len_param].index
                 self._track_df = self._track_df[self._track_df['track_id'].isin(valid_tracks)]
                 logger.info(f"Tracks after length filter: {self._track_df['track_id'].nunique()}")


            return self._track_df.copy() # Return a copy

        except Exception as e:
            logger.error(f"Error during tracking pipeline: {e}", exc_info=True)
            # Return empty DataFrame on error
            self._track_df = pd.DataFrame(columns=['track_id', 'frame', 'y', 'x'])
            return self._track_df

    def get_tracks(self):
        """
        Get the tracks DataFrame generated by the last `track()` call.

        Returns
        -------
        pandas.DataFrame
            DataFrame with track data ('track_id', 'frame', 'y', 'x').

        Raises
        ------
        ValueError
            If `track()` has not been called successfully.
        """
        if self._track_df is None:
            raise ValueError("No tracks available. Run track() first.")

        return self._track_df.copy() # Return a copy

    def track_diagnostics(self):
        """
        Calculate and return diagnostics for the tracked particles.

        Returns
        -------
        dict
            Dictionary with track diagnostics (counts, lengths, displacements).
            Returns empty dict if no tracks are available.
        """
        if self._track_df is None or self._track_df.empty:
            logger.warning("No tracks available for diagnostics.")
            return {}

        logger.info("Calculating track diagnostics...")
        try:
            # Calculate track statistics
            track_lengths = self._track_df.groupby('track_id').size()
            total_tracks = len(track_lengths)
            if total_tracks == 0: return {'total_tracks': 0} # Handle case after filtering

            mean_track_length = track_lengths.mean()
            median_track_length = track_lengths.median()
            max_track_length = track_lengths.max()
            min_track_length = track_lengths.min() # Added min length

            # Calculate frame statistics
            detections_per_frame = self._track_df.groupby('frame').size()
            mean_detections = detections_per_frame.mean()
            max_detections = detections_per_frame.max() # Added max detections

            # --- Calculate displacement statistics (Optimized) ---
            # Sort by track_id and frame is crucial for diff()
            df_sorted = self._track_df.sort_values(['track_id', 'frame'])
            # Calculate difference in x and y within each group (track)
            df_sorted['dx'] = df_sorted.groupby('track_id')['x'].diff()
            df_sorted['dy'] = df_sorted.groupby('track_id')['y'].diff()
            # Calculate frame difference (for gap detection, though not used here directly)
            # df_sorted['dframe'] = df_sorted.groupby('track_id')['frame'].diff()

            # Calculate displacement magnitude, ignore NaNs from first point of each track
            displacements = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2).dropna()

            if displacements.empty:
                mean_displacement = 0
                median_displacement = 0
                max_displacement = 0
                std_displacement = 0 # Added std dev
            else:
                mean_displacement = displacements.mean()
                median_displacement = displacements.median()
                max_displacement = displacements.max()
                std_displacement = displacements.std()
            # ----------------------------------------------------

            logger.info("Diagnostics calculation complete.")
            return {
                'total_tracks': int(total_tracks),
                'mean_track_length': float(mean_track_length),
                'median_track_length': float(median_track_length),
                'min_track_length': int(min_track_length),
                'max_track_length': int(max_track_length),
                'mean_detections_per_frame': float(mean_detections),
                'max_detections_per_frame': int(max_detections),
                'mean_displacement': float(mean_displacement),
                'median_displacement': float(median_displacement),
                'std_displacement': float(std_displacement),
                'max_displacement': float(max_displacement),
                'total_points': len(self._track_df) # Added total points
            }

        except Exception as e:
            logger.error(f"Error calculating diagnostics: {e}", exc_info=True)
            return {'error': str(e)}


    def filter_tracks(self, min_length: Optional[int] = None,
                      max_gap: Optional[int] = None,
                      roi: Optional[Tuple[float, float, float, float]] = None):
        """
        Filter tracks based on length, gaps, or region of interest.

        Parameters
        ----------
        min_length : int, optional
            Minimum track length (number of points), by default None.
        max_gap : int, optional
            Maximum allowed gap in frames between consecutive points in a track.
            A gap of 0 means frames must be consecutive. Default is None (no gap filtering).
        roi : tuple, optional
            Region of interest (x_min, y_min, x_max, y_max) in pixel coordinates.
            Tracks with *any* point outside this ROI will be removed. Default is None.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with track data. Returns empty DataFrame if no tracks remain.

        Raises
        ------
        ValueError
            If `track()` has not been called successfully.
        """
        if self._track_df is None or self._track_df.empty:
            logger.warning("No tracks available to filter.")
            return pd.DataFrame(columns=self._track_df.columns if self._track_df is not None else ['track_id', 'frame', 'y', 'x'])


        logger.info(f"Filtering tracks: min_length={min_length}, max_gap={max_gap}, roi={roi}")
        filtered_df = self._track_df.copy()
        initial_track_count = filtered_df['track_id'].nunique()

        # --- Filter by track length ---
        if min_length is not None and min_length > 1:
            track_lengths = filtered_df.groupby('track_id').size()
            valid_tracks_len = track_lengths[track_lengths >= min_length].index
            filtered_df = filtered_df[filtered_df['track_id'].isin(valid_tracks_len)]
            logger.debug(f"Tracks after length filter ({min_length}): {filtered_df['track_id'].nunique()}")

        # --- Filter by maximum gap (Optimized) ---
        if max_gap is not None and max_gap >= 0:
            if not filtered_df.empty:
                # Calculate frame difference within each track
                filtered_df = filtered_df.sort_values(['track_id', 'frame'])
                filtered_df['dframe'] = filtered_df.groupby('track_id')['frame'].diff()
                # Find tracks where the maximum frame difference exceeds the allowed gap + 1
                max_frame_diff = filtered_df.groupby('track_id')['dframe'].max()
                # Allowed difference is max_gap + 1 (e.g., max_gap=0 means dframe must be 1)
                tracks_with_large_gaps = max_frame_diff[max_frame_diff > (max_gap + 1)].index
                # Keep only tracks that *do not* have large gaps
                filtered_df = filtered_df[~filtered_df['track_id'].isin(tracks_with_large_gaps)]
                # Clean up temporary column
                if 'dframe' in filtered_df.columns:
                     filtered_df = filtered_df.drop(columns=['dframe'])
                logger.debug(f"Tracks after gap filter ({max_gap}): {filtered_df['track_id'].nunique()}")
            else:
                 logger.debug("Skipping gap filter as DataFrame is empty.")


        # --- Filter by region of interest ---
        if roi is not None:
            if len(roi) != 4:
                 logger.warning(f"Invalid ROI format: {roi}. Expected (x_min, y_min, x_max, y_max). Skipping ROI filter.")
            else:
                 x_min, y_min, x_max, y_max = roi
                 # Find tracks that have *any* point outside the ROI
                 points_outside_roi = filtered_df[
                     (filtered_df['x'] < x_min) | (filtered_df['x'] > x_max) |
                     (filtered_df['y'] < y_min) | (filtered_df['y'] > y_max)
                 ]
                 tracks_outside_roi = points_outside_roi['track_id'].unique()
                 # Keep only tracks that are *not* in the set of tracks outside the ROI
                 filtered_df = filtered_df[~filtered_df['track_id'].isin(tracks_outside_roi)]
                 logger.debug(f"Tracks after ROI filter ({roi}): {filtered_df['track_id'].nunique()}")


        final_track_count = filtered_df['track_id'].nunique()
        logger.info(f"Filtering complete. Kept {final_track_count} out of {initial_track_count} tracks.")

        # Important: Reset index after filtering if needed downstream
        return filtered_df.reset_index(drop=True)

