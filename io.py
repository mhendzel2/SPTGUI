# Modified: spt_analyzer/utils/io.py
"""
Input/output module for SPT Analysis.

Provides functions for loading/saving tracks, images, configs, and results.
Includes specific loaders for TrackMate XML and heuristics for Fiji CSV formats.
"""

import numpy as np
import pandas as pd
import os
import json
import yaml
import tifffile
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import xml.etree.ElementTree as ET # For TrackMate XML

logger = logging.getLogger(__name__)

# --- Image I/O ---

def load_image_stack(file_path: str, scale: float = 1.0, subset: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load microscopy image stack from various file formats.

    Parameters
    ----------
    file_path : str
        Path to image file (TIF, Imaris, ND2, LSM, CZI, etc.)
    scale : float, optional
        Scaling factor for image intensity, by default 1.0
    subset : tuple, optional
        Frame subset to load (start_frame, end_frame_exclusive), by default None

    Returns
    -------
    numpy.ndarray
        Image stack with shape (n_frames, height, width) or (n_frames, channels, height, width)
    """
    try:
        logger.info(f"Loading image stack from {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.ims':  # Imaris format
            import h5py
            with h5py.File(file_path, 'r') as f:
                # Navigate Imaris HDF5 structure
                data_item = f['DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data']
                image_stack = data_item[:]
                
        elif ext == '.nd2':  # Nikon ND2
            import nd2
            with nd2.ND2File(file_path) as nd2_file:
                image_stack = nd2_file.asarray()
                
        elif ext == '.lsm':  # Zeiss LSM
            from tifffile import TiffFile
            with TiffFile(file_path) as tif:
                image_stack = tif.asarray()
                metadata = tif.lsm_metadata
                
        elif ext == '.czi':  # Zeiss CZI
            import czifile
            with czifile.CziFile(file_path) as czi:
                image_stack = czi.asarray()
                
        else:  # Default to tifffile for TIF and similar formats
            with tifffile.TiffFile(file_path) as tif:
                if subset is not None:
                    start, end = subset
                    try:
                        num_pages = len(tif.pages)
                        start = max(0, start)
                        end = min(num_pages, end)
                        image_stack = tif.asarray(key=range(start, end))
                    except Exception:
                        logger.warning(f"Subset loading failed for {file_path}. Loading all frames.")
                        image_stack = tif.asarray()
                else:
                    image_stack = tif.asarray()

        # Ensure minimum 3D shape (T, Y, X)
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, :, :]
        elif image_stack.ndim > 4:
            logger.warning(f"Complex data structure detected. Attempting to reshape.")
            # Try to intelligently reshape based on metadata
            image_stack = reshape_complex_stack(image_stack)

        # Apply scaling
        if scale != 1.0:
            if not np.issubdtype(image_stack.dtype, np.floating):
                image_stack = image_stack.astype(np.float32)
            image_stack = image_stack * scale

        logger.info(f"Loaded image stack with shape {image_stack.shape} and dtype {image_stack.dtype}")
        return image_stack

    except Exception as e:
        logger.error(f"Error loading image stack from {file_path}: {str(e)}", exc_info=True)
        raise

def reshape_complex_stack(image_stack: np.ndarray) -> np.ndarray:
    """Helper function to reshape complex multi-dimensional stacks."""
    # Implementation depends on specific format requirements
    # This is a placeholder for format-specific reshaping logic
    return image_stack
    """
    Load microscopy image stack from file (TIF format).

    Parameters
    ----------
    file_path : str
        Path to image file (TIF or similar).
    scale : float, optional
        Scaling factor for image intensity, by default 1.0.
    subset : tuple, optional
        Frame subset to load (start_frame, end_frame_exclusive), by default None.

    Returns
    -------
    numpy.ndarray
        Image stack with shape (n_frames, height, width) or (n_frames, channels, height, width).

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    Exception
        For general loading errors.
    """
    try:
        logger.info(f"Loading image stack from {file_path}")

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        with tifffile.TiffFile(file_path) as tif:
            # Determine keys/pages to load
            if subset is not None:
                start, end = subset
                try:
                    num_pages = len(tif.pages)
                except Exception: # Handle cases where len(tif.pages) might fail
                    logger.warning(f"Could not determine number of pages for {file_path}. Loading all.")
                    image_stack = tif.asarray()
                else:
                    # Clamp indices to valid range
                    start = max(0, start)
                    end = min(num_pages, end)
                    if start >= end:
                        logger.warning(f"Invalid subset {subset} for stack with {num_pages} pages. Loading empty stack.")
                        # Ensure shape information is available if possible
                        try:
                             page_shape = tif.pages[0].shape
                        except IndexError:
                             page_shape = (0,0) # Fallback if no pages
                        image_stack = np.empty((0,) + page_shape, dtype=tif.pages[0].dtype if num_pages > 0 else np.float32)

                    else:
                        keys = range(start, end)
                        image_stack = tif.asarray(key=keys)
            else:
                image_stack = tif.asarray()

        # Apply scaling
        if scale != 1.0:
            # Convert to float before scaling if necessary to avoid overflow/clipping
            if not np.issubdtype(image_stack.dtype, np.floating):
                image_stack = image_stack.astype(np.float32)
            image_stack = image_stack * scale

        # Ensure minimum 3D shape (T, Y, X)
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, :, :]
        elif image_stack.ndim < 2:
             raise ValueError(f"Loaded image has too few dimensions: {image_stack.shape}")
        elif image_stack.ndim > 3:
             # Handle stacks with channels, e.g. (T, C, Y, X) or (T, Y, X, C)
             logger.warning(f"Loaded image has {image_stack.ndim} dimensions. Assuming T, ..., Y, X. Check data structure.")

        logger.info(f"Loaded image stack with shape {image_stack.shape} and dtype {image_stack.dtype}")

        return image_stack

    except Exception as e:
        logger.error(f"Error loading image stack from {file_path}: {str(e)}", exc_info=True)
        raise

def save_image_stack(image_stack: np.ndarray, file_path: str, compress: bool = True) -> str:
    """
    Save image stack to a TIF file.

    Parameters
    ----------
    image_stack : numpy.ndarray
        Image stack with shape (n_frames, height, width) or similar.
    file_path : str
        Output file path.
    compress : bool, optional
        Whether to apply compression, by default True.

    Returns
    -------
    str
        Path to saved file.

    Raises
    ------
    Exception
        For general saving errors.
    """
    try:
        logger.info(f"Saving image stack with shape {image_stack.shape} to {file_path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # Save image stack
        compression_level = 6 if compress else 0
        # tifffile can handle various shapes, including channels etc.
        tifffile.imwrite(file_path, image_stack, compress=compression_level)

        logger.info(f"Successfully saved image stack to {file_path}")

        return file_path

    except Exception as e:
        logger.error(f"Error saving image stack to {file_path}: {str(e)}", exc_info=True)
        raise

def load_image(file_path: str) -> np.ndarray:
    """
    Load a single 2D image from a TIF file.

    Parameters
    ----------
    file_path : str
        Path to image file.

    Returns
    -------
    np.ndarray
        Loaded 2D image data.
    """
    try:
        logger.info(f"Loading single image from {file_path}")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        image = tifffile.imread(file_path)
        if image.ndim != 2:
            # Handle multi-page TIFFs loaded as 3D by selecting the first page
            if image.ndim >= 3 and image.shape[0] > 0:
                logger.warning(f"Input file {file_path} is a stack; loading only the first frame/slice.")
                image = image[0]
                # If still not 2D (e.g., T,C,Y,X loaded as C,Y,X), take first channel
                if image.ndim == 3 and image.shape[0] > 0:
                     image = image[0]
            else:
                raise ValueError(f"Expected 2D image, got shape {image.shape}")
        if image.ndim != 2: # Final check after potential slicing
            raise ValueError(f"Could not extract 2D image, final shape {image.shape}")

        logger.info(f"Loaded single image with shape {image.shape} and dtype {image.dtype}")
        return image
    except Exception as e:
        logger.error(f"Error loading single image from {file_path}: {e}", exc_info=True)
        # Re-raising to indicate failure clearly
        raise ValueError(f"Could not load image from {file_path}: {e}") from e

# --- Track Data I/O ---

def load_tracks(file_path: str, format: Optional[str] = None) -> pd.DataFrame:
    """
    Load track data from various file formats.

    Standardizes column names to 'track_id', 'frame', 'x', 'y'.
    Includes specific loaders for TrackMate XML and heuristics for Fiji CSV formats.

    Parameters
    ----------
    file_path : str
        Path to track data file.
    format : str, optional
        File format ('csv', 'excel', 'hdf5', 'json', 'trackmate', 'fiji', 'fiji_mosaic').
        If None, format is inferred from extension. By default None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with track data, including columns 'track_id', 'frame', 'x', 'y'.

    Raises
    ------
    FileNotFoundError
        If the track file does not exist.
    ValueError
        If the file format is unsupported or required columns are missing.
    Exception
        For general loading errors.
    """
    inferred_format = format
    try:
        logger.info(f"Loading tracks from {file_path}")
        if not os.path.isfile(file_path): raise FileNotFoundError(f"Track file not found: {file_path}")

        # --- Format Inference ---
        if inferred_format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv': inferred_format = 'csv'
            elif ext in ['.xls', '.xlsx']: inferred_format = 'excel'
            elif ext in ['.h5', '.hdf5']: inferred_format = 'hdf5'
            elif ext == '.json': inferred_format = 'json'
            elif ext == '.xml':
                 try:
                      tree = ET.parse(file_path)
                      root = tree.getroot()
                      if root.tag == 'TrackMate': inferred_format = 'trackmate'
                      else: raise ValueError("XML not recognized as TrackMate format.")
                 except ET.ParseError: raise ValueError("Could not parse XML file.")
                 except Exception as xml_err: raise ValueError(f"Cannot infer format from XML: {xml_err}")
            else:
                 raise ValueError(f"Cannot infer format from extension: {ext}")
        # -----------------------

        # --- Load Data ---
        tracks_df = None
        if inferred_format == 'csv':
            tracks_df = pd.read_csv(file_path)
            # Check for Fiji formats after loading
            if {'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME'}.issubset(tracks_df.columns): inferred_format = 'fiji'
            elif {'Label', 'X (px)', 'Y (px)', 'Slice'}.issubset(tracks_df.columns): inferred_format = 'fiji_mosaic'
        elif inferred_format == 'excel':
            tracks_df = pd.read_excel(file_path)
            if {'TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME'}.issubset(tracks_df.columns): inferred_format = 'fiji'
            elif {'Label', 'X (px)', 'Y (px)', 'Slice'}.issubset(tracks_df.columns): inferred_format = 'fiji_mosaic'
        elif inferred_format == 'hdf5':
            try: tracks_df = pd.read_hdf(file_path, key='tracks')
            except (KeyError, TypeError, ValueError):
                logger.debug("Pandas HDF read failed, trying generic h5py.")
                with h5py.File(file_path, 'r') as f:
                    if 'tracks' in f and isinstance(f['tracks'], h5py.Dataset):
                        data = f['tracks'][:]; columns = f['tracks'].attrs.get('columns', [])
                        if isinstance(columns, bytes): columns = columns.decode('utf-8')
                        if isinstance(columns, str) and columns: columns = columns.split(',')
                        columns = list(columns) if columns else []
                        if not columns:
                            if data.dtype.names: columns = list(data.dtype.names)
                            elif data.ndim == 2: columns = [f'col_{i}' for i in range(data.shape[1])]
                            else: raise ValueError("Cannot determine columns for HDF5 dataset.")
                        tracks_df = pd.DataFrame(data, columns=columns)
                    else: raise ValueError("HDF5 file lacks recognizable 'tracks' dataset.")
        elif inferred_format == 'json':
             with open(file_path, 'r') as f: data = json.load(f)
             if isinstance(data, list) and data and isinstance(data[0], dict) and 'points' in data[0]:
                 all_points = [];
                 for track_info in data:
                      track_id = track_info.get('track_id');
                      for point in track_info.get('points', []): point['track_id'] = track_id; all_points.append(point)
                 tracks_df = pd.DataFrame(all_points)
             elif isinstance(data, list) and data and isinstance(data[0], dict): tracks_df = pd.DataFrame(data)
             elif isinstance(data, dict):
                 all_points = [];
                 for track_id, points in data.items():
                      if isinstance(points, list):
                           for point in points:
                                if isinstance(point, dict): point['track_id'] = track_id; all_points.append(point)
                 if all_points: tracks_df = pd.DataFrame(all_points)
                 else: raise ValueError("Unsupported JSON dict structure.")
             else: raise ValueError("Unsupported JSON structure.")
        elif inferred_format == 'trackmate':
            tracks_df = load_trackmate(file_path) # Use specific function
        elif inferred_format == 'fiji':
            tracks_df = load_fiji_tracks(file_path) # Use specific function
        elif inferred_format == 'fiji_mosaic':
            tracks_df = load_fiji_mosaic_tracks(file_path) # Use specific function
        else:
            raise ValueError(f"Unsupported format: {inferred_format}")
        # ---------------

        if tracks_df is None: # Handle cases where specific loaders might return None
             raise ValueError(f"Loading function for format '{inferred_format}' returned None.")

        # --- Standardize Columns ---
        column_mapping = {
            # Common variations -> standard name
            'trajectory': 'track_id', 'particle': 'track_id', 'id': 'track_id', 'track': 'track_id', 'TRACK_ID': 'track_id', 'Label': 'track_id',
            't': 'frame', 'time': 'frame', 'frame_number': 'frame', 'FRAME': 'frame', 'Slice': 'frame',
            'position_x': 'x', 'x_position': 'x', 'X': 'x', 'POSITION_X': 'x', 'X (px)': 'x',
            'position_y': 'y', 'y_position': 'y', 'Y': 'y', 'POSITION_Y': 'y', 'Y (px)': 'y',
            'position_z': 'z', 'z_position': 'z', 'Z': 'z', 'POSITION_Z': 'z', 'Z (px)': 'z',
        }
        rename_map = {k: v for k, v in column_mapping.items() if k in tracks_df.columns}
        if rename_map: logger.info(f"Renaming columns: {rename_map}"); tracks_df = tracks_df.rename(columns=rename_map)
        # ---------------------------

        # --- Validate Required Columns ---
        required_columns = ['track_id', 'frame', 'x', 'y']
        missing_columns = [col for col in required_columns if col not in tracks_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns after mapping: {missing_columns}. Available: {list(tracks_df.columns)}")
        # -------------------------------

        # --- Type Conversion & Cleanup ---
        for col in ['frame', 'x', 'y', 'z']:
            if col in tracks_df.columns:
                if tracks_df[col].dtype == object: logger.warning(f"Column '{col}' has object dtype, attempting numeric conversion.")
                tracks_df[col] = pd.to_numeric(tracks_df[col], errors='coerce')
        initial_len = len(tracks_df)
        tracks_df = tracks_df.dropna(subset=['frame', 'x', 'y']) # Drop rows if essential coords are NaN
        if len(tracks_df) < initial_len: logger.warning(f"Dropped {initial_len - len(tracks_df)} rows with NaN values.")
        # Convert frame to integer
        if 'frame' in tracks_df.columns:
             tracks_df['frame'] = tracks_df['frame'].astype(int)
        # Convert track_id to integer if possible, otherwise keep as is
        if 'track_id' in tracks_df.columns:
            try:
                 tracks_df['track_id'] = pd.to_numeric(tracks_df['track_id']).astype(int)
            except (ValueError, TypeError):
                 logger.info("track_id column contains non-numeric values, keeping original type.")
        # -------------------------------

        if not tracks_df.empty: logger.info(f"Loaded {len(tracks_df)} points, {tracks_df['track_id'].nunique()} tracks.")
        else: logger.warning("Loaded track data resulted in an empty DataFrame.")
        return tracks_df

    except Exception as e:
        logger.error(f"Error loading tracks from {file_path} (format: {inferred_format}): {str(e)}", exc_info=True)
        raise

def save_tracks(tracks_df: pd.DataFrame, file_path: str, format: Optional[str] = None) -> str:
    """ Save track data DataFrame to a specified file format. """
    # (Implementation remains the same as provided previously)
    try:
        if not isinstance(tracks_df, pd.DataFrame): raise ValueError("'tracks_df' must be a DataFrame.")
        required_columns = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_columns): raise ValueError(f"DataFrame missing required columns: {required_columns}")
        logger.info(f"Saving {tracks_df['track_id'].nunique()} tracks ({len(tracks_df)} points) to {file_path}")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        save_format = format
        if save_format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv': save_format = 'csv'
            elif ext in ['.xls', '.xlsx']: save_format = 'excel'
            elif ext in ['.h5', '.hdf5']: save_format = 'hdf5'
            elif ext == '.json': save_format = 'json'
            else: logger.warning(f"Unknown extension {ext}, saving as CSV."); save_format = 'csv'; file_path = os.path.splitext(file_path)[0] + '.csv'
        if save_format == 'csv': tracks_df.to_csv(file_path, index=False)
        elif save_format == 'excel':
            try: tracks_df.to_excel(file_path, index=False)
            except ImportError: logger.error("Saving to Excel requires 'openpyxl'."); raise
        elif save_format == 'hdf5':
            try: tracks_df.to_hdf(file_path, key='tracks', mode='w', format='fixed')
            except ImportError: logger.error("Saving to HDF5 requires 'tables'."); raise
            except Exception as hdf_err: logger.error(f"Error saving HDF5: {hdf_err}. Ensure 'tables' installed."); raise
        elif save_format == 'json': tracks_df.to_json(file_path, orient='records', indent=2)
        else: raise ValueError(f"Unsupported format for saving: {save_format}")
        logger.info(f"Successfully saved tracks to {file_path}")
        return file_path
    except Exception as e: logger.error(f"Error saving tracks to {file_path}: {str(e)}", exc_info=True); raise

# --- Specific Loaders ---
def load_trackmate(file_path: str) -> pd.DataFrame:
    """
    Load tracks from a TrackMate XML file.

    Parses the XML structure to extract spots and reconstruct tracks based on edges.

    Parameters
    ----------
    file_path : str
        Path to TrackMate XML file.

    Returns
    -------
    pd.DataFrame
        DataFrame with track data including 'track_id', 'frame', 'x', 'y',
        'z', 'spot_id', and potentially quality/intensity features.
    """
    logger.info(f"Parsing TrackMate XML file: {file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # --- Extract Spot Data ---
        spots = []
        all_spots_element = root.find('.//Model/AllSpots') # More robust XPath-like search
        if all_spots_element is None:
             logger.error("Could not find 'Model/AllSpots' element in TrackMate XML.")
             return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y']) # Return empty with standard cols

        spot_id_map = {} # Map spot ID -> spot dict
        for frame_element in all_spots_element:
             try: frame_num = int(float(frame_element.get('FRAME')))
             except (ValueError, TypeError): logger.warning(f"Skipping frame, invalid FRAME: {frame_element.attrib}"); continue

             for spot_element in frame_element:
                  spot_id = spot_element.get('ID')
                  if not spot_id: continue

                  try:
                      spot_info = {
                           'track_id': -1, # Default track ID
                           'frame': frame_num,
                           'x': float(spot_element.get('POSITION_X')),
                           'y': float(spot_element.get('POSITION_Y')),
                           'z': float(spot_element.get('POSITION_Z', 0.0)),
                           'quality': float(spot_element.get('QUALITY', -1.0)),
                           'radius': float(spot_element.get('RADIUS', 1.0)),
                           'intensity_mean': float(spot_element.get('MEAN_INTENSITY', np.nan)),
                           'spot_id': spot_id,
                      }
                      spot_id_map[spot_id] = spot_info
                  except (ValueError, TypeError) as spot_err:
                       logger.warning(f"Skipping spot {spot_id} due to parsing error: {spot_err}. Attributes: {spot_element.attrib}")

        if not spot_id_map:
            logger.warning("No valid spots found in TrackMate XML.")
            return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y'])

        logger.info(f"Extracted {len(spot_id_map)} spots from XML.")

        # --- Reconstruct Tracks from Edges ---
        all_tracks_element = root.find('.//Model/AllTracks')
        if all_tracks_element is not None:
            trackmate_id_to_spt_id = {} # Map TrackMate's TRACK_ID to our sequential ID
            next_spt_id = 0

            for track_element in all_tracks_element:
                tm_track_id_str = track_element.get('TRACK_ID')
                if tm_track_id_str is None: continue # Skip tracks without ID

                # Assign a consistent spt_analyzer track_id
                if tm_track_id_str not in trackmate_id_to_spt_id:
                     trackmate_id_to_spt_id[tm_track_id_str] = next_spt_id
                     spt_track_id = next_spt_id
                     next_spt_id += 1
                else:
                     spt_track_id = trackmate_id_to_spt_id[tm_track_id_str]

                # Assign the spt_track_id to all spots connected by edges within this track
                edges = track_element.findall('Edge')
                spot_ids_in_track = set()
                # Add nodes first (spots might exist without edges in a track)
                # TrackMate XML structure might vary, adjust findall path if needed
                nodes = track_element.findall('.//Spot') # Find spots within this track element
                if not nodes: # Fallback if Spot elements are not nested directly
                     # Try finding spots based on edges if nodes aren't listed explicitly
                     for edge in edges:
                          source_id = edge.get('SPOT_SOURCE_ID')
                          target_id = edge.get('SPOT_TARGET_ID')
                          if source_id: spot_ids_in_track.add(source_id)
                          if target_id: spot_ids_in_track.add(target_id)
                else:
                     for node in nodes:
                          spot_id = node.get('ID') # Assuming Spot elements have ID attribute here
                          if spot_id: spot_ids_in_track.add(spot_id)

                # Add spots connected by edges (ensure uniqueness)
                for edge in edges:
                    source_id = edge.get('SPOT_SOURCE_ID')
                    target_id = edge.get('SPOT_TARGET_ID')
                    if source_id: spot_ids_in_track.add(source_id)
                    if target_id: spot_ids_in_track.add(target_id)

                # Assign track_id to the collected spots
                for spot_id in spot_ids_in_track:
                    if spot_id in spot_id_map:
                         spot_id_map[spot_id]['track_id'] = spt_track_id
                    else:
                         logger.warning(f"Spot ID '{spot_id}' found in track edges/nodes but not in AllSpots.")
        else:
            logger.warning("Could not find 'Model/AllTracks' element. Track IDs will not be assigned from XML.")

        # Create DataFrame from all extracted spots
        all_spots_list = list(spot_id_map.values())
        tracks_df = pd.DataFrame(all_spots_list)

        # Filter out spots that were not assigned to any track
        tracks_df = tracks_df[tracks_df['track_id'] != -1].copy()

        if tracks_df.empty:
             logger.warning("No tracks reconstructed from TrackMate XML edges/tracks.")
        else:
             logger.info(f"Reconstructed {tracks_df['track_id'].nunique()} tracks from TrackMate XML.")

        # Ensure standard columns exist
        for col in ['track_id', 'frame', 'x', 'y']:
             if col not in tracks_df.columns:
                  tracks_df[col] = pd.Series(dtype=float if col in ['x', 'y'] else int)

        return tracks_df

    except ET.ParseError as e:
        logger.error(f"Error parsing TrackMate XML file {file_path}: {e}", exc_info=True)
        raise ValueError(f"Invalid XML file: {file_path}") from e
    except Exception as e:
        logger.error(f"Error loading TrackMate file {file_path}: {e}", exc_info=True)
        raise

def load_fiji_tracks(file_path: str) -> pd.DataFrame:
    """ Load tracks from a standard Fiji TrackMate CSV/Excel file using load_tracks. """
    logger.info("Attempting to load Fiji TrackMate CSV/Excel using generic load_tracks.")
    # The format hint helps load_tracks prioritize Fiji column names
    return load_tracks(file_path, format='fiji')

def load_fiji_mosaic_tracks(file_path: str) -> pd.DataFrame:
    """ Load tracks from a Fiji MOSAIC Suite CSV/Excel file using load_tracks. """
    logger.info("Attempting to load Fiji MOSAIC CSV/Excel using generic load_tracks.")
    # The format hint helps load_tracks prioritize MOSAIC column names
    return load_tracks(file_path, format='fiji_mosaic')

# --- Config I/O ---

def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    (Implementation remains the same)
    """
    try:
        logger.info(f"Loading configuration from {file_path}")
        if not os.path.isfile(file_path): raise FileNotFoundError(f"Config file not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f: config = yaml.safe_load(f)
        elif ext == '.json':
            with open(file_path, 'r') as f: config = json.load(f)
        else: raise ValueError(f"Unsupported config format: {ext}")
        if not isinstance(config, dict): raise TypeError(f"Config file {file_path} did not load as dict.")
        logger.info(f"Loaded config with {len(config)} top-level entries")
        return config
    except Exception as e: logger.error(f"Error loading config from {file_path}: {e}", exc_info=True); raise

def save_config(config: Dict[str, Any], file_path: str) -> str:
    """
    Save configuration dictionary to YAML or JSON file.
    (Implementation remains the same)
    """
    try:
        if not isinstance(config, dict): raise ValueError("Config must be a dictionary.")
        logger.info(f"Saving configuration to {file_path}")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif ext == '.json':
            with open(file_path, 'w') as f: json.dump(config, f, indent=2)
        else: raise ValueError(f"Unsupported config format for saving: {ext}")
        logger.info(f"Saved config with {len(config)} top-level entries")
        return file_path
    except Exception as e: logger.error(f"Error saving config to {file_path}: {e}", exc_info=True); raise

# --- Analysis Results I/O ---

# Custom JSON encoder/decoder to handle numpy arrays and NaN/Inf
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj): return "NaN"
            if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating):
                 temp_list = obj.tolist()
                 return ["NaN" if isinstance(item, float) and np.isnan(item) else "Infinity" if isinstance(item, float) and np.isinf(item) and item > 0 else "-Infinity" if isinstance(item, float) and np.isinf(item) and item < 0 else item for item in temp_list]
            else: return obj.tolist()
        if isinstance(obj, (np.bool_)): return bool(obj)
        return super(NpEncoder, self).default(obj)

def _np_json_hook(dct):
    """Helper for JSON decoder to convert 'NaN', 'Infinity' back."""
    for key, value in dct.items():
        if isinstance(value, str):
            if value == "NaN": dct[key] = np.nan
            elif value == "Infinity": dct[key] = np.inf
            elif value == "-Infinity": dct[key] = -np.inf
        elif isinstance(value, list): dct[key] = [_np_json_list_hook_item(item) for item in value]
        elif isinstance(value, dict): dct[key] = _np_json_hook(value)
    return dct

def _np_json_list_hook_item(item):
     """Helper for _np_json_hook to process items in lists."""
     if isinstance(item, str):
          if item == "NaN": return np.nan
          if item == "Infinity": return np.inf
          if item == "-Infinity": return -np.inf
     elif isinstance(item, list): return [_np_json_list_hook_item(sub_item) for sub_item in item]
     elif isinstance(item, dict): return _np_json_hook(item)
     return item

def load_analysis_results(file_path: str) -> Dict[str, Any]:
    """
    Load analysis results from HDF5 or JSON file.
    (Implementation remains the same)
    """
    try:
        logger.info(f"Loading analysis results from {file_path}")
        if not os.path.isfile(file_path): raise FileNotFoundError(f"Results file not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        results = {}
        if ext in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'r') as f:
                def _load_hdf_item(name, obj):
                    current_dict = results; path_parts = name.split('/'); key = path_parts[-1]; group_path = "/".join(path_parts[:-1])
                    if group_path:
                         for part in group_path.split('/'): current_dict = current_dict.setdefault(part, {})
                    if isinstance(obj, h5py.Dataset):
                        if obj.attrs.get('type') == 'DataFrame':
                            data = obj[:]; columns = obj.attrs.get('columns', [])
                            if isinstance(columns, np.ndarray): columns = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in columns]
                            elif isinstance(columns, bytes): columns = columns.decode('utf-8').split(',')
                            elif isinstance(columns, str) and columns: columns = columns.split(',')
                            columns = list(columns) if columns else None
                            if columns is None: logger.warning(f"Missing 'columns' for DataFrame '{name}'")
                            try: current_dict[key] = pd.DataFrame(data, columns=columns)
                            except Exception as pd_err: logger.warning(f"Could not construct DataFrame '{name}': {pd_err}. Loading as array."); current_dict[key] = data
                        else: current_dict[key] = obj[:]
                    elif isinstance(obj, h5py.Group):
                        current_dict.setdefault(key, {})
                        for attr_key, attr_val in obj.attrs.items():
                             if isinstance(attr_val, bytes): attr_val = attr_val.decode('utf-8')
                             if isinstance(attr_val, str) and attr_val == "None": attr_val = None
                             if isinstance(current_dict[key], dict): current_dict[key][attr_key] = attr_val
                             else: logger.warning(f"Cannot add attr '{attr_key}' to non-dict item '{key}'")
                f.visititems(_load_hdf_item)
                for attr_key, attr_val in f.attrs.items():
                     if isinstance(attr_val, bytes): attr_val = attr_val.decode('utf-8')
                     if isinstance(attr_val, str) and attr_val == "None": attr_val = None
                     results[attr_key] = attr_val
        elif ext == '.json':
            with open(file_path, 'r') as f: raw_results = json.load(f, object_hook=_np_json_hook)
            results = {}
            for key, value in raw_results.items():
                if isinstance(value, dict) and all(k in value for k in ['columns', 'data']):
                    try:
                        if isinstance(value['data'], list): results[key] = pd.DataFrame(value['data'], columns=value['columns'])
                        else: logger.warning(f"Data for DataFrame '{key}' not list. Keeping dict."); results[key] = value
                    except Exception as df_err: logger.warning(f"Could not convert dict '{key}' to DataFrame: {df_err}. Keeping dict."); results[key] = value
                else: results[key] = value
        else: raise ValueError(f"Unsupported results format: {ext}")
        logger.info(f"Loaded analysis results with {len(results)} top-level entries")
        return results
    except Exception as e: logger.error(f"Error loading analysis results from {file_path}: {e}", exc_info=True); raise

def save_analysis_results(results: Dict[str, Any], file_path: str) -> str:
    """
    Save analysis results dictionary to HDF5 or JSON file.
    (Implementation remains the same)
    """
    try:
        if not isinstance(results, dict): raise ValueError("Results must be a dictionary.")
        logger.info(f"Saving analysis results with {len(results)} entries to {file_path}")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.h5', '.hdf5']:
            with h5py.File(file_path, 'w') as f:
                def _save_hdf_item(group, key, item):
                    key_str = str(key)
                    if isinstance(item, pd.DataFrame):
                        try:
                            df_to_save = item.copy()
                            for col in df_to_save.select_dtypes(include='object').columns: df_to_save[col] = df_to_save[col].astype(str)
                            ds = group.create_dataset(key_str, data=df_to_save.values)
                            ds.attrs['type'] = 'DataFrame'; ds.attrs['columns'] = [str(c).encode('utf-8') for c in df_to_save.columns]
                        except Exception as df_save_err: logger.warning(f"Could not save DataFrame '{key_str}': {df_save_err}. Skipping.")
                    elif isinstance(item, np.ndarray):
                        try:
                            if item.dtype == object: item_str = item.astype(str); ds = group.create_dataset(key_str, data=item_str.astype('S')); ds.attrs['original_dtype'] = 'object'
                            else: group.create_dataset(key_str, data=item)
                        except Exception as arr_save_err: logger.warning(f"Could not save NumPy array '{key_str}': {arr_save_err}. Skipping.")
                    elif isinstance(item, dict):
                        new_group = group.create_group(key_str);
                        for sub_key, sub_item in item.items(): _save_hdf_item(new_group, sub_key, sub_item)
                    elif item is None: group.attrs[key_str] = "None"
                    elif isinstance(item, (str, int, float, bool, np.number, np.bool_)):
                         try:
                              if isinstance(item, str) and len(item) > 64: group.create_dataset(key_str, data=item.encode('utf-8'))
                              else: group.attrs[key_str] = item
                         except TypeError as attr_err:
                                 logger.warning(f"Could not save simple item '{key_str}' type {type(item)} as attr: {attr_err}. Saving as string dataset.")
                                 try: group.create_dataset(key_str, data=str(item).encode('utf-8'))
                                 except Exception as ds_err: logger.error(f"Failed to save '{key_str}' as string dataset: {ds_err}")
                    elif isinstance(item, (list, tuple)):
                         try:
                              arr = np.array(item)
                              if arr.dtype == object: arr_str = arr.astype(str); ds = group.create_dataset(key_str, data=arr_str.astype('S')); ds.attrs['original_type'] = str(type(item)); ds.attrs['original_dtype'] = 'object'
                              else: group.create_dataset(key_str, data=arr); group[key_str].attrs['original_type'] = str(type(item))
                         except Exception as list_conv_err:
                              logger.warning(f"Could not convert list/tuple '{key_str}' to NumPy array: {list_conv_err}. Storing as string attribute.")
                              try: group.attrs[key_str] = str(item)
                              except TypeError: logger.error(f"Could not save list/tuple '{key_str}' as string attribute.")
                    else:
                        logger.warning(f"Unsupported type for HDF5: '{type(item)}' for key '{key_str}'. Storing as string attribute.")
                        try: group.attrs[key_str] = str(item)
                        except TypeError: logger.error(f"Could not save item '{key_str}' as string attribute.")
                for main_key, main_value in results.items(): _save_hdf_item(f, main_key, main_value)
        elif ext == '.json':
             json_serializable_results = {}
             def _prepare_dict_for_json(d: Dict) -> Dict:
                 new_dict = {}
                 for k, v in d.items():
                     if isinstance(v, pd.DataFrame): new_dict[k] = {'type': 'DataFrame', 'columns': v.columns.tolist(), 'data': v.values.tolist()}
                     elif isinstance(v, dict): new_dict[k] = _prepare_dict_for_json(v)
                     else: new_dict[k] = v
                 return new_dict
             for key, value in results.items():
                 if isinstance(value, pd.DataFrame): json_serializable_results[key] = {'type': 'DataFrame', 'columns': value.columns.tolist(), 'data': value.values.tolist()}
                 elif isinstance(value, dict): json_serializable_results[key] = _prepare_dict_for_json(value)
                 else: json_serializable_results[key] = value
             with open(file_path, 'w') as f: json.dump(json_serializable_results, f, indent=2, cls=NpEncoder)
        else: raise ValueError(f"Unsupported results format for saving: {ext}")
        logger.info(f"Successfully saved analysis results to {file_path}")
        return file_path
    except Exception as e: logger.error(f"Error saving analysis results to {file_path}: {e}", exc_info=True); raise
