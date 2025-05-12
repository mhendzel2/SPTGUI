# microcompartment.py
"""
Unified microenvironment analysis module
# ... (rest of docstring) ...
"""

# ---------------------------------------------------------------------
# Imports (deduplicated union of all three source files)
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import json
import uuid
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import cv2 # Keep for potential future use in visualization/loading
from scipy import ndimage
from skimage import filters, morphology, segmentation, exposure, util # Added exposure, util
from skimage.measure import regionprops, label
import skimage.segmentation as sk_seg

# Import necessary I/O utility if defined elsewhere, or implement basic loading here
try:
     # Assuming utils/io.py exists for loading multi-channel images
     from ..utils import io as spt_io
     # Import visualization utils if they exist
     from ..visualization import utils as viz_utils
     from ..visualization import tracks as track_viz
except ImportError:
     logger.warning("Could not import spt_analyzer.utils.io or .visualization. Some I/O/Viz features may be limited.")
     spt_io = None
     viz_utils = None
     track_viz = None


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Global configuration (from microcompartment.py, left unchanged)
# ---------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Visualisation
    "default_color": (0.5, 0.5, 0.5, 1.0), # Changed to RGBA
    "visualization_alpha": 0.5,
    # Morphological operations
    "opening_iterations": 1,
    "closing_iterations": 1,
    # Percentile thresholds to pre-compute
    "percentile_thresholds": [
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.85, 0.9,
        0.95, 0.99,
    ],
}

# ---------------------------------------------------------------------
# Core compartment-definition machinery (shared by both analysers)
# ---------------------------------------------------------------------
class CompartmentDefinition:
    """Identical to the original; governs intensity-rule-based masking."""
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        # Default color now RGBA
        color: Optional[Tuple[float, float, float, float]] = None,
    ):
        if not name:
            raise ValueError("Compartment name cannot be empty")

        self.name: str = name
        self.description: str = description or ""
        self.rules: List[Dict[str, Any]] = rules or []
        default_color = DEFAULT_CONFIG.get("default_color", (0.5, 0.5, 0.5, 1.0))
        # Ensure color has 4 components (RGBA)
        provided_color = color if color is not None else default_color
        if len(provided_color) == 3:
             self.color: Tuple[float, float, float, float] = (*provided_color, 1.0) # Add alpha if missing
        elif len(provided_color) == 4:
             self.color: Tuple[float, float, float, float] = provided_color
        else:
             logger.warning(f"Invalid color tuple length for {name}. Using default.")
             self.color = default_color
        self.id: str = str(uuid.uuid4())

    def add_rule(
        self,
        channel_name: str,
        min_percentile: Optional[float] = None,
        max_percentile: Optional[float] = None,
        min_absolute: Optional[float] = None,
        max_absolute: Optional[float] = None,
    ):
        if not channel_name:
            raise ValueError("Channel name cannot be empty")
        if min_percentile is not None and not (0 <= min_percentile <= 1):
            raise ValueError("min_percentile must be between 0 and 1")
        if max_percentile is not None and not (0 <= max_percentile <= 1):
            raise ValueError("max_percentile must be between 0 and 1")
        if (
            min_percentile is not None
            and max_percentile is not None
            and min_percentile > max_percentile
        ):
            raise ValueError("min_percentile cannot be greater than max_percentile")
        if (
            min_absolute is not None
            and max_absolute is not None
            and min_absolute > max_absolute
        ):
             raise ValueError("min_absolute cannot be greater than max_absolute")

        rule = {
            "channel": channel_name,
            "min_percentile": min_percentile,
            "max_percentile": max_percentile,
            "min_absolute": min_absolute,
            "max_absolute": max_absolute,
        }
        self.rules.append(rule)
        return rule # Return rule dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rules": self.rules.copy(),
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompartmentDefinition:
        if not data or not isinstance(data, dict):
            raise ValueError("Invalid data format for compartment definition")
        if "name" not in data:
            raise ValueError("Compartment definition must include a name")

        default_color = DEFAULT_CONFIG.get("default_color", (0.5, 0.5, 0.5, 1.0))
        instance = cls(
            name=data["name"],
            description=data.get("description", ""),
            rules=data.get("rules", []),
            color=data.get("color", default_color), # Uses updated default
        )
        instance.id = data.get("id", str(uuid.uuid4()))
        return instance

    def generate_mask(
        self,
        channel_images: Dict[str, np.ndarray],
        thresholds: Dict[str, Dict[str, float]],
    ) -> Optional[np.ndarray]:
        if not self.rules:
            logger.debug(f"No rules defined for compartment '{self.name}'. Returning None.")
            return None
        if not channel_images:
            logger.warning("generate_mask called with no channel images.")
            return None

        # Find the first valid channel image to determine mask shape
        mask_shape = None
        initial_mask_set = False
        mask = None

        for rule in self.rules:
            channel = rule["channel"]
            if channel not in channel_images:
                logger.warning(f"Channel '{channel}' for rule in '{self.name}' not found in provided images.")
                continue
            image = channel_images[channel]
            if image is None or image.size == 0:
                logger.warning(f"Image data for channel '{channel}' is empty or None.")
                continue

            # Initialize mask on first valid channel
            if not initial_mask_set:
                mask_shape = image.shape
                mask = np.ones(mask_shape, dtype=bool)
                initial_mask_set = True

            if image.shape != mask_shape:
                 logger.error(f"Shape mismatch for channel '{channel}' ({image.shape}) vs expected ({mask_shape}). Skipping rule.")
                 continue

            # --- Apply thresholding rules ---
            temp_mask = np.ones_like(image, dtype=bool) # Mask for this rule

            # min percentile
            if rule.get("min_percentile") is not None:
                p = rule["min_percentile"]
                key = f"p{p:.2f}"
                threshold = thresholds.get(channel, {}).get(key)
                if threshold is None:
                    if np.any(image):
                        threshold = np.percentile(image[image > 0], p * 100) # Calculate only on non-zero if needed?
                    else: threshold = 0
                    logger.debug(f"Calculated min threshold {threshold:.3f} for {channel} (p{p*100:.1f})")
                temp_mask &= (image >= threshold)

            # max percentile
            if rule.get("max_percentile") is not None:
                p = rule["max_percentile"]
                key = f"p{p:.2f}"
                threshold = thresholds.get(channel, {}).get(key)
                if threshold is None:
                    if np.any(image):
                        threshold = np.percentile(image[image > 0], p * 100)
                    else: threshold = 0
                    logger.debug(f"Calculated max threshold {threshold:.3f} for {channel} (p{p*100:.1f})")
                temp_mask &= (image <= threshold)

            # min absolute
            if rule.get("min_absolute") is not None:
                temp_mask &= (image >= rule["min_absolute"])

            # max absolute
            if rule.get("max_absolute") is not None:
                temp_mask &= (image <= rule["max_absolute"])

            # Combine rule mask with overall mask using AND logic
            mask &= temp_mask
            # --- End thresholding rules ---

        if mask is None: # If no valid channels/rules were processed
             logger.warning(f"Could not generate mask for '{self.name}', no valid rules applied.")
             return None

        if not np.any(mask):
            logger.warning(f"Generated mask for '{self.name}' is empty (all False). Check rules and thresholds.")

        return mask


class CompartmentConfigManager:
    """Manager class unchanged (merged once)."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.compartment_definitions: Dict[str, CompartmentDefinition] = {}
        self.current_config_name: str = "Default"
        self.configs: Dict[str, Dict[str, CompartmentDefinition]] = {
            "Default": self._create_default_definitions()
        }
        self.compartment_definitions = self.configs["Default"]

        # Avoid modifying global config here
        # if config is not None:
        #     if not isinstance(config, dict):
        #         raise ValueError("config must be a dictionary")
        #     DEFAULT_CONFIG.update(config)

    def _create_default_definitions(self) -> Dict[str, CompartmentDefinition]:
        return {}

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    def switch_config(self, name: str) -> bool:
        if name in self.configs:
            self.current_config_name = name
            self.compartment_definitions = self.configs[name]
            logger.info(f"Switched to configuration: {name}")
            return True
        else:
            logger.error(f"Configuration '{name}' not found.")
            return False

    def save_active_config(self, name: Optional[str] = None) -> None:
        save_name = name if name is not None else self.current_config_name
        self.configs[save_name] = self.compartment_definitions.copy()
        if name and name != self.current_config_name:
             self.switch_config(name)
        logger.info(f"Saved current definitions as configuration: {save_name}")

    def delete_config(self, name: str) -> bool:
        if name == "Default":
            logger.error("Cannot delete the Default configuration.")
            return False
        if name in self.configs:
            del self.configs[name]
            logger.info(f"Deleted configuration: {name}")
            if name == self.current_config_name:
                 self.switch_config("Default")
            return True
        else:
            logger.error(f"Configuration '{name}' not found for deletion.")
            return False

    def create_definition(
        self,
        name: str,
        description: Optional[str] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        color: Optional[Tuple[float, float, float, float]] = None, # Updated color type hint
    ) -> CompartmentDefinition:
        definition = CompartmentDefinition(name, description, rules, color)
        self.compartment_definitions[definition.id] = definition
        logger.info(f"Created compartment definition: {name} (ID: {definition.id}) in config '{self.current_config_name}'")
        return definition

    def get_definition(self, definition_id: str) -> Optional[CompartmentDefinition]:
        return self.compartment_definitions.get(definition_id)

    def get_definition_by_name(self, name: str) -> Optional[CompartmentDefinition]:
         for definition in self.compartment_definitions.values():
              if definition.name == name:
                   return definition
         return None

    def update_definition(self, definition_id: str, **kwargs) -> bool:
        definition = self.get_definition(definition_id)
        if definition:
            for key, value in kwargs.items():
                if hasattr(definition, key):
                    # Special handling for color to ensure RGBA
                    if key == 'color' and isinstance(value, tuple):
                         if len(value) == 3: value = (*value, 1.0)
                         elif len(value) != 4:
                              logger.warning(f"Invalid color tuple length for update: {value}. Skipping color update.")
                              continue
                    setattr(definition, key, value)
                else:
                     logger.warning(f"Cannot update unknown attribute '{key}' for definition {definition_id}")
            logger.info(f"Updated compartment definition: {definition.name} (ID: {definition_id})")
            return True
        logger.error(f"Compartment definition not found for update: {definition_id}")
        return False

    def delete_definition(self, definition_id: str) -> bool:
        if definition_id in self.compartment_definitions:
            del self.compartment_definitions[definition_id]
            logger.info(f"Deleted compartment definition: {definition_id}")
            return True
        logger.error(f"Compartment definition not found for deletion: {definition_id}")
        return False

    def get_all_definitions(self) -> List[CompartmentDefinition]:
        return list(self.compartment_definitions.values())

    def save_configs_to_file(self, filepath: str) -> None:
        """Saves all configurations to a file."""
        data_to_save = {
             "configs": {
                 name: {def_id: definition.to_dict() for def_id, definition in definitions.items()}
                 for name, definitions in self.configs.items()
             },
             "current_config": self.current_config_name
        }
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data_to_save, f, indent=2)
            logger.info(f"Saved {len(self.configs)} configurations to {filepath}")
        except IOError as e:
            logger.error(f"Error saving configurations to {filepath}: {e}")
            raise

    def load_configs_from_file(self, filepath: str) -> None:
        """Loads all configurations from a file."""
        if not os.path.exists(filepath):
             logger.error(f"Configuration file not found: {filepath}")
             self.configs = {"Default": self._create_default_definitions()}
             self.current_config_name = "Default"
             self.compartment_definitions = self.configs["Default"]
             return

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            loaded_configs = {}
            if "configs" in data and isinstance(data["configs"], dict):
                for name, definitions_data in data["configs"].items():
                    definitions = {}
                    for def_id, definition_data in definitions_data.items():
                        try:
                             definitions[def_id] = CompartmentDefinition.from_dict(definition_data)
                        except ValueError as e:
                             logger.warning(f"Skipping invalid definition data in config '{name}', ID '{def_id}': {e}")
                    loaded_configs[name] = definitions
            else:
                 logger.warning("Configuration file is missing 'configs' structure. Loading default.")
                 loaded_configs = {"Default": self._create_default_definitions()}

            self.configs = loaded_configs
            if "Default" not in self.configs:
                 self.configs["Default"] = self._create_default_definitions()

            self.current_config_name = data.get("current_config", "Default")
            if self.current_config_name not in self.configs:
                 logger.warning(f"Loaded current_config '{self.current_config_name}' not found. Switching to Default.")
                 self.current_config_name = "Default"

            self.compartment_definitions = self.configs[self.current_config_name]
            logger.info(f"Loaded {len(self.configs)} configurations from {filepath}. Current config: '{self.current_config_name}'")

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading configurations from {filepath}: {e}")
            self.configs = {"Default": self._create_default_definitions()}
            self.current_config_name = "Default"
            self.compartment_definitions = self.configs["Default"]
            raise


# ---------------------------------------------------------------------
# 1. TrackingCompartmentAnalyzer  (from microcompartment.py)
# ---------------------------------------------------------------------
class TrackingCompartmentAnalyzer:
    """
    Single-particle-tracking-centric analysis of sub-cellular compartments.
    Formerly CompartmentAnalyzer in microcompartment.py.
    """
    # ---- Implementation Added ----
    def __init__(self, config_manager: CompartmentConfigManager | None = None, config=None):
        self.config_manager = config_manager or CompartmentConfigManager(config)
        self.channel_images: Dict[str, np.ndarray] = {}
        self.channel_thresholds: Dict[str, Dict[str, float]] = {}
        self.compartment_masks: Dict[str, np.ndarray] = {}
        self.compartment_properties: Dict[str, Any] = {}
        self.tracks_df: Optional[pd.DataFrame] = None
        self.track_compartments: Optional[pd.DataFrame] = None # Stores assignment results
        # Use global config directly? Or instance config? Sticking with instance for now.
        self.config = DEFAULT_CONFIG.copy()
        if config is not None:
            if not isinstance(config, dict):
                raise ValueError("config must be a dictionary")
            self.config.update(config)

    def load_channel(self, channel_name: str, image: np.ndarray):
        """Loads a single channel image and calculates thresholds."""
        if image is None or image.ndim != 2:
             logger.error(f"Invalid image provided for channel '{channel_name}'. Must be 2D numpy array.")
             raise ValueError("Invalid image data for channel.")
        self.channel_images[channel_name] = image.copy() # Store a copy
        self.channel_thresholds[channel_name] = self._calculate_thresholds(image)
        logger.info(f"Loaded channel: '{channel_name}' with shape {image.shape}")

    def _calculate_thresholds(self, image: np.ndarray) -> Dict[str, float]:
         """Helper to calculate percentile thresholds."""
         thresholds = {}
         percentiles_to_calc = self.config.get("percentile_thresholds", []) # Use instance config
         if np.any(image): # Check if image is not all zero
             # Ensure calculation is robust against empty slices if mask is applied later
             image_positive = image[image > 0] if np.any(image > 0) else image # Use only positive values if they exist
             if image_positive.size == 0: # Handle case where image has values but none > 0
                  image_positive = image

             for p in percentiles_to_calc:
                   key = f"p{p:.2f}" # Use consistent formatting
                   try:
                        thresholds[key] = np.percentile(image_positive, p * 100)
                   except IndexError: # Can happen if image_positive becomes empty unexpectedly
                        thresholds[key] = 0.0
                        logger.warning(f"Could not calculate percentile {p*100} for an image slice, defaulting threshold to 0.")
         else: # Image is all zero or empty
              for p in percentiles_to_calc:
                   key = f"p{p:.2f}"
                   thresholds[key] = 0.0
         return thresholds

    def load_channels_from_file(self, filepath: str, channel_names: Optional[List[str]] = None):
        """Loads multi-channel images, e.g., from a multi-page TIFF."""
        if spt_io is None:
             logger.error("Cannot load channels from file: spt_analyzer.utils.io not available.")
             raise ImportError("spt_analyzer.utils.io is required for file loading.")
        try:
             image_stack = spt_io.load_image_stack(filepath)
             if image_stack.ndim != 3: # Expecting (channels/frames, H, W)
                  logger.error(f"Expected 3D stack from {filepath}, got shape {image_stack.shape}")
                  raise ValueError("Loaded file is not a 3D image stack.")

             num_channels = image_stack.shape[0]
             if channel_names is None:
                  channel_names = [f"Channel_{i}" for i in range(num_channels)]
             elif len(channel_names) != num_channels:
                  logger.warning(f"Number of provided channel names ({len(channel_names)}) does not match number of channels in stack ({num_channels}). Using default names.")
                  channel_names = [f"Channel_{i}" for i in range(num_channels)]

             self.channel_images = {} # Reset channels
             self.channel_thresholds = {}
             for i, name in enumerate(channel_names):
                  self.load_channel(name, image_stack[i])
             logger.info(f"Loaded {num_channels} channels from {filepath}")

        except FileNotFoundError:
             logger.error(f"Channel file not found: {filepath}")
             raise
        except Exception as e:
             logger.error(f"Error loading channels from {filepath}: {e}")
             raise

    def generate_compartment_masks(self, apply_morphology: bool = True):
        """Generates masks for all defined compartments using loaded channel images."""
        self.compartment_masks = {}
        definitions = self.config_manager.get_all_definitions()
        if not definitions:
            logger.warning("No compartment definitions found to generate masks.")
            return

        if not self.channel_images:
             logger.error("Cannot generate masks without loading channel images first.")
             raise RuntimeError("Channel images must be loaded before generating masks.")

        opening_iter = self.config.get("opening_iterations", 1)
        closing_iter = self.config.get("closing_iterations", 1)

        for definition in definitions:
            logger.debug(f"Generating mask for compartment: {definition.name}")
            mask = definition.generate_mask(self.channel_images, self.channel_thresholds)
            if mask is not None:
                 if apply_morphology:
                      # Apply morphological opening and closing to clean up the mask
                      if opening_iter > 0:
                           mask = ndimage.binary_opening(mask, iterations=opening_iter)
                      if closing_iter > 0:
                           mask = ndimage.binary_closing(mask, iterations=closing_iter)
                 self.compartment_masks[definition.name] = mask
            else:
                 logger.warning(f"Mask generation returned None for compartment: {definition.name}")

        logger.info(f"Generated {len(self.compartment_masks)} compartment masks.")
        # Analyze properties after generating masks
        self.analyze_compartments()

    def analyze_compartments(self):
         """Analyzes morphological properties of generated compartment masks."""
         self.compartment_properties = {}
         if not self.compartment_masks:
              logger.warning("No compartment masks generated to analyze.")
              return

         for name, mask in self.compartment_masks.items():
              if not np.any(mask): # Skip empty masks
                   self.compartment_properties[name] = {'area': 0}
                   logger.debug(f"Skipping analysis for empty mask: {name}")
                   continue

              labeled_mask, num_labels = label(mask, return_num=True)
              # Get properties for the corresponding channel image if available
              intensity_img = None
              # Find the channel primarily used for this compartment (heuristic)
              definition = self.config_manager.get_definition_by_name(name)
              if definition and definition.rules:
                   primary_channel = definition.rules[0].get('channel')
                   if primary_channel and primary_channel in self.channel_images:
                        intensity_img = self.channel_images[primary_channel]

              props = regionprops(labeled_mask, intensity_image=intensity_img)
              if props:
                   # Aggregate properties (e.g., total area, weighted centroid)
                   # Or analyze properties of the largest component
                   largest_prop = max(props, key=lambda p: p.area)
                   self.compartment_properties[name] = {
                       'name': name,
                       'num_regions': num_labels,
                       'total_area': np.sum(mask),
                       'largest_region_area': largest_prop.area,
                       'largest_region_centroid': largest_prop.centroid,
                       'mean_intensity': largest_prop.mean_intensity if intensity_img is not None else None,
                       # Add more relevant properties...
                       'eccentricity': largest_prop.eccentricity,
                       'solidity': largest_prop.solidity,
                   }
              else:
                   self.compartment_properties[name] = {'name': name, 'total_area': 0, 'num_regions': 0}
         logger.info(f"Analyzed properties for {len(self.compartment_properties)} compartments.")

    def load_tracks(self, tracks_df: pd.DataFrame):
         """Loads and validates tracking data."""
         if not isinstance(tracks_df, pd.DataFrame):
              raise TypeError("tracks_df must be a pandas DataFrame")
         required_cols = ['track_id', 'frame', 'x', 'y']
         if not all(col in tracks_df.columns for col in required_cols):
              missing = [col for col in required_cols if col not in tracks_df.columns]
              logger.error(f"Tracks DataFrame is missing required columns: {missing}")
              raise ValueError(f"tracks_df missing required columns: {missing}")

         self.tracks_df = tracks_df.copy() # Store a copy
         # Reset previous compartment assignments if new tracks are loaded
         if 'compartment' in self.tracks_df.columns:
              self.tracks_df = self.tracks_df.drop(columns=['compartment'])
         self.track_compartments = None
         logger.info(f"Loaded {len(self.tracks_df)} track points from {self.tracks_df['track_id'].nunique()} tracks.")

    def assign_tracks_to_compartments(self):
        """Assigns each track point to its corresponding compartment."""
        if self.tracks_df is None or self.tracks_df.empty:
            logger.error("Cannot assign tracks: Tracks DataFrame is not loaded or empty.")
            raise RuntimeError("Tracks DataFrame not loaded.")
        if not self.compartment_masks:
            logger.warning("Cannot assign tracks: No compartment masks generated. Tracks will be 'Outside'.")
            # Assign all as Outside if no masks exist
            self.tracks_df['compartment'] = "Outside"
            self.track_compartments = self.tracks_df[['track_id', 'frame', 'compartment']].copy()
            return

        # Create a labeled map for efficient lookup (higher index overrides lower if overlap)
        first_mask = next(iter(self.compartment_masks.values()))
        labeled_map = np.zeros_like(first_mask, dtype=np.int32)
        # Ensure consistent ordering for indexing
        comp_names = sorted(self.compartment_masks.keys()) # Sort names alphabetically
        comp_name_to_index = {name: i + 1 for i, name in enumerate(comp_names)}

        for name in comp_names:
            labeled_map[self.compartment_masks[name]] = comp_name_to_index[name]
        logger.debug("Created labeled map for compartment assignment.")

        # Prepare coordinates for lookup
        coords = self.tracks_df[['y', 'x']].round().astype(int).values
        y_coords, x_coords = coords[:, 0], coords[:, 1]

        # Create default compartment assignment ('Outside')
        assigned_compartments = np.full(len(self.tracks_df), "Outside", dtype=object)

        # Find points within image bounds
        h, w = labeled_map.shape
        valid_indices = (y_coords >= 0) & (y_coords < h) & (x_coords >= 0) & (x_coords < w)

        # Get labels for valid coordinates
        valid_labels = labeled_map[y_coords[valid_indices], x_coords[valid_indices]]

        # Map labels back to names for valid points
        valid_assignments = np.array(["Outside"] * len(valid_labels), dtype=object)
        for i, name in enumerate(comp_names, start=1):
            valid_assignments[valid_labels == i] = name

        # Update the main assignment array
        assigned_compartments[valid_indices] = valid_assignments

        # Add the 'compartment' column to the DataFrame
        self.tracks_df['compartment'] = assigned_compartments
        self.track_compartments = self.tracks_df[['track_id', 'frame', 'compartment']].copy()
        logger.info("Assigned tracks to compartments.")

    def calculate_compartment_statistics(self, stats_to_calc: List[str] | None = None) -> Dict[str, Any]:
        """Calculates statistics related to tracks within compartments."""
        if self.tracks_df is None or 'compartment' not in self.tracks_df.columns:
            logger.error("Cannot calculate stats: Track compartment assignment needed.")
            raise RuntimeError("Assign tracks to compartments before calculating statistics.")

        stats_to_calc = stats_to_calc or ['track_counts', 'point_counts', 'dwell_times'] # Default stats
        results = {}
        logger.info(f"Calculating compartment statistics: {stats_to_calc}")

        # Group by compartment
        grouped = self.tracks_df.groupby('compartment')

        if 'track_counts' in stats_to_calc:
             # Number of unique tracks that *ever* enter each compartment
             track_counts = grouped['track_id'].nunique()
             results['track_counts'] = track_counts.to_dict()

        if 'point_counts' in stats_to_calc:
             # Total number of points found within each compartment
             point_counts = grouped.size()
             results['point_counts'] = point_counts.to_dict()

        if 'dwell_times' in stats_to_calc:
             # Calculate dwell times (requires iterating through tracks)
             dwell_times_dict = {name: [] for name in self.compartment_masks.keys()}
             dwell_times_dict["Outside"] = []

             for track_id, track in self.tracks_df.groupby('track_id'):
                  current_comp = None
                  entry_frame = None
                  for idx, row in track.sort_values('frame').iterrows():
                       if row['compartment'] != current_comp:
                            # Left previous compartment or started track
                            if current_comp is not None and current_comp != "Outside": # Don't record dwell outside
                                 dwell_frames = row['frame'] - entry_frame
                                 dwell_times_dict[current_comp].append(dwell_frames)
                            # Entered new compartment
                            current_comp = row['compartment']
                            entry_frame = row['frame']
                  # Handle end of track
                  if current_comp is not None and current_comp != "Outside":
                       last_frame = track['frame'].max()
                       dwell_frames = last_frame - entry_frame + 1 # +1 to include last frame
                       dwell_times_dict[current_comp].append(dwell_frames)

             # Calculate mean dwell times (in frames)
             mean_dwell_times = {
                  comp: np.mean(times) if times else 0
                  for comp, times in dwell_times_dict.items()
             }
             results['mean_dwell_frames'] = mean_dwell_times
             results['all_dwell_frames'] = dwell_times_dict # Store all for distribution plots

        # Store overall results (could add more stats like density, avg speed per comp, etc.)
        self.compartment_properties['statistics'] = results
        logger.info("Calculated compartment statistics.")
        return results


    def visualize_compartments(self, base_channel: Optional[str] = None, show_boundaries: bool = True, **viz_kwargs):
        """Visualizes the generated compartment masks, optionally overlaid."""
        if not self.compartment_masks:
            logger.warning("No compartment masks to visualize.")
            return None
        if viz_utils is None:
             logger.error("Cannot visualize compartments: spt_analyzer.visualization.utils not available.")
             return None

        # Prepare figure using context manager
        fig = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=viz_kwargs.get('figsize', (8, 8)))

            # Display base image if provided
            base_image = None
            if base_channel and base_channel in self.channel_images:
                base_image = self.channel_images[base_channel]
                # Normalize base image for display
                p_low, p_high = np.percentile(base_image, (1, 99))
                base_image_norm = exposure.rescale_intensity(base_image, in_range=(p_low, p_high), out_range=(0, 1))
                ax.imshow(base_image_norm, cmap='gray')
            elif not base_channel and self.channel_images:
                 # Use first channel as default background
                 first_channel_name = next(iter(self.channel_images))
                 base_image = self.channel_images[first_channel_name]
                 p_low, p_high = np.percentile(base_image, (1, 99))
                 base_image_norm = exposure.rescale_intensity(base_image, in_range=(p_low, p_high), out_range=(0, 1))
                 ax.imshow(base_image_norm, cmap='gray')


            # Overlay compartment masks
            legend_handles = []
            definitions = self.config_manager.get_all_definitions()
            def_map = {d.name: d for d in definitions}

            for name, mask in self.compartment_masks.items():
                definition = def_map.get(name)
                color = definition.color if definition else self.config.get("default_color")
                alpha = self.config.get("visualization_alpha", 0.5)

                # Create colored overlay for the mask
                colored_mask = np.zeros((*mask.shape, 4)) # RGBA
                colored_mask[mask, :] = color # Apply color where mask is True
                colored_mask[..., 3][~mask] = 0 # Make non-mask areas transparent

                ax.imshow(colored_mask, alpha=alpha)

                # Add legend entry
                from matplotlib.patches import Patch
                legend_handles.append(Patch(color=color[:3], alpha=alpha, label=name))

                # Optionally show boundaries
                if show_boundaries:
                     contours = segmentation.find_boundaries(mask, mode='inner')
                     boundary_color = tuple(c * 0.8 for c in color[:3]) # Darker boundary
                     boundary_overlay = np.zeros((*mask.shape, 4))
                     boundary_overlay[contours, :3] = boundary_color
                     boundary_overlay[contours, 3] = 1.0 # Opaque boundary
                     ax.imshow(boundary_overlay)


            if legend_handles:
                ax.legend(handles=legend_handles, loc='best', fontsize='small')

            ax.set_title("Compartment Visualization")
            ax.axis('off')
            plt.tight_layout()
            # fig object is returned for potential further manipulation or saving
            return fig

        except ImportError:
             logger.error("Matplotlib is required for visualization.")
             return None
        except Exception as e:
             logger.error(f"Error during compartment visualization: {e}")
             if fig: plt.close(fig) # Close figure if error occurred during plotting
             return None

    def visualize_tracks_in_compartments(self, base_channel: Optional[str] = None, **viz_kwargs):
        """Visualizes tracks colored by the compartment they reside in."""
        if self.tracks_df is None or 'compartment' not in self.tracks_df.columns:
            logger.error("Cannot visualize tracks: Assign tracks to compartments first.")
            raise RuntimeError("Tracks not assigned to compartments.")
        if track_viz is None:
             logger.error("Cannot visualize tracks: spt_analyzer.visualization.tracks not available.")
             return None

        # Prepare figure using context manager
        fig = None
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=viz_kwargs.get('figsize', (10, 8)))

            # Display base image if provided
            base_image = None
            if base_channel and base_channel in self.channel_images:
                base_image = self.channel_images[base_channel]
                p_low, p_high = np.percentile(base_image, (1, 99))
                base_image_norm = exposure.rescale_intensity(base_image, in_range=(p_low, p_high), out_range=(0, 1))
                ax.imshow(base_image_norm, cmap='gray')
            elif not base_channel and self.channel_images:
                 first_channel_name = next(iter(self.channel_images))
                 base_image = self.channel_images[first_channel_name]
                 p_low, p_high = np.percentile(base_image, (1, 99))
                 base_image_norm = exposure.rescale_intensity(base_image, in_range=(p_low, p_high), out_range=(0, 1))
                 ax.imshow(base_image_norm, cmap='gray')


            # Get compartment colors
            definitions = self.config_manager.get_all_definitions()
            comp_colors = {d.name: d.color for d in definitions}
            comp_colors["Outside"] = (0.7, 0.7, 0.7, 0.5) # Gray for outside

            # Plot tracks, colored by compartment
            for compartment, group in self.tracks_df.groupby('compartment'):
                color = comp_colors.get(compartment, self.config.get("default_color"))
                for track_id, track in group.groupby('track_id'):
                    ax.plot(track['x'], track['y'], '-', color=color[:3], # Use RGB from RGBA
                           alpha=color[3] if len(color)==4 else 0.7, # Use alpha from color or default
                           linewidth=viz_kwargs.get('linewidth', 1))

            # Create legend handles manually
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=c[:3], lw=2, label=name)
                               for name, c in comp_colors.items() if name in self.tracks_df['compartment'].unique()]
            ax.legend(handles=legend_elements, title="Compartments", loc='best')

            ax.set_title("Tracks Colored by Compartment")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.invert_yaxis()
            ax.set_aspect('equal')
            plt.tight_layout()
            return fig

        except ImportError:
             logger.error("Matplotlib is required for visualization.")
             return None
        except Exception as e:
             logger.error(f"Error visualizing tracks in compartments: {e}")
             if fig: plt.close(fig)
             return None

    @contextmanager
    def save_visualization(self, filepath: str, fig: Optional[plt.Figure] = None, **savefig_kwargs):
        """Context manager to save a generated matplotlib figure."""
        # If no figure is passed, create a new one
        if fig is None:
             if plt is None: raise ImportError("Matplotlib required for saving visualization.")
             fig_created = True
             fig, ax = plt.subplots(figsize=savefig_kwargs.pop('figsize', (8, 8)))
             yield ax # Yield the axis for plotting
        else:
             fig_created = False
             # If fig is passed, assume plotting is done outside, just save.
             # Or yield the existing axes? Yielding axis seems more flexible.
             if not fig.axes: # If fig has no axes, add one
                  ax = fig.add_subplot(111)
             else:
                  ax = fig.axes[0] # Assume first axis
             yield ax

        # --- Save the figure ---
        try:
             # Ensure directory exists
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
             dpi = savefig_kwargs.pop('dpi', 150)
             bbox_inches = savefig_kwargs.pop('bbox_inches', 'tight')
             fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
             logger.info(f"Visualization saved to {filepath}")
        except Exception as e:
             logger.error(f"Failed to save visualization to {filepath}: {e}")
             raise # Re-raise error after logging
        finally:
             # Close the figure only if it was created within this context manager
             if fig_created:
                  plt.close(fig)


    def export_results(self, filepath: str, format: str = 'json'):
         """Exports analysis results (properties and track assignments)."""
         results_to_export = {
              "compartment_properties": self.compartment_properties,
              # Store track assignments if available
              "track_compartments": self.track_compartments.to_dict(orient='records') if self.track_compartments is not None else None,
              # Include compartment masks? Might be large. Maybe save separately.
              # "compartment_masks": {name: mask.tolist() for name, mask in self.compartment_masks.items()}, # Example if needed
         }
         # Use the utility function if available
         if spt_io:
              try:
                   # spt_io.save_results might need adaptation to handle numpy arrays etc.
                   # For now, using basic json dump with default=str for robustness
                   spt_io.save_results(results_to_export, filepath) # Assumes save_results handles formats
                   logger.info(f"Exported results using spt_io to {filepath}")
              except Exception as e:
                   logger.warning(f"spt_io.save_results failed ({e}), falling back to basic JSON dump.")
                   format = 'json' # Force JSON on fallback

         if format == 'json': # Fallback or if spt_io failed/not available
             try:
                 def default_serializer(obj):
                     if isinstance(obj, np.ndarray): return obj.tolist()
                     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                     elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                     elif isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
                     elif isinstance(obj, (np.bool_)): return bool(obj)
                     elif isinstance(obj, (np.void)): return None # Treat void as null
                     return str(obj) # Fallback to string

                 os.makedirs(os.path.dirname(filepath), exist_ok=True)
                 with open(filepath, 'w') as f:
                      json.dump(results_to_export, f, indent=2, default=default_serializer)
                 logger.info(f"Exported results as JSON to {filepath}")

             except Exception as e:
                 logger.error(f"Failed to export results to {filepath}: {e}")
                 raise
         elif format != 'json': # If spt_io failed but format wasn't json
              logger.error(f"Cannot export in format '{format}' without spt_io helper.")


# ---------------------------------------------------------------------
# 2. CellposeSegmenter  (from microenvironment.py, intact)
# ---------------------------------------------------------------------
class CellposeSegmenter:
    """Thin wrapper around Cellpose for nucleus/cell segmentation."""

    # ... (Implementation as provided previously) ...
    def __init__(self, model_type: str = "nuclei", gpu: Optional[bool] = None):
        try:
            from cellpose import models
            if gpu is None:
                try:
                     import torch
                     gpu = torch.cuda.is_available()
                     logger.info(f"PyTorch CUDA available: {gpu}")
                except ImportError:
                     logger.warning("PyTorch not found. Cannot automatically detect GPU. Assuming CPU.")
                     gpu = False

            self.model = models.Cellpose(model_type=model_type, gpu=gpu)
            logger.info(f"Initialized Cellpose with model: {model_type}, GPU: {gpu}")
        except ImportError:
            logger.error("Cellpose not installed. Install it with: pip install cellpose")
            raise

    def segment(
        self,
        image: np.ndarray,
        diameter: Optional[float] = None,
        channels: List[int] = [0, 0],
        flow_threshold: float = 0.4,
        min_size: int = 30,
        normalize: bool = True, # Let Cellpose normalize by default
    ) -> np.ndarray:
        if not hasattr(self, 'model') or self.model is None:
             logger.error("Cellpose model not initialized. Cannot segment.")
             raise RuntimeError("Cellpose model not initialized.")
        if image is None or image.size == 0:
             logger.error("Input image is empty or None.")
             raise ValueError("Input image cannot be empty.")
        if image.ndim not in [2, 3]:
             logger.error(f"Input image has unsupported dimensions: {image.ndim}")
             raise ValueError("Input image must be 2D or 3D.")

        img_to_segment = image
        if image.ndim == 3 and image.shape[0] == 1:
            img_to_segment = image[0]

        # Let Cellpose handle normalization unless specified otherwise
        # The internal normalization is generally robust.
        cellpose_normalize_flag = normalize

        try:
            masks, flows, styles = self.model.eval(
                img_to_segment,
                diameter=diameter,
                channels=channels,
                flow_threshold=flow_threshold,
                min_size=min_size,
                normalize=cellpose_normalize_flag,
            )
            logger.info(f"Cellpose segmentation found {masks.max()} potential objects.")
            return masks
        except Exception as e:
             logger.error(f"Error during Cellpose model evaluation: {e}", exc_info=True)
             raise

# ---------------------------------------------------------------------
# 3. NuclearCompartmentAnalyzer  (formerly CompartmentAnalyzer in
#    microenvironment.py, preserved but renamed to avoid collision)
# ---------------------------------------------------------------------
class NuclearCompartmentAnalyzer:
    """
    Analysis of nuclear segmentation and spatial relationships between
    compartments and nuclear boundary.
    """
    # ... (Implementation as provided previously and corrected) ...
    def __init__(self):
        self.compartments: Dict[int, np.ndarray] = {}
        self.compartment_properties: Dict[int, Any] = {}
        self.nucleus_mask: Optional[np.ndarray] = None
        self.nucleus_properties: Dict[str, Any] = {}

    def segment_nucleus(
        self,
        dapi_image: np.ndarray,
        method: str = "cellpose",
        params: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        # ...(previous corrected implementation)...
        if dapi_image is None or dapi_image.size == 0:
             logger.error("DAPI image is empty or None.")
             raise ValueError("DAPI image cannot be empty.")
        if dapi_image.ndim != 2:
             logger.error(f"DAPI image must be 2D, but got shape {dapi_image.shape}")
             raise ValueError("DAPI image must be 2D.")

        params = params or {}
        nucleus_mask = None

        try:
            logger.info(f"Segmenting nucleus using method: {method}")
            if method == 'cellpose':
                try:
                    model_type = params.get('model_type', 'nuclei')
                    diameter = params.get('diameter', 100)
                    flow_threshold = params.get('flow_threshold', 0.4)
                    min_size = params.get('min_size', 100)
                    gpu = params.get('gpu', None)

                    segmenter = CellposeSegmenter(model_type=model_type, gpu=gpu)
                    masks = segmenter.segment(
                        dapi_image, diameter=diameter, flow_threshold=flow_threshold,
                        min_size=min_size, normalize=True
                    )

                    if masks.max() > 0:
                         if masks.max() > 1:
                              logger.info(f"Cellpose found {masks.max()} objects. Selecting the largest.")
                              sizes = np.bincount(masks.flatten())
                              if len(sizes) > 1:
                                  largest_label = np.argmax(sizes[1:]) + 1
                                  nucleus_mask = (masks == largest_label)
                              else:
                                   nucleus_mask = np.zeros_like(masks, dtype=bool)
                         else:
                              nucleus_mask = (masks > 0)
                    else:
                         logger.warning("Cellpose did not find any nuclei.")
                         nucleus_mask = np.zeros_like(masks, dtype=bool)

                except ImportError:
                     logger.error("Cellpose is not installed. Cannot use 'cellpose' method.")
                     raise
                except Exception as e:
                     logger.error(f"Error during Cellpose segmentation: {e}", exc_info=True)
                     raise

            elif method == 'threshold':
                threshold_method = params.get('method', 'otsu')
                min_size = params.get('min_size', 100)
                fill_holes = params.get('fill_holes', True)
                manual_threshold = params.get('threshold_value', None)

                try:
                     from ..segmentation import SegmentationMethods
                     if threshold_method == 'manual' and manual_threshold is None:
                          logger.warning("Manual threshold method selected but no 'threshold_value' provided. Using Otsu.")
                          threshold_method = 'otsu'

                     nucleus_mask = SegmentationMethods.intensity_based(
                         dapi_image, method=threshold_method, min_size=min_size,
                         fill_holes=fill_holes, threshold_value=manual_threshold
                     )
                except ImportError:
                     logger.warning("SegmentationMethods not found. Using basic Otsu thresholding.")
                     if dapi_image.max() > dapi_image.min():
                          threshold = filters.threshold_otsu(dapi_image)
                          nucleus_mask = dapi_image > threshold
                          if min_size > 0: nucleus_mask = morphology.remove_small_objects(nucleus_mask, min_size)
                          if fill_holes: nucleus_mask = ndimage.binary_fill_holes(nucleus_mask)
                     else:
                          nucleus_mask = np.zeros_like(dapi_image, dtype=bool)
            else:
                raise ValueError(f"Unknown nucleus segmentation method: {method}")

            self.nucleus_mask = nucleus_mask
            logger.info(f"Generated nucleus mask with area: {np.sum(nucleus_mask)} pixels")

            if np.any(nucleus_mask):
                labeled_mask = label(nucleus_mask)
                props = regionprops(labeled_mask, intensity_image=dapi_image)
                if props:
                    p = props[0]
                    self.nucleus_properties = {
                        'area': p.area, 'perimeter': p.perimeter, 'centroid': p.centroid,
                        'mean_intensity': p.mean_intensity, 'max_intensity': p.max_intensity,
                        'min_intensity': p.min_intensity, 'major_axis_length': p.major_axis_length,
                        'minor_axis_length': p.minor_axis_length, 'eccentricity': p.eccentricity,
                        'solidity': p.solidity, 'equivalent_diameter': p.equivalent_diameter,
                        'orientation': p.orientation,
                    }
                    logger.info(f"Calculated properties for the segmented nucleus: Area={p.area}")
                else:
                    logger.warning("Region properties could not be calculated for the nucleus mask.")
                    self.nucleus_properties = {}
            else:
                logger.warning("Nucleus mask is empty. No properties calculated.")
                self.nucleus_properties = {}
            return nucleus_mask

        except Exception as e:
            logger.error(f"Error during nucleus segmentation: {e}", exc_info=True)
            self.nucleus_mask = None
            self.nucleus_properties = {}
            raise

    def analyze_compartment_nuclear_relationship(self, compartment_channel_idx: int):
        # ...(previous corrected implementation)...
        if self.nucleus_mask is None:
            logger.error("Nuclear segmentation must be performed first.")
            raise ValueError("Nuclear segmentation must be performed first")
        if compartment_channel_idx not in self.compartments:
             logger.error(f"No compartment mask found for channel index {compartment_channel_idx}")
             raise ValueError(f"No compartment mask found for channel index {compartment_channel_idx}")

        compartment_mask = self.compartments[compartment_channel_idx]
        if compartment_mask is None or compartment_mask.shape != self.nucleus_mask.shape:
             logger.error(f"Compartment mask for channel {compartment_channel_idx} is invalid or shape mismatch.")
             raise ValueError(f"Invalid compartment mask for channel {compartment_channel_idx}")

        try:
            nucleus_mask_bool = self.nucleus_mask.astype(bool)
            compartment_mask_bool = compartment_mask.astype(bool)

            overlap_mask = nucleus_mask_bool & compartment_mask_bool
            overlap_area = np.sum(overlap_mask)
            compartment_area = np.sum(compartment_mask_bool)
            nucleus_area = np.sum(nucleus_mask_bool)
            inside_fraction = (overlap_area / compartment_area) if compartment_area > 0 else 0.0
            occupation_fraction = (overlap_area / nucleus_area) if nucleus_area > 0 else 0.0

            nucleus_boundary = segmentation.find_boundaries(nucleus_mask_bool, mode='inner')
            mean_distance = float('nan')
            min_distance = float('nan')
            boundary_association_length = 0
            boundary_association_fraction = 0.0

            if np.any(nucleus_boundary):
                distance_from_boundary = ndimage.distance_transform_edt(~nucleus_boundary)
                compartment_pixels_indices = np.where(compartment_mask_bool)
                if compartment_pixels_indices[0].size > 0:
                    compartment_distances = distance_from_boundary[compartment_pixels_indices]
                    mean_distance = np.mean(compartment_distances)
                    min_distance = np.min(compartment_distances)

                boundary_association_mask = compartment_mask_bool & nucleus_boundary
                boundary_association_length = np.sum(boundary_association_mask)
                total_boundary_length = np.sum(nucleus_boundary)
                boundary_association_fraction = (boundary_association_length / total_boundary_length) if total_boundary_length > 0 else 0.0
            else:
                 logger.warning("Nucleus boundary could not be determined.")

            relationship = {
                'overlap_area': int(overlap_area), 'inside_fraction': float(inside_fraction),
                'occupation_fraction': float(occupation_fraction),
                'mean_distance_from_boundary': float(mean_distance),
                'min_distance_from_boundary': float(min_distance),
                'boundary_association_length': int(boundary_association_length),
                'boundary_association_fraction': float(boundary_association_fraction)
            }

            if compartment_channel_idx in self.compartment_properties:
                self.compartment_properties[compartment_channel_idx]['nuclear_relationship'] = relationship
            else:
                 self.compartment_properties[compartment_channel_idx] = {'nuclear_relationship': relationship}

            logger.info(f"Calculated nuclear relationship for compartment {compartment_channel_idx}: Overlap Area={overlap_area}, Inside Fraction={inside_fraction:.3f}")
            return relationship

        except Exception as e:
            logger.error(f"Error calculating compartment-nuclear relationship for channel {compartment_channel_idx}: {e}", exc_info=True)
            raise


# ---------------------------------------------------------------------
# Backwards-compatibility aliases (optional; comment out if undesired)
# ---------------------------------------------------------------------
# CompartmentAnalyzer = TrackingCompartmentAnalyzer

# ---------------------------------------------------------------------
# Module public interface
# ---------------------------------------------------------------------
__all__ = [
    "DEFAULT_CONFIG",
    "CompartmentDefinition",
    "CompartmentConfigManager",
    "TrackingCompartmentAnalyzer",
    "CellposeSegmenter",
    "NuclearCompartmentAnalyzer",
]