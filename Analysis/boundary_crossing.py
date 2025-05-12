# Modified: spt_analyzer/analysis/boundary_crossing.py
"""
Boundary crossing analysis module for SPT Analysis.

This module provides tools for analyzing particle behavior at boundaries
between compartments, including crossing dynamics and angular distributions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import ndimage # For distance transform and gradient
from skimage import measure # For boundary finding (alternative)

logger = logging.getLogger(__name__)


class BoundaryCrossingAnalyzer:
    def __init__(self, dt: float = 0.014):
        """
        Initialize the BoundaryCrossingAnalyzer.

        Parameters
        ----------
        dt : float, optional
            Time interval between frames in seconds, by default 0.014.
            Used for potential velocity calculations if needed later.
        """
        self.dt = dt
        # Results storage
        self.crossing_events: List[Dict[str, Any]] = []
        self.boundary_params: Dict[str, Any] = {}
        self.angular_distributions: Dict[str, Any] = {}

    def analyze_boundary_crossings(
        self,
        tracks_df: pd.DataFrame,
        compartment_masks: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Analyze boundary crossing events between compartments.

        Identifies steps where a particle moves from one compartment zone
        (defined by masks) to another, or between a compartment and 'Outside'.

        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame containing 'track_id', 'frame', 'x', 'y' columns (in pixels).
        compartment_masks : dict of {str: np.ndarray}
            Dictionary mapping compartment names to boolean or integer mask arrays.
            Masks should have the same shape.

        Returns
        -------
        List[dict]
            A list of crossing event dictionaries, each containing details about
            the crossing step.
        """
        self.crossing_events = [] # Clear previous results
        if tracks_df is None or not isinstance(tracks_df, pd.DataFrame) or tracks_df.empty:
            logger.warning("Invalid or empty Tracks DataFrame provided for boundary crossing.")
            return []
        if compartment_masks is None or not compartment_masks or not isinstance(compartment_masks, dict):
            logger.warning("Invalid or empty compartment masks provided for boundary crossing.")
            return []
        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
            logger.error(f"Tracks DataFrame missing required columns: {required_cols}")
            return []

        try:
            # --- Build Labeled Map ---
            # Find the shape from the first valid mask
            first_mask = next((m for m in compartment_masks.values() if isinstance(m, np.ndarray)), None)
            if first_mask is None:
                 raise ValueError("No valid numpy array masks found in compartment_masks.")
            img_height, img_width = first_mask.shape

            labeled_map = np.zeros((img_height, img_width), dtype=np.int32)
            compartment_names = list(compartment_masks.keys())
            name_to_label = {name: i + 1 for i, name in enumerate(compartment_names)}
            label_to_name = {i + 1: name for i, name in enumerate(compartment_names)}
            label_to_name[0] = 'Outside' # Label 0 represents outside

            for name, mask in compartment_masks.items():
                if not isinstance(mask, np.ndarray) or mask.shape != (img_height, img_width):
                    logger.warning(f"Skipping invalid or shape-mismatched mask for '{name}'.")
                    continue
                label_idx = name_to_label[name]
                labeled_map[mask > 0] = label_idx # Assign label where mask is True
            # -------------------------

            crossing_events_list: List[Dict[str, Any]] = []

            # --- Iterate Tracks ---
            for track_id, track_df in tracks_df.groupby("track_id"):
                if len(track_df) < 2: continue # Need at least two points for a step

                track_df = track_df.sort_values("frame").reset_index(drop=True)
                positions = track_df[["x", "y"]].values # Pixel coordinates
                frames = track_df["frame"].values

                for i in range(len(positions) - 1):
                    pos1_pix = positions[i]
                    pos2_pix = positions[i+1]
                    # Round to nearest pixel index for lookup in labeled_map
                    x1_idx, y1_idx = int(round(pos1_pix[0])), int(round(pos1_pix[1]))
                    x2_idx, y2_idx = int(round(pos2_pix[0])), int(round(pos2_pix[1]))

                    # Get compartment label at start and end of step
                    label1 = 0
                    if 0 <= y1_idx < img_height and 0 <= x1_idx < img_width:
                        label1 = labeled_map[y1_idx, x1_idx]

                    label2 = 0
                    if 0 <= y2_idx < img_height and 0 <= x2_idx < img_width:
                        label2 = labeled_map[y2_idx, x2_idx]

                    # Check if compartment changed
                    if label1 != label2:
                        comp1 = label_to_name[label1]
                        comp2 = label_to_name[label2]

                        crossing_events_list.append({
                            "track_id": track_id,
                            "frame_from": int(frames[i]),
                            "frame_to": int(frames[i + 1]),
                            "from_compartment": comp1,
                            "to_compartment": comp2,
                            "position_from": pos1_pix.tolist(), # Store original pixel position
                            "position_to": pos2_pix.tolist(),   # Store original pixel position
                            "dx": pos2_pix[0] - pos1_pix[0], # Displacement in pixels
                            "dy": pos2_pix[1] - pos1_pix[1]
                        })
            # --------------------

            self.crossing_events = crossing_events_list
            logger.info(f"Found {len(self.crossing_events)} boundary crossing events.")
            return self.crossing_events

        except Exception as e:
             logger.error(f"Error analyzing boundary crossings: {e}", exc_info=True)
             self.crossing_events = []
             return []

    def get_boundary_pixels(self, compartment_masks: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], np.ndarray]:
         """Identifies pixels at the interface between pairs of compartments."""
         # --- (Implementation remains the same) ---
         boundaries = {}
         names = list(compartment_masks.keys())
         if not names: return {}
         first_mask = next((m for m in compartment_masks.values() if isinstance(m, np.ndarray)), None)
         if first_mask is None: return {}
         combined_mask = np.zeros(first_mask.shape, dtype=bool)
         for mask in compartment_masks.values():
              if isinstance(mask, np.ndarray) and mask.shape == first_mask.shape:
                   combined_mask |= mask.astype(bool)
         outer_boundary = ndimage.binary_dilation(combined_mask) & (~combined_mask)
         boundaries[('Any', 'Outside')] = outer_boundary # Store boundary with outside

         for i in range(len(names)):
              for j in range(i + 1, len(names)):
                   name1, name2 = names[i], names[j]
                   mask1 = compartment_masks.get(name1)
                   mask2 = compartment_masks.get(name2)
                   if isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray) and mask1.shape == mask2.shape:
                        interface1 = ndimage.binary_dilation(mask1.astype(bool)) & mask2.astype(bool)
                        interface2 = ndimage.binary_dilation(mask2.astype(bool)) & mask1.astype(bool)
                        boundary_pixels = interface1 | interface2
                        if np.any(boundary_pixels):
                             boundaries[tuple(sorted((name1, name2)))] = boundary_pixels
         return boundaries

    def analyze_angular_distribution(self, compartment_masks: Dict[str, np.ndarray], pixel_size: float = 1.0) -> Dict[str, Any]:
        """
        Analyze the angular distribution of tracks crossing boundaries.

        Calculates the angle of the track displacement vector relative to the
        local normal of the boundary it crosses.

        Parameters
        ----------
        compartment_masks : dict
            Dictionary mapping compartment names to boolean mask arrays.
        pixel_size : float, optional
             Pixel size (e.g., um/pixel) to convert distances to physical units
             for distance transform, by default 1.0.

        Returns
        -------
        dict
            Dictionary containing:
            - 'crossing_angles': List of crossing angles in degrees [-180, 180].
              Angle is relative to the outward normal of the 'from_compartment'.
              0 degrees = moving directly out. +/-180 degrees = moving directly in.
              +/-90 degrees = moving parallel to the boundary.
            - 'boundary_normals': List of [ny, nx] unit normal vectors at crossings.
            - 'boundary_summary': Dict mapping boundary pairs to angle statistics.
        """
        logger.info("Analyzing angular distribution of boundary crossings.")
        if not self.crossing_events:
            logger.warning("No crossing events found. Run analyze_boundary_crossings first.")
            self.angular_distributions = {'status': 'No crossing events'}
            return self.angular_distributions

        crossing_angles = []
        boundary_normals = []
        boundary_summary = {} # Store stats per boundary type (canonical name)

        # Pre-calculate distance transforms and gradients for efficiency
        distance_transforms = {}
        gradients = {}
        img_shape = None

        for name, mask in compartment_masks.items():
            if not isinstance(mask, np.ndarray): continue
            if img_shape is None: img_shape = mask.shape
            elif mask.shape != img_shape:
                logger.warning(f"Skipping mask '{name}' due to shape mismatch in angular analysis.")
                continue

            mask_bool = mask.astype(bool)
            # Distance transform from *inside* the mask
            distance_transforms[name] = ndimage.distance_transform_edt(mask_bool, sampling=pixel_size)
            # Gradient of distance transform (points outwards)
            gy, gx = np.gradient(distance_transforms[name])
            gradients[name] = (gy, gx)

        if img_shape is None:
             logger.error("No valid masks found to calculate gradients.")
             self.angular_distributions = {'status': 'No valid masks'}
             return self.angular_distributions
        img_height, img_width = img_shape


        # --- Iterate Crossing Events ---
        for event in self.crossing_events:
            from_comp = event['from_compartment']
            to_comp = event['to_compartment']
            pos_from_pix = np.array(event['position_from'])
            pos_to_pix = np.array(event['position_to'])
            disp_vec_pix = np.array([event['dx'], event['dy']]) # [dx, dy] in pixels

            # Normalize displacement vector
            disp_norm = np.linalg.norm(disp_vec_pix)
            if disp_norm < 1e-6: continue # Skip zero displacement steps
            disp_unit_vec = disp_vec_pix / disp_norm # Unit vector [dx_norm, dy_norm]

            # Identify the mask defining the boundary being exited ('from' region)
            mask_name_for_normal = None
            normal_sign_flip = 1.0 # Flip normal if exiting 'Outside'

            if from_comp != 'Outside' and from_comp in gradients:
                 mask_name_for_normal = from_comp
            elif to_comp != 'Outside' and to_comp in gradients:
                 # Exiting 'Outside' into 'to_comp'. Use 'to_comp' gradient but flip it.
                 mask_name_for_normal = to_comp
                 normal_sign_flip = -1.0
            else:
                 # Crossing between two 'Outside' regions or invalid compartments
                 logger.debug(f"Skipping angle calculation for crossing: {from_comp} -> {to_comp}")
                 continue

            # Get pre-calculated gradient
            gy, gx = gradients[mask_name_for_normal]

            # Find approximate boundary point (midpoint of step in pixel coords)
            crossing_point_pix = (pos_from_pix + pos_to_pix) / 2.0
            # Use integer indices for gradient lookup
            cy_idx, cx_idx = int(round(crossing_point_pix[1])), int(round(crossing_point_pix[0])) # y, x order

            # Ensure indices are within bounds
            if not (0 <= cy_idx < img_height and 0 <= cx_idx < img_width):
                logger.debug(f"Crossing point ({cx_idx}, {cy_idx}) outside image bounds. Skipping angle.")
                continue

            # Get local normal vector [ny, nx] from gradient (points outwards from mask)
            local_normal = np.array([gy[cy_idx, cx_idx], gx[cy_idx, cx_idx]])
            local_normal *= normal_sign_flip # Flip if exiting 'Outside'

            normal_norm = np.linalg.norm(local_normal)
            if normal_norm < 1e-6:
                logger.debug(f"Zero normal vector at crossing point ({cx_idx}, {cy_idx}). Skipping angle.")
                continue # Skip if normal is zero

            normal_unit_vec = local_normal / normal_norm # Unit normal [ny_norm, nx_norm]

            # Calculate angle between displacement [dx, dy] and normal [nx, ny]
            # Swap normal components to get [nx, ny] for correct angle calculation with [dx, dy]
            normal_xy = np.array([normal_unit_vec[1], normal_unit_vec[0]]) # [nx_norm, ny_norm]
            dot_product = np.dot(disp_unit_vec, normal_xy)
            dot_product = np.clip(dot_product, -1.0, 1.0) # Ensure valid range for arccos
            angle_rad = np.arccos(dot_product)

            # Determine sign using 2D cross product: disp_x * normal_y - disp_y * normal_x
            cross_product = disp_unit_vec[0] * normal_xy[1] - disp_unit_vec[1] * normal_xy[0]
            if cross_product < 0:
                angle_rad = -angle_rad

            angle_deg = np.degrees(angle_rad)

            crossing_angles.append(angle_deg)
            boundary_normals.append(normal_unit_vec.tolist()) # Store [ny, nx] normal

            # Store angle per boundary type (canonical name)
            boundary_name = tuple(sorted((from_comp, to_comp)))
            if boundary_name not in boundary_summary:
                 boundary_summary[boundary_name] = {'angles': [], 'count': 0}
            boundary_summary[boundary_name]['angles'].append(angle_deg)
            boundary_summary[boundary_name]['count'] += 1

        # --- Calculate Summary Statistics ---
        for name, data in boundary_summary.items():
             angles = np.array(data['angles'])
             if len(angles) > 0:
                  data['mean_angle'] = np.mean(angles)
                  data['std_angle'] = np.std(angles)
                  angles_rad = np.deg2rad(angles)
                  # Circular stats
                  sin_mean = np.mean(np.sin(angles_rad))
                  cos_mean = np.mean(np.cos(angles_rad))
                  data['circular_mean_angle'] = np.degrees(np.arctan2(sin_mean, cos_mean))
                  R = np.sqrt(sin_mean**2 + cos_mean**2) # Mean resultant length
                  # Circular std dev from R (Mardia & Jupp, Directional Statistics, eq 2.3.10)
                  data['circular_std_dev_deg'] = np.degrees(np.sqrt(-2 * np.log(R))) if R > 1e-9 else np.nan
             else: # Handle case with no angles for this boundary
                  data['mean_angle'] = np.nan
                  data['std_angle'] = np.nan
                  data['circular_mean_angle'] = np.nan
                  data['circular_std_dev_deg'] = np.nan
        # ----------------------------------

        self.angular_distributions = {
            'status': 'Computed',
            'crossing_angles': crossing_angles,
            'boundary_normals': boundary_normals,
            'boundary_summary': boundary_summary
        }
        logger.info(f"Analyzed angular distribution for {len(crossing_angles)} valid crossings.")
        return self.angular_distributions

