
"""
Multi-channel management module for SPT Analysis.

This module provides functionality for handling multiple image channels, 
particularly for experiments with different fluorescent markers.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class MultiChannelManager:
    """
    Manager for multi-channel microscopy data.
    
    This class handles image channels and tracks which channels are used for
    particle tracking, compartment identification, etc.
    """
    
    def __init__(self):
        """Initialize the multi-channel manager."""
        self.channels = {}  # Dictionary to hold channel image data (e.g., numpy arrays)
        self.channel_info = {}  # Metadata about each channel (name, type, role, etc.)
        self.tracking_channels = []  # List of channel indices marked for tracking
    
    def add_channel(self, data, name, channel_type, role=None, particle_type=None):
        """
        Add a new channel to the manager.
        
        Parameters
        ----------
        data : numpy.ndarray
            The channel image data
        name : str
            Name of the channel
        channel_type : str
            Type of channel (e.g., 'DAPI', 'Tracking', etc.)
        role : str, optional
            Optional role (e.g., 'Compartment Marker'), by default None
        particle_type : str, optional
            For tracking channels, by default None
            
        Returns
        -------
        int
            The index of the new channel
        """
        channel_idx = len(self.channels)
        self.channels[channel_idx] = data
        
        is_tracking = channel_type.lower() == 'tracking'
        
        info = {
            'name': name,
            'type': channel_type,
            'role': role,
            'is_tracking': is_tracking,
            'shape': data.shape
        }
        
        if is_tracking:
            if not particle_type:
                particle_type = f"Particle{len(self.tracking_channels) + 1}"
            info['particle_type'] = particle_type
            self.tracking_channels.append(channel_idx)
        
        self.channel_info[channel_idx] = info
        
        logger.info(f"Added channel {channel_idx}: {name} ({channel_type})")
        
        return channel_idx
    
    def get_particle_types(self):
        """
        Return a list of particle types from tracking channels.
        
        Returns
        -------
        list
            List of particle types
        """
        return [self.channel_info[idx].get('particle_type') 
                for idx in self.tracking_channels]
    
    def get_tracking_channels(self):
        """
        Return a dictionary mapping channel indices to particle types.
        
        Returns
        -------
        dict
            Dictionary mapping channel indices to particle types
        """
        return {idx: self.channel_info[idx].get('particle_type') 
                for idx in self.tracking_channels}
    
    def get_channel_by_role(self, role):
        """
        Get channel indices that match a specific role.
        
        Parameters
        ----------
        role : str
            Role to match
            
        Returns
        -------
        list
            List of channel indices matching the role
        """
        return [idx for idx, info in self.channel_info.items() 
                if info.get('role') == role]
    
    def get_compartment_channels(self):
        """
        Get channels marked as compartment markers.
        
        Returns
        -------
        dict
            Dictionary mapping channel indices to compartment names
        """
        return {idx: info.get('name') for idx, info in self.channel_info.items() 
                if info.get('role') == 'Compartment Marker'}
    
    def generate_compartment_masks(self, threshold=None):
        """
        Generate binary masks for compartment channels.
        
        Parameters
        ----------
        threshold : float or dict, optional
            Threshold value(s) for mask generation, by default None
            
        Returns
        -------
        dict
            Dictionary mapping compartment names to binary masks
        """
        compartment_channels = self.get_compartment_channels()
        
        if not compartment_channels:
            return {}
        
        masks = {}
        
        for idx, name in compartment_channels.items():
            channel_data = self.channels[idx]
            
            # Determine threshold
            if threshold is None:
                # Use Otsu's method for automatic thresholding
                from skimage.filters import threshold_otsu
                try:
                    thresh_value = threshold_otsu(channel_data)
                except:
                    # Fallback to simple thresholding
                    thresh_value = np.mean(channel_data) + np.std(channel_data)
            elif isinstance(threshold, dict):
                thresh_value = threshold.get(name, np.mean(channel_data) + np.std(channel_data))
            else:
                thresh_value = threshold
            
            # Create mask
            mask = channel_data > thresh_value
            
            # Optional: Clean up mask (remove small objects, fill holes)
            try:
                from scipy import ndimage
                mask = ndimage.binary_opening(mask, iterations=2)
                mask = ndimage.binary_fill_holes(mask)
            except:
                pass
            
            masks[name] = mask
        
        return masks
    
    def save_channels(self, directory):
        """
        Save all channels to a directory.
        
        Parameters
        ----------
        directory : str
            Directory to save channels
            
        Returns
        -------
        list
            List of saved file paths
        """
        import os
        import tifffile
        
        os.makedirs(directory, exist_ok=True)
        
        saved_paths = []
        
        for idx, data in self.channels.items():
            info = self.channel_info[idx]
            name = info['name'].replace(' ', '_')
            
            file_path = os.path.join(directory, f"{name}_channel_{idx}.tif")
            
            tifffile.imwrite(file_path, data)
            saved_paths.append(file_path)
        
        return saved_paths
    
    @classmethod
    def load_channels(cls, directory, pattern="*_channel_*.tif"):
        """
        Load channels from a directory.
        
        Parameters
        ----------
        directory : str
            Directory containing channel files
        pattern : str, optional
            File pattern to match, by default "*_channel_*.tif"
            
        Returns
        -------
        MultiChannelManager
            Manager with loaded channels
        """
        import os
        import glob
        import tifffile
        import re
        
        manager = cls()
        
        file_paths = glob.glob(os.path.join(directory, pattern))
        
        for file_path in file_paths:
            # Extract name and channel index from filename
            filename = os.path.basename(file_path)
            name_match = re.match(r"(.+)_channel_(\d+)\.tif", filename)
            
            if name_match:
                name = name_match.group(1).replace('_', ' ')
                channel_idx = int(name_match.group(2))
                
                # Determine channel type and role based on name
                channel_type = 'Unknown'
                role = None
                particle_type = None
                
                if 'dapi' in name.lower() or 'hoechst' in name.lower():
                    channel_type = 'Nuclear'
                    role = 'Compartment Marker'
                elif 'gfp' in name.lower() or 'track' in name.lower():
                    channel_type = 'Tracking'
                    particle_type = name
                elif 'membrane' in name.lower() or 'cell' in name.lower():
                    channel_type = 'Membrane'
                    role = 'Compartment Marker'
                
                # Load data
                data = tifffile.imread(file_path)
                
                # Add channel
                manager.add_channel(data, name, channel_type, role, particle_type)
        
        return manager
