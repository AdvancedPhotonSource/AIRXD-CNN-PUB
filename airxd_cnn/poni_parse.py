import os
from math import pi, cos, sin, sqrt, acos, asin
import numpy as np
import json
import re

degrees = lambda  x: 180 * x / pi
radians = lambda  x: x * pi / 180

#Poni file reader
class PyFaiFile:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'Detector_config':
                        # Parse the detector config dictionary
                        import json
                        detector_config = json.loads(value)
                        self.pixel1 = detector_config['pixel1']
                        self.pixel2 = detector_config['pixel2']
                    else:
                        # Convert string values to appropriate types
                        try:
                            value = float(value)
                        except ValueError:
                            # Keep as string if not convertible to float
                            pass
                        
                        setattr(self, key.lower(), value)


def convert_to_imctrl(poni_path):
    """Convert a Geometry|PONI object to the geometry of Fit2D
    Please see the doc from Fit2dGeometry

    :param poni: azimuthal integrator, geometry or poni
    :return: same geometry as a Fit2dGeometry named-tuple
    """
    poni = PyFaiFile(poni_path)

    cos_tilt = cos(poni.rot1) * cos(poni.rot2)

    sin_tilt = sqrt(1.0 - cos_tilt * cos_tilt)
    tan_tilt = sin_tilt / cos_tilt
    # This is tilt plane rotation
    if sin_tilt == 0:
        # tilt plan rotation is undefined when there is no tilt!, does not matter
        cos_tilt = 1.0
        sin_tilt = 0.0
        cos_tpr = 1.0
        sin_tpr = 0.0

    else:
        cos_tpr = max(-1.0, min(1.0, -cos(poni.rot2) * sin(poni.rot1) / sin_tilt))
        sin_tpr = sin(poni.rot2) / sin_tilt
    directDist = 1.0e3 * poni.distance / cos_tilt
    tilt = degrees(acos(cos_tilt))

    if sin_tpr < 0:
        tpr = -degrees(acos(cos_tpr))
    else:
        tpr = degrees(acos(cos_tpr))

    centerX = (poni.poni2 + poni.distance * tan_tilt * cos_tpr) * 1e3#/ poni.pixel2
    if abs(tilt) < 1e-5:  # in degree
        centerY = (poni.poni1) / poni.pixel1
    else:
        centerY = (poni.poni1 + poni.distance * tan_tilt * sin_tpr) * 1e3#/ poni.pixel1
    out = {}
    out["distance"] = directDist
    out["center"] = [centerX, centerY]
    out["tilt"] = tilt
    out["rotation"] = tpr
    out["type"] = poni.detector
    out["pixelSize"] = [poni.pixel2 * 1e6, poni.pixel1 * 1e6]
    out["wavelength"] = poni.wavelength * 1e10

    #Putting in some dummy values because they're found in .imctrl and not in .poni
    out['IOtth'] = [0.8, 17.0]
    out['PolaVal'] = [0.9, False]
    out['azmthOff'] = 0.0
    out['DetDepth'] = 0.0
    
    return out

import numpy as np

def create_2theta_map_poni(poni_params, detector_shape):
    """
    Create 2θ map directly from PONI geometry parameters
    
    Parameters:
    -----------
    poni_params : dict
        Dictionary containing PONI parameters:
        - 'Distance': sample-to-detector distance (m)
        - 'Poni1': vertical offset of beam center (m)  
        - 'Poni2': horizontal offset of beam center (m)
        - 'Rot1': rotation about vertical axis (radians)
        - 'Rot2': rotation about horizontal axis (radians) 
        - 'Rot3': rotation about beam axis (radians)
        - 'pixel1': pixel size in Y direction (m)
        - 'pixel2': pixel size in X direction (m)
        
    detector_shape : tuple
        (height, width) of detector in pixels
        
    Returns:
    --------
    tth_map : ndarray
        2θ values in degrees for each pixel
    azm_map : ndarray  
        Azimuthal angles in degrees for each pixel
    """
    
    # Extract parameters
    dist = poni_params['Distance']
    poni1 = poni_params['Poni1'] 
    poni2 = poni_params['Poni2']
    rot1 = poni_params['Rot1']
    rot2 = poni_params['Rot2'] 
    rot3 = poni_params['Rot3']
    pixel1 = poni_params['pixel1']  # Y pixel size
    pixel2 = poni_params['pixel2']  # X pixel size
    
    height, width = detector_shape
    
    # Create pixel coordinate grids
    # In detector coordinates: j=0,1,2... (width), i=0,1,2... (height)
    j_coords, i_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to physical coordinates on detector surface (meters)
    # Detector origin is at lower-left when viewed from sample perspective
    # So we need to flip Y coordinates: Y increases upward from bottom
    x_det = j_coords * pixel2  # Physical X coordinate on detector (left to right)
    y_det = (height - 1 - i_coords) * pixel1  # Physical Y coordinate (bottom to top)
    z_det = np.zeros_like(x_det)  # All pixels are on detector surface initially
    
    # Create rotation matrices (right-hand rule)
    def rotation_matrix_x(angle):
        """Rotation about X axis (Rot2)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s], 
                        [0, s, c]])
    
    def rotation_matrix_y(angle):
        """Rotation about Y axis (Rot1)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    
    def rotation_matrix_z(angle):
        """Rotation about Z axis (Rot3)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    
    # Apply rotations in order: Rot1 (Y), then Rot2 (X), then Rot3 (Z)
    R1 = rotation_matrix_y(rot1)
    R2 = rotation_matrix_x(rot2) 
    R3 = rotation_matrix_z(rot3)
    
    # Combined rotation matrix
    R_total = R3 @ R2 @ R1
    
    # Detector is positioned at distance 'dist' along Z axis from sample
    # The point of normal incidence is at (0, 0, dist) in sample coordinates
    # But the beam actually hits at (poni2, poni1, dist) 
    
    # Transform detector coordinates to sample reference frame
    # First, translate so that point of normal incidence is at origin
    x_sample_frame = x_det - poni2
    y_sample_frame = y_det - poni1
    z_sample_frame = z_det + dist  # Detector is at +Z distance from sample
    
    # Stack coordinates for transformation
    coords = np.stack([x_sample_frame.ravel(), y_sample_frame.ravel(), z_sample_frame.ravel()], axis=0)
    
    # Apply rotation to move detector to its actual orientation
    rotated_coords = R_total @ coords
    
    # Reshape back to 2D arrays
    x_final = rotated_coords[0].reshape(detector_shape)
    y_final = rotated_coords[1].reshape(detector_shape) 
    z_final = rotated_coords[2].reshape(detector_shape)
    
    # Calculate scattering vectors from sample (at origin) to each pixel
    scattered_length = np.sqrt(x_final**2 + y_final**2 + z_final**2)
    
    # Normalize scattered beam direction vectors
    scat_x = x_final / scattered_length
    scat_y = y_final / scattered_length  
    scat_z = z_final / scattered_length
    
    # Incident beam direction: [0, 0, 1] (along +Z)
    # Calculate 2θ using dot product: cos(2θ) = incident · scattered = scat_z
    cos_2theta = scat_z
    
    # Ensure valid range for arccos
    cos_2theta = np.clip(cos_2theta, -1.0, 1.0)
    
    # Calculate 2θ in degrees
    tth_map = np.degrees(np.arccos(cos_2theta))
    
    # Calculate azimuthal angle in the scattering plane
    # Project onto plane perpendicular to incident beam
    azm_map = np.degrees(np.arctan2(scat_y, scat_x))
    
    # Ensure azimuth is in [0, 360) range
    azm_map = (azm_map + 360) % 360
    
    return tth_map, azm_map

def parse_poni_file(poni_filepath):
    """
    Parse a PONI file and extract parameters for 2θ map calculation
    
    Parameters:
    -----------
    poni_filepath : str
        Path to the PONI file
        
    Returns:
    --------
    poni_params : dict
        Dictionary containing parameters for create_2theta_map_poni():
        - 'Distance': sample-to-detector distance (m)
        - 'Poni1': vertical offset of beam center (m)  
        - 'Poni2': horizontal offset of beam center (m)
        - 'Rot1': rotation about vertical axis (radians)
        - 'Rot2': rotation about horizontal axis (radians) 
        - 'Rot3': rotation about beam axis (radians)
        - 'pixel1': pixel size in Y direction (m)
        - 'pixel2': pixel size in X direction (m)
    """
    
    poni_params = {}
    
    with open(poni_filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
            
        # Handle key: value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Parse the main parameters
            if key == 'Distance':
                poni_params['Distance'] = float(value)
            elif key == 'Poni1':
                poni_params['Poni1'] = float(value)
            elif key == 'Poni2':
                poni_params['Poni2'] = float(value)
            elif key == 'Rot1':
                poni_params['Rot1'] = float(value)
            elif key == 'Rot2':
                poni_params['Rot2'] = float(value)
            elif key == 'Rot3':
                poni_params['Rot3'] = float(value)
            elif key == 'Detector_config':
                # Parse the JSON-like detector configuration
                try:
                    # Clean up the JSON string and parse it
                    detector_config = json.loads(value)
                    poni_params['pixel1'] = detector_config['pixel1']
                    poni_params['pixel2'] = detector_config['pixel2']
                    if 'max_shape' in detector_config and detector_config['max_shape']:
                        poni_params['max_shape'] = detector_config['max_shape']
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse Detector_config: {value}")
            elif key == 'Wavelength':
                poni_params['Wavelength'] = float(value)
            elif key == 'poni_version':
                poni_params['poni_version'] = float(value)
            elif key == 'Detector':
                poni_params['Detector'] = value
    
    # Verify that all required parameters are present
    required_params = ['Distance', 'Poni1', 'Poni2', 'Rot1', 'Rot2', 'Rot3', 'pixel1', 'pixel2']
    missing_params = [param for param in required_params if param not in poni_params]
    
    if missing_params:
        raise ValueError(f"Missing required parameters in PONI file: {missing_params}")
    
    return poni_params