import os
from math import pi, cos, sin, sqrt, acos, asin

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