# Import libraries
import numpy as np
import numpy.ma as ma
from cffi import FFI
import scipy.special as sc
if __name__ == "__main__":
    from _mask import lib
else:
    from ._mask import lib
import warnings
warnings.filterwarnings('ignore')

#d stands for degrees, converted from radians
npcosd = lambda x: np.cos(x*np.pi/180.)
npsind = lambda x: np.sin(x*np.pi/180.)
nptand = lambda x: np.tan(x*np.pi/180.)
npatand = lambda x: 180.*np.arctan(x)/np.pi
npatan2d = lambda y,x: 180.*np.arctan2(y,x)/np.pi

class MASK:
    def __init__(self, controls, shape):
        self.shape = shape
        self.controls = controls

        #Grabs 2 theta values only
        self.TA = self.Make2ThetaAzimuthMap(self.controls,(0,shape[0]),(0,shape[1]))[0]

    def AutoSpotMask(self, image, esdmul=3.0, numchans=445):
        assert image.shape == self.shape, f"The image shape is different from the declared shape: {self.shape}"

        # Additional masking
        masks = {'Frames': None}
        frame = masks['Frames']

        #Boolean mask filled with False
        tam = ma.make_mask_none(image.shape)

        LUtth = np.array(self.controls['IOtth'])
        dtth = (LUtth[1]-LUtth[0])/numchans
        TThs = np.linspace(LUtth[0], LUtth[1], numchans, False)
        band = np.array(image)

        ffi = FFI()
        m, n = tam.shape
        p = TThs.shape[0]
        ptam =  ffi.new('int['+str(m*n)+']', tam.ravel().tolist())
        pta = ffi.new('double['+str(m*n)+']', self.TA.ravel().tolist())
        pband = ffi.new('double['+str(m*n)+']', band.ravel().tolist())
        ptths = ffi.new('double['+str(p)+']', TThs.tolist())
        output = ffi.new('double['+str(m*n)+']')
        print(f'Tam length: {len(tam.ravel().tolist())}')
        print(f'TA length: {len(self.TA.ravel().tolist())}')
        print(f'Band length: {len(band.ravel().tolist())}')
        print(f'TThs length: {len(TThs.tolist())}')
        print('Starting up C code')
        lib.mask(m, n, p, dtth, esdmul, ptam, pta, pband, ptths, output)
        #m*n*8 is b/c float64 is 8 bite size
        mask = np.frombuffer(ffi.buffer(output, m*n*8), dtype=np.float64)
        mask.shape = (m, n)

        # Release memory
        ffi.release(ptam)
        ffi.release(pta)
        ffi.release(pband)
        ffi.release(ptths)
        ffi.release(output)

        return mask

    def peneCorr(self, tth, dep, dist):
        return dep*(1.-npcosd(tth))*dist**2/1000.

    def makeMat(self, Angle, Axis):
        '''Make rotation matrix from Angle and Axis
        :param float Angle: in degrees
        :param int Axis: 0 for rotation about x, 1 for about y, etc.
        '''
        cs = npcosd(Angle)
        ss = npsind(Angle)
        M = np.array(([1.,0.,0.],[0.,cs,-ss],[0.,ss,cs]),dtype=np.float32)
        return np.roll(np.roll(M,Axis,axis=0),Axis,axis=1)

    def Polarization(self, Pola, Tth, Azm=0.0):
        """   Calculate angle dependent x-ray polarization correction (not scaled correctly!)

        :param Pola: polarization coefficient e.g 1.0 fully polarized, 0.5 unpolarized
        :param Azm: azimuthal angle e.g. 0.0 in plane of polarization - can be numpy array
        :param Tth: 2-theta scattering angle - can be numpy array
          which (if either) of these is "right"?
        :return: (pola, dpdPola) - both 2-d arrays
          * pola = ((1-Pola)*npcosd(Azm)**2+Pola*npsind(Azm)**2)*npcosd(Tth)**2+ \
            (1-Pola)*npsind(Azm)**2+Pola*npcosd(Azm)**2
          * dpdPola: derivative needed for least squares

        """
        cazm = npcosd(Azm)**2
        sazm = npsind(Azm)**2
        pola = ((1.0-Pola)*cazm+Pola*sazm)*npcosd(Tth)**2+(1.0-Pola)*sazm+Pola*cazm
        dpdPola = -npsind(Tth)**2*(sazm-cazm)
        return pola,dpdPola

    def GetTthAzmG2(self, x, y, data):
        '''Give 2-theta, azimuth & geometric corr. values for detector x,y position;
         calibration info in data - only used in integration - old version
        '''
        'Needs a doc string - checked OK for ellipses & hyperbola'
        #Angle detector normal makes with incident beam (-90 to 90 degrees)
        tilt = data['tilt']
        dist = data['distance']/npcosd(tilt)
        #Rotation matrix product
        #Check makeMat
        MN = -np.inner(self.makeMat(data['rotation'],2), self.makeMat(tilt,0))
        
        #Re-center x data based on detector center (from metadata)
        dx = x-data['center'][0]
        #Re-center y data based on detector center (from metadata)
        dy = y-data['center'][1]
        #Steps:
        #1. np.dstack stacks x, y and z coordinates "vertically"
        #2. The dot product applies transformation matrix to all coordinates
        #   We transpose x,y so we can multiply along third dimension in dot product
        
        #dz: z displacement from 0 (assumed planar position originally)
        dz = np.dot(np.dstack([dx.T,dy.T,np.zeros_like(dx.T)]),MN).T[2]

        #Subtraction is because we're going from a "tilted plane" to a "flat plane"
        #Our original tilted plane distance from the origin is dx**2 + dy**2
        #This "OG" tilted plane is actually the "hypotenuse" of the "true" xy distance
        #Hypotenuse = "true xy" + dz **2
        #=> "true xy" = Hypotenuse - dz**2
        xyZ = dx**2+dy**2-dz**2

        #dist = distance from sample to detector plane
        #Azimuthal angle is tth0
        tth0 = npatand(np.sqrt(xyZ)/(dist-dz))

        #Penetration correction for dz
        dzp = self.peneCorr(tth0, data['DetDepth'], dist)

        #Actual azimuthal correction for every position
        #2theta
        #dist-dz+dzp is corrected distance from sample to pixel
        #xyZ is distance from center of detector to pixel
        #npatan2d takes the sweeping angle from the vector (1,0), horizontal line
        #to the vector ending in (x,y)

        #This view is essentially looking at a cross section of sample and detector
        #2theta angle is deflection angle off of center of detector (no deflection)
        tth = npatan2d(np.sqrt(xyZ),dist-dz+dzp)
        #Azimuthal angle is rotation clockwise from (1,0) to dy/dx value in addition to azimuthal correction
        #Mod 360 in case rotation is greater than 360
        azm = (npatan2d(dy,dx)+data['azmthOff']+720.)%360.
        distsq = data['distance']**2

        #x0 is the displacement of the detector center from the sample center
        x0 = data['distance']*nptand(tilt)
        #x component of x0 correction
        x0x = x0*npcosd(data['rotation'])
        #y component of x0 correction
        x0y = x0*npsind(data['rotation'])

        #Sum of correction distances - geometric correction
        G = ((dx-x0x)**2+(dy-x0y)**2+distsq)/distsq
        return tth,azm,G

    def Make2ThetaAzimuthMap(self, data, iLim, jLim):
        'Needs a doc string'
        pixelSize = data['pixelSize']
        scalex = pixelSize[0]/1000.
        scaley = pixelSize[1]/1000.
        tay,tax = np.mgrid[iLim[0]+0.5:iLim[1]+.5,jLim[0]+.5:jLim[1]+.5]
        tax = np.asfarray(tax*scalex,dtype=np.float32).flatten()
        tay = np.asfarray(tay*scaley,dtype=np.float32).flatten()
        nI = iLim[1]-iLim[0]
        nJ = jLim[1]-jLim[0]
        TA = np.empty((4,nI,nJ))
        TA[:3] = np.array(self.GetTthAzmG2(np.reshape(tax,(nI,nJ)),np.reshape(tay,(nI,nJ)),data))
        TA[1] = np.where(TA[1]<0,TA[1]+360,TA[1])
        TA[3] = self.Polarization(data['PolaVal'][0],TA[0],TA[1]-90.)[0]
        return TA

if __name__ == "__main__":
    import imageio
    from dataset import parse_imctrl
    numChans = 445
    Controls = parse_imctrl('../data/Nickel/Si_ch3_d700-00000.imctrl')
    Image = imageio.volread('../data/Nickel/Ni83_ch3_RTto950_d700-00005.tif')
    mask = MASK(Controls, shape=(2880, 2880))
    result = mask.AutoSpotMask(Image)
    np.save('mask', result)
