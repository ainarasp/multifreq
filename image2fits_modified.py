#!/usr/bin/env python
import string,math,sys,fileinput,glob,os,time,errno
from numpy import *
from scipy import *
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pylab import *
import h5py
import sys,glob,argparse
import fnmatch

#use astropy for fits and source locations
from astropy.time import Time as aTime
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits

import ehtim as eh

def Dl(red):

    #cosmology values
    Omega = 0.27e0
    H0 = 71.e0

    #using Ue-Li Pen's analytical equation
    s3=(1.0e0-Omega)/Omega

    a=1.0e0/(1.0e0+red)

    eta1=2.0e0*sqrt(s3+1.0e0)*(1.0e0-0.1540e0*s3**(1.0e0/3.0e0)+0.4304e0*s3**(2.0e0/3.0e0)+0.19097e0*s3+0.066941e0*s3**(4.0e0/3.0e0))**(-1.0e0/8.0e0)
    eta2=2.0e0*sqrt(s3+1.0e0)*(1.0e0/a**4.0e0-0.1540e0*s3**(1.0e0/3.0e0)/a**3.0e0+0.4304e0*s3**(2.0e0/3.0e0)/a**2.0e0+0.19097e0*s3/a+0.066941e0*s3**(4.0e0/3.0e0))**(-1.0e0/8.0e0)

    #luminosity distance in Mpc
    dl=2.99792458e5/H0*(eta1-eta2)*(1+red)
    dlm=dl

    #luminosity distance in cm
    dl=dl*1.0e6*3.08e18

    #angular size distance
    da=dl/3.08e18/(1.+red)**2

    mas2pc=da*4.84e-9

    return {'dlMpc':dlm, 'dlcm': dl, 'da':da, 'mas2pc':mas2pc}

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='create fits file from image data')

parser.add_argument('path', metavar='d', type=str, nargs=1,
                    help='path to input image files from simulation')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency in Hz')

parser.add_argument('source', metavar='s', type=str, nargs=1,
                    help='standard name of the source')

args = parser.parse_args()

# Get all files
file_list = []
for i in os.listdir(args.path[0]):
    if fnmatch.fnmatch(i, 'JETNB*'):
        file_list.append(i)
file_list.sort()

# Load image
for i in range(100):
    data = np.load(args.path[0]+file_list[i])
    # Universal data: frequency, sky location, scales.
    if i == 0:
        freq = float(args.freq[0])
        ind = np.argmin(abs(data['freq']-freq))

        loc = SkyCoord.from_name(args.source[0])
        ra = loc.ra.deg
        dec = loc.dec.deg

        cosmo = Dl(data['redshift'])
        dx = float(data['xplt'][1]-data['xplt'][0])/data['emiss'].shape[0]*data['Rscale']
        dy = float(data['yplt'][1]-data['yplt'][0])/data['emiss'].shape[1]*data['Rscale']

    # Get emission and scale to Jansky
    flux = data['emiss'][:,:,ind,0]*dx*dy/(cosmo['dlcm'])**2*(1+data['redshift'])*1e23

    # Get emission into square array
    regrid=np.zeros((flux.shape[0],flux.shape[0]))
    regrid[200:600,:]=flux.T
    regrid = ndimage.rotate(regrid, 15)

    if i == 0:
        # Compute pixel scale in cm
        dx=(data['xplt'][1]-data['xplt'][0])/flux.shape[0]*data['Rscale']
        dxpc=dx/3.08e18
        dxmas=dxpc/cosmo['mas2pc']
        dxmuas=dxmas*1e3
        print(dx,dxpc,dxmas,dxmuas,eh.DEGREE)

        # Create fits header
        header = fits.Header()
        header['OBJECT'] = args.source[0]
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CDELT1'] = -dxmuas*eh.RADPERUAS/eh.DEGREE
        header['CDELT2'] = dxmuas*eh.RADPERUAS/eh.DEGREE
        header['OBSRA']=loc.ra.deg
        header['OBSDEC']=loc.dec.deg
        header['TELESCOP'] = 'VLBI'
        header['FREQ'] = args.freq[0]
        header['BUNIT'] = 'JY/PIXEL'
        header['STOKES'] = 'I'

    # Write data to fits file
    hdu = fits.PrimaryHDU(regrid, header=header)
    hdulist=[hdu]
    hdulist = fits.HDUList(hdulist)

    # Modify file name in order not to overwrite existing ones!
    hdulist.writeto('model/%03dGHz/%s_t%1.2f_%03dGHz.fits' %(args.freq[0]/1e9,args.source[0],data['time'],args.freq[0]/1e9), overwrite=True)