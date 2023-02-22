import string,math,sys,fileinput,glob,os,time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Ellipse,Rectangle
from pylab import *
import copy
import fnmatch

from astropy.time import Time as aTime
from astropy.io import fits
import numpy as np
import numpy.ma as ma

import ehtim as eh
import ehtim.scattering as so
from ehtim.const_def import *
from ehtim.observing.obs_simulate import *
from ehtim.calibrating import self_cal as sc

from skimage import data
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import fourier_shift

import itertools as it
import copy
import argparse

parser = argparse.ArgumentParser(description='create synthetic data from GRRT image')

parser.add_argument('source', metavar='s', type=str, nargs=1,
                    help='name of the source')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency in Hz')

parser.add_argument('array', metavar='obs', type=str, nargs=1,
                    help='array file with telescope postion and SEFD')

args = parser.parse_args()

###better not touch if you don't know what they are
## synthetic data parameters --> from PI-work week
seed = 7 
add_th_noise = True # if there are no sefds in obs_orig it will use the sigma for each data point
phasecal = True # if False then add random phases to simulate atmosphere
ampcal = False # if False then add random gain errors 
stabilize_scan_phase = False # if true then add a single phase error for each scan to act similar to adhoc phasing
stabilize_scan_amp = False # if true then add a single gain error at each scan
jones = True # apply jones matrix for including noise in the measurements (including leakage)
inv_jones = True # no not invert the jones matrix
frcal = True # True if you do not include effects of field rotation
dcal = True # True if you do not include the effects of leakage
dterm_offset = 0.05 # a random offset of the D terms is given at each site with this standard deviation away from 1
ttype = 'fast'
fft_pad_factor = 2

#gain_offset = {'APEX':0.1, 'ALMA':0.05, 'SMT':0.1, 'LMT':0.2, 'PV':0.1, 'SMA':0.1, 'JCMT':0.1, 'SPT':0.1, 'PDB':0.1}
gain_offset = 0.1
## the standard deviation of gain differences
#gainp = {'ALMA':0.05, 'APEX':0.05, 'SMT':0.05, 'LMT':0.5, 'PV':0.05, 'SMA':0.05, 'JCMT':0.05, 'SPT':0.05, 'PDB':0.05}  
gainp = 0.1

array = eh.array.load_txt(args.array[0])

# Get all files
file_list = []
for i in os.listdir('./model/%03dGHz/' %(args.freq[0]/1e9)):
    if fnmatch.fnmatch(i, '%s*' %(args.source[0])):
        file_list.append(i)
file_list.sort()

tmp=file_list[0].split('/')
t0 = float(tmp[-1][9:15])

data = np.load('./files/JETNBAA_slowlight_686.632_emission_80_0_rhoa_1.67e-21_Rscale_3.00e+16_ee_0.30_eb_0.10_ez_1.00_eg_1.00e+03_s_2.20_rhod_100_1.0_2.0_geotorus_4_30_50_tempt_1500.0_1.0_2.0.npz')
rscale = data['Rscale']
del data

c = 29979245800
timestep = rscale/c
timestep = timestep/86400
observing_date = '2017-04-07 0:00:00'
t=aTime(observing_date)


for i in range(len(file_list)):
    im = eh.image.load_fits('./model/%03dGHz/'%(args.freq[0]/1e9)+file_list[i])
    im.mjd=t.mjd
    obs=im.observe(array, 12, 600, 0, 24,0.5e9,sgrscat=False,add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal,stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp, gain_offset=gain_offset,gainp=gainp, jones=jones,inv_jones=inv_jones,dcal=dcal, frcal=frcal, dterm_offset=dterm_offset, ttype='fast',elevmin=10, elevmax=85,seed=seed)
    t.mjd = t.mjd + timestep
    ##create outname from imagefile
    tmp=file_list[i].split('/')
    outname=tmp[-1][:-5]

    ##store synthetic data
    obs.save_uvfits('uvfits/%03dGHz/%s.uvp' %(args.freq[0]/1e9,outname))