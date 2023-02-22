#!/usr/bin/env python
import string,math,sys,fileinput,glob,os,time,errno,argparse
from numpy import *
from scipy import *
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pylab import *
import fnmatch
import ehtim as eh
import astropy.io.fits as fits

parser = argparse.ArgumentParser(description='blur MEM images with a circular beam and compute the width')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequencies to analyze in Hz')

args = parser.parse_args()


obs_list = []
for i in os.listdir('./uvfits/%03dGHz/'%(args.freq[0]/1e9)):
    if fnmatch.fnmatch(i, '*.uvp'):
        obs_list.append(i)
obs_list.sort()

average_beam = []
for i in range(len(obs_list)):
	obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[0]/1e9) + obs_list[i])
	beamparams = obs.fit_beam()
	average_beam.append(beamparams)

average_beam = np.mean(average_beam,axis=0)
circular_radius = sqrt(average_beam[0]*average_beam[1]) # this is the DIAMETER. but blur_circ() takes the FWHM size so this should be correct

file_list = []
for i in os.listdir('./fits/'):
    if fnmatch.fnmatch(i, '*%03dGHz_mf.fits'%(args.freq[0]/1e9)):
        file_list.append(i)
file_list.sort()


outdir = './circular/'

for i in range(len(file_list)):
	outname = file_list[i][:-7]
	im = eh.image.load_fits('./fits/' + file_list[i])
	im.blur_circ(circular_radius).save_fits(outdir + '%03dGHz/'%(args.freq[0]/1e9) + outname + 'circ.fits')
	#data = im.imarr(pol='I')
	#data = ndimage.rotate(data,15,reshape=False)
	#plt.figure()
	#plt.imshow(data)
	#plt.show()