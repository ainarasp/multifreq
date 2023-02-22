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

parser = argparse.ArgumentParser(description='plot and save information on object width and peak flux')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequencies to analyze in Hz')

args = parser.parse_args()

obs_list = []
for i in os.listdir('./uvfits/%03dGHz/'%(args.freq[0]/1e9)):
		if fnmatch.fnmatch(i, '*.uvp'):
			obs_list.append(i)
obs_list.sort()


# MANUALLY FOR VISUALIZATION PURPOSES

beams = np.zeros([len(obs_list),3])

for i in range(len(obs_list)):

	# get beam size for each frequency (you could average them, but they seem to be all the same for the same frequency)
	obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[0]/1e9) + obs_list[i])
	beamparams = obs.fit_beam()
	beams[i] = beamparams*180/np.pi

print(beams)