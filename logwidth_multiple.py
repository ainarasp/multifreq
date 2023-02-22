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

parser.add_argument('range', metavar='r', type=int, nargs=1,
                    help='0 to not rule out manually chosen invalid values, 1 for yes')

parser.add_argument('freq', metavar='f', type=float, nargs='+',
                    help='frequencies to analyze in Hz')

args = parser.parse_args()

freq_nr = len(args.freq)

# MANUALLY FOR VISUALIZATION PURPOSES

forbidden_edge = 9.3
minimum_allowed = 0.07
forbidden_gap = np.array([[55,55],[29,29],[3,27],[0,14],[0,0]])

plt.subplots(1,2,sharey=True,sharex=True)
for i in range(freq_nr):
	widths = []
	file_list = []

	# get beam size for each frequency (you could average them, but they seem to be all the same for the same frequency)
	obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[i]/1e9) + 'NGC1052_t686.63_%03dGHz.uvp'%(args.freq[i]/1e9))
	beamparams = obs.fit_beam()
	beam_angle = beamparams[2] + 15*np.pi/180
	beam_width = sqrt((beamparams[0]*cos(beam_angle))**2 + (beamparams[1]*sin(beam_angle))**2) # this is in radians
	beam_width = beam_width/eh.RADPERUAS/1000

	for k in os.listdir('./values/%03dGHz/' %(args.freq[i]/1e9)):
		if fnmatch.fnmatch(k, '*.npz'):
			file_list.append(k)
	file_list.sort()


	for j in range(len(file_list)):
		data = np.load('./values/%03dGHz/'%(args.freq[i]/1e9)+file_list[j])
		width = data['width']
		widths.append(width)

	x = data['xplt']
	avg_width = np.mean(widths,axis=0)
	middle = round(avg_width.shape[0]/2)
	left = avg_width[0:middle][::-1]
	right = avg_width[middle:]
	distance = x[0:middle][::-1]
	left_jet = sqrt(left**2-beam_width**2)
	right_jet = sqrt(right**2-beam_width**2)

	if args.range[0] == 1:
		idx = (np.abs(distance - forbidden_edge)).argmin()
		print(idx)
		left_jet[0:forbidden_gap[i,0]] = np.nan
		right_jet[0:forbidden_gap[i,1]] = np.nan
		left_jet[idx:] = np.nan
		right_jet[idx:] = np.nan
		left_jet[left_jet<=minimum_allowed] = np.nan
		right_jet[right_jet<=minimum_allowed] = np.nan

	plt.subplot(1,2,1)
	plt.loglog(distance,left_jet,label='%02dGHz'%(args.freq[i]/1e9))
	plt.ylabel('Jet width[mas]')
	plt.xlabel('Distance from core [mas]')
	plt.title('Left jet')
	plt.legend()
	plt.subplot(1,2,2)
	plt.loglog(distance,right_jet,label='%02dGHz'%(args.freq[i]/1e9))
	plt.xlabel('Distance from core [mas]')
	plt.title('Right jet')
	plt.legend()

plt.show()