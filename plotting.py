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

# This gives you the option of plotting width and peak flux in log and linear scale. I will do a separate plot to perfect the logwidth plot


###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='plot and save information on object width and peak flux')

parser.add_argument('value', metavar='s', type=str, nargs=1,
                    help='value to plot. options: width, logwidth, flux, logflux')


parser.add_argument('freq', metavar='f', type=float, nargs='+',
                    help='all wanted frequencies in Hz')

args = parser.parse_args()

#obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[0]/1e9) + 'NGC1052_t686.63_%03dGHz.uvp'%(args.freq[0]/1e9))
#beamparams = obs.fit_beam()
#beam_width = sqrt((beamparams[0]*cos(beamparams[2]))**2 + (beamparams[1]*sin(beamparams[2]))**2) # this is in radians
#beam_width = beam_width/eh.RADPERUAS/1000


freq_nr = len(args.freq)
#beam_width = np.zeros([freq_nr,3])

for i in range(freq_nr):
	widths = []
	file_list = []
	peaks = []

	# get beam size for each frequency (you could average them, but they seem to be all the same for the same frequency)
	obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[i]/1e9) + 'NGC1052_t686.63_%03dGHz.uvp'%(args.freq[i]/1e9))
	beamparams = obs.fit_beam()
	beam_width = sqrt((beamparams[0]*cos(beamparams[2]))**2 + (beamparams[1]*sin(beamparams[2]))**2) # this is in radians
	beam_width = beam_width/eh.RADPERUAS/1000

	for k in os.listdir('./values/%03dGHz/' %(args.freq[i]/1e9)):
		if fnmatch.fnmatch(k, '*.npz'):
			file_list.append(k)
	file_list.sort()


	for j in range(len(file_list)):
		data = np.load('./values/%03dGHz/'%(args.freq[i]/1e9)+file_list[j])
		width = data['width']
		widths.append(width)
		peak = data['peaks']
		peaks.append(peak)

	avg_width = np.mean(widths,axis=0)
	avg_peak = np.mean(peaks,axis=0)

	if args.value[0] == 'width':
		plt.plot(data['xplt'],avg_width,label='%02dGHz'%(args.freq[i]/1e9))
		plt.ylabel('Jet width [mas]')
		plt.xlabel('x [mas]')
		plt.legend()

	if args.value[0] == 'logwidth':
		middle = round(avg_width.shape[0]/2)
		left_jet = avg_width[0:middle]
		right_jet = avg_width[middle:avg_width.shape[0]][::-1]
		distance = data['xplt'][0:middle]
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

	if args.value[0] == 'deconvolved':
		middle = round(avg_width.shape[0]/2)
		left = avg_width[0:middle]
		right = avg_width[middle:avg_width.shape[0]][::-1]
		left_jet = sqrt(left**2-beam_width**2)
		right_jet = sqrt(right**2-beam_width**2)
		distance = data['xplt'][0:middle]
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

	if args.value[0] == 'masked':
		middle = round(avg_width.shape[0]/2)
		left = avg_width[0:middle]
		right = avg_width[middle:avg_width.shape[0]][::-1]
		left_jet = sqrt(left**2-beam_width**2)
		right_jet = sqrt(right**2-beam_width**2)
		distance = data['xplt'][0:middle]
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

	if args.value[0] == 'logflux':
		middle = round(avg_peak.shape[0]/2)
		left_jet = avg_peak[0:middle]
		right_jet = avg_peak[middle:avg_peak.shape[0]][::-1]
		distance = data['xplt'][0:middle]
		plt.subplot(1,2,1)
		plt.loglog(distance,left_jet,label='%02dGHz'%(args.freq[i]/1e9),marker='.')
		plt.ylabel('Peak flux [Jy/pixel]')
		plt.xlabel('Distance from core [mas]')
		plt.title('Left jet')
		plt.legend()
		plt.subplot(1,2,2)
		plt.loglog(distance,right_jet,label='%02dGHz'%(args.freq[i]/1e9),marker='.')
		plt.xlabel('Distance from core[mas]')
		plt.title('Right jet')
		plt.legend()
plt.show()

stop


stop

# to include multiple frequencies in same plot, save plots, etc



for i in range(freq_nr):
	peaks = []
	file_list = []
	for k in os.listdir('./values/%03dGHz/' %(args.freq[i]/1e9)):
		if fnmatch.fnmatch(k, '%s*' %(args.source[0])):
			file_list.append(k)
	file_list.sort()

	for j in range(len(file_list)):
		data = np.load('./values/%03dGHz/'%(args.freq[i]/1e9)+file_list[j])
		peak = data['peaks']
		peak[peak==0] = float('NaN')
		peaks.append(peak)
	avg_peak = np.mean(peaks,axis=0)
	std_peak = np.std(peaks,axis=0)
	plt.semilogy(data['xplt'],avg_peak,label='%02dGHz'%(args.freq[i]/1e9))
	plt.fill_between(data['xplt'],avg_peak-std_peak,avg_peak+std_peak,alpha=0.3)

plt.ylabel('Peak flux')
plt.xlabel('x [mas]')
plt.legend()
plt.savefig('./plots/avg_peaks')
plt.show()


# Check if flux is aligned

for i in range(freq_nr):
	file_list = []
	for k in os.listdir('./values/%03dGHz/' %(args.freq[i]/1e9)):
		if fnmatch.fnmatch(k, '%s*' %(args.source[0])):
			file_list.append(k)
	file_list.sort()

	for j in range(len(file_list)):
		data = np.load('./values/%03dGHz/'%(args.freq[i]/1e9)+file_list[j])
		peak = data['peaks']
		plt.plot(data['xplt'],peak)
		plt.ylabel('Peak flux')
		plt.xlabel('x [mas]')
		plt.title('%03dGHz'%(args.freq[i]/1e9))

	plt.show()

