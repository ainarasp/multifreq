#!/usr/bin/env python
import string,math,sys,fileinput,glob,os,time,errno,argparse
from numpy import *
from scipy import *
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import *
import fnmatch
import ehtim as eh

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='plot spectral index map')

parser.add_argument('freq1', metavar='f1', type=float, nargs=1,
                    help='first frequency')

parser.add_argument('freq2', metavar='f2', type=float, nargs=1,
                    help='second frequency')

parser.add_argument('mode', metavar='m', type=int, nargs=1,
                    help='0 to create map, 1 to retrieve it')

args = parser.parse_args()

obsX = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq1[0]/1e9) + 'NGC1052_t686.63_%03dGHz.uvp'%(args.freq1[0]/1e9))
beamparamsX = obsX.fit_beam()
data = np.load('./values/%03dGHz/'%(args.freq1[0]/1e9) + 'NGC1052_t686.63_%03dGHz_mf_blur.npz'%(args.freq1[0]/1e9))
xplt = data['xplt']

if args.mode[0] == 0:

	#Retrieve pre-blurring files
	file_list_1 = []
	for i in os.listdir('./fits/'):
		if fnmatch.fnmatch(i, '*%03dGHz_mf.fits'%(args.freq1[0]/1e9)):
			file_list_1.append(i)
	file_list_1.sort()

	file_list_2 = []
	for i in os.listdir('./fits/'):
		if fnmatch.fnmatch(i, '*%03dGHz_mf.fits'%(args.freq2[0]/1e9)):
			file_list_2.append(i)
	file_list_2.sort()

	# Get 75% of average beam of lower frequency
	obs_list = []
	for i in os.listdir('./uvfits/%03dGHz/'%(args.freq1[0]/1e9)):
	    if fnmatch.fnmatch(i, '*.uvp'):
	        obs_list.append(i)
	obs_list.sort()

	beamparams = []
	for i in range(len(obs_list)):
	    obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq1[0]/1e9) + obs_list[i])
	    beam_single = obs.fit_beam()
	    beamparams.append(beam_single)

	beamparams = np.mean(beamparams,axis=0)
	print(beamparams)
	beamparams[0] = 0.75*beamparams[0]
	beamparams[1] = 0.75*beamparams[1]
	print(beamparams)
	#Blur images with the same beam and average them
	image_array_1 = []
	image_array_2 = []

	for i in range(len(file_list_1)):
		im1 = eh.image.load_fits('./fits/' + file_list_1[i])
		im2 = eh.image.load_fits('./fits/' + file_list_2[i])
		im1 = im1.blur_gauss(beamparams,1)
		im2 = im2.blur_gauss(beamparams,1)
		data1 = im1.imarr(pol='I')
		data2 = im2.imarr(pol='I')
		image_array_1.append(data1)
		image_array_2.append(data2)

	data1 = np.mean(image_array_1,axis=0)
	data2 = np.mean(image_array_2,axis=0)
	data1[data1<1e-2*ma.amax(data1)]=0
	data2[data1<1e-2*ma.amax(data1)]=0

	spectral_map = (log10(data1)-log10(data2))/(log10(args.freq1[0])-log10(args.freq2[0]))
	np.savez('./images/spectral_maps/%03dGHz_%03dGHz'%(args.freq1[0]/1e9,args.freq2[0]/1e9),map=spectral_map,data1=data1,data2=data2,beam=beamparams)
	
if args.mode[0] == 1:
	spectral_data = np.load('./images/spectral_maps/%03dGHz_%03dGHz.npz'%(args.freq1[0]/1e9,args.freq2[0]/1e9))
	spectral_map = spectral_data['map']
	data1 = spectral_data['data1']
	data2 = spectral_data['data2']
	beamparams = spectral_data['beam']

vmin = -1
vmax = 3

spectral_map[spectral_map<vmin] = np.nan
spectral_map[spectral_map>vmax] = np.nan

cmap = plt.cm.jet
cmap.set_bad('black') #replace NaN (background values) by black
i1=plt.imshow(spectral_map[100:(spectral_map.shape[0]-100),:],cmap=cmap,vmin=vmin,vmax=vmax,extent=[xplt[0],xplt[-1],xplt[xplt.shape[0]-100],xplt[100]])
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(ma.log10(data1)[100:(spectral_map.shape[0]-100),:],colors='gray',extent=[xplt[0],xplt[-1],xplt[100],xplt[xplt.shape[0]-100]])
plt.xlabel('relative RA [mas]')
plt.ylabel('relative DEC [mas]')
ell = mpl.patches.Ellipse([7.5,-4],beamparams[0]/eh.RADPERUAS/1000,beamparams[1]/eh.RADPERUAS/1000,color='white',angle=-90-beamparams[2]*180/np.pi)
plt.gca().add_patch(ell)
cbar = plt.colorbar(i1,location='top')
cbar.set_label('Spectral index')
plt.show()