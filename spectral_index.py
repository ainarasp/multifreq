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

parser.add_argument('freq1', metavar='f', type=float, nargs=1,
                    help='first frequency')

parser.add_argument('freq2', metavar='f', type=float, nargs=1,
                    help='second frequency')

args = parser.parse_args()

obsX = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq1[0]/1e9) + 'NGC1052_t686.63_%03dGHz.uvp'%(args.freq1[0]/1e9))
beamparamsX = obsX.fit_beam()
data = np.load('./values/%03dGHz/'%(args.freq1[0]/1e9) + 'NGC1052_t686.63_%03dGHz_mf_blur.npz'%(args.freq1[0]/1e9))

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


# This will be only for the first one

im1 = eh.image.load_fits('./fits/' + file_list_1[0])
im2 = eh.image.load_fits('./fits/' + file_list_2[0])
im1 = im1.blur_gauss(beamparamsX,1)
im2 = im2.blur_gauss(beamparamsX,1)
data1 = im1.imarr(pol='I')
data2 = im2.imarr(pol='I')
data1[data1<1e-2*ma.amax(data1)]=0
data2[data1<1e-2*ma.amax(data1)]=0

spectral_map = (ma.log10(data1)-ma.log10(data2))/(log10(args.freq1[0])-log10(args.freq2[0]))

vmin = -1
vmax = 3
xplt = data['xplt']

spectral_map[spectral_map<vmin] = np.nan
spectral_map[spectral_map>vmax] = np.nan

print(beamparamsX[0]/eh.RADPERUAS/1000)
print(beamparamsX[1]/eh.RADPERUAS/1000)
print(beamparamsX[2]*180/np.pi)

cmap = plt.cm.jet
cmap.set_bad('black')

i1=plt.imshow(spectral_map[100:(spectral_map.shape[0]-100),:],cmap=cmap,vmin=vmin,vmax=vmax,extent=[xplt[0],xplt[-1],xplt[xplt.shape[0]-100],xplt[100]])
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.contour(ma.log10(data1)[100:(spectral_map.shape[0]-100),:],colors='gray',extent=[xplt[0],xplt[-1],xplt[100],xplt[xplt.shape[0]-100]])
plt.xlabel('relative RA [mas]')
plt.ylabel('relative DEC [mas]')
ell = mpl.patches.Ellipse([7.5,-4],beamparamsX[0]/eh.RADPERUAS/1000,beamparamsX[1]/eh.RADPERUAS/1000,color='white',angle=-beamparamsX[2]*180/np.pi-90)
plt.gca().add_patch(ell)
cbar = plt.colorbar(i1,location='top')
cbar.set_label('Spectral index')
plt.show()