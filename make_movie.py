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
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description='create fits file from image data')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency in Hz')


args = parser.parse_args()

file_list = []
for i in os.listdir('./fits/'):
    if fnmatch.fnmatch(i, '*%03dGHz_mf_blur.fits'%(args.freq[0]/1e9)):
        file_list.append(i)
file_list.sort()

fig = plt.figure()

i = 0
image = eh.image.load_fits('./fits/' + file_list[i])
frame = image.imarr(pol='I')
size = frame.shape[0]
im = plt.imshow(ma.log10(frame)[100:size-100,:], animated=True,vmin=-7.2,vmax=-2.6)
#,cmap='afmhot'
plt.colorbar()

def updatefig(*args):
	global i
	if i < len(file_list):
		i += 1
	else:
		i = 0
	image = eh.image.load_fits('./fits/' + file_list[i])
	frame = image.imarr(pol='I')
	im.set_array(ma.log10(frame)[100:size-100,:])
	return im,

ani = animation.FuncAnimation(fig, updatefig, interval=100, blit=True)
plt.show()
