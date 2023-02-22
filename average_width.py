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

# To create and save width, peak flux, physical dimensions

def gauss(x, A, x0, sigma):
    y = (A/sqrt(2*np.pi*sigma**2)) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return y

def computewidth(data,x):

	contour_top = np.zeros(data.shape[1])
	contour_btm = np.zeros(data.shape[1])
	gaussian_zeros = np.zeros(data.shape[1])
	width = np.zeros(data.shape[1])
	gauss_fits = np.zeros([data.shape[1],data.shape[0]])
	peak = np.amax(data)
	fit_peaks = np.zeros(data.shape[1])

	for i in range(data.shape[1]):
		vert_slice = ma.fix_invalid(data[:,i])
		avg = ma.mean(vert_slice)
		parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[avg,0,1],maxfev=4000)
		y = gauss(x,parameters[0],parameters[1],parameters[2])
		gauss_fits[i,:] = y
		fwhm = 2*sqrt(2*log(2))*abs(parameters[2])
		width[i] = fwhm
		fit_peaks[i] = max(y)
		# For the contour
		contour_top[i] = parameters[1] + fwhm/2
		contour_btm[i] = parameters[1] - fwhm/2
		gaussian_zeros[i] = parameters[1]

	return contour_top, contour_btm, width, fit_peaks, gauss_fits, gaussian_zeros

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='compute and save width and peak flux information')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency in Hz')

args = parser.parse_args()


# Load files and compute width

file_list = []
for i in os.listdir('./circular/%03dGHz/'%(args.freq[0]/1e9)):
    if fnmatch.fnmatch(i, '*%03dGHz_circ.fits'%(args.freq[0]/1e9)):
        file_list.append(i)
file_list.sort()

im = eh.image.load_fits('./circular/%03dGHz/'%(args.freq[0]/1e9) + file_list[0])
data = im.imarr(pol='I')
data = ndimage.rotate(data,15,reshape=False)
x = linspace(data.shape[0]/2*im.psize/eh.RADPERUAS/1000,-data.shape[0]/2*im.psize/eh.RADPERUAS/1000,data.shape[0])

for i in range(len(file_list)):
	im = eh.image.load_fits('./circular/%03dGHz/'%(args.freq[0]/1e9)+file_list[i])
	data = im.imarr(pol='I')
	data = ndimage.rotate(data, 15, reshape=False,cval=0)
	contour_top,contour_btm,width,fit_peaks,fittings,zeros = computewidth(data,x)

	np.savez('./circular_values/%03dGHz/'%(args.freq[0]/1e9)+file_list[i][0:-5]+'.npz',width=width,peaks=fit_peaks,xplt=x,contour_top=contour_top,contour_btm=contour_btm,zeros=zeros)



