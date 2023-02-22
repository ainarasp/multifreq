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

# This script shows the width calculation for a single slice - use for troubleshooting purposes


def gauss(x, A, x0, sigma):
    y = (A/sqrt(2*np.pi*sigma**2)) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    return y

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='compute and save width and peak flux information')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency in Hz')

args = parser.parse_args()


# Load files and compute width

file_list = []
for i in os.listdir('./fits/'):
    if fnmatch.fnmatch(i, '*%03dGHz_mf_blur.fits'%(args.freq[0]/1e9)):
        file_list.append(i)
file_list.sort()

### DO ONLY ON THE FIRST FRAME FOR NOW

im = eh.image.load_fits('./fits/' + file_list[0])
data = im.imarr(pol='I')
data = ndimage.rotate(data,15,reshape=False)
x = linspace(data.shape[0]/2*im.psize/eh.RADPERUAS/1000,-data.shape[0]/2*im.psize/eh.RADPERUAS/1000,data.shape[0])
print(x)
contour_top = np.zeros(data.shape[1])
contour_btm = np.zeros(data.shape[1])
width = np.zeros(data.shape[1])
gauss_fits = np.zeros([data.shape[1],data.shape[0]])
gauss_zeros = np.zeros(data.shape[1])

for i in range(data.shape[1]):
	vert_slice = ma.fix_invalid(data[:,i])
	avg = ma.mean(vert_slice)
	parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[avg,0,1])
	y = gauss(x,parameters[0],parameters[1],parameters[2])
	gauss_zeros[i] = parameters[1]
	gauss_fits[i,:] = y
	fwhm = 2*sqrt(2*log(2))*abs(parameters[2])
	width[i] = fwhm
	# For the contour
	contour_top[i] = parameters[1] + fwhm/2
	contour_btm[i] = parameters[1] - fwhm/2

plt.figure()
plt.imshow(data,extent=[x[0],x[-1],x[-1],x[-0]]) # make quality check at around ~160 index
plt.plot(x,contour_top,'r')
plt.plot(x,contour_btm,'r')
plt.plot(x,gauss_zeros,'w',linewidth=1)
plt.show(block=False)

# Look at specific slice
x_value = 1.13# Approximate value
idx = (np.abs(x - x_value)).argmin()
vert_slice = ma.fix_invalid(data[:,idx])
parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[0.1,0,1])
y = gauss(x,parameters[0],parameters[1],parameters[2])
plt.figure()
plt.plot(x,vert_slice,label='Data')
plt.plot(x,y,label='Gaussian fit')
plt.axvline(0,label='Zero',color='green')
plt.axvline(parameters[1],label='x0',color='black')
plt.legend()
plt.show()

stop


# TEST 1: Quality check. Around the areas where it seems like the flux is not too low
stop
choose = [160,190,220,250,280,310,340]
x_dim = x[choose]
for i in range(len(choose)):
	index = choose[i]
	oned = data[:,index]
	parameters, covariance = curve_fit(gauss, x, oned, p0=[avg,0,1])
	y = gauss(x,parameters[0],parameters[1],parameters[2])
	plt.plot(x,oned)
	plt.plot(x,y)
	plt.show()

stop

# TEST 2: when is the flux too low?

plt.loglog(x,width)
plt.show()


plt.semilogy(x,data[256,:])
plt.show()
plt.semilogy(x,data[256,:]/max(data[256,:]))
plt.show()
plt.semilogy(x,data[256,:]/max(data[256,:]))
plt.semilogy(x,width/max(width))
plt.show()




stop

# Average of all frames
#im = eh.image.load_fits('./fits/%s_avg_%03dGHz_reconsblur-1.00_00.fits' %(args.source[0],args.freq[0]/1e9))
# A random one
im = eh.image.load_fits('./fits/%03dGHz/'%(args.freq[0]/1e9)+file_list[0])
data = im.imarr(pol='I')
hdul = fits.open('./fits/%03dGHz/'%(args.freq[0]/1e9)+file_list[0])
hdr = hdul[0].header
beamx = hdr['beamx']
print(beamx/1000)
hdul.close()
#x = linspace(0,data.shape[0]-1,data.shape[0])
x = linspace(data.shape[0]/2*im.psize/eh.RADPERUAS/1000,-data.shape[0]/2*im.psize/eh.RADPERUAS/1000,data.shape[0])

methods = []


for i in range(len(file_list)):
	im = eh.image.load_fits('./fits/%03dGHz/'%(args.freq[0]/1e9)+file_list[i])
	data = im.imarr(pol='I')
	data = ndimage.rotate(data, 15, reshape=False,cval=0)
	contour_top,contour_btm,width,fit_peaks,fittings = computewidth(data,x,methods,beamx)
	plt.imshow(data,vmin=1e-4,extent=[x[0],x[-1],x[-1],x[-0]])
	plt.colorbar()
	plt.plot(x,contour_top,'r')
	plt.plot(x,contour_btm,'r')
	plt.savefig('./regularization/%03dGHz/tv2_rotated/%02d_map' %(args.freq[0]/1e9,i))
	plt.show()

	plt.semilogy(x,data[256,:]/max(data[256,:]))
	plt.savefig('./regularization/%03dGHz/tv2_rotated/%02d_flux' %(args.freq[0]/1e9,i))
	plt.show()

	plt.plot(x,width)
	plt.show()
		
	#plt.plot(x,fit_peaks)
	#plt.show()

	#np.savez('./values/%03dGHz/'%(args.freq[0]/1e9)+file_list[i][0:-5]+'.npz',width=width,peaks=fit_peaks,xplt=x,beam=beamx,freq=args.freq[0])

# Look at normalized flux in the spine
stop

# Look at specific slice
x_value = 1.14# Approximate value
idx = (np.abs(x - x_value)).argmin()
vert_slice = ma.fix_invalid(data[:,idx])
parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[0.1,0,1])
y = gauss(x,parameters[0],parameters[1],parameters[2])
plt.semilogy(x,vert_slice)
plt.semilogy(x,fittings[idx,:])
plt.show()


def computewidth(data,x,methods,beam):

	contour_top = np.zeros(data.shape[1])
	contour_btm = np.zeros(data.shape[1])
	width = np.zeros(data.shape[1])
	gauss_fits = np.zeros([data.shape[1],data.shape[0]])
	peak = np.amax(data)
	fit_peaks = np.zeros(data.shape[1])
	beamfactor = 1

	if 'gap' in methods:
		half = int(data.shape[1]/2)#data[int(data.shape[0]/2)
		max_left = np.unravel_index(np.argmax(data[:,0:half]),data[:,0:half].shape) # Indices of maxima
		max_right = np.unravel_index(np.argmax(data[:,half:data.shape[1]]),data[:,half:data.shape[1]].shape)
		beam = beam/1000 # microarcsecs to mas
		xmax_left = x[max_left[1]] # x coordinates of maxima
		xmax_right = x[max_right[1]+half]
		idx_left = (np.abs(x - (xmax_left - beamfactor*beam))).argmin() # Index of left boundary
		idx_right = (np.abs(x - (xmax_right + beamfactor*beam))).argmin() # Remember x is inverted! It should go inwards
		for i in range(data.shape[1]):
			if i < idx_left or i > idx_right:
				vert_slice = ma.fix_invalid(data[:,i])
				avg = ma.mean(vert_slice)
				parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[avg,0,1], maxfev=5000)
				y = gauss(x,parameters[0],parameters[1],parameters[2])
				gauss_fits[i,:] = y
				fwhm = 2*sqrt(2*log(2))*abs(parameters[2])
				width[i] = fwhm
				fit_peaks[i] = max(y)
				# For the contour
				contour_top[i] = parameters[1] + fwhm/2
				contour_btm[i] = parameters[1] - fwhm/2
	else:
		if 'mingap' in methods:
			half = int(data.shape[1]/2)#data[int(data.shape[0]/2)
			max_left = int(np.unravel_index(np.argmax(data[:,0:half]),data[:,0:half].shape)[1]) # Indices of maxima
			max_right = int(np.unravel_index(np.argmax(data[:,half:data.shape[1]]),data[:,half:data.shape[1]].shape)[1]) + half
			x_min = np.argmin(data[half,max_left:max_right]) # Search for the minimum in the jet axis
			central_min = x_min + max_left
			beam = beam/1000 # microarcsecs to mas
			xmin = x[central_min]
			idx_left = (np.abs(x - (xmin + beamfactor*beam/2))).argmin() # Index of left boundary
			idx_right = (np.abs(x - (xmin - beamfactor*beam/2))).argmin() # Remember x is inverted!
			for i in range(data.shape[1]):
				if i < idx_left or i > idx_right:
					vert_slice = ma.fix_invalid(data[:,i])
					avg = ma.mean(vert_slice)
					parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[avg,0,1],maxfev=5000)
					y = gauss(x,parameters[0],parameters[1],parameters[2])
					gauss_fits[i,:] = y
					fwhm = 2*sqrt(2*log(2))*abs(parameters[2])
					width[i] = fwhm
					fit_peaks[i] = max(y)
					# For the contour
					contour_top[i] = parameters[1] + fwhm/2
					contour_btm[i] = parameters[1] - fwhm/2
		else:
			for i in range(data.shape[1]):
				vert_slice = ma.fix_invalid(data[:,i])
				avg = ma.mean(vert_slice)
				parameters, covariance = curve_fit(gauss, x, vert_slice, p0=[avg,0,1],maxfev=5000)
				y = gauss(x,parameters[0],parameters[1],parameters[2])
				gauss_fits[i,:] = y
				fwhm = 2*sqrt(2*log(2))*abs(parameters[2])
				width[i] = fwhm
				fit_peaks[i] = max(y)
				# For the contour
				contour_top[i] = parameters[1] + fwhm/2
				contour_btm[i] = parameters[1] - fwhm/2

	if 'minpeak' in methods:
		for i in range(data.shape[1]):
			gaussian_fit= gauss_fits[i,:]
			if max(gaussian_fit) < peak/30:
				width[i] = 0
				contour_top[i] = 0
				contour_btm[i] = 0
	
	

	#plt.semilogy(x,data[256,:]/max(data[256,:]))
	#plt.axvline(x[idx_left])
	#plt.axvline(x[idx_right])
	#plt.show()
	#plt.savefig('./regularization/simple/%02d_flux' %(i))

	return contour_top, contour_btm, width, fit_peaks, gauss_fits