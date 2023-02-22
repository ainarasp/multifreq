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

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='plotting the results of circular vs elliptical beam')

parser.add_argument('mode', metavar='r', type=int, nargs=1,
                    help='0 to plot together, 1 for ciruclar, 2 for elliptical. 1 and 2 for finding invalid values')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency to analyze in Hz')

args = parser.parse_args()


data_ell = np.load('./values/%03dGHz/NGC1052_avg_%03dGHz.npz'%(args.freq[0]/1e9,args.freq[0]/1e9))
data_circ = np.load('./circular_values/%03dGHz/NGC1052_circular_avg_%03dGHz.npz'%(args.freq[0]/1e9,args.freq[0]/1e9))
frame_ell = np.load('./values/%03dGHz/NGC1052_frame_%03dGHz.npy'%(args.freq[0]/1e9,args.freq[0]/1e9))
frame_circ = np.load('./circular_values/%03dGHz/NGC1052_circular_frame_%03dGHz.npy'%(args.freq[0]/1e9,args.freq[0]/1e9))
frame_ell_nb = np.load('./values/%03dGHz/NGC1052_frame_nonblur_%03dGHz.npy'%(args.freq[0]/1e9,args.freq[0]/1e9))
frame_circ_nb = np.load('./circular_values/%03dGHz/NGC1052_circular_frame_nonblur_%03dGHz.npy'%(args.freq[0]/1e9,args.freq[0]/1e9))

x = data_ell['xplt']
width_ell = data_ell['width']
width_circ = data_circ['width']
deconv_ell = data_ell['dewidth']
deconv_circ = data_circ['dewidth']
std_ell = data_ell['std_width']
std_circ = data_circ['std_width']
std_ell_dc = data_ell['std_dewidth']
std_circ_dc = data_circ['std_dewidth']

zeros_ell = data_ell['zeros']
zeros_circ = data_circ['zeros']

top_ell = data_ell['contour_top']
btm_ell = data_ell['contour_btm']
top_circ = data_circ['contour_top']
btm_circ = data_circ['contour_btm']

middle = round(width_ell.shape[0]/2)
left_ell = width_ell[0:middle][::-1]
right_ell = width_ell[middle:]
left_circ = width_circ[0:middle][::-1]
right_circ = width_circ[middle:]
distance = x[0:middle][::-1]
left_dc_ell = deconv_ell[0:middle][::-1]
right_dc_ell = deconv_ell[middle:]
left_dc_circ = deconv_circ[0:middle][::-1]
right_dc_circ = deconv_circ[middle:]

top_ell_dc = np.zeros(frame_ell.shape[1])
btm_ell_dc = np.zeros(frame_ell.shape[1])

top_ell_dc[0:middle] = zeros_ell[0:middle] + left_dc_ell[::-1]/2
top_ell_dc[middle:] = zeros_ell[middle:] + right_dc_ell/2
btm_ell_dc[0:middle] = zeros_ell[0:middle] - left_dc_ell[::-1]/2
btm_ell_dc[middle:] = zeros_ell[middle:] - right_dc_ell/2

top_circ_dc = np.zeros(frame_ell.shape[1])
btm_circ_dc = np.zeros(frame_ell.shape[1])

top_circ_dc[0:middle] = zeros_circ[0:middle] + left_dc_circ[::-1]/2
top_circ_dc[middle:] = zeros_circ[middle:] + right_dc_circ/2
btm_circ_dc[0:middle] = zeros_circ[0:middle] - left_dc_circ[::-1]/2
btm_circ_dc[middle:] = zeros_circ[middle:] - right_dc_circ/2


if args.mode[0] == 0:
    ## add std to plot?

    plt.subplots(1,2,sharey=True,sharex=True)
    plt.subplot(1,2,1)
    plt.loglog(distance,left_ell,label='Elliptical')
    plt.loglog(distance,left_circ,label='Circular')
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance from core [mas]')
    plt.title('Left jet')
    plt.legend()
    plt.subplot(1,2,2)
    plt.loglog(distance,right_ell)
    plt.loglog(distance,right_circ)
    plt.xlabel('Distance from core [mas]')
    plt.title('Right jet')
    plt.suptitle('Width')
    plt.show(block=False)

    plt.subplots(1,2,sharey=True,sharex=True)
    plt.subplot(1,2,1)
    plt.loglog(distance,left_dc_ell,label='Elliptical')
    plt.loglog(distance,left_dc_circ,label='Circular')
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance from core [mas]')
    plt.title('Left jet')
    plt.legend()
    plt.subplot(1,2,2)
    plt.loglog(distance,right_dc_ell)
    plt.loglog(distance,right_dc_circ)
    plt.xlabel('Distance from core [mas]')
    plt.title('Right jet')
    plt.suptitle('Deconvolved width')
    plt.show(block=False)

    plt.figure()
    plt.imshow(ma.log10(frame_ell_nb),extent=[x[0],x[-1],x[-1],x[-0]],vmin=-5)
    plt.plot(x,top_ell_dc,'r')
    plt.plot(x,btm_ell_dc,'r')
    plt.title('Elliptical map')
    plt.colorbar()
    plt.show(block=False)

    plt.figure()
    plt.imshow(ma.log10(frame_circ_nb),extent=[x[0],x[-1],x[-1],x[-0]],vmin=-5)
    plt.plot(x,top_circ_dc,'r')
    plt.plot(x,btm_circ_dc,'r')
    plt.title('Circular map')
    plt.colorbar()
    plt.show()


if args.mode[0] == 1:
    forbidden_edge = 9 # mas
    forbidden_gap = np.array([[1,1.9],[0,0],[0,0],[0,0],[0,0]]) 
    forbidden_gap = forbidden_gap[4]
    idx_edge = (np.abs(distance - forbidden_edge)).argmin()
    idx_left = (np.abs(distance - forbidden_gap[0])).argmin()
    idx_right = (np.abs(distance - forbidden_gap[1])).argmin()
    print(idx_edge)
    print(idx_left)
    print(idx_right)
    plt.subplots(1,2,sharey=True,sharex=True)
    plt.subplot(1,2,1)
    plt.loglog(distance,left_dc_circ,label='Circular')
    plt.axvline(x=forbidden_edge,color='r',linestyle='--')
    plt.axvline(x=forbidden_gap[0],color='r',linestyle='--')
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance from core [mas]')
    plt.title('Left jet')
    plt.legend()
    plt.subplot(1,2,2)
    plt.loglog(distance,right_dc_circ)
    plt.axvline(x=forbidden_edge,color='r',linestyle='--')
    plt.axvline(x=forbidden_gap[1],color='r',linestyle='--')
    plt.xlabel('Distance from core [mas]')
    plt.title('Right jet')
    plt.suptitle('Deconvolved width')
    plt.show(block=False)

    left_dc_circ[0:idx_left] = np.nan
    right_dc_circ[0:idx_right] = np.nan
    left_dc_circ[idx_edge:] = np.nan
    right_dc_circ[idx_edge:] = np.nan

    peaks = data_circ['peaks']
    peaks_left = peaks[0:middle][::-1]
    peaks_right = peaks[middle:]

    plt.subplots(1,2,sharey=True,sharex=True)
    plt.subplot(1,2,1)
    plt.loglog(distance,left_dc_circ,label='Circular')
    plt.ylabel('Jet width [mas]')
    plt.xlabel('Distance from core [mas]')
    plt.title('Left jet')
    plt.legend()
    plt.subplot(1,2,2)
    plt.loglog(distance,right_dc_circ)
    plt.xlabel('Distance from core [mas]')
    plt.title('Right jet')
    plt.suptitle('Deconvolved width')
    plt.show(block=False)

    plt.subplots(1,2,sharey=True,sharex=True)
    plt.subplot(1,2,1)
    plt.loglog(distance,peaks_left,label='Circular')
    plt.axvline(x=forbidden_edge,color='r',linestyle='--')
    plt.axvline(x=forbidden_gap[0],color='r',linestyle='--')
    plt.ylabel('Peak flux')
    plt.xlabel('Distance from core [mas]')
    plt.title('Left jet')
    plt.legend()
    plt.subplot(1,2,2)
    plt.loglog(distance,peaks_right)
    plt.axvline(x=forbidden_edge,color='r',linestyle='--')
    plt.axvline(x=forbidden_gap[1],color='r',linestyle='--')
    plt.xlabel('Distance from core [mas]')
    plt.title('Right jet')
    plt.suptitle('Peak flux')
    plt.show(block=False)

    plt.figure()
    plt.imshow(ma.log10(frame_circ_nb),extent=[x[0],x[-1],x[-1],x[-0]],vmin=-5)
    plt.plot(x,top_circ_dc,'r')
    plt.plot(x,btm_circ_dc,'r')
    plt.axvline(x=forbidden_edge)
    plt.axvline(x=-forbidden_edge)
    plt.axvline(x=forbidden_gap[0])
    plt.axvline(x=-forbidden_gap[1])
    plt.title('Blurred circular map')
    plt.colorbar()
    plt.show()
    