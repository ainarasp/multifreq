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

# Try manually ruling out invalid bits of data vs physical arguments (beam size in the center for the emission gap (find min!), low flux at the ends)

###include argument parsing --> for later use
parser = argparse.ArgumentParser(description='test the deconvoluted logwidth plot and do adjustements')

parser.add_argument('freq', metavar='f', type=float, nargs=1,
                    help='frequency to analyze in Hz')

args = parser.parse_args()

# Create average image

file_list = []
for i in os.listdir('./circular/%03dGHz/'%(args.freq[0]/1e9)):
    if fnmatch.fnmatch(i, '*%03dGHz_circ.fits'%(args.freq[0]/1e9)):
        file_list.append(i)
file_list.sort()

image_array = []
for i in range(len(file_list)):
    im = eh.image.load_fits('./circular/%03dGHz/'%(args.freq[0]/1e9) + file_list[i])
    data = im.imarr(pol='I')
    data = ndimage.rotate(data,15,reshape=False)
    image_array.append(data)

frame = np.mean(image_array,axis=0)
#data[data1<1e-2*ma.amax(data)]=0
del image_array

# Pre-blurring
file_list_2 = []
for i in os.listdir('./fits/'):
    if fnmatch.fnmatch(i, '*%03dGHz_mf.fits'%(args.freq[0]/1e9)):
        file_list_2.append(i)
file_list_2.sort()

image_array = []
for i in range(len(file_list_2)):
    im = eh.image.load_fits('./fits/' + file_list_2[i])
    data = im.imarr(pol='I')
    data = ndimage.rotate(data,15,reshape=False)
    image_array.append(data)

frame_nonblur = np.mean(image_array,axis=0)
#data[data1<1e-2*ma.amax(data)]=0
del image_array


# Average width

widths = []
file_list = []
contour_top = []
contour_btm = []
peaks = []
deconvolved_widths = []
beam_width = []
gaussian_zeros = []

# Average beam
obs_list = []
for i in os.listdir('./uvfits/%03dGHz/'%(args.freq[0]/1e9)):
    if fnmatch.fnmatch(i, '*.uvp'):
        obs_list.append(i)
obs_list.sort()

# # rotation angle 15 degrees
for k in os.listdir('./circular_values/%03dGHz/' %(args.freq[0]/1e9)):
    if fnmatch.fnmatch(k, 'NGC1052_t*'):
        file_list.append(k)
file_list.sort()


for i in range(len(file_list)):
    # First get the jet parameters for this timestep
    obs = eh.obsdata.load_uvfits('./uvfits/%03dGHz/'%(args.freq[0]/1e9) + obs_list[i])
    beamparams = obs.fit_beam()
    circular_diameter = sqrt(beamparams[0]*beamparams[1])
    width_beam = circular_diameter/eh.RADPERUAS/1000

    # Load computed values
    data = np.load('./circular_values/%03dGHz/'%(args.freq[0]/1e9)+file_list[i])
    width = data['width']
    widths.append(width)
    peak = data['peaks']
    peaks.append(peak)
    cntr_top = data['contour_top']
    cntr_btm = data['contour_btm']
    x = data['xplt']
    contour_top.append(cntr_top)
    contour_btm.append(cntr_btm)
    gaussian_mean = data['zeros']
    gaussian_zeros.append(gaussian_mean)

    # Deconvolve width
    deconvolved = sqrt(width**2-width_beam**2)
    #deconvolved = np.nan_to_num(deconvolved,nan=width_beam)
    deconvolved_widths.append(deconvolved)
    # Save beam parameters to average and plot
    beam_width.append(width_beam)

avg_width = np.mean(widths,axis=0)
std_width = np.std(widths,axis=0)
avg_peak = np.mean(peaks,axis=0)
std_peak = np.std(peaks,axis=0)
contour_top = np.nanmean(contour_top,axis=0)
contour_btm = np.nanmean(contour_btm,axis=0)
avg_deconvolved = np.nanmean(deconvolved_widths,axis=0)
std_deconvolved = np.nanstd(deconvolved_widths,axis=0)
beam_width = np.mean(beam_width)
gaussian_zeros = np.mean(gaussian_zeros,axis=0)

np.savez('./circular_values/%03dGHz/'%(args.freq[0]/1e9)+'NGC1052_circular_avg_%03dGHz.npz'%(args.freq[0]/1e9),width=avg_width,std_width=std_width,xplt=x,
    peaks=avg_peak,std_peaks=std_peak,contour_top=contour_top,contour_btm=contour_btm,dewidth=avg_deconvolved,std_dewidth=std_deconvolved,
    beam_width=beam_width,zeros=gaussian_zeros)

np.save('./circular_values/%03dGHz/'%(args.freq[0]/1e9)+'NGC1052_circular_frame_%03dGHz.npz'%(args.freq[0]/1e9),frame)
np.save('./circular_values/%03dGHz/'%(args.freq[0]/1e9)+'NGC1052_circular_frame_nonblur_%03dGHz.npz'%(args.freq[0]/1e9),frame_nonblur)

plt.figure()
plt.imshow(frame,extent=[x[0],x[-1],x[-1],x[-0]])
plt.plot(x,contour_top,'r')
plt.plot(x,contour_btm,'r')
ell = mpl.patches.Circle([7.5,-4],radius=beam_width/2)
# negative because Ellipse is clockwise. 90 because it starts horizontally. 15 from rotating the image back
plt.gca().add_patch(ell)
plt.title('Blurred map with width')
plt.colorbar()
plt.show(block=False)

middle = round(avg_width.shape[0]/2)
left = avg_width[0:middle][::-1] # Of blurred jet
right = avg_width[middle:] # Of blurred jet
left_peak = avg_peak[0:middle][::-1] # Of blurred jet
right_peak = avg_peak[middle:] # Of blurred jet
left_jet = avg_deconvolved[0:middle][::-1] # Of deconvolved jet
right_jet = avg_deconvolved[middle:] # Of deconvolved jet
distance = x[0:middle][::-1]
plt.subplots(1,2,sharey=True,sharex=True)
plt.subplot(1,2,1)
plt.loglog(distance,left,label='%02dGHz'%(args.freq[0]/1e9))
plt.fill_between(distance,left+std_width[0:middle][::-1],left-std_width[0:middle][::-1],alpha=0.5)
plt.axhline(beam_width,color='red')
#plt.loglog(distance,left_peak)
plt.ylabel('Jet width[mas]')
plt.xlabel('Distance from core [mas]')
plt.title('Left jet')
plt.legend()
plt.subplot(1,2,2)
plt.loglog(distance,right,label='%02dGHz'%(args.freq[0]/1e9))
plt.fill_between(distance,right+std_width[middle:],right-std_width[middle:],alpha=0.5)
plt.axhline(beam_width,color='red')
#plt.loglog(distance,right_peak)
plt.xlabel('Distance from core [mas]')
plt.title('Right jet')
plt.legend()
plt.suptitle('Width of blurred jet')
plt.show(block=False)

plt.subplots(1,2,sharey=True,sharex=True)
plt.subplot(1,2,1)
plt.loglog(distance,left_jet,label='%02dGHz'%(args.freq[0]/1e9))
plt.fill_between(distance,left_jet+std_deconvolved[0:middle][::-1],left_jet-std_deconvolved[0:middle][::-1],alpha=0.5)
#plt.loglog(distance,left_peak)
plt.ylabel('Jet width[mas]')
plt.xlabel('Distance from core [mas]')
plt.title('Left jet')
plt.legend()
plt.subplot(1,2,2)
plt.loglog(distance,right_jet,label='%02dGHz'%(args.freq[0]/1e9))
plt.fill_between(distance,right_jet+std_deconvolved[middle:],right_jet-std_deconvolved[middle:],alpha=0.5)
#plt.loglog(distance,right_peak)
plt.xlabel('Distance from core [mas]')
plt.title('Right jet')
plt.legend()
plt.suptitle('Deconvolved width of jet')
plt.show(block=False)

# Plot new width on the PRE-BLURRING images

#gaussian_zeros = data['zeros']
preblur_top = np.zeros(frame_nonblur.shape[1])
preblur_btm = np.zeros(frame_nonblur.shape[1])

preblur_top[0:middle] = gaussian_zeros[0:middle] + left_jet[::-1]/2
preblur_top[middle:] = gaussian_zeros[middle:] + right_jet/2
preblur_btm[0:middle] = gaussian_zeros[0:middle] - left_jet[::-1]/2
preblur_btm[middle:] = gaussian_zeros[middle:] - right_jet/2

plt.figure()
plt.imshow(ma.log10(frame_nonblur),extent=[x[0],x[-1],x[-1],x[-0]],vmin=-6)
plt.colorbar()
plt.plot(x,preblur_top,'r')
plt.plot(x,preblur_btm,'r')
ell = mpl.patches.Circle([7.5,-4],radius=beam_width/2)
plt.gca().add_patch(ell)
plt.title('MEM map with deconvolved width')
plt.show()
