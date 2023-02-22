# Multifrequency imager on MOJAVE datasets
# Produce results for figures 11-12
# Andrew Chael, October 2022

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from ehtim.calibrating import self_cal as sc
from ehtim.image import blur_mf
import time

def add_noise(obs,noisefrac=0,noisefloor=0):
   ##add noise to data
   systematic_noise=dict(zip(obs.tarr['site'],[noisefrac for i in range(len(obs.tarr['site']))]))
    
   ##include station dependen systematic noise in the data
   for d in obs.data:
      for key in systematic_noise:
         if d[2]==key or d[3]==key:
            d[-4] = (d[-4]**2 + np.abs(systematic_noise[key]*d[-8])**2+noisefloor**2)**0.5
            d[-3] = (d[-3]**2 + np.abs(systematic_noise[key]*d[-8])**2+noisefloor**2)**0.5
            d[-2] = (d[-2]**2 + np.abs(systematic_noise[key]*d[-8])**2+noisefloor**2)**0.5
            d[-1] = (d[-1]**2 + np.abs(systematic_noise[key]*d[-8])**2+noisefloor**2)**0.5


   ##include station dependen systematic noise in the data
   obs.reorder_baselines()

   return obs

plt.close('all')

# data and regularizer terms
ep = 1e-8
#data_term={'cphase':20, 'logcamp':20,'amp':1}
#data_term2={'cphase':20, 'logcamp':1,'amp':10}
#data_term3={'cphase':20, 'amp':20}
data_term={'cphase':20,'amp':20}


reg_term_mf1 = {'l1':1,'l2_alpha':0,'tv_alpha':.75}
reg_term_mf2 = {'l1':1,'tv':1,'l2_alpha':0,'tv_alpha':.75}
mf_which_solve = (1,1,0)

# PUT READING OF FILES HERE
#############################################

theory_X = eh.image.load_fits('./model/022GHz/NGC1052_t686.63_022GHz.fits')
theory_Y = eh.image.load_fits('./model/043GHz/NGC1052_t686.63_043GHz.fits')
theory_J = eh.image.load_fits('./model/086GHz/NGC1052_t686.63_086GHz.fits')

# use image.regrid with fov of the reconstruction and the larger npix
# load_fits -> data.fovx() and data.xdim()

# output directory
outdir = './output/' 

#######################################################
# Load the observations
#######################################################
avertime = 30*60
obsX = eh.obsdata.load_uvfits('./uvfits/NGC1052_t686.63_022GHz.uvp')
obsY = eh.obsdata.load_uvfits('./uvfits/NGC1052_t686.63_043GHz.uvp')
obsJ = eh.obsdata.load_uvfits('./uvfits/NGC1052_t686.63_086GHz.uvp')
obsX = obsX.avg_coherent(avertime)
obsY = obsY.avg_coherent(avertime)
obsJ = obsJ.avg_coherent(avertime)
obsX = add_noise(obsX,noisefrac=0.01,noisefloor=0.01)
obsY = add_noise(obsY,noisefrac=0.01,noisefloor=0.01)
obsJ = add_noise(obsJ,noisefrac=0.01,noisefloor=0.01)
# re-scale the noise to ensure correct statistics on closure triangles
# rescaling factors can be obtained from obs.estimate_noise_rescale_factor()
# but this takes a very long time on these large datasets
#obsX.data['sigma'] *= 1
#obsY.data['sigma'] *= 1.517

# apply post-priori amplitude scaling to the 12.1 GHz data
# Source: Matt Lister, private communication
#obsJ.data['vis'] *= 1.1
#obsJ.data['sigma'] *= 1.1
    
# shift the visibilitity phases/the image centroid
#shift =  -10000*eh.RADPERUAS/2.
#obsX.data['vis'] *= np.exp(1j*2*np.pi*shift*obsX.data['u'])
#obsY.data['vis'] *= np.exp(1j*2*np.pi*shift*obsY.data['u'])
#obsJ.data['vis'] *= np.exp(1j*2*np.pi*shift*obsJ.data['u'])

source = 'NGC1052'

rflist = [obsX.rf, obsY.rf, obsJ.rf]
labellist = ['x','y','j']

#######################################################
# flux, resolution and prior
#######################################################

####
# 8GHz ~0.2e8
# 12GHz ~0.2e8
# 15GHz ~0.2e8
# 22GHz ~0.2e8
# 43GHz ~0.2e8
# 86GHz ~1.2e8

#obsX.plotall('uvdist','amp')
#plt.show()

#stop

# zero baseline flux from short-baseline visibility median
zblJ = np.median(obsJ.flag_uvdist(uv_min=1.2e8,output='flagged').unpack(['amp'])['amp'])
zblY = np.median(obsY.flag_uvdist(uv_min=0.2e8,output='flagged').unpack(['amp'])['amp'])
zblX = np.median(obsX.flag_uvdist(uv_min=0.2e8,output='flagged').unpack(['amp'])['amp'])
zbllist = [zblX,zblY,zblJ]
reffreq = obsY.rf

# resolution
beamparamsX = obsX.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
resX = obsX.res() # nominal array resolution, 1/longest baseline
beamparamsY = obsY.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
resY = obsY.res() # nominal array resolution, 1/longest baseline
beamparamsJ = obsJ.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
resJ = obsJ.res() # nominal array resolution, 1/longest baseline
print("Nominal Resolution: " ,resJ,resY,resX)

# Construct an intial image
# fov for 8 to 15GHz -> 24000 (no beta)

fov = 17000 #try 20000
npix = 300

flatprior =  eh.image.make_square(obsY, npix, fov*eh.RADPERUAS)
flatprior.imvec += 1.e-6
flatprior = flatprior.add_gauss(zblY,(200*eh.RADPERUAS,17000*eh.RADPERUAS,0,0,0)) #keep in full field
flatprior.imvec *= np.sum(zblY)/np.sum(flatprior.imvec)

flatprior.save_fits(outdir + source +'_prior_low.fits')

# mask upper and lower bands of the image
bandsize_pixel = 100
mask = flatprior.imarr()
mask[0:bandsize_pixel,:]=0
mask[npix-bandsize_pixel:npix,:]=0
flatprior.imvec = mask.flatten()
        
#####################################################################################
# Image three frequencies together with spectral index
#####################################################################################
# determine the unresolved spectral index
xfit = np.log(np.array(rflist) / reffreq)
yfit = np.log(np.array(zbllist))
coeffs = np.polyfit(xfit,yfit,2)
#plt.plot(xfit,yfit)
#plt.
#plt.show()
#stop
alpha1 = coeffs[0]
beta1 = coeffs[1]
# add the unresolved spectral index to the initial image    
rprior = flatprior
#rprior = rprior.add_const_mf(0.,0.)
rprior = rprior.add_const_mf(alpha1,beta1)
# set up the imager
imgr  = eh.imager.Imager([obsX, obsY, obsJ], rprior, rprior, zblY,
                         data_term = data_term,
                         reg_term=reg_term_mf1,
                         mf_which_solve=mf_which_solve,
                         show_updates=False,norm_reg=True,
                         maxit=100, ttype='fast',
                         clipfloor=0)
imgr.regparams['epsilon_tv'] = ep

# blur and reimage
for i in range(3): 
    imgr.make_image_I(mf=True,show_updates=False)
    out = imgr.out_last()
    imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
    imgr.maxit_next *= 2
    imgr.maxit_next = np.min((imgr.maxit_next,10000))
    imgr.reg_term_next=reg_term_mf2
    
# self-calibrate
out = imgr.out_last()
#plt.imshow(out._mflist[0].reshape((300,300)))
#plt.colorbar()
#plt.show()
#plt.imshow(out._mflist[1].reshape((300,300)))
#plt.colorbar()
#plt.show()
#stop
#imX_sc = out0.get_image_mf(obsX.rf)
#imY_sc = out0.get_image_mf(obsY.rf)
#imJ_sc = out0.get_image_mf(obsJ.rf)

#obsX_sc = eh.self_cal.self_cal(obsX, imX_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)
##obsY_sc = eh.self_cal.self_cal(obsY, imY_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)
#obsJ_sc = eh.self_cal.self_cal(obsJ, imJ_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)

# reimage with visibility amplitudes
#rprior2 = out0.blur_circ(resX)
#imgr  = eh.imager.Imager([obsX, obsY, obsJ], rprior2, rprior2, zblY,
#                        data_term=data_term2,
#                        reg_term=reg_term_mf2,
#                        mf_which_solve=mf_which_solve,
#                        show_updates=False,norm_reg=True,
#                        maxit=100, ttype='fast',
#                        clipfloor=0)
#imgr.regparams['epsilon_tv'] = ep

#for i in range(3): 
#   imgr.make_image_I(mf=True,show_updates=False)
#   out = imgr.out_last()
#   imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
#   imgr.maxit_next *= 2
#   imgr.maxit_next = np.min((imgr.maxit_next,10000))

# self-calibrate
#out1 = imgr.out_last()
#imX_sc = out1.get_image_mf(obsX.rf)
#imY_sc = out1.get_image_mf(obsY.rf)
#imJ_sc = out1.get_image_mf(obsJ.rf)

#obsX_sc = eh.self_cal.self_cal(obsX, imX_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)
##obsY_sc = eh.self_cal.self_cal(obsY, imY_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)
#obsJ_sc = eh.self_cal.self_cal(obsJ, imJ_sc, method='both',
#                                processes=8,ttype='fast',use_grad=True)

# reimage with complex visibilities
#rprior3 = out1.blur_circ(resX)
#imgr  = eh.imager.Imager([obsX, obsY, obsJ], rprior3, rprior3, zblY,
#                        data_term=data_term3,
#                        reg_term=reg_term_mf2,
#                        mf_which_solve=mf_which_solve,
#                        show_updates=False,norm_reg=True,
#                        maxit=100, ttype='fast',
#                        clipfloor=0)
#imgr.regparams['epsilon_tv'] = ep

#for i in range(2): # blur and reimage
#   print(imgr.maxit_next)
#   imgr.make_image_I(mf=True,show_updates=False)
#   out = imgr.out_last()
#   imgr.init_next = blur_mf(out, rflist, resX, fit_order=1)
#   imgr.maxit_next *= 2
#   imgr.maxit_next = np.min((imgr.maxit_next,10000))
   
# save results
#out = imgr.out_last()    
out.get_image_mf(obsX.rf).save_fits(outdir + source +'_22_mf_default.fits')
out.get_image_mf(obsY.rf).save_fits(outdir + source +'_43_mf_default.fits')
out.get_image_mf(obsJ.rf).save_fits(outdir + source +'_86_mf_default.fits')

out.get_image_mf(obsX.rf).blur_circ(resX).save_fits(outdir + source +'_22_mf_blur_default.fits')
out.get_image_mf(obsY.rf).blur_circ(resY).save_fits(outdir + source +'_43_mf_blur_default.fits')
out.get_image_mf(obsJ.rf).blur_circ(resJ).save_fits(outdir + source +'_86_mf_blur_default.fits')

plt.close('all')

## RESCALE HERE

# use image.regrid with fov of the reconstruction and the larger npix
# load_fits -> data.fovx() and data.xdim()

theory_fov = theory_X.fovx()
npix_theory = theory_X.xdim
fov_rad = fov*eh.RADPERUAS

theory_X.regrid_image(fov_rad,npix_theory) # choose largest one? generally it should be theory
theory_X.save_fits(outdir + source + '_22_theory.fits')
theory_Y.regrid_image(fov_rad,npix_theory) 
theory_Y.save_fits(outdir + source + '_43_theory.fits')
theory_J.regrid_image(fov_rad,npix_theory) 
theory_J.save_fits(outdir + source + '_86_theory.fits')
# merge them together