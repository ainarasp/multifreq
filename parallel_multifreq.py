#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import string,math,sys,fileinput,glob,os,time
import fnmatch
import sys,glob,argparse
import matplotlib.pyplot as plt

##numpy and scipy packages
from numpy import *
from scipy import *
import numpy as np
import numpy.ma as ma

#load mpi environment
from mpi4py import MPI

#use astropy for fits and source locations
from astropy.time import Time as aTime
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits

import ehtim as eh
from ehtim.calibrating import self_cal as sc
from ehtim.image import blur_mf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# v1 alpha 500
# v2 alpha 1000
# v3 alpha 1
# v4 alpha 0
# v5 alpha 1e6
# v6 FOV 21000 alpha 500
# v7 FOV 21000 alpha 0
# v8 FOV 21000 alpha 1e6

def func(th_list_X,th_list_Y,th_list_J,th_list_K,th_list_L,obs_list_X,obs_list_Y,obs_list_J,obs_list_K,obs_list_L):
	
	# Parameters
	
	ep = 1e-8
	data_term={'vis':20,'amp':20}
	reg_term_mf1 = {'l1':1,'l2_alpha':0,'tv_alpha':.75}
	reg_term_mf2 = {'l1':1,'tv':1,'l2_alpha':0,'tv_alpha':.75}
	mf_which_solve = (1,1,1)
	
	avertime = 10*60
	labellist = ['x','y','j','k','l']
	
	fov = 21000 # try lowering it 21000? 
	npix = 512
	
	outdir = './fits/'

	# condition 1: check that all the file lists of diff frequencies in a single core are of the same size

	if len(th_list_X)==len(th_list_Y)==len(th_list_J)==len(th_list_K)==len(th_list_L)==len(obs_list_X)==len(obs_list_Y)==len(obs_list_J)==len(obs_list_K)==len(obs_list_L):
		for i in range(len(th_list_X)):

			# condition 2: check that you are always working with the same time step
			# -12 takes the NGC1052_t686.63 in NGC1052_t686.63_008GHz.fits
			# -11 for .uvp files

			tmpX=obs_list_X[i].split('/')
			obs_stepX=tmpX[-1][:-11]
			tmpY=obs_list_Y[i].split('/')
			obs_stepY=tmpY[-1][:-11]
			tmpJ=obs_list_J[i].split('/')
			obs_stepJ=tmpJ[-1][:-11]
			tmpK=obs_list_K[i].split('/')
			obs_stepK=tmpK[-1][:-11]
			tmpL=obs_list_L[i].split('/')
			obs_stepL=tmpL[-1][:-11]
			# overwrite for thery files
			tmpX=th_list_X[i].split('/')
			th_stepX=tmpX[-1][:-12]
			tmpY=th_list_Y[i].split('/')
			th_stepY=tmpY[-1][:-12]
			tmpJ=th_list_J[i].split('/')
			th_stepJ=tmpJ[-1][:-12]
			tmpK=th_list_K[i].split('/')
			th_stepK=tmpK[-1][:-12]
			tmpL=th_list_L[i].split('/')
			th_stepL=tmpL[-1][:-12]

			if th_stepX==th_stepY==th_stepJ==th_stepK==th_stepL==obs_stepX==obs_stepY==obs_stepJ==obs_stepK==obs_stepL:

				# conditions right to do the imaging

				# Load the theory
				theory_X = eh.image.load_fits('./model/008GHz/' + th_list_X[i])
				theory_Y = eh.image.load_fits('./model/015GHz/' + th_list_Y[i])
				theory_J = eh.image.load_fits('./model/022GHz/' + th_list_J[i])
				theory_K = eh.image.load_fits('./model/043GHz/' + th_list_K[i])
				theory_L = eh.image.load_fits('./model/086GHz/' + th_list_L[i])

				# Load the observations, average and add noise
				
				obsX = eh.obsdata.load_uvfits('./uvfits/008GHz/' + obs_list_X[i])
				obsY = eh.obsdata.load_uvfits('./uvfits/015GHz/' + obs_list_Y[i])
				obsJ = eh.obsdata.load_uvfits('./uvfits/022GHz/' + obs_list_J[i])
				obsK = eh.obsdata.load_uvfits('./uvfits/043GHz/' + obs_list_K[i])
				obsL = eh.obsdata.load_uvfits('./uvfits/086GHz/' + obs_list_L[i])
				obsX = obsX.avg_coherent(avertime)
				obsY = obsY.avg_coherent(avertime)
				obsJ = obsJ.avg_coherent(avertime)
				obsK = obsK.avg_coherent(avertime)
				obsL = obsL.avg_coherent(avertime)
				obsX = add_noise(obsX,noisefrac=0.01,noisefloor=0.01)
				obsY = add_noise(obsY,noisefrac=0.01,noisefloor=0.01)
				obsJ = add_noise(obsJ,noisefrac=0.01,noisefloor=0.01)
				obsK = add_noise(obsK,noisefrac=0.01,noisefloor=0.01)
				obsL = add_noise(obsL,noisefrac=0.01,noisefloor=0.01)

				outnameX=tmpX[-1][:-5]
				outname=tmpX[-1][:-12] # For the alpha and the beta
				outnameY=tmpY[-1][:-5]
				outnameJ=tmpJ[-1][:-5]
				outnameK=tmpK[-1][:-5]
				outnameL=tmpL[-1][:-5]

				# Baseline flux

				rflist = [obsX.rf, obsY.rf, obsJ.rf, obsK.rf, obsL.rf]
				flux_X = theory_X.total_flux()
				flux_Y = theory_Y.total_flux()
				flux_J = theory_J.total_flux()
				flux_K = theory_K.total_flux()
				flux_L = theory_L.total_flux()
				zbllist = 1.5*np.array([flux_X,flux_Y,flux_J,flux_K,flux_L]) # Watch out for errors in total flux, might have to add a 1.5 factor
				reffreq = obsJ.rf

				# Resolution

				beamparamsX = obsX.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
				print(beamparamsX)
				stop
				resX = obsX.res() # nominal array resolution, 1/longest baseline
				beamparamsY = obsY.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
				resY = obsY.res() # nominal array resolution, 1/longest baseline
				beamparamsJ = obsJ.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
				resJ = obsJ.res() # nominal array resolution, 1/longest baseline
				beamparamsK = obsK.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
				resK = obsK.res() # nominal array resolution, 1/longest baseline
				beamparamsL = obsL.fit_beam() # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
				resL = obsL.res()

				# Construct an intial image

				flatprior =  eh.image.make_square(obsJ, npix, fov*eh.RADPERUAS)
				flatprior.imvec += 1.e-6
				flatprior = flatprior.add_gauss(zbllist[2],(100*eh.RADPERUAS,10000*eh.RADPERUAS,-15*np.pi/180,0,0))
				flatprior.imvec *= np.sum(zbllist[2])/np.sum(flatprior.imvec)

				#flatprior.save_fits(outdir + source + '_prior.fits')

				# mask upper and lower bands of the image
				bandsize_pixel = 100
				mask = flatprior.imarr()
				mask[0:bandsize_pixel,:]=0
				mask[npix-bandsize_pixel:npix,:]=0
				flatprior.imvec = mask.flatten()


				# determine the unresolved spectral index
				xfit = np.log(np.array(rflist) / reffreq)
				yfit = np.log(np.array(zbllist))
				coeffs = np.polyfit(xfit,yfit,2)
				alpha1 = coeffs[0]
				beta1 = coeffs[1]
				# add the unresolved spectral index to the initial image    
				rprior = flatprior
				rprior = rprior.add_const_mf(alpha1,beta1)
				#rprior.save_fits(outdir + 'priors/' + outnameX + '_prior.fits')
				# set up the imager
				imgr  = eh.imager.Imager([obsX, obsY, obsJ, obsK, obsL], rprior, rprior, zbllist[2],
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

				# SAVE RESULTS
				# cleaned image 
				out.get_image_mf(obsX.rf).save_fits(outdir + outnameX +'_mf.fits')
				out.get_image_mf(obsY.rf).save_fits(outdir + outnameY +'_mf.fits')
				out.get_image_mf(obsJ.rf).save_fits(outdir + outnameJ +'_mf.fits')
				out.get_image_mf(obsK.rf).save_fits(outdir + outnameK +'_mf.fits')
				out.get_image_mf(obsL.rf).save_fits(outdir + outnameL +'_mf.fits')

				# cleaned and blurred image
				out.get_image_mf(obsX.rf).blur_gauss(beamparamsX,1).save_fits(outdir + outnameX +'_mf_blur.fits')
				out.get_image_mf(obsY.rf).blur_gauss(beamparamsY,1).save_fits(outdir + outnameY +'_mf_blur.fits')
				out.get_image_mf(obsJ.rf).blur_gauss(beamparamsJ,1).save_fits(outdir + outnameJ +'_mf_blur.fits')
				out.get_image_mf(obsK.rf).blur_gauss(beamparamsK,1).save_fits(outdir + outnameK +'_mf_blur.fits')
				out.get_image_mf(obsL.rf).blur_gauss(beamparamsL,1).save_fits(outdir + outnameL +'_mf_blur.fits')

				## alpha and beta
				tmp = out.get_image_mf(obsX.rf).copy()
				tmp.imvec = out._mflist[0]
				tmp.save_fits(outdir + outname + '_alpha.fits')
				tmp.imvec = out._mflist[1]
				tmp.save_fits(outdir + outname + '_beta.fits')

				# resize and save theory images
				theory_fov = theory_X.fovx()
				npix_theory = theory_X.xdim
				fov_rad = fov*eh.RADPERUAS

				# save resized theory images
				theory_X.regrid_image(fov_rad,npix_theory) # choose largest one? generally it should be theory
				theory_X.save_fits(outdir + outnameX + '_theory.fits')
				theory_Y.regrid_image(fov_rad,npix_theory) 
				theory_Y.save_fits(outdir + outnameY + '_theory.fits')
				theory_J.regrid_image(fov_rad,npix_theory) 
				theory_J.save_fits(outdir + outnameJ + '_theory.fits')
				theory_K.regrid_image(fov_rad,npix_theory) 
				theory_K.save_fits(outdir + outnameK + '_theory.fits')
				theory_L.regrid_image(fov_rad,npix_theory) 
				theory_L.save_fits(outdir + outnameL + '_theory.fits')   

				# save blurred theory images
				theory_X.blur_gauss(beamparamsX,1).save_fits(outdir + outnameX + '_theory_blur.fits')
				theory_Y.blur_gauss(beamparamsY,1).save_fits(outdir + outnameY + '_theory_blur.fits')
				theory_J.blur_gauss(beamparamsJ,1).save_fits(outdir + outnameJ + '_theory_blur.fits')
				theory_K.blur_gauss(beamparamsK,1).save_fits(outdir + outnameK + '_theory_blur.fits')
				theory_L.blur_gauss(beamparamsL,1).save_fits(outdir + outnameL + '_theory_blur.fits')

			else:
				print('Error! Files not properly ordered.')
				print('8GHz theory: ' + th_list_X)
				print('15GHz theory: ' + th_list_Y)
				print('22GHz theory: ' + th_list_J)
				print('43GHz theory: ' + th_list_K)
				print('86GHz theory: ' + th_list_L)
				print('8GHz obs: ' + obs_list_X)
				print('15GHz obs: ' + obs_list_Y)
				print('22GHz obs: ' + obs_list_J)
				print('43GHz obs: ' + obs_list_K)
				print('86GHz obs: ' + obs_list_L)
				stop

	else:
		print('Error! Files divided into chunks of unequal sizes.')
		print(len(th_list_X))
		print(len(th_list_Y))
		print(len(th_list_J))
		print(len(th_list_K))
		print(len(th_list_L))
		print(len(obs_list_X))
		print(len(obs_list_Y))
		print(len(obs_list_J))
		print(len(obs_list_K))
		print(len(obs_list_L))
		stop

		# add additional condition for timestamps not being the same


def getchunks_cont(items, maxbaskets=3, item_count=None):
	'''
	generates balanced baskets from iterable, contiguous contents
	provide item_count if providing a iterator that doesn't support len()
	'''
	item_count = item_count or len(items)
	baskets = min(item_count, maxbaskets)
	items = iter(items)
	floor = item_count // baskets 
	ceiling = floor + 1
	stepdown = item_count % baskets
	for x_i in range(baskets):
		length = ceiling if x_i < stepdown else floor
		yield [items.__next__() for _ in range(length)]

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


if rank==0:

	##load all files
	th_list_X = []
	th_list_Y = []
	th_list_J = []
	th_list_K = []
	th_list_L = []

	obs_list_X = []
	obs_list_Y = []
	obs_list_J = []
	obs_list_K = []
	obs_list_L = []

	for i in os.listdir('./model/008GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			th_list_X.append(i)
	th_list_X.sort()
	print('root',len(th_list_X))
	#create n-size chunks
	fpos_th_X=list(getchunks_cont(th_list_X,size))

	for i in os.listdir('./model/015GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			th_list_Y.append(i)
	th_list_Y.sort()
	print('root',len(th_list_Y))
	#create n-size chunks
	fpos_th_Y=list(getchunks_cont(th_list_Y,size))

	for i in os.listdir('./model/022GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			th_list_J.append(i)
	th_list_J.sort()
	print('root',len(th_list_J))
	#create n-size chunks
	fpos_th_J=list(getchunks_cont(th_list_J,size))

	for i in os.listdir('./model/043GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			th_list_K.append(i)
	th_list_K.sort()
	print('root',len(th_list_K))
	#create n-size chunks
	fpos_th_K=list(getchunks_cont(th_list_K,size))

	for i in os.listdir('./model/086GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			th_list_L.append(i)
	th_list_L.sort()
	print('root',len(th_list_L))
	#create n-size chunks
	fpos_th_L=list(getchunks_cont(th_list_L,size))

	for i in os.listdir('./uvfits/008GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			obs_list_X.append(i)
	obs_list_X.sort()
	print('root',len(obs_list_X))
	#create n-size chunks
	fpos_obs_X=list(getchunks_cont(obs_list_X,size))

	for i in os.listdir('./uvfits/015GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			obs_list_Y.append(i)
	obs_list_Y.sort()
	print('root',len(obs_list_Y))
	#create n-size chunks
	fpos_obs_Y=list(getchunks_cont(obs_list_Y,size))

	for i in os.listdir('./uvfits/022GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			obs_list_J.append(i)
	obs_list_J.sort()
	print('root',len(obs_list_J))
	#create n-size chunks
	fpos_obs_J=list(getchunks_cont(obs_list_J,size))

	for i in os.listdir('./uvfits/043GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			obs_list_K.append(i)
	obs_list_K.sort()
	print('root',len(obs_list_K))
	#create n-size chunks
	fpos_obs_K=list(getchunks_cont(obs_list_K,size))

	for i in os.listdir('./uvfits/086GHz/'):
		if fnmatch.fnmatch(i, 'NGC1052*'):
			obs_list_L.append(i)
	obs_list_L.sort()
	print('root',len(obs_list_L))
	#create n-size chunks
	fpos_obs_L=list(getchunks_cont(obs_list_L,size))

else:
	##create variables on other cpus
	fpos_th_X=None
	fpos_th_Y=None
	fpos_th_J=None
	fpos_th_K=None
	fpos_th_L=None
	fpos_obs_X=None
	fpos_obs_Y=None
	fpos_obs_J=None
	fpos_obs_K=None
	fpos_obs_L=None

fpos_th_X = comm.scatter(fpos_th_X, root=0)
fpos_th_Y = comm.scatter(fpos_th_Y, root=0)
fpos_th_J = comm.scatter(fpos_th_J, root=0)
fpos_th_K = comm.scatter(fpos_th_K, root=0)
fpos_th_L = comm.scatter(fpos_th_L, root=0)
fpos_obs_X = comm.scatter(fpos_obs_X, root=0)
fpos_obs_Y = comm.scatter(fpos_obs_Y, root=0)
fpos_obs_J = comm.scatter(fpos_obs_J, root=0)
fpos_obs_K = comm.scatter(fpos_obs_K, root=0)
fpos_obs_L = comm.scatter(fpos_obs_L, root=0)

func(fpos_th_X,fpos_th_Y,fpos_th_J,fpos_th_K,fpos_th_L,fpos_obs_X,fpos_obs_Y,fpos_obs_J,fpos_obs_K,fpos_obs_L)
