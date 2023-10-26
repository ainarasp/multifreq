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

import ehtim as eh

import emissionanalysis as ea

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def func(file_list,indices):

	freq = 86e9
	
	for i in range(len(file_list)):
		file = ea.loademission('./slowlight/good/' + file_list[i])
		file.movieimages(freq,indices[i],save='yes')


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


if rank==0:

	##load all files
	file_list = []

	for i in os.listdir('./slowlight/good/'):
		if fnmatch.fnmatch(i, 'JETNB*'):
			file_list.append(i)
	file_list.sort()
	print('root',len(file_list))

	# Get last n elements
	file_list = file_list[-4:]
	# Create integer array from 0 to n - 1
	indices = np.arange(len(file_list))

	#create n-size chunks
	fpos_list = list(getchunks_cont(file_list,size))
	fpos_indices = list(getchunks_cont(indices,size))

else:
	##create variables on other cpus
	fpos_list = None
	fpos_indices = None

fpos_list = comm.scatter(fpos_list, root=0)
fpos_indices = comm.scatter(fpos_indices, root=0)


func(fpos_list,fpos_indices)
