from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh
from ehtim.calibrating import self_cal as sc
from ehtim.image import blur_mf
import time

theory_X = eh.image.load_fits('./model/005GHz/NGC1052_t686.63_005GHz.fits')
theory_Y = eh.image.load_fits('./model/008GHz/NGC1052_t686.63_008GHz.fits')
theory_J = eh.image.load_fits('./model/015GHz/NGC1052_t686.63_015GHz.fits')
theory_K = eh.image.load_fits('./model/022GHz/NGC1052_t686.63_022GHz.fits')
theory_L = eh.image.load_fits('./model/043GHz/NGC1052_t686.63_043GHz.fits')
theory_M = eh.image.load_fits('./model/086GHz/NGC1052_t686.63_086GHz.fits')
blabla
flux_X = theory_X.total_flux()
flux_Y = theory_Y.total_flux()
flux_J = theory_J.total_flux()
flux_K = theory_K.total_flux()
flux_L = theory_L.total_flux()
flux_M = theory_M.total_flux()

fluxes = [flux_X,flux_Y,flux_J,flux_K,flux_L,flux_M]
print(fluxes)
freqs = [theory_X.rf,theory_Y.rf,theory_J.rf,theory_K.rf,theory_L.rf,theory_M.rf]
plt.plot(freqs,fluxes)
plt.show()