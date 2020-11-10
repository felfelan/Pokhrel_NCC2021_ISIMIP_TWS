#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap,shiftgrid
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from collections import OrderedDict
from latlon2xy import cylHLFdegxy
from matplotlib import gridspec
from netCDF4 import Dataset
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import fnmatch
import os

import matplotlib.dates as mdates
import statsmodels.api as sm
import sys

# My module libraries
from ISIMIPmod import Glaciers1xEnlargedMASK, NOgreenlndNOantarcticaMASK

r"""
#%%------------------------------------------------------------------
     Description:   ISIMIP2b Project Figure 1: for multiple/one model
				    

     First Version: 05/29/2018
     Update:        07/08/2020 (Adapting the same code for one model)

     Developer:	    Farshid Felfelani

	 Outstanding Issues:
					1- 
---------------------------------------------------------------------
#%%
"""

figDIR  = '../figs/'
hlfDIR  = '../0.5_hydro_data/'
srcDIR  = '../spatial_mean/'

WeightedMean_CONT = '../WeightedMean_CONT_Masks/woMATSIRO/'
Weighted = True
Median   = False

global_lndMASK = np.fromfile(hlfDIR + 'lndara.GSWP2.hlf',np.float32).byteswap().reshape(360,720) #-180~180 in meter^2
glob_lnd_area = global_lndMASK.sum(1).sum(0)

models        = ['clm45','cwatm','h08','lpjml','mpi-hm','pcr-globwb','watergap2']

GCMs          = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5']
climate       = ['picontrol','historical','rcp26','rcp60']
CONTs         = ['nAmerica','sAmerica','Europe','Africa','Asia','Australia']
CONTs_name    = ['N-America','S-America','Europe','Africa','Asia','Australia']
baseline      = '1976_2005'
midcentury    = '2030_2059'
latecentury   = '2070_2099'

base_sYR      = baseline[0:4]
base_eYR      = baseline[5:9]
mid_sYR       = midcentury[0:4]
mid_eYR       = midcentury[5:9]
late_sYR      = latecentury[0:4]
late_eYR      = latecentury[5:9]

r""" remove glaciers, greenland, and antarctica"""
NOgrnlndNOglcrsNOntctc = Glaciers1xEnlargedMASK() * NOgreenlndNOantarcticaMASK()

r""" Choose between one model e.g., 'lpjml' and 'all' """
whichModel = 'all'

#%%------------------------------------------------------------------ Reading the continental weights for weighted mean

kk = 0
for mod in models:
	for gcm in GCMs:
		if os.path.isfile(WeightedMean_CONT + mod + '_' + gcm + '.npy'):
			print(mod + '_' + gcm)
			wFile0 = np.load(WeightedMean_CONT + mod + '_' + gcm + '.npy').reshape(1,360,720)
			if kk == 0:
				wFile = wFile0
				kk = 1
			else:
				wFile = np.concatenate([wFile, wFile0],axis = 0)

def datConcat(argList):
	print('')
	
	for file in sorted(os.listdir(srcDIR)):
		ii = 0
		for arg in argList:
			ii += 1 if arg in file else 0

		if ii == len(argList): # check if all conditions met
			dat0 = np.ma.masked_less(np.load(srcDIR + file),0.0).reshape(1,360,720)
			print(file)
			if 'gfdl' in file:
				dat = dat0
			else:
				dat = np.ma.concatenate([dat,dat0],axis = 0)

	return dat

#%%------------------------------------------------------------------

if whichModel == 'all':
	for mod in models:
	
		if mod == 'clm45':
			twsMOD_hist   = datConcat([mod, 'historical', baseline,    '_2005soc_'])	
		else:
			twsMOD_hist   = datConcat([mod, 'historical', baseline,    '_histsoc_'])
	
		twsMOD_MID26  = datConcat([mod, '_rcp26_'   , midcentury,  '_2005soc_'])
		twsMOD_LATE26 = datConcat([mod, '_rcp26_'   , latecentury, '_2005soc_'])
		twsMOD_MID60  = datConcat([mod, '_rcp60_'   , midcentury,  '_2005soc_'])
		twsMOD_LATE60 = datConcat([mod, '_rcp60_'   , latecentury, '_2005soc_'])
		
		MID26_DIFF  = twsMOD_MID26  - twsMOD_hist
		LATE26_DIFF = twsMOD_LATE26 - twsMOD_hist
		MID60_DIFF  = twsMOD_MID60  - twsMOD_hist
		LATE60_DIFF = twsMOD_LATE60 - twsMOD_hist
	    
		if mod == 'clm45':
	
			tws_DIFF_TOT_MID26  = deepcopy(MID26_DIFF)
			tws_DIIF_TOT_LATE26 = deepcopy(LATE26_DIFF)
			tws_DIFF_TOT_MID60  = deepcopy(MID60_DIFF)
			tws_DIFF_TOT_LATE60 = deepcopy(LATE60_DIFF)
	
		else:
	
			tws_DIFF_TOT_MID26  = np.ma.concatenate([tws_DIFF_TOT_MID26, deepcopy(MID26_DIFF)],axis = 0)
			tws_DIIF_TOT_LATE26 = np.ma.concatenate([tws_DIIF_TOT_LATE26, deepcopy(LATE26_DIFF)],axis = 0)
			tws_DIFF_TOT_MID60  = np.ma.concatenate([tws_DIFF_TOT_MID60, deepcopy(MID60_DIFF)],axis = 0)
			tws_DIFF_TOT_LATE60 = np.ma.concatenate([tws_DIFF_TOT_LATE60, deepcopy(LATE60_DIFF)],axis = 0)

print('************************************************')

if whichModel != 'all':
	if whichModel == 'clm45':
		twsONE_hist   = datConcat([whichModel, 'historical', baseline,    '_2005soc_'])		
	else:
		twsONE_hist   = datConcat([whichModel, 'historical', baseline,    '_histsoc_'])
	
	
	twsONE_MID26  = datConcat([whichModel, '_rcp26_'   , midcentury,  '_2005soc_'])
	twsONE_LATE26 = datConcat([whichModel, '_rcp26_'   , latecentury, '_2005soc_'])
	twsONE_MID60  = datConcat([whichModel, '_rcp60_'   , midcentury,  '_2005soc_'])
	twsONE_LATE60 = datConcat([whichModel, '_rcp60_'   , latecentury, '_2005soc_'])
	
	ONE_MID26_DIFF  = twsONE_MID26  - twsONE_hist
	ONE_LATE26_DIFF = twsONE_LATE26 - twsONE_hist
	ONE_MID60_DIFF  = twsONE_MID60  - twsONE_hist
	ONE_LATE60_DIFF = twsONE_LATE60 - twsONE_hist

	ONE_MModel_TWS_DIFF = np.ma.concatenate([ONE_MID26_DIFF.mean(0).reshape(1,360,720), ONE_LATE26_DIFF.mean(0).reshape(1,360,720), \
                                         ONE_MID60_DIFF.mean(0).reshape(1,360,720), ONE_LATE60_DIFF.mean(0).reshape(1,360,720)],axis = 0 )

if whichModel == 'all':

	if Weighted:
		Muti_Model_TWS_DIFF = np.ma.concatenate([(tws_DIFF_TOT_MID26 * wFile).sum(0).reshape(1,360,720), (tws_DIIF_TOT_LATE26 * wFile).sum(0).reshape(1,360,720), \
											  (tws_DIFF_TOT_MID60 * wFile).sum(0).reshape(1,360,720), (tws_DIFF_TOT_LATE60 * wFile).sum(0).reshape(1,360,720)],axis = 0 )
		wLabel = 'Weighted'
	
	elif Median:
		Muti_Model_TWS_DIFF = np.ma.concatenate([np.ma.median((tws_DIFF_TOT_MID26), axis =0).reshape(1,360,720), np.ma.median((tws_DIIF_TOT_LATE26), axis =0).reshape(1,360,720), \
											  np.ma.median((tws_DIFF_TOT_MID60), axis =0).reshape(1,360,720), np.ma.median((tws_DIFF_TOT_LATE60), axis =0).reshape(1,360,720)],axis = 0 )
		wLabel = 'Median'
		
	else:
		Muti_Model_TWS_DIFF = np.ma.concatenate([tws_DIFF_TOT_MID26.mean(0).reshape(1,360,720), tws_DIIF_TOT_LATE26.mean(0).reshape(1,360,720), \
											  tws_DIFF_TOT_MID60.mean(0).reshape(1,360,720), tws_DIFF_TOT_LATE60.mean(0).reshape(1,360,720)],axis = 0 )
		wLabel = 'nonWeighted'

#%%------------------------------------------------------------------  Plotting    

titles = ['Mid Century (RCP2.6)', 'Late Century (RCP2.6)', 'Mid Century (RCP6.0)', 'Late Century (RCP6.0)']

gs = gridspec.GridSpec(2, 4, height_ratios = [1,1], width_ratios = [10,2,10,2]) # the middle row is to place the colorbar axes
gs.update(bottom=0.25, top=0.98,wspace=0., hspace = 0.0)
fig = plt.figure(num=1, figsize=(8.5,3.38)) #figsize: w,h letter size

ax = []
axx = []

cmap = plt.cm.jet_r
cmap2 = plt.cm.brg_r
cmap3 = plt.cm.rainbow_r

cmaplist1 = [cmap(i) for i in range(cmap.N)]
cmaplist2 = [cmap2(i) for i in range(cmap2.N)]
cmaplist3 = [cmap3(i) for i in range(cmap3.N)]

cmaplist = np.concatenate([cmaplist1[:-60],cmaplist2[-30:],cmaplist3[-30:]],axis=0)
cmaplist[120:150,] = np.array([.9, .9, .9, 1.]) # for gray

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_tick = np.array([-300,-200,-100,-50,-10,10,50,100,200,300])
bounds      = np.array([-300,-200,-100,-50,-10,10,50,100,200,300])
r""" I changed the ticks labels to string to have variable number of digits to the right of decimal point """
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)



sim = ['rcp26','rcp26','rcp60','rcp60']
timeWindow = [midcentury,latecentury,midcentury,latecentury]
bbox_lft = [0.0,0.08,0.34,0.38,0.92,0.65] #nAmerica, sAmerica,Europe, Africa, Asia, Australia
bbox_bot = [0.4,0.0,0.76,-0.01,0.43,-0.01]
abcd = ['a','b','c','d']
for ii in range(4):
	print('======================== ii =====', ii)
	ax.append(fig.add_subplot(gs[2*ii]))

	nxa=-180
	nxb=180
	nya=90
	nyb=-60
	res=1
	LT1 = 0.3
	LT2 = 0.4
	
	upLatIdx = cylHLFdegxy(nya,nxa)[1]
	loLatIdx = cylHLFdegxy(nyb,nxa)[1]

	map=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			urcrnrlon  = nxb,  \
			llcrnrlat  = nyb,  \
			urcrnrlat  =  nya,  \
			resolution = "c")
	map.drawcoastlines(linewidth=LT1, color='black')
	map.drawcountries(linewidth=LT2, color='grey')

	r""" to cover the previous fake plot with a white background """
	cs_4_backgrnd = map.imshow(np.zeros((loLatIdx,720)),vmin = -2, vmax = 2,cmap = plt.cm.bwr) # for the white background

	if whichModel == 'all':

		TWS_DIFF = Muti_Model_TWS_DIFF[ii,upLatIdx:loLatIdx,:] * NOgrnlndNOglcrsNOntctc[upLatIdx:loLatIdx,:]

	elif whichModel != 'all':

		TWS_DIFF = ONE_MModel_TWS_DIFF[ii,upLatIdx:loLatIdx,:] * NOgrnlndNOglcrsNOntctc[upLatIdx:loLatIdx,:]
		wLabel = whichModel
		
	cs = map.imshow(TWS_DIFF,origin='upper', cmap=cmap, norm=norm, interpolation='nearest')

	ax[-1].set_title('(' + abcd[ii] + ') ' + titles[ii],fontsize=8,y=.92)

	ax[-1].axis('off')

cbar_ax = fig.add_axes([0.33, 0.13, 0.35, 0.04])    
cb = plt.colorbar(cs, cax=cbar_ax, cmap = cmap, norm=norm,spacing='uniform', ticks=bounds_tick, boundaries=bounds, orientation='horizontal')
cbar_ax.set_xticklabels(['<-300','-200','-100','-50','-10','10','50','100','200','>300'])

cb.set_label('($mm$)',fontsize=8)
cb.ax.tick_params(labelsize=8) 

cb.outline.set_visible(False) # make the frame of the colorbar invisible

cb.ax.tick_params(color = 'none') # make the ticks invisible for the bottom most colorbar

plt.savefig(figDIR + 'NCC_Fig1_v1_Global_' + wLabel + 'MEAN_TWS_NOantrctNOgrnlndNOglcrs_xEnlarged_DSI_ZonalMean_2x2.png', bbox_inches='tight', dpi=500)
plt.close()