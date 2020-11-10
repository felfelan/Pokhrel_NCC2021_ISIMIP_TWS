#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap,shiftgrid
from matplotlib.ticker import PercentFormatter
from shapely.geometry.polygon import Polygon
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from collections import OrderedDict
from shapely.geometry import Point
from latlon2xy import cylHLFdegxy
from matplotlib import gridspec
from netCDF4 import Dataset
from copy import deepcopy
from pylab import *

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import scipy.stats
import fnmatch
import sys
import os

r"""
#%%------------------------------------------------------------------
     Description:   ISIMIP2b Project Figure 5: Plotting the PDFs
				    

     First Version: 05/29/2018
     Update:        05/29/2018

     Developer:	    Farshid Felfelani

	 Outstanding Issues:
					1- 
---------------------------------------------------------------------
#%%
"""

figDIR        = '../scripts4paper/figs/'
globOUTDIR    = '../out_new/temporal_SREXregions_tws_DSI/'
IPCCDIR       = '../src/IPCC_SREX_Regions_Raster/'
srcDIR        = '../src/'
aDIR05        = '../src/'

basarea05         = aDIR05 + 'area.hlf'
basarea05         = np.fromfile(basarea05,np.float32).byteswap().reshape(360,720)	
basinMASK         = '../src/basin_mask/'
WeightedMean_CONT = '../src/WeightedMean_CONT_Masks/woMATSIRO/'
Weighted          = True

models        = ['clm45', 'cwatm','h08','lpjml','mpi-hm','pcr-globwb','watergap2']
GCMs          = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] 

masconCSR     = '/egr/research-hydro/felfelan/GRACE/MASCON/CSR_GRACE_RL05_Mascons/CSR_GRACE_RL05_Mascons_v01.nc'

years         = '1861_2099'

sDateH = '1976-01'
eDateH = '2005-12'

sDateP1 = '2030-01'
eDateP1 = '2059-12'

sDateP2 = '2070-01'
eDateP2 = '2099-12'

# tspan   = 'mid'
tspan   = 'late'

#%%------------------------------------------------------------------ Reading the continental weights for weighted mean

kk = 0
modgcm = []
for mod in models:
	for gcm in GCMs:
		if os.path.isfile(WeightedMean_CONT + mod + '_' + gcm + '.npy'):
			print(mod + '_' + gcm)
			wFile0 = np.load(WeightedMean_CONT + mod + '_' + gcm + '.npy').reshape(1,360,720)
			if kk == 0:
				wFile = wFile0
				modgcm = modgcm + [mod + '_' + gcm]
				kk = 1
			else:
				wFile = np.concatenate([wFile, wFile0],axis = 0)
				modgcm = modgcm + [mod + '_' + gcm]


#%%------------------------------------------------------------------ Plotting

fig = plt.figure(num=1, figsize=(7.8,3.87)) #figsize: w,h letter size
gs = gridspec.GridSpec(1, 1) # the middle row is to place the colorbar axes
gs.update(bottom=0.0, top = 1,left = 0, right=1)
ax = []
axx = []

SREX_names = ['ALA','AMZ','CAM','CAS', \
			  'CEU','CNA','EAF','EAS', \
			  'ENA','MED','NAS','NAU','NEB', \
			  'NEU','SAF','SAS','SAU', \
			  'SEA','SSA','WAF','WAS', \
			  'WNA', 'Glob']

bbox_lft   = [0.01,0.2 ,0.01 ,0.7, \
			   0.35, 0.13,0.6 , 0.85, \
			   0.2, 0.39, 0.85, 0.88, 0.4, \
			   0.46, 0.53,0.65 , 0.78, \
			   0.8, 0.27,0.42, 0.59, \
			   0.01, 0.05]

bbox_bot   = [0.8,0.35,0.35,0.66, \
			   0.73, 0.55, 0.45, 0.65, \
			   0.8, 0.55 , 0.82, 0.28, 0.15, \
			   0.84, 0.23, 0.25, 0.05, \
			   0.45, 0.05, 0.35, 0.8, \
			   0.6,0.05]

ax.append(fig.add_subplot(gs[0]))

nxa=-180
nxb=180
nya=90
nyb=-90
res=1
LT1 = 0.3
LT2 = 0.7
map=Basemap( projection ='cyl',  \
		llcrnrlon  = nxa,  \
		 urcrnrlon  = nxb,  \
		 llcrnrlat  = nyb,  \
		 urcrnrlat  =  nya,  \
		 resolution = "c")

map.drawcoastlines(linewidth=LT1, color='gray')
map.drawmapboundary(fill_color='azure')
map.fillcontinents(color='moccasin',lake_color='azure')
SREX_info = map.readshapefile(srcDIR + 'IPCC_SREX_Regions/referenceRegions', 'referenceRegions', drawbounds = False)

patches = []
for info, shape in zip(map.referenceRegions_info, map.referenceRegions):
	if info['LAB'] not in ['CAR*', 'CGI', 'SAH', 'TIB', 'WSA', 'ANT*', 'ARC*', 'NTP*', 'STP*', 'ETP*', 'WIO*']:
		pol = Polygon(shape)
		patches.append(pol)
		if info['LAB'] == 'ALA':
			ax[-1].text(pol.centroid.x + 6, pol.centroid.y - 3, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'CNA':
			ax[-1].text(pol.centroid.x, pol.centroid.y + 2, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'NAS':
			ax[-1].text(pol.centroid.x, pol.centroid.y + 2, info['LAB'],horizontalalignment='center',fontsize=6)	
		elif info['LAB'] == 'CAM':
			ax[-1].text(pol.centroid.x + 3, pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)	
		elif info['LAB'] == 'AMZ':
			ax[-1].text(pol.centroid.x, pol.centroid.y-4, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'ENA':
			ax[-1].text(pol.centroid.x - 6, pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'MED':
			ax[-1].text(pol.centroid.x + 6, pol.centroid.y-3, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'WAF':
			ax[-1].text(pol.centroid.x + 10, pol.centroid.y + 2, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SAF':
			ax[-1].text(pol.centroid.x, pol.centroid.y + 6, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'EAF':
			ax[-1].text(pol.centroid.x - 6, pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SAS':
			ax[-1].text(pol.centroid.x + 2, pol.centroid.y + 3, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SEA':
			ax[-1].text(pol.centroid.x - 22, pol.centroid.y + 8, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SAU':
			ax[-1].text(pol.centroid.x - 20, pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'NEU':
			ax[-1].text(pol.centroid.x-15, pol.centroid.y-6, info['LAB'],horizontalalignment='center',fontsize=6)
		else:		
			ax[-1].text(pol.centroid.x,pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)

		x, y = zip(*shape) 
		map.plot(x, y, marker=None,color='darkgray', linewidth=1.)

plt.setp(ax[-1].spines.values(), color='darkgray')

LineColors = ['gray','royalblue','orange', 'red']
ShadColors = ['silver','deepskyblue','lemonchiffon','mistyrose']

reg = []
for file2 in sorted(os.listdir(IPCCDIR)):
	reg = reg + [file2]
reg = reg + ['HLF_mask_Glob.npy']


ii = 0 
for file2 in reg:
	print('------------------------------', file2.split('.')[0].split('_')[-1])
	yy = 0
	xx = 0
	zz = 0
	for file in sorted(os.listdir(globOUTDIR)):

		if ('matsiro' not in file):
			r""" Readig Future Projections Simulations"""
			if ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('picontrol_picontrol' in file) and (years in file) and ('2005soc' in file):
				print(file)		
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('IPCCS_SREX_mean_tws_DSI_','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if xx == 0:
					df_rivP = df_riv0.copy()
					xx = 1
				else:
					df_rivP = df_rivP.merge(df_riv0, how = 'outer', on=['Date'])

			r""" Readig Historical Simulations"""
			elif ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('historical_rcp26' in file) and (years in file) and ('2005soc' in file):
				print(file)
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('IPCCS_SREX_mean_tws_DSI_','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if yy == 0:
					df_rivR26 = df_riv0.copy()
					yy = 1
				else:
					df_rivR26 = df_rivR26.merge(df_riv0, how = 'outer', on=['Date'])

			elif ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('historical_rcp60' in file) and (years in file) and ('2005soc' in file):
				print(file)
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('IPCCS_SREX_mean_tws_DSI_','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if zz == 0:
					df_rivR60 = df_riv0.copy()
					zz = 1
				else:
					df_rivR60 = df_rivR60.merge(df_riv0, how = 'outer', on=['Date'])				

	df_dummyR26 = df_rivR26.copy()
	df_dummyR60 = df_rivR60.copy()
	df_dummyP   = df_rivP.copy()

	df_H  = df_dummyR26[(df_dummyR26['Date'] >= sDateH) & (df_dummyR26['Date'] <= eDateH)].copy()


	if tspan == 'mid':
		
		df_R26  = df_dummyR26[(df_dummyR26['Date'] >= sDateP1) & (df_dummyR26['Date'] <= eDateP1)].copy()
		df_R60  = df_dummyR60[(df_dummyR60['Date'] >= sDateP1) & (df_dummyR60['Date'] <= eDateP1)].copy()	
		df_P    = df_dummyP[(df_dummyP['Date'] >= sDateP1) & (df_dummyP['Date'] <= eDateP1)].copy()


	elif tspan == 'late':
		df_R26  = df_dummyR26[(df_dummyR26['Date'] >= sDateP2) & (df_dummyR26['Date'] <= eDateP2)].copy()	
		df_R60  = df_dummyR60[(df_dummyR60['Date'] >= sDateP2) & (df_dummyR60['Date'] <= eDateP2)].copy()
		df_P    = df_dummyP[(df_dummyP['Date'] >= sDateP2) & (df_dummyP['Date'] <= eDateP2)].copy()
	

	H = np.array(df_H.iloc[:,1:])
	H = H.reshape(np.size(H))
	H_kde = scipy.stats.gaussian_kde(H)	

	
	R26 = np.array(df_R26.iloc[:,1:])
	R26 = R26.reshape(np.size(R26))
	R26_kde = scipy.stats.gaussian_kde(R26)

	
	R60 = np.array(df_R60.iloc[:,1:])
	R60 = R60.reshape(np.size(R60))
	R60_kde = scipy.stats.gaussian_kde(R60)
	
	P = np.array(df_P.iloc[:,1:])
	P = P.reshape(np.size(P))
	P_kde = scipy.stats.gaussian_kde(P)


	t_range = np.linspace(-4,4,200)
	
	#%%------------------------------------------------------------------ Plotting
	PGmin = 0
	PGmax = 0

	if file2.split('.')[0].split('_')[-1] == 'Glob':
		axx = inset_axes(ax[-1],height="20%",width="20%",loc='lower left', bbox_to_anchor=(bbox_lft[ii],bbox_bot[ii],1,1), bbox_transform=ax[-1].transAxes)	
	else:
		axx = inset_axes(ax[-1],height="15%",width="10%",loc='lower left', bbox_to_anchor=(bbox_lft[ii],bbox_bot[ii],1,1), bbox_transform=ax[-1].transAxes)
	
	if tspan == 'mid':	
		Plabel = ['PIC (2030-2059)', 'HIST (1976-2005)', 'RCP2.6 (2030-2059)', 'RCP6.0 (2030-2059)']

	elif tspan == 'late':
		Plabel = ['PIC (2070-2099)', 'HIST (1976-2005)', 'RCP2.6 (2070-2099)', 'RCP6.0 (2070-2099)']
		
	axx.fill_between(t_range,P_kde(t_range),np.zeros((len(t_range))),facecolor = ShadColors[0], alpha = 0.8)
	axx.fill_between(t_range,H_kde(t_range),np.zeros((len(t_range))),facecolor = ShadColors[1], alpha = 0.4)
	axx.fill_between(t_range,R26_kde(t_range),np.zeros((len(t_range))),facecolor = ShadColors[2], alpha = 0.5)
	axx.fill_between(t_range,R60_kde(t_range),np.zeros((len(t_range))),facecolor = ShadColors[3], alpha = 0.5)

		
	axx.plot(t_range, P_kde(t_range)   ,lw=.8, color = LineColors[0], linestyle='-',  label=Plabel[0])
	axx.plot(t_range, H_kde(t_range)   ,lw=.8, color = LineColors[1], linestyle='-',  label=Plabel[1])
	axx.plot(t_range, R26_kde(t_range) ,lw=.8, color = LineColors[2], linestyle='-',  label=Plabel[2])
	axx.plot(t_range, R60_kde(t_range) ,lw=.8, color = LineColors[3], linestyle='-',  label=Plabel[3])



	axx.set_xticks([-2.0,-1.6,-1.3,-0.8,-0.5,0.5,0.8,1.3,1.6,2.0])		

	if file2.split('.')[0].split('_')[-1] == 'Glob':
		axx.set_xticklabels([-2.0,-1.6,-1.3,-0.8,-0.5,0.5,0.8,1.3,1.6,2.0], fontsize=4., rotation=-90)
	else: 
		axx.set_xticklabels([])
		axx.set_yticklabels([])			

	axx.tick_params(axis='both',labelsize = 6,direction = 'inout')
	axx.spines['right'].set_visible(False)
	axx.spines['top'].set_visible(False)
	
	
	if file2.split('.')[0].split('_')[-1] == 'Glob':
		axx.text(1.3,0.5,'GLOBE',fontsize = 6)
		axx.set_xlim(-3, 3)
		axx.set_ylim(0, 0.7)
		axx.set_yticks([0, 0.2, 0.4, 0.6])
	else:
		axx.text(1.3,0.5,file2.split('.')[0].split('_')[-1],fontsize = 6)
		axx.set_xlim(-3, 3)
		axx.set_ylim(0, 0.7)
		axx.set_yticks([0, 0.2, 0.4, 0.6])

	del df_dummyR26, df_rivR26
	del df_dummyR60, df_rivR60
	del df_dummyP, df_rivP

	ii = ii + 1

fig.show()
plt.legend(loc='lower left', bbox_to_anchor=(1.6,-0.35),ncol=2,fontsize=8)

plt.savefig( figDIR + 'NCC_Fig5_v1_IPCC_Group2_DSI_nonWeighted_PDF_SREXregions_' + tspan + '_defaultBW.png', bbox_inches='tight', dpi=500 )
plt.close()

