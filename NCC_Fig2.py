#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap,shiftgrid
from sklearn.metrics import mean_squared_error
from shapely.geometry.polygon import Polygon
from matplotlib.gridspec import GridSpec
from collections import OrderedDict
from shapely.geometry import Point
from matplotlib import gridspec
from netCDF4 import Dataset
from copy import deepcopy
from scipy import stats

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

import scipy.stats
import datetime
import sys
import os

r"""
#%%------------------------------------------------------------------
     Description:   ISIMIP2b Project Figure 2: the monthly TWS 
	                Quantile for SREX regions, averging across HMs 
					and GCMs To quantify the uncertainty contribution 
					from HMs and GCMs
				    

     First Version: 04/05/2020
     Update:        04/05/2020

     Developer:	    Farshid Felfelani

	 Outstanding Issues:
					1- 
---------------------------------------------------------------------
#%%
"""

figDIR            = '../figs/'
globOUTDIR        = '../temporal_SREXregions_TWS_Quantile/'
mskDIR            = '../src/basin_mask/'
srcDIR            = '../src/'

models        = ['clm45', 'cwatm', 'h08', 'lpjml', 'mpi-hm', 'pcr-globwb', 'watergap2']
climate       = ['picontrol', 'rcp26', 'rcp60']
GCMs          = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5'] 

years   = '1861_2099'
sDate   = '1976-01'
eDate   = '2099-12'

sDateH  = '1976-01'
eDateH  = '2005-12'

sDateMC = '2030-01'
eDateMC = '2059-12'

sDateLC = '2070-01'
eDateLC = '2099-12'

SREXSet = 'sel_allSREX_list'
blist   = open(srcDIR + SREXSet + '.txt').readlines()
	
#%%------------------------------------------------------------------ Create a Dataframe for all Models and Simulations

def df_concat(clm_clm):
	yy = 0
	for ii, basin0 in enumerate(blist):
		basin = basin0.split('.')[0].split('_')[-1]
		print('-------------------------', basin)
		count = 0
		for file in sorted(os.listdir(globOUTDIR)):
			r""" Readig Historical Simulations"""
			if (basin + '_' in file) and (clm_clm in file) and ('matsiro' not in file):
				print(file)
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df0 = pd.DataFrame(columns=['Date'])
				df0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df0.Date = df0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df0[file.split('_')[0] + '_' + file.split('_')[1] + '_'  + file.split('_')[2]] = df_dummy_ts.loc[:,'TWS_Q']  
				count = count + 1
				r""" remove the repeating key words from the column title"""
				if yy == 0:
					df = df0.copy()
					yy = 1
				else:
					df = df.merge(df0, how = 'outer', on=['Date'])

		print('Tot count:', count)
		print('')
	return df

def df_basinExtract(df):
	df_basin = df[['Date'] + [col for col in list(df) if (col.find('_' + basin) != -1)]]
	df_basin = df_basin.set_index('Date')
	df_basin = df_basin.set_index(pd.to_datetime(df_basin.index))
	return df_basin

def df_HMgrouped(df):
	df_grouped = pd.DataFrame(index = df.index)
	for mod in models:
		df_grouped[mod] = df[[col for col in list(df) if (col.find(mod + '_') != -1)]].mean(axis = 1)
	MinofCol = df_grouped.min(axis = 1)
	MaxofCol = df_grouped.max(axis = 1)
	df_grouped['_Range'] = MaxofCol - MinofCol			
	return df_grouped

def df_GCMgrouped(df):
	df_grouped = pd.DataFrame(index = df.index)
	for gcm in GCMs:
		df_grouped[gcm] = df[[col for col in list(df) if (col.find(gcm + '_') != -1)]].mean(axis = 1)
	MinofCol = df_grouped.min(axis = 1)
	MaxofCol = df_grouped.max(axis = 1)
	df_grouped['_Range'] = MaxofCol - MinofCol	
	return df_grouped

def R_intgral(df):
	Rmean = np.ones((3)) * -99
	df_H  = df[(df.index >= sDateH) & (df.index <= eDateH)].copy()
	df_MC = df[(df.index >= sDateMC) & (df.index <= eDateMC)].copy()
	df_LC = df[(df.index >= sDateLC) & (df.index <= eDateLC)].copy()
	
	Rmean[0] = df_H[[col for col in list(df_H) if (col.find('_Range') != -1)]].values.mean()
	Rmean[1] = df_MC[[col for col in list(df_MC) if (col.find('_Range') != -1)]].values.mean()	
	Rmean[2] = df_LC[[col for col in list(df_LC) if (col.find('_Range') != -1)]].values.mean()
	return Rmean

df_dummyHR26 = df_concat('historical_rcp26')
df_dummyHR60 = df_concat('historical_rcp60')
df_dummyP    = df_concat('picontrol_picontrol')

df_dummyHR26 = df_dummyHR26[(df_dummyHR26['Date'] >= sDate) & (df_dummyHR26['Date'] <= eDate)].copy()
df_dummyHR60 = df_dummyHR60[(df_dummyHR60['Date'] >= sDate) & (df_dummyHR60['Date'] <= eDate)].copy()
df_dummyP    = df_dummyP[(df_dummyP['Date'] >= sDate) & (df_dummyP['Date'] <= eDate)].copy()

r""" sort based on the Date and reindex"""
df_dummyHR26 = df_dummyHR26.sort_values(["Date"])
df_dummyHR26 = df_dummyHR26.reset_index(drop=True)

df_dummyHR60 = df_dummyHR60.sort_values(["Date"])
df_dummyHR60 = df_dummyHR60.reset_index(drop=True)

df_dummyP = df_dummyP.sort_values(["Date"])
df_dummyP = df_dummyP.reset_index(drop=True)

#%%------------------------------------------------------------------ Processing

GCM_R26_mat = np.ones((len(blist),3)) * -99
GCM_R60_mat = np.ones((len(blist),3)) * -99
HMs_R26_mat = np.ones((len(blist),3)) * -99
HMs_R60_mat = np.ones((len(blist),3)) * -99

basinList = []

for ii, basin0 in enumerate(blist):
	basin = basin0.split('.')[0].split('_')[-1]
	basinList = basinList + [basin]

	df_basinHR26 = df_basinExtract(df_dummyHR26)
	df_basinHR60 = df_basinExtract(df_dummyHR60)

	df_basinHR26_HM  = df_HMgrouped(df_basinHR26)
	df_basinHR26_GCM = df_GCMgrouped(df_basinHR26)

	df_basinHR60_HM  = df_HMgrouped(df_basinHR60)
	df_basinHR60_GCM = df_GCMgrouped(df_basinHR60)

	GCM_R26_mat[ii,] = R_intgral(df_basinHR26_GCM)
	HMs_R26_mat[ii,] = R_intgral(df_basinHR26_HM)
	GCM_R60_mat[ii,] = R_intgral(df_basinHR60_GCM)
	HMs_R60_mat[ii,] = R_intgral(df_basinHR60_HM)

#%%------------------------------------------------------------------ Plotting

fig = plt.figure(figsize=(4, 4))
gs  = gridspec.GridSpec(1, 4, height_ratios=[1], width_ratios=[1,1, 1, 1], wspace=0.05,hspace=0.1)
gs.update(bottom=0.15, top=.95, left = 0.05, right = .8,hspace = 0.1,wspace = 0.05)

cmap     = plt.cm.Greys
cmap     = plt.cm.hot_r
cmap     = plt.cm.afmhot_r

cmaplist = [cmap(i) for i in range(cmap.N)]
cmap     = cmap.from_list('Custom cmap', cmaplist[:-40], cmap.N)
bounds   = np.array([0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55, 0.6,0.65,0.7])
norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)

periods = ['1976-2005', '2030-2059', '2070-2099']
for ii in range(4):
	ax = plt.subplot(gs[ii])
	if ii == 0:
		XYmap = ax.imshow(GCM_R26_mat,cmap = cmap,norm=norm)
		ax.text(0.05,1.02,'RCP2.6', fontsize = 8, transform=ax.transAxes)
		ax.text(-0.3,.47,'GCM', fontsize = 8, transform=ax.transAxes,rotation=90)
	elif ii == 1:
		XYmap = ax.imshow(GCM_R60_mat,cmap = cmap,norm=norm) 
		ax.text(0.05,1.02,'RCP6.0', fontsize = 8, transform=ax.transAxes)
	elif ii == 2:
		ax.text(0.05,1.02,'RCP2.6', fontsize = 8, transform=ax.transAxes)		
		XYmap = ax.imshow(HMs_R26_mat,cmap = cmap,norm=norm) 
		ax.text(-0.3,.42,'GHM/LSM', fontsize = 8, transform=ax.transAxes,rotation=90)
	elif ii == 3:
		XYmap = ax.imshow(HMs_R60_mat,cmap = cmap,norm=norm) 
		ax.text(0.05,1.02,'RCP6.0', fontsize = 8, transform=ax.transAxes)
	if ii in [1,3]:

		major_yticks = np.arange(0., len(blist) , 1)
		ax.set_yticks(major_yticks)
		ax.set_yticklabels(basinList, ha = 'left', position = (-0.5,0))
		ax.tick_params(axis = 'y', which = 'major', labelsize = 7.8,length=0,labelrotation=0)
	else:
		ax.set_yticks([])
		ax.set_yticklabels([])

	major_xticks = np.arange(0., len(periods) , 1)
	ax.set_xticks(major_xticks)
	ax.set_xticklabels(periods)
	ax.tick_params(axis = 'x', which = 'major', labelsize = 7,length=3,labelrotation=90)

	minor_xticks = np.arange(-0.5, len(periods) + 0.5, 1)
	ax.set_xticks(minor_xticks, minor = True)
	ax.set_xticklabels([], minor = True)
	ax.tick_params(axis = 'x', which = 'minor',length=0)

	minor_yticks = np.arange(-0.5, len(blist) + 0.5, 1)
	ax.set_yticks(minor_yticks, minor = True)
	ax.set_yticklabels([], minor = True)
	ax.tick_params(axis = 'y', which = 'minor',length=0)

cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.6]) 
cb = plt.colorbar(XYmap, cax=cbar_ax, cmap = cmap, norm=norm,spacing='uniform', ticks=bounds, boundaries=bounds)

cb.ax.tick_params(color = 'k',labelsize=8)
cb.set_label('Uncertainty',fontsize=10, labelpad=-45)

plt.savefig( figDIR + 'NCC_Fig2_v1_AllSREX_TWS_Quantile_Matrix_'  + years + '.png', bbox_inches='tight', dpi=500 )
plt.close()