#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap,shiftgrid
from shapely.geometry.polygon import Polygon
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point
from matplotlib import gridspec
from netCDF4 import Dataset

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.stats
import datetime
import sys
import os

r"""
#%%------------------------------------------------------------------
     Description:   ISIMIP2b Project Figure 3: Global TWS Seasonal 
	                Cyle MODELS and GRACE
				    

     First Version: 01/17/2019
     Update:        01/17/2019

     Developer:	    Farshid Felfelani

	 Outstanding Issues:
					1- 
---------------------------------------------------------------------
#%%
"""

srcDIR        = '../src/'
figDIR        = '../scripts4paper/figs/'
globOUTDIR    = '../out_new/IPCC_Group2_SREXregions_negMASKED/'
IPCCDIR       = '../src/IPCC_SREX_Regions_Raster/'

aDIR05        = '../src/'
basarea05     = aDIR05 + 'area.hlf'
basarea05     = np.fromfile(basarea05,np.float32).byteswap().reshape(360,720)	
basinMASK     = '../src/basin_mask/'

masconCSR         = '../CSR_GRACE_RL05_Mascons/CSR_GRACE_RL05_Mascons_v01.nc'
WeightedMean_CONT = '../src/WeightedMean_CONT_Masks/woMATSIRO/'
Weighted          = True

models        = ['clm45', 'cwatm','h08','lpjml','mpi-hm','pcr-globwb','watergap2']

GCMs          = ['gfdl-esm2m','hadgem2-es','ipsl-cm5a-lr','miroc5'] 

soec          = ['2005soc','histsoc']
years         = '1861_2099'

sDateH = '1976-01'
eDateH = '2005-12'

sDateFu = '2070-01'
eDateFu = '2099-12'

sDateP = '2070-01' # for picontrol
eDateP = '2099-12'

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


def weightedMEAN(df):
	
	bmask = np.load(IPCCDIR + file2)
	w3DArray = np.ma.masked_equal(wFile * bmask ,0.0)
	df_model_Weighted = df.copy()
	wSum = 0
	
	for column in df_model_Weighted.columns:
		if column != 'MonStr':
			modd = column.split('_')[0]
			gcmm = column.split('_')[1]
			
			w2DArray = w3DArray[modgcm.index(modd + '_' + gcmm),]
			
			if w2DArray.max() == w2DArray.min() and w2DArray.max() != 0.0:
				weight = w2DArray.max()
		
				df_model_Weighted.loc[:,column] = df.loc[:,column] * weight
				wSum = wSum + weight
			else:
				print('=========CAUCION=============', file2, modd, gcmm)
				df_model_Weighted.loc[:,column] = df.loc[:,column] * w2DArray.mean()
				wSum = wSum + w2DArray.mean()

	print(file2, '     wSum: ', wSum)

	return df_model_Weighted


def weightedMEAN_GLOB(df):
		
	r""" to calculate the global average weights from continental weights"""
	wGLOB = np.ma.masked_equal(wFile,0.0).sum(2).sum(1)/ np.ma.masked_equal(wFile[0],0.0).count()

	df_model_Weighted = df.copy()
	wSum = 0
	
	for column in df_model_Weighted.columns:
		if column != 'MonStr':
			modd = column.split('_')[0]
			gcmm = column.split('_')[1]
			
			weight = wGLOB[modgcm.index(modd + '_' + gcmm),]
			df_model_Weighted.loc[:,column] = df.loc[:,column] * weight
			wSum = wSum + weight
	print('wSum: ', wSum)
	return df_model_Weighted

#%%------------------------------------------------------------------ Create a Dataframe for all Models and Simulations

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
			  'WNA', 'GLOB']

bbox_lft   = [0.03,0.2 ,0.05 ,0.73, \
			   0.35, 0.14,0.6 , 0.85, \
			   0.2, 0.39, 0.85, 0.88, 0.4, \
			   0.46, 0.56,0.71 , 0.85, \
			   0.8, 0.27,0.42, 0.59, \
			   0.01, 0.05]

bbox_bot   = [0.8,0.35,0.35,0.66, \
			   0.73, 0.55, 0.45, 0.65, \
			   0.8, 0.55 , 0.82, 0.28, 0.15, \
			   0.84, 0.2, 0.27, 0.05, \
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
			ax[-1].text(pol.centroid.x + 9, pol.centroid.y - 3, info['LAB'],horizontalalignment='center',fontsize=6)
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
			ax[-1].text(pol.centroid.x - 6, pol.centroid.y -3, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SAS':
			ax[-1].text(pol.centroid.x + 2, pol.centroid.y + 3, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SEA':
			ax[-1].text(pol.centroid.x - 22, pol.centroid.y + 8, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'SAU':
			ax[-1].text(pol.centroid.x - 20, pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'NEU':
			ax[-1].text(pol.centroid.x-15, pol.centroid.y-6, info['LAB'],horizontalalignment='center',fontsize=6)
		elif info['LAB'] == 'NAU':
			ax[-1].text(pol.centroid.x-3, pol.centroid.y-1, info['LAB'],horizontalalignment='center',fontsize=6)
		else:		
			ax[-1].text(pol.centroid.x,pol.centroid.y, info['LAB'],horizontalalignment='center',fontsize=6)

		x, y = zip(*shape) 
		map.plot(x, y, marker=None,color='darkgray', linewidth=1.)
		
plt.setp(ax[-1].spines.values(), color='darkgray')
colors2  = ['grey','royalblue','orange','firebrick','black']

reg = []
for file2 in sorted(os.listdir(IPCCDIR)):
	reg = reg + [file2]
reg = reg + ['HLF_mask_GLOB.npy']

df_TOTratio = pd.DataFrame(columns=['SREX_reg' , 'Mean PIC - PIC', 'Mean RCP6.0 - PIC', 'Ratio'])
df_TOTrcp6  = pd.DataFrame(columns=['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
df_TOTpicH  = pd.DataFrame(columns=['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
df_TOTpicF  = pd.DataFrame(columns=['Jan' , 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

ii = 0 
for file2 in reg:
	yy = 0
	xx = 0
	zz = 0
	for file in sorted(os.listdir(globOUTDIR)):
		
		if ('matsiro' not in file):
			r""" Readig Future Projections Simulations"""
			if ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('_picontrol_' in file):
				print(file)		
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('_IPCC_SREX_mean_tws_2002_2016_ANMLYbaseline_1861_2099','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if xx == 0:
					df_rivP = df_riv0.copy()
					xx = 1
				else:
					df_rivP = df_rivP.merge(df_riv0, how = 'outer', on=['Date'])
	
			r""" Readig Historical Simulations"""
			elif ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('_HiRCP26_' in file):
				print(file)
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('_IPCC_SREX_mean_tws_2002_2016_ANMLYbaseline_1861_2099','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if yy == 0:
					df_rivHR26 = df_riv0.copy()
					yy = 1
				else:
					df_rivHR26 = df_rivHR26.merge(df_riv0, how = 'outer', on=['Date'])
	
			elif ('_' + file2.split('.')[0].split('_')[-1] + '_' in file) and ('_HiRCP60_' in file):
				print(file)
				df_dummy_ts = pd.read_csv(globOUTDIR + file,sep=',',header=0, error_bad_lines=False)
				
				df_riv0 = pd.DataFrame(columns=['Date'])
				df_riv0.Date = pd.date_range('1/' + years[:4], '12/' + years[-4:] , freq='MS') #create the date range 
				df_riv0.Date = df_riv0['Date'].dt.strftime('%Y-%m') #just keep year and month
				df_riv0[file.replace('_IPCC_SREX_mean_tws_2002_2016_ANMLYbaseline_1861_2099','')] = df_dummy_ts.iloc[:,2]  # remove the repeating key words from the column title
				if zz == 0:
					df_rivHR60 = df_riv0.copy()
					zz = 1
				else:
					df_rivHR60 = df_rivHR60.merge(df_riv0, how = 'outer', on=['Date'])				



	df_dummyHR26 = df_rivHR26.copy()
	df_dummyHR60 = df_rivHR60.copy()
	df_dummyP    = df_rivP.copy()
	df_dummyPH   = df_rivP.copy()


	df_dummyH1   = df_dummyHR26[(df_dummyHR26['Date'] >= sDateH) & (df_dummyHR26['Date'] <= eDateH)].copy()
	df_dummyR26  = df_dummyHR26[(df_dummyHR26['Date'] >= sDateFu) & (df_dummyHR26['Date'] <= eDateFu)].copy()	

	df_dummyH2   = df_dummyHR60[(df_dummyHR60['Date'] >= sDateH) & (df_dummyHR60['Date'] <= eDateH)].copy()
	df_dummyR60  = df_dummyHR60[(df_dummyHR60['Date'] >= sDateFu) & (df_dummyHR60['Date'] <= eDateFu)].copy()	
	df_dummyP    = df_dummyP[(df_dummyP['Date'] >= sDateP) & (df_dummyP['Date'] <= eDateP)].copy()
	df_dummyPH   = df_dummyPH[(df_dummyPH['Date'] >= sDateH) & (df_dummyPH['Date'] <= eDateH)].copy()


	r""" sort based on the Date and reindex"""
	df_dummyH1 = df_dummyH1.sort_values(["Date"])
	df_dummyH1 = df_dummyH1.reset_index(drop=True)

	df_dummyH2 = df_dummyH2.sort_values(["Date"])
	df_dummyH2 = df_dummyH2.reset_index(drop=True)

	df_dummyR26 = df_dummyR26.sort_values(["Date"])
	df_dummyR26 = df_dummyR26.reset_index(drop=True)

	df_dummyR60 = df_dummyR60.sort_values(["Date"])
	df_dummyR60 = df_dummyR60.reset_index(drop=True)
	
	df_dummyP = df_dummyP.sort_values(["Date"])
	df_dummyP = df_dummyP.reset_index(drop=True)

	df_dummyPH = df_dummyPH.sort_values(["Date"])
	df_dummyPH = df_dummyPH.reset_index(drop=True)

	r""" To calculate the seasonal cycle:
	     Insert month column after date (index = 1) """
	df_dummyH1.insert(1, 'Date.month', pd.to_datetime(df_dummyH1['Date']).dt.strftime('%m'))
	df_dummyH2.insert(1, 'Date.month', pd.to_datetime(df_dummyH2['Date']).dt.strftime('%m'))
	df_dummyR26.insert(1, 'Date.month', pd.to_datetime(df_dummyR26['Date']).dt.strftime('%m'))  # take out the two-digit month
	df_dummyR60.insert(1, 'Date.month', pd.to_datetime(df_dummyR60['Date']).dt.strftime('%m'))  # take out the two-digit month
	df_dummyP.insert(1, 'Date.month', pd.to_datetime(df_dummyP['Date']).dt.strftime('%m'))
	df_dummyPH.insert(1, 'Date.month', pd.to_datetime(df_dummyPH['Date']).dt.strftime('%m'))


	sea_cycle_H1 = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_H1.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_H1.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyH1.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyH1.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyH1.groupby('Date.month')[x].mean().index
			
			sea_cycle_H1 = sea_cycle_H1.merge(df_dummy_seaCYCLE,on=['month'])


	sea_cycle_H2 = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_H2.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_H2.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyH2.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyH2.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyH2.groupby('Date.month')[x].mean().index
			
			sea_cycle_H2 = sea_cycle_H2.merge(df_dummy_seaCYCLE,on=['month'])

	sea_cycle_R26 = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_R26.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_R26.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyR26.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyR26.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyR26.groupby('Date.month')[x].mean().index
			
			sea_cycle_R26 = sea_cycle_R26.merge(df_dummy_seaCYCLE,on=['month'])

	sea_cycle_R60 = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_R60.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_R60.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyR60.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyR60.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyR60.groupby('Date.month')[x].mean().index
			
			sea_cycle_R60 = sea_cycle_R60.merge(df_dummy_seaCYCLE,on=['month'])			

			
	sea_cycle_P = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_P.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_P.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyP.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyP.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyP.groupby('Date.month')[x].mean().index

			sea_cycle_P = sea_cycle_P.merge(df_dummy_seaCYCLE,on=['month'])


			
	sea_cycle_PH = pd.DataFrame(columns = ['MonStr','month'])
	sea_cycle_PH.MonStr = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	sea_cycle_PH.month  = ['01','02','03','04','05','06','07','08','09','10','11','12']
	for x in df_dummyPH.columns:
		if (x != 'Date') and (x != 'Date.month'):
			df_dummy_seaCYCLE = pd.DataFrame(columns=['month', x])
			df_dummy_seaCYCLE[x] = df_dummyPH.groupby('Date.month')[x].mean()
			df_dummy_seaCYCLE['month'] = df_dummyPH.groupby('Date.month')[x].mean().index

			sea_cycle_PH = sea_cycle_PH.merge(df_dummy_seaCYCLE,on=['month'])
			
	#%%------------------------------------------------------------------ Plotting

	PGmin = 0
	PGmax = 0

	if file2.split('.')[0].split('_')[-1] == 'GLOB':
		axx = inset_axes(ax[-1],height="20%",width="15%",loc='lower left', bbox_to_anchor=(bbox_lft[ii],bbox_bot[ii],1,1), bbox_transform=ax[-1].transAxes)
	else:
		axx = inset_axes(ax[-1],height="15%",width="10%",loc='lower left', bbox_to_anchor=(bbox_lft[ii],bbox_bot[ii],1,1), bbox_transform=ax[-1].transAxes)	

	df_basin_dummyP = sea_cycle_P[['MonStr'] + [col for col in list(sea_cycle_P) if (col.find('picontrol') != -1)]]
	df_basin_dummyPH = sea_cycle_PH[['MonStr'] + [col for col in list(sea_cycle_PH) if (col.find('picontrol') != -1)]]
	df_basin_dummyH1 = sea_cycle_H1[['MonStr'] + [col for col in list(sea_cycle_H1) if (col.find('HiRCP26') != -1)]]
	df_basin_dummyR26 = sea_cycle_R26[['MonStr'] + [col for col in list(sea_cycle_R26) if (col.find('HiRCP26') != -1)]]	
	df_basin_dummyR60 = sea_cycle_R60[['MonStr'] + [col for col in list(sea_cycle_R60) if (col.find('HiRCP60') != -1)]]	


	if file2.split('.')[0].split('_')[-1] == 'GLOB':

		df_basin_P   = weightedMEAN_GLOB(df_basin_dummyP)
		df_basin_PH  = weightedMEAN_GLOB(df_basin_dummyPH)
		df_basin_H1  = weightedMEAN_GLOB(df_basin_dummyH1)	
		df_basin_R26 = weightedMEAN_GLOB(df_basin_dummyR26)	
		df_basin_R60 = weightedMEAN_GLOB(df_basin_dummyR60)	
		
	else:
		df_basin_P   = weightedMEAN(df_basin_dummyP)
		df_basin_PH  = weightedMEAN(df_basin_dummyPH)
		df_basin_H1  = weightedMEAN(df_basin_dummyH1)	
		df_basin_R26 = weightedMEAN(df_basin_dummyR26)	
		df_basin_R60 = weightedMEAN(df_basin_dummyR60)	


	if Weighted:	
		axx.plot(df_basin_P.index,df_basin_P.iloc[:,1:].sum(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[0],alpha = 1.0, label= 'PIC (2070-2099)')
		axx.plot(np.zeros((len(df_basin_P))),color='k', linestyle=':', linewidth=.5)
		
		axx.plot(df_basin_PH.index,df_basin_PH.iloc[:,1:].sum(axis=1) , linestyle='--', \
									linewidth=.8, color = colors2[0],alpha = 1.0, label= 'PIC (1976-2005)')
									
		axx.plot(df_basin_H1.index,df_basin_H1.iloc[:,1:].sum(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[1],alpha = 1.0, label= 'HIST (1976-2005)')		
	
	
		axx.plot(df_basin_R26.index,df_basin_R26.iloc[:,1:].sum(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[2],alpha = 1.0, label= 'RCP2.6 (2070-2099)')
	
		axx.plot(df_basin_R60.index,df_basin_R60.iloc[:,1:].sum(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[3],alpha = 1.0, label= 'RCP6.0 (2070-2099)')
	
	
		PGmin = min(df_basin_P.sum(axis=1).min(), \
		df_basin_PH.sum(axis=1).min(),
		df_basin_H1.sum(axis=1).min(), \
		df_basin_R26.sum(axis=1).min(), \
		df_basin_R60.sum(axis=1).min())
		
		PGmax = max(df_basin_P.sum(axis=1).max(), \
		df_basin_PH.sum(axis=1).min(),
		df_basin_H1.sum(axis=1).max(), \
		df_basin_R26.sum(axis=1).max(), \
		df_basin_R60.sum(axis=1).max())

		wLabel = 'Weighted'
	else:

		axx.plot(df_basin_dummyP.index,df_basin_dummyP.iloc[:,1:].mean(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[0],alpha = 1.0, label= 'PIC (2070-2099)')
		axx.plot(np.zeros((len(df_basin_dummyP))),color='k', linestyle=':', linewidth=.5)
		
		axx.plot(df_basin_dummyPH.index,df_basin_dummyPH.iloc[:,1:].mean(axis=1) , linestyle='--', \
									linewidth=.8, color = colors2[0],alpha = 1.0, label= 'PIC (1976-2005)')
									
		axx.plot(df_basin_dummyH1.index,df_basin_dummyH1.iloc[:,1:].mean(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[1],alpha = 1.0, label= 'HIST (1976-2005)')		
	
	
		axx.plot(df_basin_dummyR26.index,df_basin_dummyR26.iloc[:,1:].mean(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[2],alpha = 1.0, label= 'RCP2.6 (2070-2099)')
	
		axx.plot(df_basin_dummyR60.index,df_basin_dummyR60.iloc[:,1:].mean(axis=1) , linestyle='-', \
									linewidth=.8, color = colors2[3],alpha = 1.0, label= 'RCP6.0 (2070-2099)')
	
	
		PGmin = min(df_basin_dummyP.mean(axis=1).min(), \
		df_basin_dummyPH.mean(axis=1).min(),
		df_basin_dummyH1.mean(axis=1).min(), \
		df_basin_dummyR26.mean(axis=1).min(), \
		df_basin_dummyR60.mean(axis=1).min())
		
		PGmax = max(df_basin_dummyP.mean(axis=1).max(), \
		df_basin_dummyPH.mean(axis=1).min(),
		df_basin_dummyH1.mean(axis=1).max(), \
		df_basin_dummyR26.mean(axis=1).max(), \
		df_basin_dummyR60.mean(axis=1).max())

		wLabel = 'nonWeighted'		
		
		

	axx.set_xticks(df_basin_P.index)		
	if file2.split('.')[0].split('_')[-1] == 'GLOB':
		axx.set_xticklabels(['Jan','','Mar','','May','','Jul','','Sep','','Nov',''], fontsize=6, rotation=-45)
		axx.tick_params(axis='y',labelsize = 6)
		axx.set_ylabel('(mm)', fontsize=8)
	else: 
		axx.set_xticklabels([])	

	axx.tick_params(axis='both',labelsize = 6,direction = 'inout')
	axx.spines['right'].set_visible(False)
	axx.spines['top'].set_visible(False)

	axx.set_ylim(PGmin-10,PGmax+10)
	
	axx.set_xlim(0, len(df_basin_P))
	if file2.split('.')[0].split('_')[-1] == 'CAM':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'WAF':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'SAS':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'EAS':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'SEA':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'SAU':
		axx.text(.2,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)
	elif file2.split('.')[0].split('_')[-1] == 'GLOB':
		axx.text(.5,0.8,'GLOBE',fontsize = 6, transform=axx.transAxes)		
	else:
		axx.text(.5,0.8,file2.split('.')[0].split('_')[-1],fontsize = 6, transform=axx.transAxes)


	PICratio = df_basin_dummyP.iloc[:,1:].mean(axis=1) - df_basin_dummyPH.iloc[:,1:].mean(axis=1)
	RCPratio = df_basin_dummyR60.iloc[:,1:].mean(axis=1) - df_basin_dummyPH.iloc[:,1:].mean(axis=1)
	
	TOTratio = PICratio.mean() / RCPratio.mean()
	
	df_TOTratio.loc[ii, 'SREX_reg'] = file2
	df_TOTratio.loc[ii, 'Ratio'] = TOTratio
	df_TOTratio.loc[ii, 'Mean PIC - PIC'] = PICratio.mean()
	df_TOTratio.loc[ii, 'Mean RCP6.0 - PIC'] = 	RCPratio.mean()

	df_TOTrcp6.loc[ii, :]  = np.array(df_basin_dummyR60.iloc[:,1:].mean(axis=1))
	df_TOTpicH.loc[ii, :]  = np.array(df_basin_dummyPH.iloc[:,1:].mean(axis=1))
	df_TOTpicF.loc[ii, :]  = np.array(df_basin_dummyP.iloc[:,1:].mean(axis=1))


	del df_dummyHR26, df_rivHR26
	del df_dummyHR60, df_rivHR60
	del df_dummyP, df_rivP
	del df_dummyPH

	ii = ii + 1

fig.show()
plt.legend(loc='lower left', bbox_to_anchor=(2.2,-0.35),ncol=2,fontsize=8)

plt.savefig( figDIR + 'NCC_Fig3_v1_IPCC_Group2_' + wLabel + '_TWS_SeasonalCycle_SREXregions.png', bbox_inches='tight', dpi=500 )
plt.close()
