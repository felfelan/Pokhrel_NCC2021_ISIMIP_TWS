#!/egr/research-hydro/felfelan/installed_soft/anaconda/DIR/bin/ipython

from __future__ import division # force division to be floating point in Python
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import fnmatch
import sys
import os

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap,shiftgrid
from scipy.interpolate import griddata
from collections import OrderedDict
from matplotlib import gridspec
from netCDF4 import Dataset
from copy import deepcopy
from scipy import stats
from pylab import mpl

from latlon2xy import cylHLFdegxy
from ISIMIPmod import Glaciers1xEnlargedMASK, NOgreenlndNOantarcticaMASK
from ISIMIPmod import Global_LndAreaMASK, CONTs_LndMASK

r"""
#%%------------------------------------------------------------------
     Description:   ISIMIP2b Project Figure 4: Plotting the spatial map 
					of DSI trend + histograms + timeseries
				    

     First Version: 07/09/2019
     Update:        50/08/2020 (Re-organizing, using functions, Make 
								Modular etc.)

     Developer:	    Farshid Felfelani

	 Outstanding Issues:
					1- 
---------------------------------------------------------------------
#%%
"""

CNTNTLdsiDIR      = '../out_new/temporal_CNTNTL_tws_DSI_1861_2099/'
figDIR            = '../scripts4NCC/figs/'
hlfDIR            = '../0.5_hydro_data/'
hlfPOP            = '../gpw-v4-population-count-rev11_totpop_30_min_nc/'
srcDIR            = '../out_new/DSI_TS_DynFracAreaPop_AbnDryRemoved/'
PopDIR            = '../src/population/'
WeightedMean_CONT = '../WeightedMean_CONT_Masks/woMATSIRO/'

models      = ['clm45', 'cwatm', 'h08', 'lpjml', 'mpi-hm', 'pcr-globwb', 'watergap2']
GCMs        = ['gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5']
climate     = ['picontrol', 'historical', 'rcp26', 'rcp60']
SOECs       = OrderedDict([('clm45', '2005soc'),
						('cwatm', 'histsoc'),
						('h08', 'histsoc'),
						('lpjml', 'histsoc'),
						('mpi-hm', 'histsoc'),
						('pcr-globwb', 'histsoc'),						
						('watergap2', 'histsoc')])

CONTs       = ['Africa','Asia','Australia','Europe','nAmerica','sAmerica',]
CONTs_name  = OrderedDict([('Asia', 'Asia'),
						('Europe', 'Europe'),
						('Africa', 'Africa'),
						('nAmerica', 'North America'),
						('sAmerica', 'South America'),
						('Australia', 'Australia')])
CONTcode    = OrderedDict([('Asia', 1),
						('Europe', 2),
						('Africa', 3),
						('nAmerica', 4),
						('sAmerica', 5),
						('Australia', 6)])

YRS        = '1861_2099'
sDate_TR   = '2006-01-01'
eDate_TR   = '2099-12-31'

r""" to get the 30-yr mean values for baseline period: sDate_TS   = '1976-01-01'
                              For plotting the figure: sDate_TS   = '2006-01-01'
"""
sDate_TS   = '2006-01-01'
eDate_TS   = '2099-12-31'

baseline      = '1976_2005'
midcentury    = '2030_2059'
latecentury   = '2070_2099'

Weighted  = True
FigFormat = 'PNG'           # choose between 'PDF' and 'PNG' and 'None'
RCP_Versn = 'rcp60'         # choose from 'rcp26' and 'rcp60'
LTmethod  = 'SciPylsq'      # choosing between 'SciPylsq', 'SciPyTheilSen', 'np.corrcoef', 'pymannkendallTheilSen'
Sigmethd  = 'pymannkendall' # choosing among 'pymannkendall', 'SciPyWald', 'mkt_sg.mk_test', 'mkt.test'

TRsrcDIR  = '../Drought_TrendMap_AbnDryRemoved_Trn_' + LTmethod + '/'
SigsrcDIR = '../Drought_TrendMap_AbnDryRemoved_Sig_' + Sigmethd + '/'

glcr_mask_enlarged = Glaciers1xEnlargedMASK()
NOgrnlnd           = NOgreenlndNOantarcticaMASK()

global_lndMASK  = Global_LndAreaMASK() * NOgrnlnd #-180~180 in meter^2
totCONT_lndMASK = CONTs_LndMASK()                 #-180~180 in meter^2

r"""remove glaciers, greenland, and antarctica"""
NOgrnlndNOglcrsNOntctc = NOgrnlnd * glcr_mask_enlarged * totCONT_lndMASK

#%%------------------------------------------------------------------ Functions

def df_generate(mat,interval,Des_latsXY,Des_lonsXY,tresh):

	df = pd.DataFrame(columns = ['latsXY','lonsXY','maskVAL'])
	df['latsXY']  = Des_latsXY.reshape(int(180/interval)*int(360/interval))
	df['lonsXY']  = Des_lonsXY.reshape(int(180/interval)*int(360/interval))
	df['maskVAL'] = mat.reshape(int(180/interval)*int(360/interval))
	df = df[(df['maskVAL'] > tresh)]

	return df

def df_slice(df, sDate_slice, eDate_slice):
	df_slice = df[(df['Date'] >= sDate_slice) & (df['Date'] <= eDate_slice)].copy()
	df_slice = df_slice.reset_index(drop=True) # reindexing
	return df_slice

def df_slice_mean(df, col, tperiod):
	val = df[col][(df['Date'] >= tperiod[:4] + '-01-01') & (df['Date'] <= tperiod[-4:] + '-12-31')].mean()
	return val 

def df_mean(df, cols):

	dfmean  = pd.DataFrame(columns = cols) 
	dfmean.Date = df['Date'].copy()
	dfmean.Mean = meanFUNC(df)
	if len(cols) == 3:
		dfmean.Pop = meanFUNC(df) * PopArray.sum(2).sum(1)

	return dfmean


def isfile(file):
	return os.path.isfile('/egr/research-hydro/felfelan/ISIMIP2b/out_new/DSI_TS_DynFracAreaPop_AbnDryRemoved/' + file)

def meanFUNC(df):
	mean = df.iloc[:,1:].mean(axis = 1)
	return mean

def stdFUNC(df):
	std  = df.iloc[:,1:].std(axis = 1)
	return std

def trendMAP(mat,global_lndMASK):
	w, h = 360, 720
	coeff = [[-999.0 for x in range(h)] for y in range(w)] 
	coeff = np.array((coeff))

	for ii in range(360):
		for jj in range(720):
			if global_lndMASK[ii,jj] != 0.0:
				
				slope, intercept, r_value, p_value, std_err = stats.linregress(range(1,len(mat) + 1),mat[:,ii,jj])
				coeff[ii,jj] = slope				
	return coeff

def trendVal(df):

	Period1 = '2006-2049' 
	Period2 = '2050-2099'

	df_sec1 = df[(df.Date >= Period1[:4] + '-01-01') & (df.Date <= Period1[-4:] + '-12-01')].copy()
	df_sec2 = df[(df.Date >= Period2[:4] + '-01-01') & (df.Date <= Period2[-4:] + '-12-01')].copy()

	r""" sort based on the Date and reindex"""
	df_sec1 = df_sec1.sort_values(["Date"])
	df_sec1 = df_sec1.reset_index(drop=True)
	df_sec2 = df_sec2.sort_values(["Date"])
	df_sec2 = df_sec2.reset_index(drop=True)

	mean1 = df_sec1.iloc[:,1:].mean(axis = 1)
	mean2 = df_sec2.iloc[:,1:].mean(axis = 1)

	slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df_sec1.index, mean1)
	slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df_sec2.index, mean2)

	return slope1, p_value1, slope2, p_value2

def latlonXY(interval):
	lats_out = np.flip(np.arange(-90 + interval / 2, 90, interval), axis=0)
	lons_out = np.arange(-180 + interval / 2, 180, interval)
	latsXY_out = np.tile(lats_out.reshape(int(180 / interval), 1), (1, int(360 / interval)))
	lonsXY_out = np.tile(lons_out.reshape(1, int(360 / interval)), (int(180 / interval), 1))
	return latsXY_out, lonsXY_out
	
	
def df_DroughtTSconcat(RCP, Drought_str):
	for mod in models:
		for gcm in GCMs:
			fName = mod + '_' + gcm + '_ewembi_historical_' + RCP + '_' + SOECs[mod] + '_2005soc_' + Drought_str + '_'  + YRS			
			if isfile(fName):
				print('')
				print('======================== ' + RCP, Drought_str)
				print(mod + '_' + gcm + '_' + RCP + '_' + SOECs[mod])
				
				mod_gcm = mod + '_' + gcm
				df = pd.read_csv(srcDIR + fName, sep=',',header=0, error_bad_lines=False)
				df = df.rename(columns={'Fraction' : mod_gcm}).copy()	
				if 'clm45_gfdl-esm2m' in fName:
					df_concat  = df
				else:
					df_concat  = df_concat.merge(df, how='outer', on=['Date'])
	return df_concat

#%%------------------------------------------------------------------ Initiation

r""" Population Calculation"""
df_yymm_pop = pd.DataFrame(columns=['Date'])
df_yymm_pop.Date = pd.date_range('1/' + sDate_TS[:4], '12/' + eDate_TS[:4] , freq='MS')
PopArray = np.zeros((len(df_yymm_pop),360,720))

for ii in np.arange(2005,2105,5):
	popFile = np.load(PopDIR +  str(ii) + '_Gridded_SEDACbased_SSP2growth_Population_2005basis.npy')
	print(str(ii) + '_Gridded_SEDACbased_SSP2growth_Population_2005basis.npy')

	r""" two years before and two years after gets the population count of a given year"""
	if ii == 2005:
		idx = df_yymm_pop[(df_yymm_pop['Date'] < str(ii + 3))].index
		
	elif ii == 2100:
		idx = df_yymm_pop[(df_yymm_pop['Date'] >= str(ii - 2))].index
		
	else:
		idx = df_yymm_pop[(df_yymm_pop['Date'] >= str(ii - 2)) & (df_yymm_pop['Date'] < str(ii + 3))].index
		
	
	PopArray[idx,] = popFile

r"""Reading the continental weights for weighted mean"""
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

#%%------------------------------------------------------------------ Trend Maps

r"""reading trend maps"""
for mg in modgcm:
	
	print(mg + '_trend_' + RCP_Versn)
	
	trend_Mod0 = np.ma.masked_equal(np.load(TRsrcDIR + mg + '_trend_' + RCP_Versn + '_Mod_' + \
										 sDate_TR[:4] + '_' + eDate_TR[:4] + '.npy'),-999.0).reshape(1,360,720) * 30.5 #month/year ---> days/year
	trend_Ex0 = np.ma.masked_equal(np.load(TRsrcDIR + mg + '_trend_' + RCP_Versn + '_Ex_' + \
										 sDate_TR[:4] + '_' + eDate_TR[:4] + '.npy'),-999.0).reshape(1,360,720) * 30.5 #month/year ---> days/year

	signi_Mod0 = np.ma.masked_equal(np.load(SigsrcDIR + mg + '_signi_' + RCP_Versn + '_Mod_' + \
										 sDate_TR[:4] + '_' + eDate_TR[:4] + '.npy'),-999.0).filled(0).reshape(1,360,720)
	signi_Ex0 = np.ma.masked_equal(np.load(SigsrcDIR + mg + '_signi_' + RCP_Versn + '_Ex_' + \
										 sDate_TR[:4] + '_' + eDate_TR[:4] + '.npy'),-999.0).filled(0).reshape(1,360,720)

	if mg == "clm45_gfdl-esm2m":
		trend_Mod = deepcopy(trend_Mod0)
		trend_Ex  = deepcopy(trend_Ex0)

		signi_Mod = deepcopy(signi_Mod0)
		signi_Ex  = deepcopy(signi_Ex0)	

	else:
		trend_Mod = np.ma.concatenate([trend_Mod, trend_Mod0],axis = 0)		
		trend_Ex  = np.ma.concatenate([trend_Ex, trend_Ex0],axis = 0)		
		
		signi_Mod = np.ma.concatenate([signi_Mod, signi_Mod0],axis = 0)		
		signi_Ex  = np.ma.concatenate([signi_Ex, signi_Ex0],axis = 0)	

tot_mean = np.ma.concatenate([(trend_Mod * wFile).sum(0).reshape(1,360,720), \
							(trend_Ex * wFile).sum(0).reshape(1,360,720)],axis = 0)* NOgrnlnd

r"""Agreement in Trend Significance"""
signi_Ex_sum  = signi_Ex.sum(0)
signi_Mod_sum = signi_Mod.sum(0) 

signi_Ex_sum[signi_Ex_sum <= 0.5 * len(signi_Ex) ]   = 0.0
signi_Ex_sum[signi_Ex_sum > 0.5 * len(signi_Ex)  ]  = 1.

signi_Mod_sum[signi_Mod_sum <= 0.5 * len(signi_Mod) ]   = 0.0
signi_Mod_sum[signi_Mod_sum > 0.5 * len(signi_Mod)  ]  = 1.

r"""Agreement in Trend Sign"""
diff_Mod_Late = deepcopy(trend_Mod) 
diff_Ex_Late  = deepcopy(trend_Ex)

sign_mask_diff_Mod_Late = deepcopy(diff_Mod_Late)
sign_mask_diff_Ex_Late = deepcopy(diff_Ex_Late)

sign_mask_diff_Mod_Late[(sign_mask_diff_Mod_Late > 0.0)  & (sign_mask_diff_Mod_Late.mask==False)] = 1.0
sign_mask_diff_Mod_Late[(sign_mask_diff_Mod_Late < 0.0)  & (sign_mask_diff_Mod_Late.mask==False)] = -1.0

sign_mask_diff_Ex_Late[(sign_mask_diff_Ex_Late > 0.0) & (sign_mask_diff_Ex_Late.mask==False)] = 1.0
sign_mask_diff_Ex_Late[(sign_mask_diff_Ex_Late < 0.0) & (sign_mask_diff_Ex_Late.mask==False)] = -1.0


tot_mean_mask_bipolar = deepcopy(tot_mean)
tot_mean_mask_bipolar[(tot_mean_mask_bipolar > 0.0) & (tot_mean_mask_bipolar.mask==False)] = 1.0 * len(diff_Mod_Late) 
tot_mean_mask_bipolar[(tot_mean_mask_bipolar < 0.0) & (tot_mean_mask_bipolar.mask==False)] = -1.0 * len(diff_Mod_Late)


Models_sign_sum     = np.ma.concatenate([sign_mask_diff_Mod_Late.sum(0).reshape(1,360,720) , \
						sign_mask_diff_Ex_Late.sum(0).reshape(1,360,720)], axis = 0)

Multi_Model_sign_betwM1P1  = Models_sign_sum / np.ma.masked_equal(tot_mean_mask_bipolar,0.0)

r"""to map from (-1,+1) to (0,+1):   x-xmin/(xmax-xmin)"""
Multi_Model_sign_betw0P1  = (Multi_Model_sign_betwM1P1 + 1) / 2

Multi_Model_sign = deepcopy(Multi_Model_sign_betw0P1)

r"""regrid hlf degree sign_mask to 2 degree; hatch can be better plotted on coarse resolution"""
intervalHLF = 0.5
interval2dg = 2

latsXYhlf, lonsXYhlf = latlonXY(intervalHLF)
latsXY2dg, lonsXY2dg = latlonXY(interval2dg)

points = np.zeros((360*720,2))
points[:,0] = np.array(latsXYhlf.reshape(360*720))
points[:,1] = np.array(lonsXYhlf.reshape(360*720))  

sign_mask2d_diff_Mod_Late = griddata(points, Multi_Model_sign[0].filled(0).reshape(360*720), (latsXY2dg, \
                                                lonsXY2dg), method='linear')
sign_mask2d_diff_Ex_Late = griddata(points, Multi_Model_sign[1].filled(0).reshape(360*720), (latsXY2dg, \
                                                lonsXY2dg), method='linear')

signi_Mod_Late = griddata(points, signi_Mod_sum.reshape(360*720), (latsXY2dg, \
                                                lonsXY2dg), method='linear')
signi_Ex_Late = griddata(points, signi_Ex_sum.reshape(360*720), (latsXY2dg, \
                                                lonsXY2dg), method='linear')


df_mask_diff_Mod_Late1 = df_generate(sign_mask2d_diff_Mod_Late, interval2dg, latsXY2dg, lonsXY2dg, 0.5)
df_mask_diff_Ex_Late1  = df_generate(sign_mask2d_diff_Ex_Late,  interval2dg, latsXY2dg, lonsXY2dg, 0.5)

df_mask_diff_Mod_Late2 = df_generate(sign_mask2d_diff_Mod_Late, interval2dg, latsXY2dg, lonsXY2dg, 0.75)
df_mask_diff_Ex_Late2  = df_generate(sign_mask2d_diff_Ex_Late,  interval2dg, latsXY2dg, lonsXY2dg, 0.75)

r"""originl data is 0 or 1, regridded data can be between 0 and 1, above 0.1 is acceptable"""
df_mask_signi_Mod_Late = df_generate(signi_Mod_Late, interval2dg, latsXY2dg, lonsXY2dg, 0.1)
df_mask_signi_Ex_Late  = df_generate(signi_Ex_Late,  interval2dg, latsXY2dg, lonsXY2dg, 0.1)

my_dict1 = {}
my_dict1['df_mask_Mod_Late'] = df_mask_diff_Mod_Late1
my_dict1['df_mask_Ex_Late'] = df_mask_diff_Ex_Late1

my_dict2 = {}
my_dict2['df_mask_Mod_Late'] = df_mask_diff_Mod_Late2
my_dict2['df_mask_Ex_Late'] = df_mask_diff_Ex_Late2

my_dict3 = {}
my_dict3['df_mask_Mod_Late'] = df_mask_signi_Mod_Late
my_dict3['df_mask_Ex_Late'] = df_mask_signi_Ex_Late

dict_name = ['df_mask_Mod_Late', 'df_mask_Ex_Late']

#%%------------------------------------------------------------------For DSI Histograms

xx = 0
for file in sorted(os.listdir(CNTNTLdsiDIR)):
	if ('_mean_tws_DSI_' in file) and ('matsiro' not in file):
		print(file)
		df = pd.read_csv(CNTNTLdsiDIR + file,sep=',',header=0, error_bad_lines=False)
		df = df.drop(['Unnamed: 0'],axis=1)

		if xx == 0.0:
			df_sum = df.rename(columns={'CONT_mean_tws' : file}).copy()
			xx = 1.0
		else:
			df = df.rename(columns={'CONT_mean_tws' : file})
			df_sum = df.merge(df_sum, how='outer', on=['Date'])

r"""sort based on the Date and reindex"""
df_sum = df_sum.sort_values(["Date"])
df_sum = df_sum.reset_index(drop=True) 

Dict_TS = {}
for cont in CONTs:
	r"""accounting for weights for histograms is useless!!!!!
	because we don't take the mean! All members DSI are considred to generate the histograms"""
	CONT_lnd_mask = deepcopy(NOgrnlndNOglcrsNOntctc)
	CONT_lnd_mask[CONT_lnd_mask != CONTcode[cont]] = 0.0
	CONT_lnd_mask[CONT_lnd_mask == CONTcode[cont]] = 1.0
	
	w3DArray = np.ma.masked_equal(wFile * CONT_lnd_mask ,0.0)

	for clmt in ['_rcp26_', '_rcp60_']:
		filter_col1 = [col for col in list(df_sum) if (cont in col)]
		filter_col2 = [col for col in filter_col1 if (clmt in col)]
		
		df_model = df_sum[['Date'] + filter_col2].copy()
		df_model_dummy = df_model.copy()
		
		wSum = 0		
		for column in df_model_dummy.columns:
			if column != 'Date':
				mod = column.split('_')[0]
				gcm = column.split('_')[1]
				
				w2DArray = w3DArray[modgcm.index(mod + '_' + gcm),]
				
				if w2DArray.max() == w2DArray.min() and w2DArray.max() != 0.0:
					weight = w2DArray.max()
			
					df_model_dummy.loc[:,column] = df_model.loc[:,column] * weight
					wSum = wSum + weight
				else:
					'=========CAUCIOUS============='
	
		print(cont, '     wSum: ', wSum)
	
		df_model['meanOFcols'] = df_model_dummy.iloc[:,1:].sum(axis = 1)
	   
		r"""remove nan values checking all columns Except for Date
		if all values in a row is Nan; drop the row"""
		df_model = df_model.dropna(subset= df_model.columns[df_model.columns != 'Date'],how = 'all')
		
	
		Dict_TS[cont + '_' + clmt[1:-1]] = df_model

#%%------------------------------------------------------------------ For Time Series

if RCP_Versn == 'rcp60':
	df_sum_HiRCP60_ExDroughtArea  = df_DroughtTSconcat('rcp60', 'ExDroughtArea')
	df_sum_HiRCP60_ExDroughtPop   = df_DroughtTSconcat('rcp60', 'ExDroughtPop')
	df_sum_HiRCP60_ModDroughtArea = df_DroughtTSconcat('rcp60', 'ModDroughtArea')
	df_sum_HiRCP60_ModDroughtPop  = df_DroughtTSconcat('rcp60', 'ModDroughtPop')

	df_slice_ExDroughtArea  = df_slice(df_sum_HiRCP60_ExDroughtArea, sDate_TS, eDate_TS)
	df_slice_ExDroughtPop   = df_slice(df_sum_HiRCP60_ExDroughtPop, sDate_TS, eDate_TS)
	df_slice_ModDroughtArea = df_slice(df_sum_HiRCP60_ModDroughtArea, sDate_TS, eDate_TS)
	df_slice_ModDroughtPop  = df_slice(df_sum_HiRCP60_ModDroughtPop, sDate_TS, eDate_TS)
	plotlabel = ['RCP6.0 Extreme-Exceptional','RCP6.0 Moderate-Severe' , r'$\pm$' + '1SD', r'$\pm$' + '1SD']

elif RCP_Versn == 'rcp26':

	df_sum_HiRCP26_ExDroughtArea    = df_DroughtTSconcat('rcp26', 'ExDroughtArea')
	df_sum_HiRCP26_ExDroughtPop     = df_DroughtTSconcat('rcp26', 'ExDroughtPop')
	df_sum_HiRCP26_ModDroughtArea   = df_DroughtTSconcat('rcp26', 'ModDroughtArea')
	df_sum_HiRCP26_ModDroughtPop    = df_DroughtTSconcat('rcp26', 'ModDroughtPop')

	df_slice_ExDroughtArea  = df_slice(df_sum_HiRCP26_ExDroughtArea, sDate_TS, eDate_TS)
	df_slice_ExDroughtPop   = df_slice(df_sum_HiRCP26_ExDroughtPop, sDate_TS, eDate_TS)
	df_slice_ModDroughtArea = df_slice(df_sum_HiRCP26_ModDroughtArea, sDate_TS, eDate_TS)
	df_slice_ModDroughtPop  = df_slice(df_sum_HiRCP26_ModDroughtPop, sDate_TS, eDate_TS)
	plotlabel = ['RCP2.6 Extreme-Exceptional','RCP2.6 Moderate-Severe' , r'$\pm$' + '1SD', r'$\pm$' + '1SD']

df_ExDroughtArea  = df_mean(df_slice_ExDroughtArea, ['Date','Mean'])
df_ExDroughtPop   = df_mean(df_slice_ExDroughtPop, ['Date','Mean', 'Pop'])
df_ModDroughtArea = df_mean(df_slice_ModDroughtArea, ['Date','Mean'])
df_ModDroughtPop  = df_mean(df_slice_ModDroughtPop, ['Date','Mean', 'Pop'])

ExDroughtArea_slope2049 , ExDroughtArea_Pvalue2049 , ExDroughtArea_slope2099 , ExDroughtArea_Pvalue2099  = trendVal(df_slice_ExDroughtArea)
ExDroughtPop_slope2049  , ExDroughtPop_Pvalue2049  , ExDroughtPop_slope2099  , ExDroughtPop_Pvalue2099   = trendVal(df_slice_ExDroughtPop)
ModDroughtArea_slope2049, ModDroughtArea_Pvalue2049, ModDroughtArea_slope2099, ModDroughtArea_Pvalue2099 = trendVal(df_slice_ModDroughtArea)
ModDroughtPop_slope2049 , ModDroughtPop_Pvalue2049 , ModDroughtPop_slope2099 , ModDroughtPop_Pvalue2099  = trendVal(df_slice_ModDroughtPop)

#%%------------------------------------------------------------------ Plotting    

fig = plt.figure(num=1, figsize=(8.,9)) #figsize: w,h letter size	

gs = gridspec.GridSpec(2, 1, height_ratios = [1,1]) # the middle row is to place the colorbar axes
gs.update(bottom=0.3, top=0.98, left = 0.0, right = 0.74,hspace = 0.15)			

gs2 = gridspec.GridSpec(6, 1, height_ratios = [1,1,1,1,1,1]) # the middle row is to place the colorbar axes
gs2.update(bottom=0.3, top=0.98, left = 0.80, right = 1.,hspace = 0.2)

gs3 = gridspec.GridSpec(2, 1, height_ratios = [1,1]) # the middle row is to place the colorbar axes
gs3.update(bottom=0.03, top=0.26, left = 0.05, right = .98,hspace = 0.1)


cmap = plt.cm.coolwarm
cmaplist = [cmap(i) for i in range(cmap.N)]

cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
bounds_tick1 = np.array([-2,-1.6,-1.2,-0.8,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.8,1.2,1.6,2])

norm1 = mpl.colors.BoundaryNorm(bounds_tick1, cmap.N)
titles = ['(a)','(b)', '(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']	

lwCOAST = 0.4
lwCONTRS = 0.4
for ii in range(2):
	ax = plt.subplot(gs[ii])

	nxa=-180
	nxb=180
	nya=90
	nyb=-60
	res=1
	map=Basemap( projection ='cyl',  \
			llcrnrlon  = nxa,  \
			 urcrnrlon  = nxb,  \
			 llcrnrlat  = nyb,  \
			 urcrnrlat  =  nya,  \
			 resolution = "c")
	map.drawcoastlines(linewidth=lwCOAST, color='black')
	map.drawcountries(linewidth=lwCONTRS, color='grey')
	cs = map.imshow(tot_mean[ii,:cylHLFdegxy(nyb,0)[1],:] \
				 ,origin='upper',norm = norm1, interpolation='nearest',cmap=cmap)	

	t1 = mpl.markers.MarkerStyle(marker='_')
	t1._transform = t1.get_transform().rotate_deg(45)

	t2 = mpl.markers.MarkerStyle(marker='_')
	t2._transform = t2.get_transform().rotate_deg(-45)
	
	if FigFormat == 'PNG' or FigFormat == 'None':
		map.scatter(my_dict1[dict_name[ii]]['lonsXY'],my_dict1[dict_name[ii]]['latsXY'],latlon='True',marker=t1, \
					  s=6,linewidth='0.2',facecolors='k',edgecolors='none',alpha=1.) # for PNG

		map.scatter(my_dict2[dict_name[ii]]['lonsXY'],my_dict2[dict_name[ii]]['latsXY'],latlon='True',marker=t2, \
					  s=6,linewidth='0.2',facecolors='k',edgecolors='none',alpha=1.) # for PNG


		map.scatter(my_dict3[dict_name[ii]]['lonsXY'],my_dict3[dict_name[ii]]['latsXY'],latlon='True',marker='8', \
					  s=4,alpha=0.5, facecolor = 'k', edgecolor = 'none') # for PNG

	elif FigFormat == 'PDF':
		map.scatter(my_dict1[dict_name[ii]]['lonsXY'],my_dict1[dict_name[ii]]['latsXY'],latlon='True',marker=t1, \
					  s=6,linewidth='0.1',facecolors='k',edgecolors='none',alpha=.5) # for PDF

		map.scatter(my_dict2[dict_name[ii]]['lonsXY'],my_dict2[dict_name[ii]]['latsXY'],latlon='True',marker=t2, \
					  s=6,linewidth='0.1',facecolors='k',edgecolors='none',alpha=.5) # for PDF


		map.scatter(my_dict3[dict_name[ii]]['lonsXY'],my_dict3[dict_name[ii]]['latsXY'],latlon='True',marker='8', \
					  s=4,alpha=0.5, facecolor = 'k', edgecolor = 'none') # for PDF


	ax.text(-180,85,titles[ii], fontsize = 10)
	ax.axis('off')

	if ii == 0:
		cbar_ax = fig.add_axes([0.15, 0.66, 0.5, 0.015])    
		cb = fig.colorbar(cs, cax=cbar_ax,spacing='uniform',orientation='horizontal',norm=norm1, ticks=bounds_tick1, boundaries=bounds_tick1,)
		cb.ax.tick_params(labelsize=8)
		cb.outline.set_visible(False)
		cb.ax.set_xticklabels(bounds_tick1)
		cb.set_label('($days/year$)',fontsize=8,labelpad=-22, x=1.1)

for jj in range(6):
	inset_ax_cnt = plt.subplot(gs2[jj])
	r""" rcp 60 / 26 histograms"""
	df_cnt = Dict_TS[CONTs[jj] + '_' + RCP_Versn][(Dict_TS[CONTs[jj] + '_' + RCP_Versn]['Date'] >= latecentury[:4] + '-01') & \
					(Dict_TS[CONTs[jj] + '_' + RCP_Versn]['Date'] <= latecentury[-4:] + '-12')]
	aa = df_cnt[df_cnt.columns[df_model.columns != 'Date'] & df_cnt.columns[df_model.columns != 'meanOFcols']]
	
	r""" No need to do weighted mean for the histograms, because we consider all the members!!!!"""
	aa = np.array(aa).reshape(np.shape(aa)[0]*np.shape(aa)[1])

	r"""hist histograms"""
	df_cnt_hist = Dict_TS[CONTs[jj] + '_rcp26'][(Dict_TS[CONTs[jj] + '_rcp26']['Date'] >= baseline[:4] + '-01') & \
					(Dict_TS[CONTs[jj] + '_rcp26']['Date'] <= baseline[-4:] + '-12')]
	bb = df_cnt_hist[df_cnt_hist.columns[df_model.columns != 'Date'] & df_cnt_hist.columns[df_model.columns != 'meanOFcols']]
	bb = np.array(bb).reshape(np.shape(bb)[0]*np.shape(bb)[1])

	cm = plt.cm.coolwarm_r
	cm = plt.cm.BrBG
	
	cmaplist2 = [cm(i) for i in range(cm.N)]
	cm = cmap.from_list('Custom cmap', cmaplist2[40:-40], cm.N)

	n1, bins1, patches1 = inset_ax_cnt.hist(aa,bins = [-3,-2.0,-1.6,-1.3, -0.8,-0.5, 0.5,0.8,1.3,1.6,2.0,3], \
											 color='firebrick',ec='firebrick',density=True, alpha=.5, label = '2070-2099' )
										 
	n2, bins2, patches2 = inset_ax_cnt.hist(bb,bins = [-3,-2.0,-1.6,-1.3, -0.8,-0.5, 0.5,0.8,1.3,1.6,2.0,3], \
										 color='royalblue',ec='royalblue',density=True, alpha=.5, label = 'HIST')

	inset_ax_cnt.text(-2.7,0.5,titles[jj + 2] + ' ' + CONTs_name[CONTs[jj]], fontsize = 8)
	print('')
	print(CONTs[jj])
	binlist1 = []
	for i, p in enumerate(n1):
		inset_ax_cnt.text(bins1[i]+0.2, 0.2, str(round(p,4)), color='firebrick', fontsize=5, rotation=90.0)
		binlist1 = binlist1 + [round(p,4)]	

	
	binlist2 = []	
	for i, p in enumerate(n2):
		inset_ax_cnt.text(bins2[i]+0.2, 0.45, str(round(p,4)), color='royalblue', fontsize=5, rotation=90.0)
		binlist2 = binlist2 + [round(p,4)]		

	for ll in range(len(n1)):
		print(binlist2[ll], binlist1[ll])
	
	inset_ax_cnt.set_xlim(-3.,3)
	inset_ax_cnt.set_ylim(0,0.6)

	inset_ax_cnt.set_xticks([-3,-2.0,-1.6,-1.3, -0.8,-0.5, 0.5,0.8,1.3,1.6,2.0,3])
	inset_ax_cnt.set_xticklabels([])

	if jj == 0:
		plt.legend(loc='lower left',bbox_to_anchor=(0., 1.02, 1., .102), ncol =2,fontsize = 7, mode="expand" , borderaxespad=0.)
	
	if jj == 5:
		inset_ax_cnt.set_xticklabels(['-3.0','-2.0','-1.6','-1.3', '-0.8','-0.5', '0.5','0.8','1.3','1.6','2.0','3.0'], fontsize=3, rotation=-90.0)


	inset_ax_cnt.tick_params(axis = 'both', which = 'major', labelsize = 6)

	inset_ax_cnt.spines['right'].set_visible(False)
	inset_ax_cnt.spines['top'].set_visible(False)

LineColors = ['firebrick','royalblue']
ShadColors = ['firebrick','royalblue']
axx = plt.subplot(gs3[0])

r"""Because plt.plot can plot more than one line at once, it returns a list of line2D objects, even if you only plot one line 
(i.e. in your case, a list of length 1). When you grab its handle for the legend, you want to only use the first item of 
this list (the actual line2D object)."""

dd = [df_slice_ExDroughtArea.index[df_slice_ExDroughtArea.Date == '2006-01-01'][0]]
for yr in np.arange(2010,2100,5):
	dd = dd + [df_slice_ExDroughtArea.index[df_slice_ExDroughtArea.Date == str(yr) + '-01-01'][0]]
dd = dd + [df_slice_ExDroughtArea.index[df_slice_ExDroughtArea.Date == '2099-01-01'][0]]

r"""add a comma after normplt (line plot) when you call plt.plot, to only store the first item from the list in normplt"""
p1, = axx.plot(df_slice_ModDroughtArea['Date'], meanFUNC(df_slice_ModDroughtArea), color = LineColors[1]  ,linewidth = 0.8,label = plotlabel[1])

p2 = axx.fill_between(df_slice_ModDroughtArea['Date'], meanFUNC(df_slice_ModDroughtArea) + stdFUNC(df_slice_ModDroughtArea), \
									 meanFUNC(df_slice_ModDroughtArea) - stdFUNC(df_slice_ModDroughtArea) , color = ShadColors[1],alpha = 0.5,label = plotlabel[3])

p3, = axx.plot(df_slice_ExDroughtArea['Date'], meanFUNC(df_slice_ExDroughtArea), color = LineColors[0]  ,linewidth = 0.8,label = plotlabel[0])
p4 = axx.fill_between(df_slice_ExDroughtArea['Date'], meanFUNC(df_slice_ExDroughtArea) + stdFUNC(df_slice_ExDroughtArea), \
									 meanFUNC(df_slice_ExDroughtArea) - stdFUNC(df_slice_ExDroughtArea) , color = ShadColors[0],alpha = 0.5,label = plotlabel[2])

axx.tick_params(axis = 'both', which = 'major', direction = 'inout', labelsize = 8, right = True, labelright=False)
axx.grid(linestyle = '--')
axx.set_xticks(dd)
axx.set_xticklabels([])
axx.set_ylim(bottom = 0.0, top = 0.29)
axx.set_xlim(df_slice_ExDroughtArea.index[0], df_slice_ExDroughtArea.index[-1])

axx.set_ylabel('Fractional\nArea', fontsize=8)
axx.text(0.01,0.85,titles[8], fontsize = 10, transform=axx.transAxes)

axxx = fig.add_subplot(gs3[1])

axxx.plot(df_slice_ModDroughtPop['Date'], meanFUNC(df_slice_ModDroughtPop), color = LineColors[1]  ,linewidth = 0.8,label = plotlabel[1])
axxx.fill_between(df_slice_ModDroughtPop['Date'], meanFUNC(df_slice_ModDroughtPop) + stdFUNC(df_slice_ModDroughtPop), \
									 meanFUNC(df_slice_ModDroughtPop) - stdFUNC(df_slice_ModDroughtPop) , color = ShadColors[1],alpha = 0.5,label = plotlabel[3])


axxx.plot(df_slice_ExDroughtPop['Date'], meanFUNC(df_slice_ExDroughtPop), color = LineColors[0]  ,linewidth = 0.8,label = plotlabel[0])
axxx.fill_between(df_slice_ExDroughtPop['Date'], meanFUNC(df_slice_ExDroughtPop) + stdFUNC(df_slice_ExDroughtPop), \
									 meanFUNC(df_slice_ExDroughtPop) - stdFUNC(df_slice_ExDroughtPop) , color = ShadColors[0],alpha = 0.5,label = plotlabel[2])

axxx.tick_params(axis = 'both', which = 'major', direction = 'inout', labelsize = 8, right = True, labelright=False)
axxx.set_ylim(bottom = 0.0, top = 0.29)
axxx.grid(linestyle = '--')
axxx.set_xticks(dd)
axxx.set_xticklabels(pd.to_datetime(df_slice_ExDroughtArea['Date']).dt.year[dd], fontsize=8, rotation=-45)

for tick in axxx.xaxis.get_majorticklabels():
	tick.set_horizontalalignment('center')

axxx.set_ylabel('Fractional\nPopulation', fontsize=8)
axxx.set_xlim(df_slice_ExDroughtArea.index[0], df_slice_ExDroughtArea.index[-1])
axxx.text(0.01,0.85,titles[9], fontsize = 10, transform=axxx.transAxes)

fontP = FontProperties()
fontP.set_size(8)	
axx.legend(handles=[p1,p2,p3,p4], ncol=4, prop=fontP,bbox_to_anchor=(-0.238, 1.165, 1., .102))

plt.setp(axx.spines.values(), color='darkgray')
plt.setp(axxx.spines.values(), color='darkgray')

if FigFormat == 'PNG':
	plt.savefig( figDIR + 'NCC_Fig4_v1_' + RCP_Versn + '_DSI_Weighted_TrendMAP_' + LTmethod + '_signi_' + Sigmethd + '_50&75percAgree_50percSigni_DynPop_' +  sDate_TR[:4] + '_' + eDate_TR[:4] + '_Continents_HISTOG.png', bbox_inches='tight', dpi=500 )
	plt.close()

elif FigFormat == 'PDF':
	plt.savefig( figDIR + 'NCC_Fig4_v1_' + RCP_Versn + '_DSI_Weighted_TrendMAP_' + LTmethod + '_signi_' + Sigmethd +  '_50&75percAgree_50percSigni_DynPop_' +  sDate_TR[:4] + '_' + eDate_TR[:4] + '_Continents_HISTOG.pdf', bbox_inches='tight' )
	plt.close()