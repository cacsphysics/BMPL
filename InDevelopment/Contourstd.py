
import scipy.io as spio
import matplotlib.pylab as plt
import numpy as np
import os
from load_hdf5 import load_hdf5
import spectrum_wwind as spec
import indexfinderfuncs as iff
import get_corr as gc
import scipy.integrate as sp
from scipy.interpolate import interp1d

from scipy.signal import butter, sosfiltfilt, sosfreqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

import smooth as sm

def iter_smooth(array,loops=6,window_len=3):
    for l in np.arange(loops):
        array = sm.smooth(array,window_len=window_len)
    return array


# Sample rate and desired cutoff frequencies (in Hz).
fs = 125e6
lowcut = 50e3
highcut = 2e6

directory='C:\\Users\\dschaffner\\Dropbox\\Data\\BMPL\\BMX\\2023\\07182023\\'
#datafilename='Dataset_06282023_course_radscan.h5'
#datafilename='Dataset_07132023_blockertest.h5'
datafilename= 'Dataset_07182023_full_radscan.h5'
data=load_hdf5(directory+datafilename,verbose=True)


time_s = data['time']['time_s']
timeB_s = time_s[1:]
time_us = data['time']['time_us']
timeB_us = time_us[1:]
analysis_start_time = 60
analysis_end_time = 160
start_time_index = iff.tindex_min(analysis_start_time,timeB_us)
end_time_index = iff.tindex_min(analysis_end_time,timeB_us)
timerange_limit = 3e-6#s
port_sep = 0.0254#m

numshots=10




directions = ['r','t','z']
spec=np.zeros([7,5,20,3,6251])
for r in np.arange(7):
    for probe in np.arange(5):
        for n in np.arange(20):
            for d in directions: 
                arr=data['mag_probe'][d]['b'][r,probe,n,:]
                f,f0,comp1,pwr,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(arr[start_time_index:end_time_index],timeB_s[start_time_index:end_time_index],window='hanning')
                spec[r,probe,n,d,:]=pwr
                
                
        data1_1=data['mag_probe']['r']['b'][1,probe,x,:]
        f,f0,comp1,pwr,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(data1_1[start_time_index:end_time_index],timeB_s[start_time_index:end_time_index],window='hanning')
        [1,]
        
        data1_2=data['mag_probe']['r']['b'][2,probe,x,:]
        data1_3=data['mag_probe']['r']['b'][3,probe,x,:]
        data1_4=data['mag_probe']['r']['b'][4,probe,x,:]
        data1_5=data['mag_probe']['r']['b'][5,probe,x,:]
        data1_6=data['mag_probe']['r']['b'][6,probe,x,:]
        
        data2_0=data['mag_probe']['t']['b'][0,probe,x,:]
        data2_1=data['mag_probe']['t']['b'][1,probe,x,:]
        data2_2=data['mag_probe']['t']['b'][2,probe,x,:]
        data2_3=data['mag_probe']['t']['b'][3,probe,x,:]
        data2_4=data['mag_probe']['t']['b'][4,probe,x,:]
        data2_5=data['mag_probe']['t']['b'][5,probe,x,:]
        data2_6=data['mag_probe']['t']['b'][6,probe,x,:]
        
        data3_0=data['mag_probe']['z']['b'][0,probe,x,:]
        data3_1=data['mag_probe']['z']['b'][1,probe,x,:]
        data3_2=data['mag_probe']['z']['b'][2,probe,x,:]
        data3_3=data['mag_probe']['z']['b'][3,probe,x,:]
        data3_4=data['mag_probe']['z']['b'][4,probe,x,:]
        data3_5=data['mag_probe']['z']['b'][5,probe,x,:]
        data3_6=data['mag_probe']['z']['b'][6,probe,x,:]
        
        # create the total magnetic field
        dataB_0 = np.sqrt(data1_0**2 + data2_0**2 + data3_0**2)
        dataB_1 = np.sqrt(data1_1**2 + data2_1**2 + data3_1**2)
        dataB_2 = np.sqrt(data1_2**2 + data2_2**2 + data3_2**2)
        dataB_3 = np.sqrt(data1_3**2 + data2_3**2 + data3_3**2)
        dataB_4 = np.sqrt(data1_4**2 + data2_4**2 + data3_4**2)
        dataB_5 = np.sqrt(data1_5**2 + data2_5**2 + data3_5**2)
        dataB_6 = np.sqrt(data1_6**2 + data2_6**2 + data3_6**2)
        
        stdB_0 = np.std(dataB_0[start_time_index:end_time_index])
        stdB_1 = np.std(dataB_1[start_time_index:end_time_index])
        stdB_2 = np.std(dataB_2[start_time_index:end_time_index])
        stdB_3 = np.std(dataB_3[start_time_index:end_time_index])
        stdB_4 = np.std(dataB_4[start_time_index:end_time_index])
        stdB_5 = np.std(dataB_5[start_time_index:end_time_index])
        stdB_6 = np.std(dataB_6[start_time_index:end_time_index])

        meanB_0 = np.mean(dataB_0[start_time_index:end_time_index])
        meanB_1 = np.mean(dataB_1[start_time_index:end_time_index])
        meanB_2 = np.mean(dataB_2[start_time_index:end_time_index])
        meanB_3 = np.mean(dataB_3[start_time_index:end_time_index])
        meanB_4 = np.mean(dataB_4[start_time_index:end_time_index])
        meanB_5 = np.mean(dataB_5[start_time_index:end_time_index])
        meanB_6 = np.mean(dataB_6[start_time_index:end_time_index])

        storearr_B_0[x]=stdB_0/meanB_0
        storearr_B_1[x]=stdB_1/meanB_1
        storearr_B_2[x]=stdB_2/meanB_2
        storearr_B_3[x]=stdB_3/meanB_3
        storearr_B_4[x]=stdB_4/meanB_4
        storearr_B_5[x]=stdB_5/meanB_5
        storearr_B_6[x]=stdB_6/meanB_6
    
        
    shotmeanB_0 = np.mean(storearr_B_0) 
    shotmeanB_1 = np.mean(storearr_B_1)
    shotmeanB_2 = np.mean(storearr_B_2)
    shotmeanB_3 = np.mean(storearr_B_3)
    shotmeanB_4 = np.mean(storearr_B_4)
    shotmeanB_5 = np.mean(storearr_B_5)
    shotmeanB_6 = np.mean(storearr_B_6)

        
     

    # plt.title("Average Magnitude of Magnetic Field at Position 15 [G]")
    # plt.ylabel("Average Magnitude of Magnetic Field [G]")
    # plt.xlabel("Time [$\mu s$]")
    
    
    # plt.plot(timeB_us, meanB_0, label='Magnetic Field at Center')
    # plt.plot(timeB_us, meanB_1, label='Magnetic Field 0.5 Inches Away From Center')
    # plt.plot(timeB_us, meanB_2, label='Magnetic Field 1 Inch Away From Center')
    # plt.plot(timeB_us, meanB_3, label='Magnetic Field 1.5 Inches Away From Center')
    # plt.plot(timeB_us, meanB_4, label='Magnetic Field 2 Inches Away From Center')
    # plt.plot(timeB_us, meanB_5, label='Magnetic Field 2.5 Inches Away From Center')
    # plt.plot(timeB_us, meanB_6, label='Magnetic Field 3 Inches Away From Center')
    # plt.legend(loc='upper right',fontsize=4)
    
    ###SECOND GRAPH CODE BEGINS HERE
    
    meanB_0_160 = np.std((dataB_0[start_time_index:end_time_index]))
    meanB_1_160 = np.std((dataB_1[start_time_index:end_time_index]))
    meanB_2_160 = np.std((dataB_2[start_time_index:end_time_index]))
    meanB_3_160 = np.std((dataB_3[start_time_index:end_time_index]))
    meanB_4_160 = np.std((dataB_4[start_time_index:end_time_index]))
    meanB_5_160 = np.std((dataB_5[start_time_index:end_time_index]))
    meanB_6_160 = np.std((dataB_6[start_time_index:end_time_index]))
    
    if probe==0:
        z[0,probe]=shotmeanB_0
        z[1,probe]=shotmeanB_1
        z[2,probe]=shotmeanB_2
        z[3,probe]=shotmeanB_3
        z[4,probe]=shotmeanB_4
        z[5,probe]=shotmeanB_5
        z[6,probe]=shotmeanB_6
    if probe>=2:
        z[0,probe-1]=shotmeanB_0
        z[1,probe-1]=shotmeanB_1
        z[2,probe-1]=shotmeanB_2
        z[3,probe-1]=shotmeanB_3
        z[4,probe-1]=shotmeanB_4
        z[5,probe-1]=shotmeanB_5
        z[6,probe-1]=shotmeanB_6


"""
y=np.arange(7)*0.5
x=np.arange(4)


#Plot Details###############################
plt.rc('axes',linewidth=0.5)
plt.rc('xtick.major',width=0.5)
plt.rc('ytick.major',width=0.5)
plt.rc('xtick.minor',width=0.5)
plt.rc('ytick.minor',width=0.5)
plt.rc('lines',markersize=2.0,markeredgewidth=0.0,linewidth=0.5)
fig=plt.figure(num=1,figsize=(5,5),dpi=300,facecolor='w',edgecolor='k')


left  = 0.13  # the left side of the subplots of the figure
right = 0.92    # the right side of the subplots of the figure
bottom = 0.22  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax1=plt.subplot(1,1,1)   






cs=plt.contourf(x,y,z,levels=100,cmap='plasma')
cbar=plt.colorbar()
cbar.set_label(r'$\Delta B / B$', rotation=270, labelpad=15)

plt.title(r"Average $\Delta B/B$ over 20 shots, 60-160us",fontsize=9)
plt.ylabel("Radial Position from Central Axis [in]",fontsize=8)
plt.xlabel("Axial Distance from Block [in]", fontsize=8)
plt.xlim(-1.0,3)
plt.xticks(np.array([-1,0,1,2,3]), [
           0,1,2,3,4], fontsize=9)

plt.vlines(-1, 1.5, 3, color='grey', linestyle='solid', linewidth=50.0)

savefile = 'blocker_scan_contour_deltaBoverB.png'
plt.savefig(savefile, dpi=300, facecolor='w', edgecolor='k')


#plt.yticks(np.array([0, 0.5, 1, 1.5,2, 2.5, 3]), fontsize=9)
"""