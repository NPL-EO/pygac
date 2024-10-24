#!/usr/bin/env python

# Copyright (c) 2014-2015, 2019 Pytroll Developers

# Author(s):

#   Jonathan Mittaz <j.mittaz@reading.ac.uk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Uncertainty information based on the noaa.py calibration for IR channels
"""
from __future__ import division

import numpy as np
import xarray as xr
from importlib.resources import files
import argparse
import matplotlib.pyplot as plt

from pygac import get_reader_class
from pygac.calibration.noaa import Calibrator
from pygac.utils import allan_deviation

class convBT(object):
    """Routine to covert temperature to radiance and visa-versa"""
    def t_to_rad(self,tprt):

        tsBB = self.A + self.B*tprt
        return self.nBB_num / (np.exp(self.c2_nu_c / tsBB) - 1.0)
        
    def rad_to_t(self,rad):

        corrT = self.c2_nu_c/np.log((self.nBB_num/rad)+1.)
        return (corrT-self.A)/self.B

    def rad_to_t_uncert(self,rad,urad):

        T = self.rad_to_t(rad)
        T1 = self.rad_to_t(rad+urad)
        T2 = self.rad_to_t(rad-urad)

        return T,(np.abs(T-T1)+np.abs(T-T2))/2.

    def t_to_rad_uncert(self,T,uT):

        rad = self.t_to_rad(T)
        rad1 = self.t_to_rad(T+uT)
        rad2 = self.t_to_rad(T-uT)

        return rad,(np.abs(rad-rad1)+np.abs(rad-rad2))/2.

    def __init__(self,cal,chan):

        # constants
        self.c1 = 1.1910427e-5  # mW/m^2/sr/cm^{-4}
        self.c2 = 1.4387752  # cm K
        # coefficients
        self.A = cal.to_eff_blackbody_intercept[chan]
        self.B = cal.to_eff_blackbody_slope[chan]
        self.nu_c = cal.centroid_wavenumber[chan]
        self.nBB_num = self.c1 * (self.nu_c**3)
        self.c2_nu_c = self.c2 * self.nu_c

def get_bad_space_counts(sp_data,ict_data):
    """Find bad space count data (space count data is voltage clamped so
    should have very close to the same value close to 950 - 960
    Written by J.Mittaz / University of Reading 6 Oct 2024"""
    
    #
    # Use robust estimators to get thresholds for space counts
    # Use 4 sigma threshold from median
    # Ensure only for good data
    #
    gd = np.isfinite(sp_data)
    if np.sum(gd) == 0:
        sp_bad_data = np.zeros(sp_data.shape,dtype=bool)
        sp_bad_data[:,:] = True
        return sp_bad_data

    quantile = np.quantile(sp_data[gd].flatten(),[0.25,0.75])
    if quantile[0] == quantile[1]:
        quantile[1] = quantile[1]+0.5
    std = (quantile[1]-quantile[0])/1.349
    sp_bad_data = np.zeros(sp_data.shape,dtype=bool)
    sp_bad_data[:,:] = True
    gd = np.isfinite(sp_data)
    sp_bad_data[gd] = ~((np.abs(sp_data[gd] - np.median(sp_data[gd].flatten()))/\
                         std < 4.)&(ict_data[gd] > 0))

    return sp_bad_data

def get_noise(total_space,total_ict,window,twelve_micron):
    """Get noise estimates from the counts"""

    #
    # Find bad space view data
    #
    bad_data_1 = get_bad_space_counts(total_space[:,:,0],total_ict[:,:,0])
    bad_data_2 = get_bad_space_counts(total_space[:,:,1],total_ict[:,:,1])
    if twelve_micron:
        bad_data_3 = get_bad_space_counts(total_space[:,:,2],total_ict[:,:,2])
    bad_scans = np.zeros(total_space.shape[0],dtype=np.int8)
    if twelve_micron:
        for i in range(len(bad_scans)):
            if np.any(bad_data_1[i,:]) or np.any(bad_data_2[i,:]) or \
               np.any(bad_data_3[i,:]):
                bad_scans[i] = 1
    else:
        for i in range(len(bad_scans)):
            if np.any(bad_data_1[i,:]) or np.any(bad_data_2[i,:]):
                bad_scans[i] = 1

    #
    # Estimate noise using the Allan deviation plus the digitisation 
    # uncertainty
    #
    # 3.7 micron space counts
    #
    noise1 = allan_deviation(ds["total_space_counts"].values[:,:,0],bad_scan=bad_scans)
    noise1 = np.sqrt(noise1*noise1 + 1./3)

    #
    # 11 micron space counts
    #
    noise2 = allan_deviation(ds["total_space_counts"].values[:,:,1],bad_scan=bad_scans)
    noise2 = np.sqrt(noise2*noise2 + 1./3)

    #
    # 12 micron space counts
    #
    if twelve_micron:
        noise3 = allan_deviation(ds["total_space_counts"].values[:,:,2],bad_scan=bad_scans)
        noise3 = np.sqrt(noise3*noise3 + 1./3)
    else:
        noise3 = None

    #
    # 3.7 micron ICT counts
    #
    ict_noise1 = allan_deviation(ds["total_ict_counts"].values[:,:,0],bad_scan=bad_scans)
    ict_noise1 = np.sqrt(ict_noise1*ict_noise1 + 1./3)

    #
    # 11 micron ICT counts
    #
    ict_noise2 = allan_deviation(ds["total_ict_counts"].values[:,:,1],bad_scan=bad_scans)
    ict_noise2 = np.sqrt(ict_noise2*ict_noise2 + 1./3)

    #
    # 12 micron ICT counts
    #
    if twelve_micron:
        ict_noise3 = allan_deviation(ds["total_ict_counts"].values[:,:,2],bad_scan=bad_scans)
        ict_noise3 = np.sqrt(ict_noise3*ict_noise3 + 1./3)
    else:
        ict_noise3 = None

    #
    # Calculate the uncertainty after averaging - note 10 measurements per
    # scanline
    #
    sqrt_window = np.sqrt(window*10)
    av_noise1 = noise1/sqrt_window
    av_noise2 = noise2/sqrt_window
    if twelve_micron:
        av_noise3 = noise3/sqrt_window
    else:
        av_noise3 = None
    av_ict_noise1 = ict_noise1/sqrt_window
    av_ict_noise2 = ict_noise2/sqrt_window
    if twelve_micron:
        av_ict_noise3 = ict_noise3/sqrt_window
    else:
        av_ict_noise3 = None

    return noise1,noise2,noise3,av_noise1,av_noise2,av_noise3,\
        av_ict_noise1,av_ict_noise2,av_ict_noise3,bad_scans

def get_uICT(gainval,CS,CICT,Tict,NS,convT,bad_scans):
    """Get ICT temperature gradient uncertainty based on analysis of the
    gain variations in the 3.7 micron channel"""
    
    gd = (bad_scans == 0)
    Lict = convT.t_to_rad(Tict[gd])
    gain = (Lict-NS)/(CS[gd]-CICT[gd])

    sp_ict = CS[gd]-CICT[gd]

    Tcorr = convT.A+convT.B*Tict[gd]
    dGain_dICT = (convT.B/sp_ict)*convT.nBB_num*np.exp(convT.c2_nu_c/Tcorr)*\
                      (convT.c2_nu_c/(Tcorr**2))/\
                      (np.exp(convT.c2_nu_c/Tcorr)-1.)**2
    dgain = (gain-gainval)
    dT = dgain/dGain_dICT

    uICT = np.std(dT)

    return uICT

def get_ict_uncert(Tict,prt_random,prt_bias,uICT,convT):
    """Get uncertainty in radiance/temperature of ICT on the basis of 
    ICT uncertainties"""

    #
    # Note in operational calibration prts and averaged over 4
    # so input prt uncertainties need to be divided by 2
    #
    rad,urand = convT.t_to_rad_uncert(Tict,prt_random/2.)

    #
    # Systematic doesn't average down
    #
    prt_sys = np.sqrt(prt_bias**2 + uICT**2)


    return urand,prt_sys,rad

def get_random(channel,noise,av_noise,ict_noise,ict_random,Lict,CS,CE,CICT,NS,\
               c1,c2):
    """Get the random parts of the IR calibration uncertainty. Done per 
    scanline"""
    #
    # Gain part for all noise sources
    #
    dLlin_dCS = (Lict-NS)*(CS-CE)/(CS-CICT)**2 + (Lict-NS)/(CS-CICT)
    dLlin_dCICT = -(Lict-NS)*(CS-CE)/(CS-CICT)**2 
    dLlin_dCE = -(Lict-NS)/(CS-CICT)
    dLlin_dLict = (CS-CE)/(CS-CICT)

    #
    # If channel = 2,3 (11/12) then add non-linear part
    #
    if channel == 1:
        uncert = (dLlin_dCS**2)*(av_noise**2) + \
                 (dLlin_dCICT**2)*(ict_noise**2) + \
                 (dLlin_dCE**2)*(noise**2) + \
                 (dLlin_dLict**2)*(ict_random**2)
    else:
        Llin = NS + (Lict-NS)*(CS-CE)/(CS-CICT)
        dLE_dCS = dLlin_dCS * (1.+c1+c2*Llin)
        dLE_dCICT = dLlin_dCICT * (1.+c1+c2*Llin)
        dLE_dCE = dLlin_dCE * (1.+c1+c2*Llin)
        dLE_dLict = dLlin_dLict * (1.+c1+c2*Llin)

        uncert = (dLE_dCS**2)*(av_noise**2) + \
                 (dLE_dCICT**2)*(ict_noise**2) + \
                 (dLE_dCE**2)*(noise**2) + \
                 (dLE_dLict**2)*(ict_random**2)
        
    return np.sqrt(uncert)

def get_sys(channel,uICT,Tict,CS,CE,CICT,NS,\
            c1,c2,convT):
    """Get the random parts of the IR calibration uncertainty. Done per 
    scanline"""
    #
    # Gain part for all noise sources
    #
    Lict,uradTict = convT.t_to_rad_uncert(Tict,uICT)
    dLlin_dLict = (CS-CE)/(CS-CICT)

    #
    # If channel = 2,3 (11/12) then add non-linear part
    #
    if channel == 1:
        uncert = (dLlin_dLict**2)*(uradTict**2)
    else:
        Llin = NS + (Lict-NS)*(CS-CE)/(CS-CICT)
        dLE_dLict = dLlin_dLict * (1.+c1+c2*Llin)

        uncert = (dLE_dLict**2)*(uradTict**2)

    return np.sqrt(uncert)

def get_vars(ds,channel,convT,wlength,prt_threshold,ict_threshold,\
             space_threshold,gac,cal,mask,out_prt=False):
    """Get variables from xarray"""

    space = ds['space_counts'].values[:,channel]
    prt = ds["prt_counts"].values[:]
    ict = ds['ict_counts'].values[:,channel]
    ce = ds['channels'].values[:,:,channel-3]
    line_numbers = ds["scan_line_index"].data

    #
    # Set nan's to value to be caught by interpolation routines
    #
    gd = ~np.isfinite(prt)
    if np.sum(gd) > 0:
        prt[gd] = 0
    gd = ~np.isfinite(ict)
    if np.sum(gd) > 0:
        ict[gd] = 0
    gd = ~np.isfinite(space)
    if np.sum(gd) > 0:
        space[gd] = 0

    #
    # PRT index check
    #
    # PRTs. See reader.get_telemetry implementations.

    for offset in range(5):
        # According to the KLM Guide the fill value between PRT measurments is 0, but we search
        # for the first measurement gap using the threshold, because the fill value is in practice
        # not always exactly 0.
        if np.median(prt[(line_numbers - line_numbers[0]) % 5 == offset]) < prt_threshold:
            break
    else:
        raise IndexError("No PRT 0-index found!")

    # get the PRT index, iprt equals to 0 corresponds to the measurement gaps
    # Note for GAC we have PRT nos 3 1 4 2 not 1 2 3 4
    if gac:
        prt_nos = [3,1,4,2]
    else:
        prt_nos = [1,2,3,4]
    iprt_orig = (line_numbers - line_numbers[0] + 5 - offset) % 5
    iprt = np.copy(iprt_orig)
    for i in [1,2,3,4]:
        gd = (iprt_orig == i)
        iprt[gd] = prt_nos[i-1]

    #
    # Interpolate over bad prt values - from pygac calibrate_thermal
    #
    # fill measured values below threshold by interpolation
    #
    prt_threshold = 50  # empirically found and set by Abhay Devasthale
    ifix = np.where(np.logical_and(iprt == 1, prt <= prt_threshold))
    if len(ifix[0]):
        inofix = np.where(np.logical_and(iprt == 1, prt > prt_threshold))
        if len(inofix[0]):
            prt[ifix] = np.interp(ifix[0], inofix[0], prt[inofix])
        else:
            raise IndexError('No good prt1 data')

    ifix = np.where(np.logical_and(iprt == 2, prt <= prt_threshold))
    if len(ifix[0]):
        inofix = np.where(np.logical_and(iprt == 2, prt > prt_threshold))
        if len(inofix[0]):
            prt[ifix] = np.interp(ifix[0], inofix[0], prt[inofix])
        else:
            raise IndexError('No good prt2 data')

    ifix = np.where(np.logical_and(iprt == 3, prt <= prt_threshold))
    if len(ifix[0]):
        inofix = np.where(np.logical_and(iprt == 3, prt > prt_threshold))
        if len(inofix[0]):
            prt[ifix] = np.interp(ifix[0], inofix[0], prt[inofix])
        else:
            raise IndexError('No good prt3 data')

    ifix = np.where(np.logical_and(iprt == 4, prt <= prt_threshold))
    if len(ifix[0]):
        inofix = np.where(np.logical_and(iprt == 4, prt > prt_threshold))
        if len(inofix[0]):
            prt[ifix] = np.interp(ifix[0], inofix[0], prt[inofix])    
        else:
            raise IndexError('No good prt4 data')
        
    #
    # Convert to temperature
    #
    # calculate PRT temperature using equation (7.1.2.4-1) KLM Guide
    # Tprt = d0 + d1*Cprt + d2*Cprt^2 + d3*Cprt^3 + d4*Cprt^4
    # Note: First dimension of cal.d are the five coefficient indicees
    #
    tprt = np.polynomial.polynomial.polyval(prt, cal.d[:, iprt], tensor=False)

    #
    # Get interpolated values as done in pygac
    #
    tprt_interp = np.copy(tprt)
    zeros = iprt == 0
    nonzeros = np.logical_not(zeros)

    tprt_interp[zeros] = np.interp((zeros).nonzero()[0],
                            (nonzeros).nonzero()[0],
                            tprt[nonzeros])
    #
    # Interpolate over each PRT number
    #
    tprt1_interp = np.copy(tprt)
    zeros = (iprt == 0)|(iprt != 1)
    nonzeros = np.logical_not(zeros)

    tprt1_interp[zeros] = np.interp((zeros).nonzero()[0],
                            (nonzeros).nonzero()[0],
                            tprt[nonzeros])

    tprt2_interp = np.copy(tprt)
    zeros = (iprt == 0)|(iprt != 2)
    nonzeros = np.logical_not(zeros)

    tprt2_interp[zeros] = np.interp((zeros).nonzero()[0],
                            (nonzeros).nonzero()[0],
                            tprt[nonzeros])

    tprt3_interp = np.copy(tprt)
    zeros = (iprt == 0)|(iprt != 3)
    nonzeros = np.logical_not(zeros)

    tprt3_interp[zeros] = np.interp((zeros).nonzero()[0],
                            (nonzeros).nonzero()[0],
                            tprt[nonzeros])

    tprt4_interp = np.copy(tprt)
    zeros = (iprt == 0)|(iprt != 4)
    nonzeros = np.logical_not(zeros)

    tprt4_interp[zeros] = np.interp((zeros).nonzero()[0],
                            (nonzeros).nonzero()[0],
                            tprt[nonzeros])
    
    # Thresholds to flag missing/wrong data for interpolation
    # Remove masked data
    ict[mask] = 0
    space[mask] = 0
    zeros = ict < ict_threshold
    nonzeros = np.logical_not(zeros)
    try:
        ict[zeros] = np.interp((zeros).nonzero()[0],
                               (nonzeros).nonzero()[0],
                               ict[nonzeros])
    except ValueError: # 3b has no valid data
        raise IndexError('Channel 3b has no valid data')
    zeros = space < space_threshold
    nonzeros = np.logical_not(zeros)

    space[zeros] = np.interp((zeros).nonzero()[0],
                             (nonzeros).nonzero()[0],
                             space[nonzeros])
    
    #
    # Make averages and do using pygacs method at this point
    #
    weighting_function = np.ones(wlength, dtype=float) / wlength
    tprt_convolved = np.convolve(tprt_interp, weighting_function, "same")
    tprt1_convolved = np.convolve(tprt1_interp, weighting_function, "same")
    tprt2_convolved = np.convolve(tprt2_interp, weighting_function, "same")
    tprt3_convolved = np.convolve(tprt3_interp, weighting_function, "same")
    tprt4_convolved = np.convolve(tprt4_interp, weighting_function, "same")
    ict_convolved = np.convolve(ict, weighting_function, "same")
    space_convolved = np.convolve(space, weighting_function, "same")

    # take care of the beginning and end
    tprt_convolved[0:(wlength - 1) // 2] = tprt_convolved[(wlength - 1) // 2]
    tprt1_convolved[0:(wlength - 1) // 2] = tprt1_convolved[(wlength - 1) // 2]
    tprt2_convolved[0:(wlength - 1) // 2] = tprt2_convolved[(wlength - 1) // 2]
    tprt3_convolved[0:(wlength - 1) // 2] = tprt3_convolved[(wlength - 1) // 2]
    tprt4_convolved[0:(wlength - 1) // 2] = tprt4_convolved[(wlength - 1) // 2]
    ict_convolved[0:(wlength - 1) // 2] = ict_convolved[(wlength - 1) // 2]
    space_convolved[0:(wlength - 1) // 2] = space_convolved[(wlength - 1) // 2]
    tprt_convolved[-(wlength - 1) // 2:] = tprt_convolved[-((wlength + 1) // 2)]
    tprt1_convolved[-(wlength - 1) // 2:] = tprt1_convolved[-((wlength + 1) // 2)]
    tprt2_convolved[-(wlength - 1) // 2:] = tprt2_convolved[-((wlength + 1) // 2)]
    tprt3_convolved[-(wlength - 1) // 2:] = tprt3_convolved[-((wlength + 1) // 2)]
    tprt4_convolved[-(wlength - 1) // 2:] = tprt4_convolved[-((wlength + 1) // 2)]
    ict_convolved[-(wlength - 1) // 2:] = ict_convolved[-((wlength + 1) // 2)]
    space_convolved[-(wlength - 1) // 2:] = space_convolved[-((wlength + 1) // 2)]

    if out_prt:
        return space_convolved,ict_convolved,ce,tprt_convolved,\
            tprt1_convolved,tprt2_convolved,tprt3_convolved,tprt4_convolved
    else:
        return space_convolved,ict_convolved,ce,tprt_convolved

def get_gainval(time,avhrr,ict1,ict2,ict3,ict4,CS,CICT,NS,bad_scan,convT):
    """Estimate gain value at smallest stdev point in orbit either from
    file or estimate it from data"""
    #
    # Open file containing 3.7mu gain value and interpolate over time
    #
    coef_file = files("pygac") / "data/{0}_uncert.nc".format(avhrr)
    with xr.open_dataset(coef_file) as d:
        intime = d["time"].values[:]
        ingain = d["max_gain"].values[:]

    timediff = (intime-time)/np.timedelta64(1,'s')
    timediff = np.abs(timediff)
    timediff_min = timediff.min()
    #
    # Within a day at worst
    #
    if timediff_min < 86400.: 
        pos = np.nonzero(timediff == timediff_min)[0][0]
        return ingain[pos]
    else:
        #
        # No nearby gain estimate so calculate from data using min std
        # of prts
        #
        prt1 = prt1[gd]
        prt2 = prt2[gd]
        prt3 = prt3[gd]
        prt4 = prt4[gd]
        Lict = Lict[gd]
        CS = CS[gd]
        CICT = CICT[gd]
        #
        # Get stdev
        #
        X = np.zeros((len(prt1),4))
        X[:,0] = prt1
        X[:,1] = prt2
        X[:,2] = prt3
        X[:,3] = prt4
        stdev = np.std(X,axis=1)
        pos = np.nonzero(stdev == stdev.min())[0][0]
        Lict = convT.t_to_rad(np.mean(X,axis=1))
        gain = (Lict-NS)/(CS-CICT)
        return gain[pos]

def get_pixel(Lict,CS,CE,CICT,NS,c0,c1,c2):
    """Get radiance of pixel using calibration"""

    Llin = NS + (Lict-NS)*(CS-CE)/(CS-CICT)
    if NS != 0.:
        LE = Llin + c0 + c1*Llin + c2*Llin*Llin
        return LE
    else:
        return Llin

def ir_uncertainty(ds,mask,plot=False):
    """Create the uncertainty components for the IR channels. These include
    
    1) Random 
          a) Noise
          b) Digitisation
          c) ICT PRT Noise
    2) Systematic
          a) ICT Temperature uncertainty
          b) PRT Bias
          c) Calibration coefs/measurement equation uncertainty
    
    Inputs:
          ds : Input xarray dataset containing data for calibration
        mask : pygac mask from reader
    Outputs:
      uncert : xarray dataset containing random and systematic uncertainty
               components
    """
    #
    # Define averaging kernel based on value in noaa.py
    # Also set PRT uncertainty components
    #
    window=51
    prt_bias = 0.01
    prt_sys = 0.1
    prt_threshold = 50
    ict_threshold = 100
    space_threshold = 100

    if ds['channels'].values.shape[1] == 409:
        gacdata = True
    else:
        gacdata = False

    avhrr_name = ds.attrs["spacecraft_name"]

    #
    # Get calibration coefficients
    #
    cal = Calibrator(
        ds.attrs["spacecraft_name"])
    NS_1 = cal.space_radiance[0]
    NS_2 = cal.space_radiance[1]
    NS_3 = cal.space_radiance[2]
    c0_1 = cal.b[0,0]
    c1_1 = cal.b[0,1]
    c2_1 = cal.b[0,2]
    c0_2 = cal.b[1,0]
    c1_2 = cal.b[1,1]
    c2_2 = cal.b[1,2]
    c0_3 = cal.b[2,0]
    c1_3 = cal.b[2,1]
    c2_3 = cal.b[2,2]

    # Is the twelve micron channel there
    if ds.attrs["spacecraft_name"] == "tirosn" or \
       ds.attrs["spacecraft_name"] == "noaa06" or \
       ds.attrs["spacecraft_name"] == "noaa08" or \
       ds.attrs["spacecraft_name"] == "noaa10":
        twelve_micron = False
    else:
        twelve_micron = True

    # Temperature to radiance etc.
    convT1 = convBT(cal,0)
    convT2 = convBT(cal,1)
    if twelve_micron:
        convT3 = convBT(cal,2)

    #
    # Get variables for 10 sampled case
    #
    total_space = ds['total_space_counts'].values[:,:,:]
    total_ict = ds['total_ict_counts'].values[:,:,:]
    
    #
    # Noise elements
    #
    noise1,noise2,noise3,av_noise1,av_noise2,av_noise3,\
        av_ict_noise1,av_ict_noise2,av_ict_noise3,bad_scan \
               = get_noise(total_space,total_ict,window,twelve_micron)

    #
    # Get variables used on the calibration
    #
    if plot:
        plt.figure(1)

    CS_1,CICT_1,CE_1,Tict,ict1,ict2,ict3,ict4 = get_vars(ds,0,convT1,\
                                                           window,\
                                                           prt_threshold,\
                                                           ict_threshold,\
                                                           space_threshold,\
                                                           gacdata,\
                                                           cal,\
                                                           mask,\
                                                           out_prt=True)
    if plot:
        plt.subplot(131)
        plt.plot(np.arange(len(CS_1)),CS_1,',')
        plt.title('3.7$\mu$m')
        plt.ylabel('Space Cnts')
        plt.xlabel('Scanline')

    CS_2,CICT_2,CE_2,Tict = get_vars(ds,1,convT2,window,prt_threshold,\
                                     ict_threshold,\
                                     space_threshold,\
                                     gacdata,cal,mask)
    if plot:
        plt.subplot(132)
        plt.plot(np.arange(len(CS_2)),CS_2,',')
        plt.title('11$\mu$m')
        plt.ylabel('Space Cnts')
        plt.xlabel('Scanline')
        
    if twelve_micron:
        CS_3,CICT_3,CE_3,Tict = get_vars(ds,2,convT3,window,\
                                         prt_threshold,\
                                         ict_threshold,\
                                         space_threshold,\
                                         gacdata,cal,mask)
        if plot:
            plt.subplot(133)
            plt.plot(np.arange(len(CS_3)),CS_3,',')
            plt.title('12$\mu$m')
            plt.ylabel('Space Cnts')
            plt.xlabel('Scanline')
    if plot:
        plt.tight_layout()

    #
    # Systematic components - uICT from gain variation in 3.7mu channel
    #
    gd = np.isfinite(ds['times'].values)
    time = ds['times'].values[gd][0]
    gain_37 = get_gainval(time,avhrr_name,ict1,ict2,ict3,ict4,CS_1,CICT_1,\
                          0.,bad_scan,convT1)
    uICT = get_uICT(gain_37,CS_1,CICT_1,Tict,0.,convT1,bad_scan)

    #
    # Loop round scanlines
    #
    bt_rand_37 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    bt_rand_11 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    bt_rand_12 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    if not twelve_micron:
        bt_rand_12[:,:] = np.nan
    bt_sys_37 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    bt_sys_11 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    bt_sys_12 = np.zeros(CE_2.shape,dtype=CE_2.dtype)
    if not twelve_micron:
        bt_sys_12[:,:] = np.nan
    for i in range(len(CS_2)):
        #
        # Check for bad scanlines
        #
        if bad_scan[i] == 1:
            bt_rand_37[i,:] = np.nan
            bt_rand_11[i,:] = np.nan
            if twelve_micron:
                bt_rand_12[i,:] = np.nan
            bt_sys_37[i,:] = np.nan
            bt_sys_11[i,:] = np.nan
            if twelve_micron:
                bt_sys_12[i,:] = np.nan
            continue
        #
        # Get uncertainty in ICT temperature from PRT measurements
        #
        ict_random1, ict_sys1, Lict_1 = get_ict_uncert(Tict[i],prt_bias,prt_sys,\
                                               uICT,convT1)
        ict_random2, ict_sys2, Lict_2 = get_ict_uncert(Tict[i],prt_bias,prt_sys,\
                                               uICT,convT2)
        if twelve_micron:
            ict_random3, ict_sys3, Lict_3 = get_ict_uncert(Tict[i],prt_bias,prt_sys,\
                                                   uICT,convT3)

        #
        # get pixel radiance
        #
        rad_37 = get_pixel(Lict_1,CS_1[i],\
                           CE_1[i,:],CICT_1[i],0.,0.,0.,0.)
        rad_11 = get_pixel(Lict_2,CS_2[i],\
                           CE_2[i,:],CICT_2[i],NS_2,c0_2,c1_2,c2_2)
        if twelve_micron:
            rad_12 = get_pixel(Lict_3,CS_3[i],\
                               CE_3[i,:],CICT_3[i],NS_3,c0_3,c1_3,c2_3)

        #
        # Get noise in radiance space
        #
        rad_noise_37 = get_random(1,noise1,av_noise1,av_ict_noise1,\
                                  ict_random1,Lict_1,CS_1[i],\
                                  CE_1[i,:],CICT_1[i],0.,0.,0.)
        rad_noise_11 = get_random(2,noise2,av_noise2,av_ict_noise2,\
                                  ict_random2,Lict_2,CS_2[i],\
                                  CE_2[i,:],CICT_2[i],NS_2,c1_2,c2_2)
        if twelve_micron:
            rad_noise_12 = get_random(3,noise3,av_noise3,\
                                      av_ict_noise3,\
                                      ict_random3,Lict_3,CS_3[i],\
                                      CE_3[i,:],CICT_3[i],NS_3,c1_3,c2_3)

        #
        # Convert to BT space uncertainty
        #
        T,bt_rand_37[i,:] = convT1.rad_to_t_uncert(rad_37,rad_noise_37)
        T,bt_rand_11[i,:] = convT2.rad_to_t_uncert(rad_11,rad_noise_11)
        if twelve_micron:
            T,bt_rand_12[i,:] = convT3.rad_to_t_uncert(rad_12,rad_noise_12)

        #
        # Get systematic uncertainty through the measurement equation
        # Note measurement equation uncertainty set to 0.5
        #
        rad_sys_37 = get_sys(1,ict_sys1,Tict[i],CS_1[i],\
                             CE_1[i,:],CICT_1[i],0.,0.,0.,convT1)
        rad_sys_11 = get_sys(2,ict_sys2,Tict[i],CS_2[i],\
                             CE_2[i,:],CICT_2[i],NS_2,c1_2,c2_2,convT2)
        if twelve_micron:
            rad_sys_12 = get_sys(3,ict_sys3,Tict[i],CS_3[i],\
                                 CE_3[i,:],CICT_3[i],NS_3,c1_3,c2_3,convT3)

        T,bt_sys_37[i,:] = convT1.rad_to_t_uncert(rad_37,rad_sys_37)
        T,bt_sys_11[i,:] = convT2.rad_to_t_uncert(rad_11,rad_sys_11)
        if twelve_micron:
            T,bt_sys_12[i,:] = convT3.rad_to_t_uncert(rad_12,rad_sys_12)

        #
        # Add 0.5K for measurement equation uncertainty
        #
        bt_sys_37[i,:] = np.sqrt(bt_sys_37[i,:]**2+0.5**2)
        bt_sys_11[i,:] = np.sqrt(bt_sys_11[i,:]**2+0.5**2)
        if twelve_micron:
            bt_sys_12[i,:] = np.sqrt(bt_sys_12[i,:]**2+0.5**2)

    if plot:
        if twelve_micron:
            plt.figure(2)
            plt.subplot(231)
            plt.hist(bt_rand_37.flatten(),bins=100)
            plt.title('3.7$\mu$m')
            
            plt.subplot(232)
            plt.hist(bt_rand_11.flatten(),bins=100)
            plt.title('11$\mu$m (Random)')

            plt.subplot(233)
            plt.hist(bt_rand_12.flatten(),bins=100)
            plt.title('12$\mu$m')
            
            plt.subplot(234)
            plt.hist(bt_sys_37.flatten(),bins=100)
            plt.title('3.7$\mu$m')
            
            plt.subplot(235)
            plt.hist(bt_sys_11.flatten(),bins=100)
            plt.title('11$\mu$m (Systematic)')
            plt.xlabel('Uncertainty / K')

            plt.subplot(236)
            plt.hist(bt_sys_12.flatten(),bins=100)
            plt.title('12$\mu$m')
            plt.tight_layout()

            plt.figure(3)
            plt.subplot(231)
            im=plt.imshow(bt_rand_37)
            plt.colorbar(im)
            plt.title('3.7$\mu$m')

            plt.subplot(232)
            im=plt.imshow(bt_rand_11)
            plt.colorbar(im)
            plt.title('11$\mu$m (Random)')

            plt.subplot(233)
            im=plt.imshow(bt_rand_12)
            plt.colorbar(im)
            plt.title('12$\mu$m')

            plt.subplot(234)
            im=plt.imshow(bt_sys_37)
            plt.colorbar(im)
            plt.title('3.7$\mu$m')

            plt.subplot(235)
            im=plt.imshow(bt_sys_11)
            plt.colorbar(im)
            plt.title('11$\mu$m (Systematic)')

            plt.subplot(236)
            im=plt.imshow(bt_sys_12)
            plt.colorbar(im)
            plt.title('12$\mu$m')
            plt.tight_layout()
        else:
            plt.figure(2)
            plt.subplot(221)
            plt.hist(bt_rand_37.flatten(),bins=100)
            plt.title('3.7$\mu$m (Random)')
            
            plt.subplot(222)
            plt.hist(bt_rand_11.flatten(),bins=100)
            plt.title('11$\mu$m (Random)')

            plt.subplot(223)
            plt.hist(bt_sys_37.flatten(),bins=100)
            plt.title('3.7$\mu$m (Systematic)')
            plt.xlabel('Uncertainty / K')
            
            plt.subplot(224)
            plt.hist(bt_sys_11.flatten(),bins=100)
            plt.title('11$\mu$m (Systematic)')
            plt.xlabel('Uncertainty / K')

            plt.tight_layout()
        plt.show()
    #
    # Output uncertainties
    #
    random = np.zeros((bt_rand_11.shape[0],bt_rand_11.shape[1],3))
    systematic = np.zeros((bt_rand_11.shape[0],bt_rand_11.shape[1],3))

    random[:,:,0] = bt_rand_37
    random[:,:,1] = bt_rand_11
    if twelve_micron:
        random[:,:,2] = bt_rand_12
    else:
        random[:,:,2] = np.nan
    systematic[:,:,0] = bt_sys_37
    systematic[:,:,1] = bt_sys_11
    if twelve_micron:
        systematic[:,:,2] = bt_sys_12
    else:
        systematic[:,:,2] = np.nan

    time = (ds["times"].values - np.datetime64("1970-01-01 00:00:00"))/\
           np.timedelta64(1,'s')
    time_da = xr.DataArray(time,dims=["times"],attrs={"long_name":"scanline time",\
                                                     "units":"seconds since 1970-01-01"})
    across_da = xr.DataArray(np.arange(random.shape[1]),dims=["across_track"])
    ir_channels_da = xr.DataArray(np.array([3,4,5]),dims=["ir_channels"])
    random_da = xr.DataArray(random,dims=["times","across_track","ir_channels"],\
                             attrs={"long_name":"Random uncertainties","units":"K"})
    sys_da = xr.DataArray(systematic,dims=["times","across_track","ir_channels"],\
                          attrs={"long_name":"Systematic uncertainties","units":"K"})

    uncertainties = xr.Dataset(dict(times=time_da,across_track=across_da,\
                               ir_channels=ir_channels_da,\
                                    random=random_da,systematic=sys_da))

    return uncertainties

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    args = parser.parse_args()

    #
    # Read data
    #
    reader_cls = get_reader_class(args.filename)

    reader = reader_cls(tle_dir="/gws/nopw/j04/nceo_uor/users/jmittaz/NPL/AVHRR/TLE",
                        tle_name="TLE_%(satname).txt",
                        calibration_method="noaa",
                        adjust_clock_drift=False)
    reader.read(args.filename)
    ds = reader.get_calibrated_dataset()
    mask = reader.mask
    uncert = ir_uncertainty(ds,mask,plot=True)
    print(uncert)

