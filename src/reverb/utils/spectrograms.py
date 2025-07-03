# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: spectrogram.py
#  Purpose: Plotting spectrogram of Seismograms.
#   Author: Christian Sippl, Moritz Beyreuther
#    Email: sippl@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Christian Sippl
# --------------------------------------------------------------------
"""
Plotting spectrogram of seismograms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import os
import math

import h5py
import numpy as np
from scipy.interpolate import interp2d

from matplotlib import mlab

def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrogram(data, samp_rate, per_lap=0.9, wlen=None, 
                save_output=None, mult=8.0,  downsample_dims = None):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to a
        window length matching 128 samples.
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type save_output: bool
    :param save_output: Save raw spectrogram array to a hdf5 file.
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = 128 / samp_rate

    npts = len(data)

    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))

    if npts < nfft:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, sampling rate '
               f'{samp_rate} Hz)')
        raise ValueError(msg)

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    if len(time) < 2:
        msg = (f'Input signal too short ({npts} samples, window length '
               f'{wlen} seconds, nfft {nfft} samples, {nlap} samples window '
               f'overlap, sampling rate {samp_rate} Hz)')
        raise ValueError(msg)

    # interpolate to desired shape
    if downsample_dims is not None:
        interpolater = interp2d(time,freq,specgram, kind="linear")

        time = np.linspace(time[0], time[-1], downsample_dims[1])

        freq = np.linspace(freq[0], freq[-1], downsample_dims[0])[:-1]

        specgram = interpolater(time, freq)


    # db scale and remove zero/offset for amplitude
    specgram = 10 * np.log10(specgram)

    if save_output is not None:
        file = h5py.File(save_output, 'w')
        file.create_dataset("spectrogram_array", data = specgram)
    
    return specgram, time, freq

from scipy import signal

def resample_trace(tr, dt, method, lanczos_a=20):
    """
    resample ObsPy Trace (tr) with dt as delta (1/sampling_rate).
    This code is from LASIF repository (Lion Krischer) with some
    minor modifications.
    :param tr:
    :param dt:
    :param method:
    :param lanczos_a:
    :return:
    """
    while True:
        if method == "decimate":
            decimation_factor = int(dt / tr.stats.delta)
        elif method == "lanczos":
            decimation_factor = float(dt) / tr.stats.delta
        # decimate in steps for large sample rate reductions.
        if decimation_factor > 5:
            decimation_factor = 5
        if decimation_factor > 1:
            new_nyquist = tr.stats.sampling_rate / 2.0 / decimation_factor
            zerophase_chebychev_lowpass_filter(tr, new_nyquist)
            if method == "decimate":
                tr.decimate(factor=decimation_factor, no_filter=True)
            elif method == "lanczos":
                tr.taper(max_percentage=0.01)
                current_sr = float(tr.stats.sampling_rate)
                tr.interpolate(
                    method="lanczos",
                    sampling_rate=current_sr / decimation_factor,
                    a=lanczos_a,
                )
        else:
            return tr

def zerophase_chebychev_lowpass_filter(trace, freqmax):
    """
    Custom Chebychev type two zerophase lowpass filter useful for
    decimation filtering.
    This filter is stable up to a reduction in frequency with a factor of
    10. If more reduction is desired, simply decimate in steps.
    Partly based on a filter in ObsPy.
    :param trace: The trace to be filtered.
    :param freqmax: The desired lowpass frequency.
    Will be replaced once ObsPy has a proper decimation filter.
    This code is from LASIF repository (Lion Krischer).
    """
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    # stop band frequency
    ws = freqmax / (trace.stats.sampling_rate * 0.5)
    # pass band frequency
    wp = ws

    while True:
        if order <= 12:
            break
        wp *= 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

    b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

    # Apply twice to get rid of the phase distortion.
    trace.data = signal.filtfilt(b, a, trace.data)



class SpectrogramCalculator:

    def __init__(self, config):
        self.config = config
    
    def remove_response(self, st, resp):
        st.attach_response(resp)
        st = resample_trace(st, dt=1.0 / self.config['trace_resampling_rate'], method="lanczos")
        st = st.remove_response(
            output=self.config['trace_unit'],
            pre_filt=self.config['trace_prefilter'],
            water_level=self.config['trace_waterlevel'],
            zero_mean=True,
            taper=True,
            taper_fraction=self.config['trace_taper'][0],
        )
        return st
    
    def compute_and_save(self, st, output_dir, starttime, endtime):
        st = st.slice(starttime, endtime)

        output_name =  f'{st.stats.station}_{st.stats.channel}_{starttime.strftime("%Y.%m.%d-%H%M%S")}-{endtime.strftime("%Y.%m.%d-%H%M%S")}.h5'
        output_name = output_dir / output_name
        specgram, time, freq = spectrogram(
                st.data,
                st.stats.sampling_rate,
                per_lap=self.config['overlap'],
                wlen=self.config['window_length'],
                save_output=output_name,
                downsample_dims = self.config['downsample_dims']
            )
        
        return specgram, time, freq
