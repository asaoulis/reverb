import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def get_spectrogram_colors(specgram):

    fig = plt.figure(figsize=(15, 10))
    ax_spec = fig.add_axes([0.1, 0.1, 0.7, 0.60])
    
    ax_spec.xaxis_date()

    clip = [0.0, 1.0]
    vmin, vmax = clip

    zorder = None
    cmap = 'jet'
    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
            if v is not None}



    cut_specgram = specgram

    _range = float(cut_specgram.max() - cut_specgram.min())
    vmin = cut_specgram.min() + vmin * _range
    vmax = cut_specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    ax_spec.pcolormesh(specgram, norm=norm, **kwargs)

    ax_spec.axis('tight')
    #ax_spec.set_xlim(0, end)
    ax_spec.grid(False)

    ax_spec.set_xlabel('Time [s]')
    ax_spec.set_ylabel('Frequency [Hz]')


    # ax_spec.set_ylim(ax_spec_y_lims)
    vvmin= -250
    vvmax= -150

    ax_spec.collections[0].set_clim(
            vmin=vvmin, vmax=vvmax
        )  # Find the quadmesh/pcolormesh created by the spectrogram call, and then change its clims
    
    mesh = ax_spec.collections[0]
    return mesh.to_rgba(mesh.get_array().reshape(specgram.shape))

