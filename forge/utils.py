import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt


def tone_burst(dt, centre_freq, n_cycles, n_samples, amplitude,
               envelope='gaussian', offset=0, phase=lambda x: 0,
               plot=False):
    """
    Generate a tone burst signal with specified parameters.

    Parameters
    ----------
    dt : float
        The time step of the signal in seconds.
    centre_freq : float
        The center frequency of the tone burst in Hz.
    n_cycles : int
        The number of cycles in the tone burst.
    n_samples : int
        The total number of samples in the output signal.
    amplitude : float
        The amplitude of the tone burst.
    envelope : {'gaussian', 'rectangular'}, optional
        The envelope type of the tone burst. Default is 'gaussian'.
    offset : int, optional
        The offset in samples to place the tone burst within the signal.
        Default is 0.
    phase : function, optional
        The phase function of the tone burst. It should take time array
        as input and return phase values. Default is lambda x: 0.
    plot : bool, optional
        Whether to plot the generated tone burst. Default is False.

    Returns
    -------
    ndarray
        The generated tone burst signal.

    """
    tone_length = n_cycles/centre_freq

    time_array, step = np.linspace(0, tone_length, int(tone_length/dt+1),
                                   retstep=True, endpoint=False)

    tone_burst = np.sin(2*np.pi*centre_freq*time_array+2*np.pi *
                        centre_freq*phase(time_array))
    n_tone = len(tone_burst)

    if envelope == 'gaussian':
        limit = 3
        window_x = np.linspace(-limit, limit, n_tone)
        window = np.exp(-window_x**2/2)

    elif envelope == 'rectangular':
        window = np.ones((tone_burst.shape[0],))

    tone_burst = np.multiply(tone_burst, window)

    window = scipy.signal.get_window(('tukey', 0.05), n_tone, False)
    tone_burst = np.multiply(tone_burst, window)

    signal = np.pad(tone_burst, ((offset, n_samples-offset-n_tone),),
                    mode='constant', constant_values=0)
    source = amplitude*np.asarray(signal, dtype=np.float32)

    if plot:
        plt.plot(source)
        plt.title('source wavelet')
        plt.xlabel('time steps')
        plt.ylabel('pressure')
        plt.show()

    return source


def generate_data(model, source, s_pos, bs):
    """
    Generate data for all shots when the number of shots exceeds the
    batch size.

    Parameters
    ----------
    model : FullWaveformInversion instance
        An instance of the FullWaveformInversion class.
    source : ndarray
        One-dimensional array containing source wavelet pressure field.
    s_pos : ndarray
        Array of two-dimensional source coordinates.
    bs : int
        Batch size represents the number of shots to model
        simultaneously on the GPU or the shots per run.

    Returns
    -------
    torch.Tensor
        The forward modelled data recorded at the receivers for all
        shots defined in s_pos.

    """
    num_complete_runs = int(np.floor(len(s_pos)/bs))

    d = torch.zeros(len(s_pos), model.num_rec, len(source))
    for i in range(num_complete_runs):
        print(f'Batch {i+1}/{num_complete_runs}')
        model.forward(s_pos[i*bs:i*bs+bs], source)
        d[i*bs:i*bs+bs] = model.d

    if len(s_pos) % bs > 0:
        model.forward(s_pos[int(num_complete_runs*bs):], source)
        d[int(num_complete_runs*bs):] = model.d

    return d


def resample(model, old_spacing, new_spacing, **kwargs):
    """
    Resample models to a new grid spacing.

    Parameters
    ----------
    model : ndarray
        model to resample.
    old_spacing : float
        Old grid spacing.
    new_spacing : tuple of float
        New grid spacing.
    **kwargs : additional keyword arguments
        order : int, optional
            Order of the interpolation. Defaults to 3.
        prefilter : bool, optional
            Whether prefiltering needs to be applied before interpolation.
            If downsampling, this defaults to False as an anti-aliasing filter
            will be applied instead. If upsampling, this defaults to True.
        anti_alias : bool, optional
            Whether a Gaussian filter is applied to smooth the model before
            interpolation. Defaults to True. This is only applied when
            downsampling.

    Returns
    -------
    ndarray
        Resampled model.

    """
    order = kwargs.pop('order', 3)
    prefilter = kwargs.pop('prefilter', True)

    zoom = old_spacing/new_spacing

    # Anti-aliasing is only required for down-sampling interpolation
    if zoom < 1:
        anti_alias = kwargs.pop('anti_alias', True)

        if anti_alias:
            anti_alias_sigma = np.maximum(0, (1/zoom-1)/2)
            model = scipy.ndimage.gaussian_filter(model, anti_alias_sigma)

            # Prefiltering is not necessary if anti-alias filter used
            prefilter = False

    interp = np.clip(scipy.ndimage.zoom(model, zoom,
                                        order=order, prefilter=prefilter),
                     model.min(), model.max())

    return interp


def trace_plot(data, shot, **kwargs):
    """
    Plot seismic traces from shot gathers.

    Parameters
    ----------
    data : array-like
        3D array containing acoustic data. Dimensions should be
        [num shots, num traces/receivers per shot, num time steps].
    shot : int
        Index of the shot to plot.
    **kwargs : additional keyword arguments
        scale_fac : float, optional
            Scaling factor for trace amplitudes. Default is 1.
        size : tuple, optional
            Size of the plot figure. Default is (8, 4).
        line_color : str, optional
            Color of the trace lines. Default is 'black'.
        linewidth : float, optional
            Width of the trace lines. Default is 1.
        fill : bool, optional
            Whether to fill between traces. Default is True.
        fill_color : str, optional
            Color for filling between traces. Default is 'black'.
        fill_color2 : str, optional
            Second color for filling between traces if alternating colors are
            desired. Default is None.
        fill_fac : float, optional
            Fill factor threshold. Default is 1e-2.
        save : str or None, optional
            File path to save the plot as an image. Default is None.

    Returns
    -------

    """
    scale_fac = kwargs.pop('scale_fac', 1)
    size = kwargs.pop('size', (8, 4))
    line_color = kwargs.pop('line_color', 'black')
    linewidth = kwargs.pop('linewidth', 1)
    fill = kwargs.pop('fill', True)
    fill_color = kwargs.pop('fill_color', 'black')
    fill_color2 = kwargs.pop('fill_color2', None)
    fill_fac = kwargs.pop('fill_fac', 1e-2)
    save = kwargs.pop('save', None)

    num = data.shape[2]
    time = np.arange(0, num, 1)
    fig, ax1 = plt.subplots(figsize=size)
    for i in range(len(data[shot])):
        ax1.plot((scale_fac * data[shot][i]) + i, time, color=line_color,
                 linewidth=linewidth)
        if fill:
            if fill_color2:
                if i % 2 == 0:
                    ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                      where=(((scale_fac * data[shot][i]) + i)
                                             > i + fill_fac),
                                      color=fill_color)
                else:
                    ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                      where=(((scale_fac * data[shot][i]) + i)
                                             > i + fill_fac),
                                      color=fill_color2)
            else:
                ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                  where=(((scale_fac * data[shot][i]) + i)
                                         > i + fill_fac),
                                  color=fill_color)
    ax1.invert_yaxis()
    plt.xlim((-1, data.shape[1]))
    plt.ylim((num-1, 0))
    plt.xlabel('trace number')
    plt.ylabel('time-step')
    if save:
        plt.savefig(f'{save}.png', dpi=500)
    plt.show()


def normalize(data, type='trace'):
    """
    Normalize acoustic data along specified dimensions.

    Parameters
    ----------
    data : torch.Tensor
        Seismic data tensor.
    type : str or tuple or None, optional
        Type of normalization. If 'trace', normalize along the last dimension.
        If 'shot', normalize along the last two dimensions. If a tuple,
        normalize along the specified dimensions. Default is None.

    Returns
    -------
    torch.Tensor
        Normalized acoustic data tensor.

    """
    if type == 'trace':
        dim = -1
    elif type == 'shot':
        dim = (-2, -1)
    elif isinstance(type, tuple):
        dim = type
    else:
        raise ValueError("Invalid value for 'type'. It must be 'trace', \
'shot', or a tuple.")

    return data/torch.norm(data, dim=dim, keepdim=True)
