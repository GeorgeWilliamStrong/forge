import torch
import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def butter_filter(data, threshold, fs, btype='lowpass', order=5):
    nyq = 0.5*fs
    thresh = threshold/nyq
    b, a = scipy.signal.butter(order, thresh, btype=btype)
    return np.array(scipy.signal.filtfilt(b, a, data), dtype=np.float32)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return np.asarray(scipy.signal.lfilter(b, a, data), dtype=np.float32)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = scipy.signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    return np.array(scipy.signal.sosfilt(sos, data), dtype=np.float32)


def tone_burst(dt, centre_freq, n_cycles, n_samples, amplitude, envelope='gaussian', offset=0, phase=lambda x:0, plot=False):
    tone_length = n_cycles/centre_freq
    time_array, step = np.linspace(0, tone_length, int(tone_length/dt+1), retstep=True, endpoint=False)
    tone_burst = np.sin(2*np.pi*centre_freq*time_array+2*np.pi*centre_freq*phase(time_array))
    n_tone = tone_burst.shape[0]
    if envelope=='gaussian':
        limit = 3
        window_x = np.linspace(-limit, limit, n_tone)
        window = np.exp(-window_x**2/2)
    elif envelope=='rectangular':
        window = np.ones((tone_burst.shape[0],))
    else:
        raise Exception('Envelope type not implemented')
    tone_burst = np.multiply(tone_burst, window)
    window = scipy.signal.get_window(('tukey', 0.05), n_tone, False)
    tone_burst = np.multiply(tone_burst, window)
    signal = np.pad(tone_burst, ((offset, n_samples-offset-n_tone),), mode='constant', constant_values=0)
    source = amplitude*np.asarray(signal, dtype=np.float32)
    if plot:
        plt.plot(source)
        plt.title('source wavelet')
        plt.xlabel('time steps')
        plt.ylabel('pressure')
        plt.show()
    return source


def demo_model(dx, min_vel=1480, med_vel=1550, max_vel=1650, size=0.07, outer_radius=0.021, inner_radius=0.016, plot=False):
    x = np.arange(0, size, dx)
    m = np.zeros((len(x),len(x)))
    m[:,:] = min_vel
    m0 = m.copy()
    m0[:,:] = min_vel
    for i in range(len(x)):
        for j in range(len(x)):
            if (x[i]-size/2)**2+(x[j]-size/2)**2<outer_radius**2: 
                m[i,j] = max_vel
            if (x[i]-size/2)**2+(x[j]-size/2)**2<inner_radius**2: 
                m[i,j] = med_vel
    if plot:
        plt.figure(figsize=(9,3))
        plt.subplot(1,2,1)
        plt.imshow(m, vmin=min_vel, vmax=max_vel)
        plt.title('true model')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(m0, vmin=min_vel, vmax=max_vel)
        plt.title('starting model')
        plt.colorbar()
        plt.show()
    return m, m0


def demo_geometry(model, n_elements=30, hicks=True, rad=0.46, plot=False):
    a = model.shape[0]*rad
    angle = np.zeros((n_elements))
    if hicks:
        pos = np.zeros((n_elements, 2), dtype=np.float32)
        for i in range(n_elements):
            angle[i] = ((i)*360.0/n_elements)*2*np.pi/360.0
            pos[i, 1] = a*np.cos(angle[i])+model.shape[1]/2-0.5
            pos[i, 0] = a*np.sin(angle[i])+model.shape[0]/2-0.5
    else:
        pos = np.zeros((n_elements, 2), dtype=int)
        for i in range(n_elements):
            angle[i] = ((i)*360.0/n_elements)*2*np.pi/360.0
            pos[i, 1] = int(round(a*np.cos(angle[i])+model.shape[1]/2-0.5))
            pos[i, 0] = int(round(a*np.sin(angle[i])+model.shape[0]/2-0.5))
    if plot:
        plt.figure(figsize=(4,3))
        plt.imshow(model, vmin=model.min(), vmax=model.max())
        plt.colorbar()
        geometry = np.zeros_like(model)
        geometry[np.asarray(np.round(pos), dtype=int)[:,0],np.asarray(np.round(pos), dtype=int)[:,1]]=1000
        geometry = scipy.ndimage.gaussian_filter(geometry, 1)
        geometry[geometry<1] = np.nan
        plt.imshow(geometry, cmap='Reds_r')
        plt.title('geometry')
        plt.show()
    return pos


def generate_data(model, source, s_pos, bs):
    num_complete_runs = int(np.floor(len(s_pos)/bs))
    d = torch.zeros(len(s_pos), model.num_rec, len(source))
    for i in range(num_complete_runs):
        print(f'Batch {i+1}/{num_complete_runs}')
        model.configure(s_pos[i*bs:i*bs+bs], source)
        model.forward()
        d[i*bs:i*bs+bs] = model.d
    if len(s_pos)%bs>0:
        model.configure(s_pos[int(num_complete_runs*bs):], source)
        model.forward()
        d[int(num_complete_runs*bs):] = model.d
    return d


def interpolate_models(vp, m0, r_geometry, s_geometry, dx, frequency, gp_per_wavelgth=6, rho=False, logQ=False):
    min_wavelength = min(vp.min(), m0.min())/frequency
    dxi = min_wavelength/gp_per_wavelgth
    new_x = int(np.ceil(vp.shape[0]*dx/dxi))
    zoom = new_x/vp.shape[0]
    vp = np.clip(scipy.ndimage.zoom(vp, zoom), vp.min(), vp.max())
    m0 = np.clip(scipy.ndimage.zoom(m0, zoom), m0.min(), m0.max())
    r_pos = r_geometry/dxi
    s_pos = s_geometry/dxi
    if type(rho) != bool:
        rho = np.clip(scipy.ndimage.zoom(rho, zoom), rho.min(), rho.max())
    if type(logQ) != bool:
        logQ = np.clip(scipy.ndimage.zoom(logQ, zoom), logQ.min(), logQ.max())
    return dx, vp, m0, r_pos, s_pos, rho, logQ


def animate(wavefield, shot=0, vmin=-2e7, vmax=2e7, interval=50):
    fig = plt.figure(figsize=(4,4))
    ims = []
    for i in range(wavefield.numpy()[shot].shape[0]):
        im = plt.imshow(wavefield.numpy()[shot][i], animated=True, vmin=vmin, vmax=vmax, 
                        cmap='Greys', interpolation='bilinear')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    plt.close()
    return ani


def trace_plot(data, shot, scale_fac=1, size=(8,4), line_color='black', linewidth=1, fill=True, fill_color='black', fill_color2=None, fill_fac=1e-2, save=None):
    time = np.arange(0, data.shape[2], 1)
    fig, ax1 = plt.subplots(figsize=size)
    for i in range(len(data[shot])):
        ax1.plot((scale_fac * data[shot][i]) + i, time, color=line_color, linewidth=linewidth)
        if fill:
            if fill_color2:
                if i%2 == 0:
                    ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                 where=(((scale_fac * data[shot][i]) + i) > i + fill_fac), color=fill_color)
                else:
                    ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                 where=(((scale_fac * data[shot][i]) + i) > i + fill_fac), color=fill_color2)
            else:
                ax1.fill_betweenx(time, i, (scale_fac * data[shot][i]) + i,
                                 where=(((scale_fac * data[shot][i]) + i) > i + fill_fac), color=fill_color)
    ax1.invert_yaxis()
    plt.xlim((-1, data.shape[1]))
    plt.ylim((data.shape[2]-1, 0))
    plt.xlabel('trace number')
    plt.ylabel('time-step')
    if save:
        plt.savefig(f'{save}.png', dpi=500) 
    plt.show()
                     

def trace_normalize(data, dim=-1):
    return data/torch.norm(data, dim=dim, keepdim=True)


def shot_normalize(data):
    return data/torch.norm(data, dim=(-2,-1), keepdim=True)