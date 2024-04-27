import numpy as np
import scipy
import matplotlib.pyplot as plt


def demo_model(dx, min_vel=1480, med_vel=1550, max_vel=1650, size=0.07,
               outer_radius=0.021, inner_radius=0.016, plot=False):
    """
    Generate a demonstration velocity model.

    Parameters
    ----------
    dx : float
        Grid spacing in meters.
    min_vel : float, optional
        Minimum velocity in the model. Default is 1480m/s.
    med_vel : float, optional
        Medium velocity in the model. Default is 1550m/s.
    max_vel : float, optional
        Maximum velocity in the model. Default is 1650m/s.
    size : float, optional
        Size of the model in meters. Default is 0.07m.
    outer_radius : float, optional
        Outer radius of the high-velocity region in meters.
        Default is 0.021m.
    inner_radius : float, optional
        Inner radius of the medium-velocity region in meters.
        Default is 0.016m.
    plot : bool, optional
        Whether to plot the generated model. Default is False.

    Returns
    -------
    m : ndarray
        True velocity model.
    m0 : ndarray
        Starting velocity model.

    """
    x = np.arange(0, size, dx)
    m = np.zeros((len(x), len(x)))
    m[:, :] = min_vel
    m0 = m.copy()
    m0[:, :] = min_vel
    for i in range(len(x)):
        for j in range(len(x)):
            if (x[i]-size/2)**2+(x[j]-size/2)**2 < outer_radius**2:
                m[i, j] = max_vel
            if (x[i]-size/2)**2+(x[j]-size/2)**2 < inner_radius**2:
                m[i, j] = med_vel
    if plot:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(m, vmin=min_vel, vmax=max_vel, interpolation='bilinear')
        plt.title('true model')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(m0, vmin=min_vel, vmax=max_vel, interpolation='bilinear')
        plt.title('starting model')
        plt.colorbar()
        plt.show()
    return m, m0


def demo_geometry(model, n_elements=30, hicks=True, rad=0.46, plot=False):
    """
    Generate a demonstration geometry for finite-difference simulations.

    Parameters
    ----------
    model : ndarray
        Velocity model array.
    n_elements : int, optional
        Number of elements in the geometry. Default is 30.
    hicks : bool, optional
        If True, generate coordinates as floats. If False, round them to
        the nearest integer. Default is True.
    rad : float, optional
        Radius parameter for geometry generation.
        Default is 0.46 grid cells.
    plot : bool, optional
        Whether to plot the generated geometry. Default is False.

    Returns
    -------
    pos : ndarray
        Array containing positions of elements in the geometry.

    """
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
        plt.figure(figsize=(4, 3))
        plt.imshow(model, vmin=model.min(), vmax=model.max(),
                   interpolation='bilinear')
        plt.colorbar()
        geometry = np.zeros_like(model)
        geometry[np.asarray(np.round(pos), dtype=int)[:, 0],
                 np.asarray(np.round(pos), dtype=int)[:, 1]] = 1000
        geometry = scipy.ndimage.gaussian_filter(geometry, 0.7)
        geometry[geometry < 1] = np.nan
        plt.imshow(geometry, cmap='Reds_r', interpolation='bilinear')
        plt.title('geometry')
        plt.show()
    return pos
