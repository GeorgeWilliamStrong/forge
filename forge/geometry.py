import torch
import numpy as np


def create_s_pos(new_s_pos, bp):
    """
    Generate new source positions, corrected for boundary points.

    Parameters
    ----------
    new_s_pos : ndarray, dtype=int
        Array of new two-dimensional source coordinates.
    bp : int
        Number of additional boundary points for absorbing boundaries.

    Returns
    -------
    torch.Tensor
        Tensor of two-dimensional source coordinates, prepared for
        parallel forward modelling.

    """
    s_pos = np.zeros((new_s_pos.shape[0], 3), dtype=int)

    for i in range(new_s_pos.shape[0]):
        s_pos[i, 0] = i
        s_pos[i, 1] = new_s_pos[i, 0]+bp
        s_pos[i, 2] = new_s_pos[i, 1]+bp

    return torch.from_numpy(s_pos)


def create_hicks_s_pos(new_s_pos, shape, bp, b=6.31, r=8):
    """
    Generate new Hicks interpolated source positions using
    Kaiser-windowed sinc interpolation.

    Parameters
    ----------
    new_s_pos : ndarray, dtype=float
        Array of new two-dimensional source coordinates.
    shape : list-like
        The shape of the full model domain (inner and outer).
    bp : int
        Number of additional boundary points for absorbing boundaries.
    **kwargs : additional keyword arguments
        b : float, optional
            Parameter that controls the shape of the Kaiser window.
            Defaults to optimal value of 6.31.
        r : int, optional
            Kaiser window size in grid cells. Defaults to 8.

    Returns
    -------
    torch.Tensor
        Tensor of two-dimensional hicks interpolated source coordinates,
        prepared for parallel forward modelling.
    torch.Tensor
        Corresponding tensor of two-dimensional Kaiser-windowed sinc
        interpolation weightings.

    """
    pos = new_s_pos+bp

    y = np.arange(0, shape[0], 1)
    x = np.arange(0, shape[1], 1)
    z = np.arange(0, r+1, 1)-r/2

    kaiser = np.kaiser(r+1, b)

    new_s_pos = []
    new_s_pos_values = []
    for i in range(pos.shape[0]):
        tempy = y-pos[i, 0]
        tempx = x-pos[i, 1]
        mesh = np.meshgrid(tempx, tempy)
        t = np.sqrt(mesh[0]**2+mesh[1]**2)

        hicks = np.interp(t, z, kaiser, left=0, right=0)*np.sinc(t)
        hicks_pos = np.transpose(np.nonzero(hicks))

        for j in range(len(hicks_pos)):
            new_s_pos.append(np.insert(hicks_pos[j], 0, i))
            new_s_pos_values.append(hicks[hicks_pos[j, 0], hicks_pos[j, 1]])

    return torch.tensor(np.array(new_s_pos), dtype=torch.long), \
        torch.tensor(np.array(new_s_pos_values), dtype=torch.float32)


def create_hicks_r_pos(r_pos, shape, b=6.31, r=8):
    """
    Generate Hicks interpolated receiver positions using
    Kaiser-windowed sinc interpolation.

    Parameters
    ----------
    r_pos : ndarray, dtype=float
        Array of two-dimensional receiver coordinates.
    shape : list-like
        The shape of the full model domain (inner and outer).
    **kwargs : additional keyword arguments
        b : float, optional
            Parameter that controls the shape of the Kaiser window.
            Defaults to optimal value of 6.31.
        r : int, optional
            Kaiser window size in grid cells. Defaults to 8.

    Returns
    -------
    torch.Tensor
        Tensor of two-dimensional hicks interpolated receiver coordinates,
        prepared for parallel forward modelling.
    torch.Tensor
        Corresponding tensor of two-dimensional Kaiser-windowed sinc
        interpolation weightings.
    torch.Tensor
        Tensor containing the number of elements in each of the Hicks
        interpolated receivers.

    """
    pos = r_pos.numpy()

    y = np.arange(0, shape[0], 1)
    x = np.arange(0, shape[1], 1)
    z = np.arange(0, r+1, 1)-r/2

    kaiser = np.kaiser(r+1, b)

    new_r_pos = []
    new_r_pos_values = []
    new_r_pos_sizes = []
    for i in range(pos.shape[0]):
        tempy = y-pos[i, 0]
        tempx = x-pos[i, 1]
        mesh = np.meshgrid(tempx, tempy)
        t = np.sqrt(mesh[0]**2+mesh[1]**2)

        hicks = np.interp(t, z, kaiser, left=0, right=0)*np.sinc(t)
        hicks_pos = np.transpose(np.nonzero(hicks))
        new_r_pos_sizes.append(len(hicks_pos))

        for j in range(len(hicks_pos)):
            new_r_pos.append(hicks_pos[j])
            new_r_pos_values.append(hicks[hicks_pos[j, 0], hicks_pos[j, 1]])

    return torch.tensor(np.array(new_r_pos)), \
        torch.tensor(np.array(new_r_pos_values), dtype=torch.float32), \
        torch.tensor(np.array(new_r_pos_sizes))


def d_hicks_to_d(data, r_pos_sizes, num_srcs, num_true_rec, num):
    """
    Transform Hicks interpolated recorded values of data at receivers
    from 'r_pos_sizes' to a single trace per receiver per shot.

    Parameters
    ----------
    data : torch.Tensor
        Three-dimensional Hicks interpolated acoustic data tensor.
    r_pos_sizes : torch.Tensor
        Tensor containing the number of elements in each of the Hicks
        interpolated receivers.
    num_srcs : int
        Number of physical sources.
    num_true_rec : int
        Number of physical receivers.
    num : int
        Number of time samples.

    Returns
    -------
    torch.Tensor
        Three-dimensional acoustic data tensor.

    """
    output_data = torch.zeros(num_srcs, num_true_rec, num).float()

    count = 0
    for j in range(len(r_pos_sizes)):
        output_data[:, j, :] = \
            data[:, count:count+r_pos_sizes[j], :].sum(dim=1)
        count += r_pos_sizes[j]

    return output_data


def adjoint_hicks(adjoint_source, r_pos_sizes, num_srcs, num_true_rec,
                  num_rec, num):
    """
    Transform adjoint source defined at physical receiver locations to
    Hicks interpolated adjoint source values.

    Parameters
    ----------
    adjoint_source : torch.Tensor
        Three-dimensional tensor containing the adjoint source pressure
        field to be injected for each receiver and for each shot.
    r_pos_sizes : torch.Tensor
        Tensor containing the number of elements in each of the Hicks
        interpolated receivers.
    num_srcs : int
        Number of physical sources.
    num_true_rec : int
        Number of physical receivers.
    num_rec : int
        Number of Hicks interpolation receivers.
    num : int
        Number of time samples.

    Returns
    -------
    torch.Tensor
        Three-dimensional tensor containing the adjoint source pressure
        field to be injected for each Hicks interpolated receiver location
        for each shot.

    """
    output_adjoint_source = torch.zeros(num_srcs, num_rec, num)

    count = 0
    for i in range(num_true_rec):
        output_adjoint_source[:, count:count+r_pos_sizes[i]] = \
            adjoint_source[:, i].expand(r_pos_sizes[i],
                                        num_srcs,
                                        num).transpose(0, 1)
        count += r_pos_sizes[i]

    return output_adjoint_source
