import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from .laplacian import laplacian
from .boundary import predictive_boundary
from .geometry import create_hicks_r_pos, create_hicks_s_pos, create_s_pos, \
    d_hicks_to_d
from .utils import *


class WaveInversion:
    def __init__(self, model, dx, dt, r_pos, **kwargs):
        """
        Initialise instance variables and allocate memory.

        Parameters
        ----------
        model : ndarray
            Two-dimensional acoustic velocity model in m/s.
        dx : float
            Temporal increment.
        dt : float
            Temporal increment.
        r_pos : ndarray
            Array of two-dimensional receiver coordinates.
        **kwargs : additional keyword arguments
            device : str, optional
                Name of the CUDA device to utilise. Defaults to 'cuda:0' or
                falls back to CPU if CUDA unavailable.
            boundary_points : int, optional
                Number of additional boundary points for absorbing boundaries.
                Defaults to 45.
            damping_factor : float, optional
                Variable that controls the strength of damping applied in
                absorbing boundary layer. Defaults to 0.0053.
            alpha : ndarray, optional
                Attenuation coefficient model in Np/m derived as follows:
                alpha = (pi*f)/(Q*vp),
                where f is the frequency in Hz, Q is the dimensionsless quality
                factor and vp is the acoustic velocity in m/s. Defaults to
                None.
            rho : ndarray, optional
                Density model in kg/m^3. Defaults to None.
            sampling_rate : int, optional
                Wavefield sampling rate. Defaults to 1.
            pred_bc : float, optional
                Predictive boundary coefficient. Defaults to 1.
            ot4 : bool, optional
                4th order accurate in time scheme. Defaults to True.

        Returns
        -------

        """
        device = kwargs.pop('device', 'cuda:0')
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        print(f'device = {self.device}')

        self._set_grid(dx, dt, **kwargs)
        self._set_media(model, **kwargs)
        self._set_operators(**kwargs)
        self._set_problem(r_pos, **kwargs)

    def _advance(self, a, b, t, adjoint_source=None):
        """
        Advance the first wavefield (a) by 1 time increment using finite
        differences that are 10th order accurate in space and 2nd or 4th
        order accurate in time, then inject source pressure field at
        pre-defined source/receiver positions. Finally, apply predictive
        boundary conditions to suppress unwanted boundary reflections.

        Parameters
        ----------
        a : torch.Tensor
            First pressure wavefield for time-stepping
        b : torch.Tensor
            Second pressure wavefield for time-stepping
        t : int
            Current position in time as an integer step.
        **kwargs : additional keyword arguments
            adjoint_source : torch.Tensor, optional
                Three-dimensional tensor containing the adjoint source pressure
                field to be injected for each receiver and for each shot. If
                this is passed in, the adjoint equation is used. Otherwise, the
                forward equation is adopted to propagate the source wavelet
                (self.s). Defaults to None.

        Returns
        -------

        """
        # Step forward in time and update the wavefield
        a[:, self.e:-self.e, self.e:-self.e] = \
            ((self.dt_sq /
             (self.m[self.e:-self.e, self.e:-self.e]**2)) *
                laplacian(b, self.dx_sq, self.b) +
                (2-self.q[self.e:-self.e, self.e:-self.e]**2) *
                b[:, self.e:-self.e, self.e:-self.e] -
                (1-self.q[self.e:-self.e, self.e:-self.e]) *
                a[:, self.e:-self.e, self.e:-self.e]) /\
            (1+self.q[self.e:-self.e, self.e:-self.e])

        # ot4 step
        if self.ot4:
            a[:, self.e2:-self.e2, self.e2:-self.e2] += \
                (((self.dt**4)/(12*self.dx_sq *
                                self.m[self.e2:-self.e2,
                                       self.e2:-self.e2]**4)) *
                    (laplacian(b[:, 2:, 1:-1],
                               self.dx_sq,
                               None) +
                     laplacian(b[:, :-2, 1:-1],
                               self.dx_sq,
                               None) +
                     laplacian(b[:, 1:-1, 2:],
                               self.dx_sq,
                               None) +
                     laplacian(b[:, 1:-1, :-2],
                               self.dx_sq,
                               None) -
                     4*laplacian(b[:, 1:-1, 1:-1],
                                 self.dx_sq,
                                 None))) / \
                (1+self.q[self.e2:-self.e2, self.e2:-self.e2])

        if adjoint_source is None:
            # Inject the source wavelet at source positions
            a[self.s_pos[:, 0], self.s_pos[:, 1], self.s_pos[:, 2]] += \
                (((self.dt_sq*self.s[t]) /
                  ((self.m[self.s_pos[:, 1], self.s_pos[:, 2]]**2) *
                    self.dx_sq)) /
                 (1-self.q[self.s_pos[:, 1], self.s_pos[:, 2]])) * \
                self.s_kaiser_sinc

        else:
            # Inject the adjoint source at all receiver positions
            a[:, self.r_pos[:, 0], self.r_pos[:, 1]] += \
                (((self.dt_sq*adjoint_source[:, :, t]) /
                  ((self.m[self.r_pos[:, 0], self.r_pos[:, 1]]**2) *
                   self.dx_sq)) /
                 (1-self.q[self.r_pos[:, 0], self.r_pos[:, 1]])) * \
                self.r_kaiser_sinc

        # Apply predictive boundary conditions
        self.pred(a, b, self.dt, self.dx, self.m)

    def forward(self, s_pos, source):
        """
        Solve the forward problem, G(m) = d, given a batch of source
        geometry positions and a source wavelet. A down-sampled forward
        wavefield is stored according to the sampling rate.

        Parameters
        ----------
        s_pos : ndarray
            Array of two-dimensional source coordinates.
        source : ndarray
            One-dimensional array containing source wavelet pressure field.

        Returns
        -------

        """
        # Initialize source geometry positions, wavefields and data arrays
        self._set_batch(s_pos, source)

        # Zero all wavefield and recorded data values
        self.u2[:, :, :] = 0
        self.u1[:, :, :] = 0
        self.d[:, :, :] = 0
        self.wavefield[:, :, :, :] = 0

        # Begin time-stepping
        for t in tqdm(range(len(self.s)), colour='blue', ncols=60,
                      mininterval=0.03):

            # Alternate wavefield updates between u1 and u2 to avoid storing a
            # third wavefield
            if t % 2 == 0:

                # Record current wavefield at receiver locations
                self.d[:, :, t] = \
                    self.u1[:, self.r_pos[:, 0], self.r_pos[:, 1]] * \
                    self.r_kaiser_sinc

                # Store u2 for calculating the partial derivative wavefield
                if t % self.s_rate == 0:
                    self.wavefield[:, int(t/self.s_rate), :, :] = \
                        self.u2[:, self.bp:-self.bp, self.bp:-self.bp]

                # Step forward in time
                self._advance(self.u2, self.u1, t)

                # Calculate and store the partial derivative wavefield
                if t % self.s_rate == 0:
                    self.wavefield[:, int(t/self.s_rate), :, :] = \
                        (2*self.m[self.bp:-self.bp, self.bp:-self.bp]) * \
                        ((self.wavefield[:, int(t/self.s_rate), :, :] -
                          (2*self.u1[:, self.bp:-self.bp, self.bp:-self.bp]) +
                          self.u2[:, self.bp:-self.bp, self.bp:-self.bp]) /
                         self.dt_sq)

            else:
                # Record current wavefield at receiver locations
                self.d[:, :, t] = \
                    self.u2[:, self.r_pos[:, 0], self.r_pos[:, 1]] * \
                    self.r_kaiser_sinc

                # Store u1 for calculating the partial derivative wavefield
                if t % self.s_rate == 0:
                    self.wavefield[:, int(t/self.s_rate), :, :] = \
                        self.u1[:, self.bp:-self.bp, self.bp:-self.bp]

                # Step forward in time
                self._advance(self.u1, self.u2, t)

                # Calculate and store the partial derivative wavefield
                if t % self.s_rate == 0:
                    self.wavefield[:, int(t/self.s_rate), :, :] = \
                        (2*self.m[self.bp:-self.bp, self.bp:-self.bp]) * \
                        ((self.wavefield[:, int(t/self.s_rate), :, :] -
                          (2*self.u2[:, self.bp:-self.bp, self.bp:-self.bp]) +
                          self.u1[:, self.bp:-self.bp, self.bp:-self.bp]) /
                         self.dt_sq)

        # If hicks interpolation is used, transform the second dimension of d
        # from self.r_pos.size(0) to self.num_rec
        if self.r_hicks:
            self.d = d_hicks_to_d(self.d,
                                  self.r_pos_sizes,
                                  self.num_rec,
                                  self.num_srcs,
                                  self.s)

    def adjoint(self, adjoint_source):
        """
        Use the adjoint-state method to calculate the gradient.
        """

        # allocate memory to the GPU and, if required, apply hicks interpolation
        adjoint_source = resid_hicks(adjoint_source, self.num_srcs, self.num_rec,
                            self.r_pos, self.r_pos_sizes, self.r_hicks).to(self.device)
        
        # zero all wavefield and gradient values
        self.u2[:, :, :] = 0
        self.u1[:, :, :] = 0
        
        # use counter to cross-correlate the partial derivative forward and backward wavefields at the correct time
        count = 1
        
        # begin time-stepping
        for t in tqdm(range(adjoint_source.size(2)-1, -1, -1), colour='magenta', ncols=60, mininterval=0.03):
            
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield
            if t % 2 == 0:
                
                # Step backwards in time
                self._advance(self.u2, self.u1, t,
                              adjoint_source=adjoint_source)
                
                # cumulatively calculate the gradient throughout backpropagation by cross-correlating forward and backward wavefields
                if t%self.s_rate==0:
                    self.m.grad[self.bp:-self.bp, self.bp:-self.bp] -= (self.u2[:, self.bp:-self.bp, self.bp:-self.bp]*self.wavefield[:, -count]).sum(0)
                    count += 1
                    
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield   
            else:
                
                # Step backwards in time
                self._advance(self.u1, self.u2, t,
                              adjoint_source=adjoint_source)
                
                # cumulatively calculate the gradient throughout backpropagation by cross-correlating forward and backward wavefields
                if t%self.s_rate==0:
                    self.m.grad[self.bp:-self.bp, self.bp:-self.bp] -= (self.u1[:, self.bp:-self.bp, self.bp:-self.bp]*self.wavefield[:, -count]).sum(0)
                    count += 1
    
    def m_out(self):
        # extract the model in acoustic velocity
        return 1/self.m[self.bp:-self.bp, self.bp:-self.bp].detach().cpu()
    
    def fit(self, data, s_pos, source, optimizer, loss, num_iter, bs, runs=1, blocks=None, grad_norm=True, hess_prwh=1e-9, model_callbacks = [], adjoint_callbacks=[], box=None, true_model=None):

        # loss function tracking
        self.loss_history = []

        # model RMSE tracking
        if true_model is not None:
            self.rmse_history = []
        
        # are multiple blocks required?
        if blocks is not None:
            num_blocks = len(blocks)
        else:
            num_blocks = 1
        
        # loop over multiscale frequency blocks
        for i in range(num_blocks):
            
            # low-pass the source and the data as required by the block frequency
            if blocks is not None:
                print(f"block {i+1}/{num_blocks} | {blocks[i]:.4g}Hz")
                s = butter_lowpass_filter(source.copy(), blocks[i], 1/self.dt, order=12)
                data_filt = torch.from_numpy(butter_lowpass_filter(data, blocks[i], 1/self.dt, order=12)).float()
            else:
                data_filt = data
                s = source.copy()
            
            # begin optimization loop 
            for j in range(num_iter):

                print(f'  iteration {j+1}/{num_iter}')

                # zero the gradient
                self.m.grad[:, :] = 0

                # zero the diagonal of the approximate Hessian
                if hess_prwh:
                    self.hess[:, :] = 0

                # empty the cache
                torch.cuda.empty_cache()

                # extract a random set of shots for this iteration
                total_batch = np.random.choice(len(s_pos), bs*runs, replace=False)

                f_n = 0

                # loop over multiple forward runs per iteration
                for k in range(runs):

                    if runs > 1:
                        print(f'    batch {k+1}/{runs}')
                    
                    # solve the forward problem G(m) = d, storing the forward wavefield
                    self.forward(s_pos[total_batch][k*bs:k*bs+bs], s)

                    # turn on gradient tracking with respect to the predicted data
                    self.d.requires_grad_(requires_grad=True)

                    # call the loss function
                    f = loss(self.d, data_filt[total_batch][k*bs:k*bs+bs])

                    # use AD to calculate the adjoint source self.d.grad
                    f.backward()

                    # turn off gradient tracking with respect to the predicted data
                    self.d.requires_grad_(requires_grad=False)
                    
                    # adjoint source callbacks e.g. for low-pass filtering
                    for i in adjoint_callbacks:
                        i(self.d.grad)
                    
                    # backpropagate adjoint source through the model, generating a gradient
                    self.adjoint(self.d.grad)

                    # clear the model.d.grad values for next iteration
                    self.d.grad = None

                    # increment the diagonal of the approximate Hessian
                    if hess_prwh:
                        self.hess += (self.wavefield**2).sum(1).sum(0)

                    # increment the sample normalized loss value
                    f_n += f.item()/(self.d.size(0)*self.d.size(1)*self.d.size(2))
                
                # precondition the gradient with the diagonal of the approximate Hessian
                if hess_prwh:
                    self.m.grad[self.bp:-self.bp, self.bp:-self.bp] /= (self.hess + hess_prwh*torch.norm(self.hess))
                
                # normalize the gradient so the maximum absolute value is 1
                if grad_norm:
                    self.m.grad /= abs(self.m.grad).max()
                
                # model/gradient callbacks e.g. for smoothing or regularisation terms
                for i in model_callbacks:
                    i(self.m)
                
                # take optimization step 
                optimizer.step()
                
                # box constraints
                if box:
                    self.m.data = torch.clamp(self.m, 1/box[1], 1/box[0])
                    
                # calculate, print and record sample normalized loss value
                f_n /= runs
                print(f'    loss = {f_n:.4g}')
                self.loss_history.append(f_n)

                if true_model is not None:
                    # calculate, print and record a sample normalized model RMSE
                    rmse = torch.sqrt(((self.m_out() - true_model)**2).mean()).item()/(true_model.shape[0]*true_model.shape[1])
                    print(f'    rmse = {rmse:.4g}')
                    self.rmse_history.append(rmse)
            
            print("______________________________________________________________________ \n")

    def _set_grid(self, dx, dt, **kwargs):
        """
        Define instance variables to handle grid sampling and boundary cells.

        Parameters
        ----------
        dx : float
            Spatial increment.
        dt : float
            Temporal increment.
        **kwargs : additional keyword arguments
            boundary_points : int, optional
                Number of additional boundary points for absorbing boundaries.
                Defaults to 45.

        Returns
        -------

        """
        self.dx = float(dx)
        self.dx_sq = float(dx**2)

        self.dt = float(dt)
        self.dt_sq = float(dt**2)

        bp = kwargs.pop('boundary_points', 45)
        assert bp > 10, 'There must be at least 10 boundary points.'
        self.bp = int(bp)

    def _set_media(self, model, **kwargs):
        """
        Define instance variables to handle various media required for FWI.

        Parameters
        ----------
        model : ndarray
            Two-dimensional acoustic velocity model in m/s.
        **kwargs : additional keyword arguments
            damping_factor : float, optional
                Variable that controls the strength of damping applied in
                absorbing boundary layer. Defaults to 0.0053.
            alpha : ndarray, optional
                Attenuation coefficient model in Np/m derived as follows:
                alpha = (pi*f)/(Q*vp),
                where f is the frequency in Hz, Q is the dimensionsless quality
                factor and vp is the acoustic velocity in m/s. Defaults to
                None.
            rho : ndarray, optional
                Density model in kg/m^3. Defaults to None.

        Returns
        -------

        """
        damping_factor = kwargs.pop('damping_factor', 0.0053)
        alpha = kwargs.pop('alpha', None)
        rho = kwargs.pop('rho', None)
        shape = model.shape

        # Store model in slowness and allocate to device
        self.m = 1/torch.from_numpy(
            np.pad(model,
                   self.bp,
                   mode='edge')).float().to(self.device)

        # Create damping array, pad with neg. exponential using optimal params
        self.damp = torch.from_numpy(np.exp(-(damping_factor**2)*((np.pad(
            np.zeros((shape[0], shape[1])),
            self.bp,
            mode='linear_ramp',
            end_values=self.bp))**2))).float()

        # Derive default attenuation model from damping array
        self.q = (1/self.damp)-1

        # Insert attenuation model if provided
        if alpha is not None:
            self.q[self.bp:-self.bp, self.bp:-self.bp] = \
                (torch.from_numpy(alpha)*self.dt)*torch.from_numpy(model)
        self.q = self.q.to(self.device)

        if rho is not None:
            # Store density model as buoyancy
            self.b = 1/torch.from_numpy(
                np.pad(rho,
                       self.bp,
                       mode='edge')).float().to(self.device)
        else:
            self.b = None

        # Initialise model gradient
        self.m.grad = torch.zeros_like(self.m)

        # Initialise diagonal of the approximate Hessian
        self.hess = torch.zeros_like(
            torch.from_numpy(model)).float().to(self.device)

    def _set_operators(self, **kwargs):
        """
        Define instance variables to handle boundary and ot4 operators.

        Parameters
        ----------
        **kwargs : additional keyword arguments
            pred_bc : float, optional
                Predictive boundary coefficient. Defaults to 1.
            ot4 : bool, optional
                4th order accurate in time scheme. Defaults to True.

        Returns
        -------

        """
        # Define strength of predictive boundaries
        self.pred_bc = float(kwargs.pop('pred_bc', 1))

        # Set 4th order accurate in time flag
        self.ot4 = kwargs.pop('ot4', True)

        if self.b is not None:
            # Variable density Laplacian shrinks u by 9 cells in each dimension
            self.e = 9
            # Set the variable density predictive boundary function
            self.pred = partial(predictive_boundary,
                                pred_bc=self.pred_bc,
                                rho=True)
        else:
            # Constant density Laplacian shrinks u by 5 cells in each dimension
            self.e = 5
            # Set the constant density predictive boundary function
            self.pred = partial(predictive_boundary,
                                pred_bc=self.pred_bc,
                                rho=False)

        if self.ot4:
            # OT4 method shrinks u by 6 cells in each dimension
            self.e2 = 6

    def _set_problem(self, r_pos, **kwargs):
        """
        Define instance variables to handle acquisitions and geometry.

        Parameters
        ----------
        r_pos : ndarray
            Array of two-dimensional receiver coordinates.
        **kwargs : additional keyword arguments
            sampling_rate : int, optional
                Wavefield sampling rate. Defaults to 1.

        Returns
        -------

        """
        # Set receiver locations as indexes and add on boundary points
        self.r_pos = torch.from_numpy(r_pos[:] + self.bp)

        # Set wavefield sampling rate
        self.s_rate = int(kwargs.pop('sampling_rate', 1))

        # Source kaiser windowed sinc function values
        self.s_kaiser_sinc = 1
        self.s_set = False  # Source set boolean
        self.num_rec = self.r_pos.size(0)  # Number of receivers

        # Check whether hicks interpolation is required
        if type(self.r_pos[0, 0].item()) != int:
            self.r_hicks = True  # Receiver hicks boolean

            # Receiver hicks interpolation
            self.r_pos, self.r_kaiser_sinc, self.r_pos_sizes = \
                create_hicks_r_pos(self.r_pos, self.m)
            self.r_kaiser_sinc = self.r_kaiser_sinc.to(self.device)
        else:
            self.r_hicks = False
            self.r_kaiser_sinc = 1
            self.r_pos_sizes = False

        # Set instance variables to None before they have been set
        self.s = None  # Source wavelet
        self.u1 = None  # First wavefield for time-stepping
        self.u2 = None  # Second wavefield for time-stepping
        self.d = None  # Data recorded at receivers
        self.wavefield = None  # Full partial derivative wavefield
        self.s_pos = None  # Source indexes
        self.num_srcs = None  # Number of sources
        self.s_hicks = None  # Source hicks boolean

    def _source_interp(self, s_pos):
        """
        Apply Hicks interpolation to source geometry coordinates, if necessary.

        Parameters
        ----------
        s_pos : ndarray
            Array of two-dimensional source coordinates.

        Returns
        -------

        """
        # If hicks interpolation is required, set s_hicks to True
        if type(s_pos[0, 0]) is not np.int64:
            # Calculate source indices for hicks interpolation and
            # corresponding kaiser windowed sinc function values
            self.s_hicks = True
            self.s_pos, s_kaiser_sinc = create_hicks_s_pos(s_pos,
                                                           self.m,
                                                           self.bp)
            self.s_kaiser_sinc = s_kaiser_sinc.to(self.device)

        # If hicks interpolation is not required, set s_hicks to False
        else:
            # Set kaiser windowed sinc function values to 1 and create
            # source indices
            self.s_hicks = False
            self.s_kaiser_sinc = 1
            self.s_pos = create_s_pos(s_pos, self.bp)

    def _set_batch(self, s_pos, source):
        """
        Initialise source locations, wavefields and data arrays for a batch.

        Parameters
        ----------
        s_pos : ndarray
            Array of two-dimensional source coordinates.
        source : ndarray
            One-dimensional array containing source wavelet pressure field.

        Returns
        -------

        """
        self.s = source.copy()

        # Check if sources have not yet been set or if there is a different
        # number of new and old source indices
        if not self.s_set or len(s_pos) != self.num_srcs:
            self.s_set = True
            self.num_srcs = s_pos.shape[0]

            # Apply hicks interpolation if necessary
            self._source_interp(self, s_pos)

            # Initialise wavefield and data arrays
            self.u2 = torch.zeros(self.num_srcs,
                                  self.m.size(0),
                                  self.m.size(1)).float().to(self.device)
            self.u1 = torch.zeros_like(self.u2)
            self.d = torch.zeros(self.num_srcs,
                                 self.r_pos.size(0),
                                 len(self.s)).float()
            self.wavefield = torch.zeros(
                self.num_srcs,
                len(self.s[::self.s_rate]),
                self.m[self.bp:-self.bp, :].shape[0],
                self.m[:, self.bp:-self.bp].shape[1]).float().to(self.device)

        else:
            # Apply hicks interpolation if necessary
            self._source_interp(self, s_pos)

            # Reinitialise data array as d_hicks_to_d changes dimensionalality
            # of 2nd dimension
            self.d = torch.zeros(self.num_srcs,
                                 self.r_pos.size(0),
                                 len(source)).float()
