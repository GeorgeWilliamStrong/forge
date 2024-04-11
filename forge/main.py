import torch
import numpy as np
from tqdm import tqdm
from .laplacian import *
from .geometry import *
from .utils import *


class WaveInversion:
    """
    Two-dimensional full-waveform inversion class comprising:
      1. Initialisation and (GPU) memory allocation
      2. Configuration of the source wavelet, source locations, wavefields and data tensors
      3. Solving the forward problem G(m) = d
      4. Calculate the gradient using the adjoint-state method
      5. Multi-scale optimization loop
    """


    def __init__(self, model, dx, dt, r_pos, **kwargs):

        """
        Initialise class variables and allocate memory.
        """

        # sampling_rate=1, bpoints=45, pred_bc=1, alpha=None, rho=None, OT4=True, dfac=0.0053, device="cuda:0"):

        device = kwargs.pop('device', 'cuda:0')
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu') # use a GPU if available
        print(f'device = {self.device}')

        self._set_grid(dx, dt, **kwargs)

        self._set_medium(model, **kwargs)

        #self._set_acquisitions(r_pos, **kwargs)

        #self._set_operators(**kwargs)

        self.r_pos = torch.from_numpy(r_pos[:]+self.bp) # receiver locations as indexes
        self.s_rate = int(sampling_rate) # wavefield sampling rate
        self.s = None # source wavelet
        self.u1 = None # first wavefield for time-stepping
        self.u2 = None # second wavefield for time-stepping
        self.d = None # data recorded at receivers
        self.wavefield = None # full partial derivative wavefield
        self.s_pos = None # source indexes
        self.num_srcs = None # number of sources
        self.s_kaiser_sinc = 1 # source kaiser windowed sinc function values 
        self.s_hicks = None # source hicks boolean
        self.s_set = False # source set boolean
        self.num_rec = self.r_pos.size(0) # number of receivers
        if type(self.r_pos[0, 0].item()) != int: # check whether hicks interpolation is required
            self.r_hicks = True # receiver hicks boolean
            self.r_pos, self.r_kaiser_sinc, self.r_pos_sizes = create_hicks_r_pos(self.r_pos, self.m) # receiver hicks interpolation
            self.r_kaiser_sinc = self.r_kaiser_sinc.to(self.device)
        else:
            self.r_hicks = False 
            self.r_kaiser_sinc = 1
            self.r_pos_sizes = False
        if self.b is not None: # check whether variable density propagator is required
            self.e = 9 # laplacian shrinks u by 9 cells in each dimension
            self.pred = pred_bc_10th_order_density # set the variable density predictive boundary function
        else: # if constant density propagator is required
            self.e = 5 # laplacian shrinks u by 5 cells in each dimension
            self.pred = pred_bc_10th_order # set the constant density predictive boundary function
        self.OT4 = OT4
        if self.OT4 == True: # check if OT4 accuracy is required
            self.e2 = 6 # OT4 method shrinks u by 6 cells in each dimension
        self.pred_bc = float(pred_bc) # strength of predictive boundaries
        self.loss_history = None # loss function tracking
        self.rmse_history = None # model RMSE tracking

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

    def _set_medium(self, model, **kwargs):
        """
        Define instance variables to handle various media required for FWI.

        Parameters
        ----------
        model : ndarray
            Two-dimensional acoustic velocity model.
        **kwargs : additional keyword arguments
            damping_factor : float
                Variable that controls the strength of damping applied in
                absorbing boundary layer. Defaults to 0.0053.
            alpha : ndarray
                Attenuation coefficient model in Np/m derived as follows:
                alpha = (pi*f)/(Q*vp),
                where f is the frequency in Hz, Q is the dimensionsless quality
                factor and vp is the acoustic velocity in m/s. Defaults to
                None.
            rho : ndarray
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

        # Populate model gradient TODO: move this to .fit()?
        self.m.grad = torch.zeros_like(self.m)

        # Diagonal of the approximate Hessian TODO: move this to .fit()?
        self.hess = torch.zeros_like(
            torch.from_numpy(model)).float().to(self.device)


    def configure(self, s_pos, source):
        """
        Initialise source locations, wavefields and data arrays.
        """
        
        # set the source wavelet
        self.s = source.copy()
        
        # check if sources have not yet been set
        if self.s_set==False:
            
            self.s_set = True # sources are now being set so this becomes true
            self.num_srcs = s_pos.shape[0] # store the number of physical sources
            
            # if hicks interpolation is required, set s_hicks to true
            if type(s_pos[0,0]) != np.int64:
                self.s_hicks = True
                
                # calculate source indexes for hicks interpolation and corresponding kaiser windowed sinc function values
                self.s_pos, s_kaiser_sinc = create_hicks_s_pos(s_pos, self.m, self.bp)
                self.s_kaiser_sinc = s_kaiser_sinc.to(self.device) # allocate memory to the GPU
            
            # if hicks interpolation is not required, set s_hicks to false
            else:
                self.s_hicks = False
                
                # set kaiser windowed sinc function values to 1 and create source indexes 
                self.s_kaiser_sinc = 1
                self.s_pos = create_s_pos(s_pos, self.bp)
            
            # initialise wavefield and data arrays
            self.u2 = torch.zeros(self.num_srcs, self.m.size(0), self.m.size(1)).float().to(self.device)
            self.u1 = torch.zeros_like(self.u2)
            self.d = torch.zeros(self.num_srcs, self.r_pos.size(0), len(self.s)).float()
            self.wavefield = torch.zeros(self.num_srcs, len(self.s[::self.s_rate]), self.m[self.bp:-self.bp, :].shape[0], self.m[:, self.bp:-self.bp].shape[1]).float().to(self.device)
        
        # check if there is the same number of new and old source indexes
        elif len(s_pos)==self.num_srcs:
            
            # if hicks interpolation is required, set s_hicks to true
            if type(s_pos[0,0]) != np.int64:
                self.s_hicks = True
                
                # calculate source indexes for hicks interpolation and corresponding kaiser windowed sinc function values
                self.s_pos, s_kaiser_sinc = create_hicks_s_pos(s_pos, self.m, self.bp)
                self.s_kaiser_sinc = s_kaiser_sinc.to(self.device) # allocate memory to the GPU
            
            # if hicks interpolation is not required, set s_hicks to false
            else:
                self.s_hicks = False
                
                # set kaiser windowed sinc function values to 1 and create source indexes 
                self.s_kaiser_sinc = 1
                self.s_v = create_s_pos(s_pos, self.bp)
            
            # reinitialise data array as d_hicks_to_d changes dimensionalality of 2nd dimension 
            self.d = torch.zeros(self.num_srcs, self.r_pos.size(0), len(source)).float()
        
        # if there is a different number of new and old source indexes
        else:
            
            self.num_srcs = s_pos.shape[0] # store the number of physical sources
            
            # if hicks interpolation is required, set s_hicks to true
            if type(s_pos[0,0]) != np.int64:
                self.s_hicks = True
                
                # calculate source indexes for hicks interpolation and corresponding kaiser windowed sinc function values
                self.s_pos, s_kaiser_sinc = create_hicks_s_pos(s_pos, self.m, self.bp)
                self.s_kaiser_sinc = s_kaiser_sinc.to(self.device) # allocate memory to the GPU
            
            # if hicks interpolation is not required, set s_hicks to false
            else:
                self.s_hicks = False
                
                # set kaiser windowed sinc function values to 1 and create source indexes
                self.s_kaiser_sinc = 1
                self.s_pos = create_s_pos(s_pos, self.bp)
            
            # reinitialise wavefield and data arrays
            self.u2 = torch.zeros(self.num_srcs, self.m.size(0), self.m.size(1)).float().to(self.device)
            self.u1 = torch.zeros_like(self.u2)
            self.d = torch.zeros(self.num_srcs, self.r_pos.size(0), len(self.s)).float()
            self.wavefield = torch.zeros(self.num_srcs, len(self.s[::self.s_rate]), self.m[self.bp:-self.bp, :].shape[0], self.m[:, self.bp:-self.bp].shape[1]).float().to(self.device)
    
    
    def forward(self):
        """
        Solve the forward problem G(m) = d, storing the forward wavefield.
        """
        
        # zero all wavefield and recorded data values
        self.u2[:, :, :] = 0
        self.u1[:, :, :] = 0
        self.d[:, :, :] = 0
        self.wavefield[:, :, :, :] = 0
        
        # begin time-stepping 
        for i in tqdm(range(len(self.s)), colour='blue', ncols=60, mininterval=0.03):
            
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield
            if i%2==0:
                
                # record current wavefield at receiver locations
                self.d[:, :, i] = self.u1[:, self.r_pos[:, 0], self.r_pos[:, 1]]*self.r_kaiser_sinc
                
                # temporarily store u2 for calculating the partial derivative wavefield
                if i%self.s_rate==0:
                    self.wavefield[:, int(i/self.s_rate), :, :] = self.u2[:, self.bp:-self.bp, self.bp:-self.bp]
                
                # step forward in time and update the wavefield 
                self.u2[:, self.e:-self.e, self.e:-self.e] = ((self.dt_sq/(self.m[self.e:-self.e, self.e:-self.e]**2))*laplacian(self.u1, self.dx_sq, self.b)+(2-self.q[self.e:-self.e, self.e:-self.e]**2)*self.u1[:, self.e:-self.e, self.e:-self.e]-(1-self.q[self.e:-self.e, self.e:-self.e])*self.u2[:, self.e:-self.e, self.e:-self.e])/(1+self.q[self.e:-self.e, self.e:-self.e])
                
                # ot4 step
                if self.OT4 == True:
                    self.u2[:, self.e2:-self.e2, self.e2:-self.e2] += (((self.dt**4)/(12*self.dx_sq*self.m[self.e2:-self.e2, self.e2:-self.e2]**4))*(laplacian(self.u1[:,2:,1:-1], self.dx_sq, None)+laplacian(self.u1[:,:-2,1:-1], self.dx_sq, None)+laplacian(self.u1[:,1:-1,2:], self.dx_sq, None)+laplacian(self.u1[:,1:-1,:-2], self.dx_sq, None)-4*laplacian(self.u1[:,1:-1,1:-1], self.dx_sq, None)))/(1+self.q[self.e2:-self.e2, self.e2:-self.e2])

                # inject the source wavelet
                self.u2[self.s_pos[:, 0], self.s_pos[:, 1], self.s_pos[:, 2]] += (((self.dt_sq*self.s[i])/((self.m[self.s_pos[:, 1], self.s_pos[:, 2]]**2)*self.dx_sq))/(1-self.q[self.s_pos[:, 1], self.s_pos[:, 2]]))*self.s_kaiser_sinc
                
                # apply predictive boundary conditions
                self.pred(self.u2, self.u1, self.dt, self.dx, self.m, self.pred_bc)
                
                # calculate and store the partial derivative wavefield
                if i%self.s_rate==0:
                    self.wavefield[:, int(i/self.s_rate), :, :] = (2*self.m[self.bp:-self.bp, self.bp:-self.bp])*((self.wavefield[:, int(i/self.s_rate), :, :]-(2*self.u1[:, self.bp:-self.bp, self.bp:-self.bp])+self.u2[:, self.bp:-self.bp, self.bp:-self.bp])/self.dt_sq)
            
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield
            else:
                
                # record current wavefield at receiver locations
                self.d[:, :, i] = self.u2[:, self.r_pos[:, 0], self.r_pos[:, 1]]*self.r_kaiser_sinc
                
                # temporarily store u1 for calculating the partial derivative wavefield
                if i%self.s_rate==0:
                    self.wavefield[:, int(i/self.s_rate), :, :] = self.u1[:, self.bp:-self.bp, self.bp:-self.bp]
                
                # step forward in time and update the wavefield 
                self.u1[:, self.e:-self.e, self.e:-self.e] = ((self.dt_sq/(self.m[self.e:-self.e, self.e:-self.e]**2))*laplacian(self.u2, self.dx_sq, self.b)+(2-self.q[self.e:-self.e, self.e:-self.e]**2)*self.u2[:, self.e:-self.e, self.e:-self.e]-(1-self.q[self.e:-self.e, self.e:-self.e])*self.u1[:, self.e:-self.e, self.e:-self.e])/(1+self.q[self.e:-self.e, self.e:-self.e])
                
                # ot4 step
                if self.OT4 == True:
                    self.u1[:, self.e2:-self.e2, self.e2:-self.e2] += (((self.dt**4)/(12*self.dx_sq*self.m[self.e2:-self.e2, self.e2:-self.e2]**4))*(laplacian(self.u2[:,2:,1:-1], self.dx_sq, None)+laplacian(self.u2[:,:-2,1:-1], self.dx_sq, None)+laplacian(self.u2[:,1:-1,2:], self.dx_sq, None)+laplacian(self.u2[:,1:-1,:-2], self.dx_sq, None)-4*laplacian(self.u2[:,1:-1,1:-1], self.dx_sq, None)))/(1+self.q[self.e2:-self.e2, self.e2:-self.e2])
                
                # inject the source wavelet
                self.u1[self.s_pos[:, 0], self.s_pos[:, 1], self.s_pos[:, 2]] += (((self.dt_sq*self.s[i])/((self.m[self.s_pos[:, 1], self.s_pos[:, 2]]**2)*self.dx_sq))/(1-self.q[self.s_pos[:, 1], self.s_pos[:, 2]]))*self.s_kaiser_sinc
                
                # apply predictive boundary conditions
                self.pred(self.u1, self.u2, self.dt, self.dx, self.m, self.pred_bc)
                
                # calculate and store the partial derivative wavefield
                if i%self.s_rate==0:
                    self.wavefield[:, int(i/self.s_rate), :, :] = (2*self.m[self.bp:-self.bp, self.bp:-self.bp])*((self.wavefield[:, int(i/self.s_rate), :, :]-(2*self.u2[:, self.bp:-self.bp, self.bp:-self.bp])+self.u1[:, self.bp:-self.bp, self.bp:-self.bp])/self.dt_sq)
        
        # if hicks interpolation is used, transform the second dimension of d from self.r_pos.size(0) to self.num_rec
        if self.r_hicks == True:
            self.d = d_hicks_to_d(self.d, self.r_pos_sizes, self.num_rec, self.num_srcs, self.s)
        
        
    def adjoint_gradient(self, adjoint_source):
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
        for i in tqdm(range(adjoint_source.size(2)-1, -1, -1), colour='magenta', ncols=60, mininterval=0.03):
            
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield
            if i%2==0:
                
                # step forward in time and update the wavefield 
                self.u2[:, self.e:-self.e, self.e:-self.e] = ((self.dt_sq/(self.m[self.e:-self.e, self.e:-self.e]**2))*laplacian(self.u1, self.dx_sq, self.b)+(2-self.q[self.e:-self.e, self.e:-self.e]**2)*self.u1[:, self.e:-self.e, self.e:-self.e]-(1-self.q[self.e:-self.e, self.e:-self.e])*self.u2[:, self.e:-self.e, self.e:-self.e])/(1+self.q[self.e:-self.e, self.e:-self.e])
                
                # ot4 step
                if self.OT4 == True:
                    self.u2[:, self.e2:-self.e2, self.e2:-self.e2] += (((self.dt**4)/(12*self.dx_sq*self.m[self.e2:-self.e2, self.e2:-self.e2]**4))*(laplacian(self.u1[:,2:,1:-1], self.dx_sq, None)+laplacian(self.u1[:,:-2,1:-1], self.dx_sq, None)+laplacian(self.u1[:,1:-1,2:], self.dx_sq, None)+laplacian(self.u1[:,1:-1,:-2], self.dx_sq, None)-4*laplacian(self.u1[:,1:-1,1:-1], self.dx_sq, None)))/(1+self.q[self.e2:-self.e2, self.e2:-self.e2])

                # inject the adjoint source at all receiver locations
                self.u2[:, self.r_pos[:, 0], self.r_pos[:, 1]] += (((self.dt_sq*adjoint_source[:, :, i])/((self.m[self.r_pos[:, 0], self.r_pos[:, 1]]**2)*self.dx_sq))/(1-self.q[self.r_pos[:, 0], self.r_pos[:, 1]]))*self.r_kaiser_sinc
                
                # apply predictive boundary conditions
                self.pred(self.u2, self.u1, self.dt, self.dx, self.m, self.pred_bc)
                
                # cumulatively calculate the gradient throughout backpropagation by cross-correlating forward and backward wavefields
                if i%self.s_rate==0:
                    self.m.grad[self.bp:-self.bp, self.bp:-self.bp] -= (self.u2[:, self.bp:-self.bp, self.bp:-self.bp]*self.wavefield[:, -count]).sum(0)
                    count += 1
                    
            # alternate wavefield updates between u1 and u2 to avoid storing a third wavefield   
            else:
                
                # step forward in time and update the wavefield 
                self.u1[:, self.e:-self.e, self.e:-self.e] = ((self.dt_sq/(self.m[self.e:-self.e, self.e:-self.e]**2))*laplacian(self.u2, self.dx_sq, self.b)+(2-self.q[self.e:-self.e, self.e:-self.e]**2)*self.u2[:, self.e:-self.e, self.e:-self.e]-(1-self.q[self.e:-self.e, self.e:-self.e])*self.u1[:, self.e:-self.e, self.e:-self.e])/(1+self.q[self.e:-self.e, self.e:-self.e])
                
                # ot4 step
                if self.OT4 == True:
                    self.u1[:, self.e2:-self.e2, self.e2:-self.e2] += (((self.dt**4)/(12*self.dx_sq*self.m[self.e2:-self.e2, self.e2:-self.e2]**4))*(laplacian(self.u2[:,2:,1:-1], self.dx_sq, None)+laplacian(self.u2[:,:-2,1:-1], self.dx_sq, None)+laplacian(self.u2[:,1:-1,2:], self.dx_sq, None)+laplacian(self.u2[:,1:-1,:-2], self.dx_sq, None)-4*laplacian(self.u2[:,1:-1,1:-1], self.dx_sq, None)))/(1+self.q[self.e2:-self.e2, self.e2:-self.e2])
                
                # inject the adjoint source at all receiver locations
                self.u1[:, self.r_pos[:, 0], self.r_pos[:, 1]] += (((self.dt_sq*adjoint_source[:, :, i])/((self.m[self.r_pos[:, 0], self.r_pos[:, 1]]**2)*self.dx_sq))/(1-self.q[self.r_pos[:, 0], self.r_pos[:, 1]]))*self.r_kaiser_sinc
                
                # apply predictive boundary conditions
                self.pred(self.u1, self.u2, self.dt, self.dx, self.m, self.pred_bc)
                
                # cumulatively calculate the gradient throughout backpropagation by cross-correlating forward and backward wavefields
                if i%self.s_rate==0:
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

                    # initialize corresponding source locations, wavefields and data arrays for modelling
                    self.configure(s_pos[total_batch][k*bs:k*bs+bs], s)
                    
                    # solve the forward problem G(m) = d, storing the forward wavefield
                    self.forward()

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
                    self.adjoint_gradient(self.d.grad)

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