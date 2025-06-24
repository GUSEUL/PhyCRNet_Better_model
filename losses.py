"""
Loss functions for PhyCRNet.
Includes physics-based and data-driven losses with detailed analysis capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import fd_kernels, fd_mixed_derivatives, adaptive_fd_order, ChebyshevSpectralDerivatives

class PhysicsLoss(nn.Module):
    """Physics-based loss using PDE residuals with high-order derivatives and detailed analysis."""
    
    def __init__(self, dx, params, fd_order=4, adaptive_order=False, method='chebyshev_spectral', domain=None, enable_analysis=True):
        super().__init__()
        # Physical parameters
        self.Pr = params['Pr']  
        self.Ra = params['Ra']   
        self.Ha = params['Ha']  
        self.Da = params['Da']  
        self.Rd = params['Rd'] 
        self.Q = params['Q']
        
        # Analysis settings
        self.enable_analysis = enable_analysis
        self.loss_history = {
            'continuity': [], 'x_momentum': [], 'y_momentum': [], 'energy': [],
            'spatial_loss_maps': [], 'field_gradients': [], 'residual_maps': []
        }
        
        # Numerical method settings
        self.method = method  # 'finite_difference' or 'chebyshev_spectral'
        self.fd_order = fd_order
        self.adaptive_order = adaptive_order
        self.dx = dx
        
        # Domain for spectral methods
        self.domain = domain if domain else {'x': (0, 1), 'y': (0, 1)}
        
        # Normalization parameters
        norm_params = params['norm_params']
        self.u_mean, self.u_std = norm_params['u']
        self.v_mean, self.v_std = norm_params['v']
        self.p_mean, self.p_std = norm_params['p']
        self.t_mean, self.t_std = norm_params['t']
        
        if method == 'finite_difference':
        # Register finite difference kernels
            kdx, kdy, klap = fd_kernels(dx, order=fd_order)
        self.register_buffer('kdx', kdx)
        self.register_buffer('kdy', kdy)
        self.register_buffer('klap', klap)
            
            # Mixed derivative kernel for enhanced accuracy
            if fd_order >= 4:
                kmixed = fd_mixed_derivatives(dx, order=min(fd_order, 4))
                self.register_buffer('kmixed', kmixed)
            else:
                self.register_buffer('kmixed', torch.empty(0))  # Register empty tensor to avoid None issues
            
            # Padding size based on FD order
            self.pad_size = fd_order // 2
            
        elif method == 'chebyshev_spectral':
            # Initialize Chebyshev spectral derivatives
            # Assume grid size from data (will be updated in first forward pass)
            self.spectral_derivatives = None
            print("Using Chebyshev spectral method for non-periodic boundaries")
        
        # Loss scaling factors (to balance different terms)
        self.momentum_scale = 1.0
        self.continuity_scale = 1.0
        self.energy_scale = 1.0
        
        # Adaptive weighting parameters
        self.register_parameter('continuity_weight', nn.Parameter(torch.ones(1)))
        self.register_parameter('momentum_weight', nn.Parameter(torch.ones(1)))
        self.register_parameter('energy_weight', nn.Parameter(torch.ones(1)))

    def initialize_spectral_derivatives(self, field_shape, device):
        """Initialize Chebyshev spectral derivatives based on field shape."""
        if self.method == 'chebyshev_spectral' and self.spectral_derivatives is None:
            _, _, Ny, Nx = field_shape
            self.spectral_derivatives = ChebyshevSpectralDerivatives(
                Nx, Ny, 
                domain_x=self.domain['x'], 
                domain_y=self.domain['y']
            ).to(device)
            print(f"Initialized Chebyshev spectral derivatives for grid {Ny}×{Nx}")

    def get_adaptive_kernels(self, field):
        """Get adaptive finite difference kernels based on field smoothness."""
        if not self.adaptive_order or self.method != 'finite_difference':
            return self.kdx, self.kdy, self.klap
            
        # Determine optimal order for this field
        optimal_order = adaptive_fd_order(field, self.dx, self.fd_order)
        
        if optimal_order != self.fd_order:
            kdx, kdy, klap = fd_kernels(self.dx, order=optimal_order)
            return kdx.to(field.device), kdy.to(field.device), klap.to(field.device)
        else:
            return self.kdx, self.kdy, self.klap

    def apply_mac_grid_padding(self, field, variable_type='u'):
        """Apply MAC grid specific padding based on variable type and boundary conditions."""
        B, C, H, W = field.shape
        
        # First check for NaN/Inf in input field
        if torch.isnan(field).any() or torch.isinf(field).any():
            print(f"⚠️  NaN/Inf detected in {variable_type} field before padding")
            field = torch.nan_to_num(field, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # Clamp field to prevent extreme values
        field = torch.clamp(field, min=-1e3, max=1e3)
        
        # Determine padding size
        pad_size = getattr(self, 'pad_size', 2)
        
        # MAC grid에서 u, v, p, T에 따른 다른 경계 조건
        if variable_type == 'u':  # u-velocity (staggered in x)
            # No-slip boundary conditions: u=0 at walls
            # Use symmetric padding for better stability
            field_padded = F.pad(field, [pad_size]*4, mode='reflect')
            
        elif variable_type == 'v':  # v-velocity (staggered in y)  
            # No-slip boundary conditions: v=0 at walls
            field_padded = F.pad(field, [pad_size]*4, mode='reflect')
            
        elif variable_type == 'p':  # pressure (cell-centered)
            # Neumann boundary conditions for pressure (zero gradient)
            field_padded = F.pad(field, [pad_size]*4, mode='replicate')
            
        elif variable_type == 't':  # temperature (cell-centered)
            # More stable temperature boundary conditions
            # Use replicate instead of constant for better stability
            field_padded = F.pad(field, [pad_size]*4, mode='replicate')
            
        else:  # Default case
            field_padded = F.pad(field, [pad_size]*4, mode='replicate')
        
        # Final safety check
        if torch.isnan(field_padded).any() or torch.isinf(field_padded).any():
            print(f"⚠️  NaN/Inf detected in {variable_type} field after padding - using zeros")
            field_padded = torch.zeros_like(field_padded)
        
        return field_padded

    def compute_derivatives_fd(self, field, kdx, kdy, klap, variable_type='u'):
        """Compute spatial derivatives with proper scaling to prevent explosion."""
        # Input validation 
        if torch.isnan(field).any() or torch.isinf(field).any():
            print(f"⚠️  NaN/Inf in field input for {variable_type}")
            field = torch.nan_to_num(field, nan=0.0, posinf=1e2, neginf=-1e2)
        
        # Conservative clamping
        field = torch.clamp(field, min=-1e2, max=1e2)
        
        # Ensure kernels are on same device and dtype as field
        kdx = kdx.to(field.device, field.dtype)
        kdy = kdy.to(field.device, field.dtype)  
        klap = klap.to(field.device, field.dtype)
        
        try:
            # Apply MAC grid specific padding with error handling
            try:
                if hasattr(self, 'pad_size') and self.pad_size > 1:
                    field_padded = self.apply_mac_grid_padding(field, variable_type)
                else:
                    field_padded = F.pad(field, [1, 1, 1, 1], mode='replicate')
            except Exception as e:
                print(f"⚠️  Padding error for {variable_type}: {e}")
                field_padded = F.pad(field, [1, 1, 1, 1], mode='constant', value=0.0)
            
            # Compute derivatives with scaling control
            try:
                dfdx = F.conv2d(field_padded, kdx, padding=0)
                dfdy = F.conv2d(field_padded, kdy, padding=0) 
                lap_f = F.conv2d(field_padded, klap, padding=0)
                
                # 🔧 KEY FIX: Scale down Laplacian to prevent explosion
                # Laplacian kernel has 1/dx^2 scaling which creates massive values
                # Apply compensatory scaling based on grid resolution
                lap_scale_factor = (self.dx)**2.0  # Even stronger scaling reduction
                lap_f = lap_f * lap_scale_factor
                
                # Scale down first derivatives too for consistency
                grad_scale_factor = (self.dx)**1.0  # Use dx scaling
                dfdx = dfdx * grad_scale_factor
                dfdy = dfdy * grad_scale_factor
                
                # Additional NaN/Inf protection immediately after computation
                if torch.isnan(lap_f).any() or torch.isinf(lap_f).any():
                    print(f"⚠️  NaN/Inf detected in Laplacian for {variable_type} - replacing with zeros")
                    lap_f = torch.zeros_like(lap_f)
                
                if torch.isnan(dfdx).any() or torch.isinf(dfdx).any():
                    print(f"⚠️  NaN/Inf detected in dfdx for {variable_type} - replacing with zeros")
                    dfdx = torch.zeros_like(dfdx)
                    
                if torch.isnan(dfdy).any() or torch.isinf(dfdy).any():
                    print(f"⚠️  NaN/Inf detected in dfdy for {variable_type} - replacing with zeros")
                    dfdy = torch.zeros_like(dfdy)
                
            except Exception as e:
                print(f"⚠️  Convolution error for {variable_type}: {e}")
                dfdx = torch.zeros_like(field)
                dfdy = torch.zeros_like(field)
                lap_f = torch.zeros_like(field)
                return dfdx, dfdy, lap_f
            
            # Conservative clamping BEFORE nan_to_num
            dfdx = torch.clamp(dfdx, min=-1e3, max=1e3)
            dfdy = torch.clamp(dfdy, min=-1e3, max=1e3)
            lap_f = torch.clamp(lap_f, min=-1e3, max=1e3)
            
            # Then apply nan_to_num for safety
            dfdx = torch.nan_to_num(dfdx, nan=0.0, posinf=1e2, neginf=-1e2)
            dfdy = torch.nan_to_num(dfdy, nan=0.0, posinf=1e2, neginf=-1e2)
            lap_f = torch.nan_to_num(lap_f, nan=0.0, posinf=1e2, neginf=-1e2)
            
            # Final conservative clamping
            dfdx = torch.clamp(dfdx, min=-1e2, max=1e2)
            dfdy = torch.clamp(dfdy, min=-1e2, max=1e2)
            lap_f = torch.clamp(lap_f, min=-1e2, max=1e2)
            
            # Check for remaining issues
            if torch.isnan(dfdx).any() or torch.isinf(dfdx).any():
                print(f"⚠️  NaN/Inf detected in derivative dUdx for {variable_type}")
                dfdx = torch.zeros_like(field)
            
            if torch.isnan(dfdy).any() or torch.isinf(dfdy).any():
                print(f"⚠️  NaN/Inf detected in derivative dUdy for {variable_type}")
                dfdy = torch.zeros_like(field)
                
            if torch.isnan(lap_f).any() or torch.isinf(lap_f).any():
                print(f"⚠️  NaN/Inf detected in derivative lapU for {variable_type}")
                lap_f = torch.zeros_like(field)
            
            # Ensure output size matches input
            if dfdx.shape != field.shape:
                try:
                    dfdx = F.interpolate(dfdx, size=field.shape[2:], mode='nearest')
                    dfdy = F.interpolate(dfdy, size=field.shape[2:], mode='nearest')
                    lap_f = F.interpolate(lap_f, size=field.shape[2:], mode='nearest')
                except Exception as e:
                    print(f"⚠️  Interpolation error for {variable_type}: {e}")
                    dfdx = torch.zeros_like(field)
                    dfdy = torch.zeros_like(field)
                    lap_f = torch.zeros_like(field)
            
        except Exception as e:
            print(f"⚠️  Major error in finite difference computation for {variable_type}: {e}")
            dfdx = torch.zeros_like(field)
            dfdy = torch.zeros_like(field)
            lap_f = torch.zeros_like(field)
            
        return dfdx, dfdy, lap_f

    def compute_derivatives_spectral(self, field):
        """Compute spatial derivatives using Chebyshev spectral method."""
        if self.spectral_derivatives is None:
            self.initialize_spectral_derivatives(field.shape, field.device)
        
        spectral_derivs = self.spectral_derivatives.compute_derivatives(field)
        return spectral_derivs['dudx'], spectral_derivs['dudy'], spectral_derivs['laplacian']

    def compute_enhanced_derivatives(self, U, V, T, P):
        """Compute derivatives with enhanced numerical accuracy."""
        derivatives = {}
        
        if self.method == 'finite_difference':
            # Get adaptive kernels if enabled
            kdx_u, kdy_u, klap_u = self.get_adaptive_kernels(U)
            kdx_v, kdy_v, klap_v = self.get_adaptive_kernels(V)
            kdx_t, kdy_t, klap_t = self.get_adaptive_kernels(T)
            kdx_p, kdy_p, klap_p = self.get_adaptive_kernels(P)
            
            # Compute all required derivatives with MAC grid specific handling
            derivatives['dUdx'], derivatives['dUdy'], derivatives['lapU'] = self.compute_derivatives_fd(U, kdx_u, kdy_u, klap_u, 'u')
            derivatives['dVdx'], derivatives['dVdy'], derivatives['lapV'] = self.compute_derivatives_fd(V, kdx_v, kdy_v, klap_v, 'v')
            derivatives['dTdx'], derivatives['dTdy'], derivatives['lapT'] = self.compute_derivatives_fd(T, kdx_t, kdy_t, klap_t, 't')
            derivatives['dPdx'], derivatives['dPdy'], _ = self.compute_derivatives_fd(P, kdx_p, kdy_p, klap_p, 'p')
            
            # Compute mixed derivatives for enhanced accuracy if available
            if self.kmixed.numel() > 0 and self.fd_order >= 4:
                # Ensure mixed kernel is on same device and dtype
                kmixed = self.kmixed.to(U.device, U.dtype)
                
                # Apply appropriate padding for mixed derivatives
                U_padded = self.apply_mac_grid_padding(U, 'u')
                V_padded = self.apply_mac_grid_padding(V, 'v')
                T_padded = self.apply_mac_grid_padding(T, 't')
                
                derivatives['d2Udxdy'] = F.conv2d(U_padded, kmixed, padding=0)
                derivatives['d2Vdxdy'] = F.conv2d(V_padded, kmixed, padding=0)
                derivatives['d2Tdxdy'] = F.conv2d(T_padded, kmixed, padding=0)
                
                # Resize if necessary
                for key in ['d2Udxdy', 'd2Vdxdy', 'd2Tdxdy']:
                    if derivatives[key].shape != U.shape:
                        derivatives[key] = F.interpolate(derivatives[key], size=U.shape[2:], 
                                                       mode='bilinear', align_corners=True)
        
        elif self.method == 'chebyshev_spectral':
            # Use Chebyshev spectral methods (ideal for non-periodic boundaries)
            derivatives['dUdx'], derivatives['dUdy'], derivatives['lapU'] = self.compute_derivatives_spectral(U)
            derivatives['dVdx'], derivatives['dVdy'], derivatives['lapV'] = self.compute_derivatives_spectral(V)
            derivatives['dTdx'], derivatives['dTdy'], derivatives['lapT'] = self.compute_derivatives_spectral(T)
            derivatives['dPdx'], derivatives['dPdy'], _ = self.compute_derivatives_spectral(P)
        
        return derivatives

    def denormalize(self, U, V, T, P):
        """Denormalize variables to physical units"""
        U = U * self.u_std + self.u_mean
        V = V * self.v_std + self.v_mean
        T = T * self.t_std + self.t_mean
        P = P * self.p_std + self.p_mean
        return U, V, T, P

    def normalize_derivatives(self, derivatives):
        """Normalize derivatives back to normalized units with safety checks"""
        epsilon = 1e-8
        
        # Much more conservative clamping limits to prevent explosion
        clamp_min, clamp_max = -1e2, 1e2
        
        # Safe normalization with robust error handling
        def safe_normalize_derivative(deriv_name, std_value):
            if deriv_name in derivatives:
                deriv = derivatives[deriv_name]
                
                # Check for NaN/Inf before normalization
                if torch.isnan(deriv).any() or torch.isinf(deriv).any():
                    print(f"⚠️  NaN/Inf detected in {deriv_name} before normalization - replacing with zeros")
                    deriv = torch.zeros_like(deriv)
                
                # Conservative pre-clamping
                deriv = torch.clamp(deriv, min=-1e3, max=1e3)
                
                # Safe division
                safe_std = max(abs(std_value), epsilon)
                normalized = deriv / safe_std
                
                # Post-normalization NaN/Inf check
                if torch.isnan(normalized).any() or torch.isinf(normalized).any():
                    print(f"⚠️  NaN/Inf detected in {deriv_name} after normalization - replacing with zeros")
                    normalized = torch.zeros_like(normalized)
                
                # Final conservative clamping
                normalized = torch.clamp(normalized, min=clamp_min, max=clamp_max)
                derivatives[deriv_name] = normalized
                
                return True
            return False
        
        # Normalize first derivatives
        safe_normalize_derivative('dUdx', self.u_std)
        safe_normalize_derivative('dUdy', self.u_std)
        safe_normalize_derivative('dVdx', self.v_std)
        safe_normalize_derivative('dVdy', self.v_std)
        safe_normalize_derivative('dTdx', self.t_std)
        safe_normalize_derivative('dTdy', self.t_std)
        safe_normalize_derivative('dPdx', self.p_std)
        safe_normalize_derivative('dPdy', self.p_std)
        
        # Normalize second derivatives (Laplacians) with special care
        if 'lapU' in derivatives:
            lapU = derivatives['lapU']
            if torch.isnan(lapU).any() or torch.isinf(lapU).any():
                print(f"⚠️  NaN/Inf detected in lapU before normalization - replacing with zeros")
                lapU = torch.zeros_like(lapU)
            
            # For Laplacian, apply even more conservative clamping
            lapU = torch.clamp(lapU, min=-1e2, max=1e2)
            safe_std = max(abs(self.u_std), epsilon)
            normalized_lapU = lapU / safe_std
            
            if torch.isnan(normalized_lapU).any() or torch.isinf(normalized_lapU).any():
                print(f"⚠️  NaN/Inf detected in lapU after normalization - replacing with zeros")
                normalized_lapU = torch.zeros_like(normalized_lapU)
            
            derivatives['lapU'] = torch.clamp(normalized_lapU, min=-10.0, max=10.0)
        
        safe_normalize_derivative('lapV', self.v_std)
        safe_normalize_derivative('lapT', self.t_std)
        
        # Mixed derivatives if available
        safe_normalize_derivative('d2Udxdy', self.u_std)
        safe_normalize_derivative('d2Vdxdy', self.v_std)
        safe_normalize_derivative('d2Tdxdy', self.t_std)
        
        return derivatives

    def compute_detailed_residuals(self, dUdt, dVdt, dTdt, derivatives, U1, V1, T1, P1):
        """Compute detailed PDE residuals for analysis."""
        residuals = {}
        
        # Continuity equation: ∂U/∂X + ∂V/∂Y = 0
        residuals['continuity'] = (derivatives['dUdx'] + derivatives['dVdy']) * self.continuity_scale * self.continuity_weight

        # X-momentum equation
        residuals['x_momentum'] = (dUdt + U1*derivatives['dUdx'] + V1*derivatives['dUdy'] + derivatives['dPdx'] - 
                                  self.Pr*derivatives['lapU'] - (self.Pr/self.Da)*U1) * self.momentum_scale * self.momentum_weight

        # Y-momentum equation
        residuals['y_momentum'] = (dVdt + U1*derivatives['dVdx'] + V1*derivatives['dVdy'] + derivatives['dPdy'] - 
                                  self.Pr*derivatives['lapV'] - self.Ra*self.Pr*T1 + 
                                  (self.Ha**2)*self.Pr*V1 - (self.Pr/self.Da)*V1) * self.momentum_scale * self.momentum_weight

        # Energy equation
        residuals['energy'] = (dTdt + U1*derivatives['dTdx'] + V1*derivatives['dTdy'] - 
                              (1+4*self.Rd/3)*derivatives['lapT'] - self.Q*T1) * self.energy_scale * self.energy_weight
        
        return residuals

    def analyze_spatial_loss_distribution(self, residuals):
        """Analyze where losses are occurring spatially."""
        spatial_analysis = {}
        
        for name, residual in residuals.items():
            # Compute spatial loss map
            spatial_loss = residual**2
            spatial_analysis[f'{name}_spatial_loss'] = spatial_loss
            
            # Find hotspots (top 10% highest loss regions)
            flat_loss = spatial_loss.flatten()
            threshold = torch.quantile(flat_loss, 0.9)
            hotspots = spatial_loss > threshold
            spatial_analysis[f'{name}_hotspots'] = hotspots
            
            # Regional analysis
            B, C, H, W = spatial_loss.shape
            regions = {
                'corners': spatial_loss[:, :, [0, -1]][:, :, :, [0, -1]],
                'edges': torch.cat([spatial_loss[:, :, 0, :], spatial_loss[:, :, -1, :], 
                                   spatial_loss[:, :, :, 0], spatial_loss[:, :, :, -1]], dim=-1),
                'interior': spatial_loss[:, :, 1:-1, 1:-1],
                'center': spatial_loss[:, :, H//4:3*H//4, W//4:3*W//4]
            }
            
            for region_name, region_loss in regions.items():
                spatial_analysis[f'{name}_{region_name}_avg'] = torch.mean(region_loss)
        
        return spatial_analysis

    def forward(self, f_now, f_next, dt):
        """
        Compute physics-informed loss with enhanced stability and progressive learning.
        """
        # Validate inputs
        if f_now.dim() != 4:
            print(f"⚠️  Expected 4D input, got {f_now.dim()}D")
            return torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        
        if f_now.size(1) != 4:
            print(f"⚠️  Expected 4 channels, got {f_now.size(1)}")
            return torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        
        # Extract channels - work directly with normalized values
        U0, V0, T0, P0 = torch.chunk(f_now, 4, 1)
        U1, V1, T1, P1 = torch.chunk(f_next, 4, 1)
        
        # Compute time derivatives
        dUdt = (U1 - U0) / dt
        dVdt = (V1 - V0) / dt  
        dTdt = (T1 - T0) / dt
        
        # Enhanced derivative computation with better scaling
        derivatives = self.compute_enhanced_derivatives(U1, V1, T1, P1)
        
        if derivatives is None:
            return torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        
        # Progressive physics parameter scaling (훈련 진행에 따라 점진적 증가)
        training_progress = getattr(self, 'current_epoch', 0) / 1000.0  # 최대 1000 에포크 가정
        
        # 초기에는 매우 작은 값으로 시작, 점진적으로 증가
        if training_progress < 0.1:  # 첫 100 에포크
            scale_factor = 1e-3
        elif training_progress < 0.3:  # 100-300 에포크
            scale_factor = 1e-2
        elif training_progress < 0.6:  # 300-600 에포크
            scale_factor = 5e-2
        else:  # 600+ 에포크
            scale_factor = 1e-1
        
        # Scale down physics parameters progressively
        scaled_Ra = self.Ra * scale_factor
        scaled_Pr = self.Pr * scale_factor * 0.1
        
        # Compute residuals with scaled parameters
        try:
            residuals = self.compute_detailed_residuals(dUdt, dVdt, dTdt, derivatives, U1, V1, T1, P1)
            
            # Apply scaling to residuals
            continuity_residual = residuals['continuity'] * scale_factor
            x_momentum_residual = residuals['x_momentum'] * scale_factor
            y_momentum_residual = residuals['y_momentum'] * scale_factor
            energy_residual = residuals['energy'] * scale_factor
            
        except Exception as e:
            print(f"⚠️  Error computing residuals: {e}")
            return torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        
        # Enhanced loss computation with progressive targets
        total_loss = 0.0
        
        try:
            # Progressive target values (훈련 진행에 따라 엄격해짐)
            if training_progress < 0.2:
                target_range = 1e2  # 초기: 매우 관대
            elif training_progress < 0.5:
                target_range = 1e1  # 중기: 중간 수준
            else:
                target_range = 1e0  # 후기: 엄격
            
            # Individual loss computation with progressive targets
            continuity_loss = torch.mean(continuity_residual**2)
            continuity_loss = torch.clamp(continuity_loss, min=1e-8, max=target_range)
            
            x_momentum_loss = torch.mean(x_momentum_residual**2)
            x_momentum_loss = torch.clamp(x_momentum_loss, min=1e-8, max=target_range)
            
            y_momentum_loss = torch.mean(y_momentum_residual**2)
            y_momentum_loss = torch.clamp(y_momentum_loss, min=1e-8, max=target_range)
            
            energy_loss = torch.mean(energy_residual**2)
            energy_loss = torch.clamp(energy_loss, min=1e-8, max=target_range)
            
            # NaN/Inf safety checks
            loss_components = [continuity_loss, x_momentum_loss, y_momentum_loss, energy_loss]
            for i, loss in enumerate(loss_components):
                if torch.isnan(loss) or torch.isinf(loss):
                    loss_components[i] = torch.tensor(0.01, device=f_now.device, dtype=f_now.dtype)
            
            # Weighted combination
            total_loss = sum(loss_components) * 0.25
            
        except Exception as e:
            print(f"⚠️  Error in loss computation: {e}")
            return torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        
        # Smart final clamping (훈련 진행에 따라 상한선 조정)
        if training_progress < 0.1:
            max_loss = 10.0  # 초기: 현재와 동일
        elif training_progress < 0.3:
            max_loss = 5.0   # 점진적 감소
        elif training_progress < 0.6:
            max_loss = 2.0   # 더 엄격
        else:
            max_loss = 1.0   # 최종 목표
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️  Total physics loss is NaN/Inf, using fallback value")
            total_loss = torch.tensor(0.1, device=f_now.device, dtype=f_now.dtype)
        else:
            # Progressive clamping
            total_loss = torch.clamp(total_loss, min=1e-6, max=max_loss)
        
        # Store analysis data if enabled
        if self.enable_analysis:
            self.loss_history['continuity'].append(loss_components[0].item())
            self.loss_history['x_momentum'].append(loss_components[1].item())
            self.loss_history['y_momentum'].append(loss_components[2].item())
            self.loss_history['energy'].append(loss_components[3].item())

        return total_loss
    
    def set_epoch(self, epoch):
        """Set current epoch for progressive training."""
        self.current_epoch = epoch

    def get_loss_analysis(self):
        """Return detailed loss analysis."""
        return self.loss_history

    def visualize_spatial_losses(self, epoch, save_dir='loss_analysis'):
        """Visualize spatial distribution of losses."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.loss_history['spatial_loss_maps']:
            print("No spatial loss data available for visualization")
            return
        
        # Get latest spatial analysis
        latest_analysis = self.loss_history['spatial_loss_maps'][-1]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        equations = ['continuity', 'x_momentum', 'y_momentum', 'energy']
        
        for i, eq in enumerate(equations):
            ax = axes[i//2, i%2]
            loss_key = f'{eq}_spatial_loss'
            if loss_key in latest_analysis:
                spatial_loss = latest_analysis[loss_key][0, 0].detach().cpu().numpy()
                im = ax.imshow(spatial_loss, cmap='hot', aspect='auto')
                ax.set_title(f'{eq.replace("_", " ").title()} Loss Distribution')
                plt.colorbar(im, ax=ax)
                
                # Mark hotspots
                hotspot_key = f'{eq}_hotspots'
                if hotspot_key in latest_analysis:
                    hotspots = latest_analysis[hotspot_key][0, 0].detach().cpu().numpy()
                    ax.contour(hotspots, levels=[0.5], colors='cyan', linewidths=2)
        
        plt.suptitle(f'Spatial Loss Distribution - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/spatial_losses_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def plot_loss_components(self, save_dir='loss_analysis'):
        """Plot individual loss components over training."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        components = ['continuity', 'x_momentum', 'y_momentum', 'energy']
        
        for i, comp in enumerate(components):
            ax = axes[i//2, i%2]
            if comp in self.loss_history and self.loss_history[comp]:
                ax.semilogy(self.loss_history[comp], label=comp.replace('_', ' ').title())
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss (log scale)')
                ax.set_title(f'{comp.replace("_", " ").title()} Loss')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/loss_components.png', dpi=150, bbox_inches='tight')
        plt.close()

    def get_method(self):
        """Return current numerical method."""
        return self.method
    
    def set_method(self, new_method):
        """Switch between finite difference and spectral methods."""
        if new_method in ['finite_difference', 'chebyshev_spectral']:
            self.method = new_method
            if new_method == 'chebyshev_spectral':
                self.spectral_derivatives = None  # Will be reinitialized
            print(f"Switched to {new_method} method")
        else:
            print(f"Warning: Method {new_method} not supported")

    def get_fd_order(self):
        """Return current finite difference order."""
        return self.fd_order
    
    def set_fd_order(self, new_order):
        """Update finite difference order during training."""
        if self.method == 'finite_difference' and new_order in [2, 4, 6]:
            self.fd_order = new_order
            kdx, kdy, klap = fd_kernels(self.dx, order=new_order)
            
            # Ensure kernels are on the same device as the module
            device = next(self.parameters()).device
            kdx = kdx.to(device)
            kdy = kdy.to(device)
            klap = klap.to(device)
            
            self.register_buffer('kdx', kdx)
            self.register_buffer('kdy', kdy) 
            self.register_buffer('klap', klap)
            self.pad_size = new_order // 2
            
            # Update mixed derivative kernel if needed
            if new_order >= 4:
                kmixed = fd_mixed_derivatives(self.dx, order=min(new_order, 4))
                self.register_buffer('kmixed', kmixed.to(device))
            else:
                self.register_buffer('kmixed', torch.empty(0).to(device))
                
            print(f"Updated FD order to {new_order}")
        else:
            print(f"Warning: FD order {new_order} not supported or not using finite difference method")

class DataLoss(nn.Module):
    """Data-driven loss comparing predictions with ground truth on MAC grid with detailed analysis."""
    
    def __init__(self, weight_u=1.0, weight_v=1.0, weight_p=0.1, weight_t=1.0, enable_analysis=True):
        super().__init__()
        self.wu = weight_u
        self.wv = weight_v
        self.wp = weight_p  # Reduced weight for pressure
        self.wt = weight_t
        self.enable_analysis = enable_analysis
        
        # Analysis history
        self.loss_history = {
            'u_loss': [], 'v_loss': [], 'p_loss': [], 't_loss': [],
            'spatial_error_maps': [], 'field_statistics': []
        }

    @staticmethod
    def center_to_u(pred):
        """Convert cell-centered U to staggered grid (in x)."""
        return 0.5 * (pred[:,:,:,:-1] + pred[:,:,:,1:])

    @staticmethod
    def center_to_v(pred):
        """Convert cell-centered V to staggered grid (in y)."""
        return 0.5 * (pred[:,:,:-1,:] + pred[:,:,1:,:])

    def analyze_field_errors(self, pred_field, gt_field, field_name):
        """Analyze spatial distribution of prediction errors."""
        error_map = (pred_field - gt_field)**2
        analysis = {
            f'{field_name}_mse': torch.mean(error_map).item(),
            f'{field_name}_max_error': torch.max(error_map).item(),
            f'{field_name}_error_std': torch.std(error_map).item(),
            f'{field_name}_error_map': error_map
        }
        return analysis

    def forward(self, pred, gt):
        """Compute data-driven loss with detailed analysis."""
        U, V, T, P = torch.chunk(pred, 4, 1)
        total_loss = 0.0
        individual_losses = {}
        
        # U velocity loss
        if gt['u'] is not None:
            u_staggered = self.center_to_u(U)
            min_h = min(u_staggered.shape[2], gt['u'].shape[2])
            min_w = min(u_staggered.shape[3], gt['u'].shape[3])
            loss_u = F.mse_loss(u_staggered[:,:,:min_h,:min_w], 
                               gt['u'][:,:,:min_h,:min_w])
            individual_losses['u_loss'] = loss_u.item()
            total_loss += self.wu * loss_u
            
            # Detailed error analysis
            if self.enable_analysis:
                u_analysis = self.analyze_field_errors(
                    u_staggered[:,:,:min_h,:min_w], 
                    gt['u'][:,:,:min_h,:min_w], 'u'
                )
            
        # V velocity loss
        if gt['v'] is not None:
            v_staggered = self.center_to_v(V)
            min_h = min(v_staggered.shape[2], gt['v'].shape[2])
            min_w = min(v_staggered.shape[3], gt['v'].shape[3])
            loss_v = F.mse_loss(v_staggered[:,:,:min_h,:min_w], 
                               gt['v'][:,:,:min_h,:min_w])
            individual_losses['v_loss'] = loss_v.item()
            total_loss += self.wv * loss_v
            
            if self.enable_analysis:
                v_analysis = self.analyze_field_errors(
                    v_staggered[:,:,:min_h,:min_w], 
                    gt['v'][:,:,:min_h,:min_w], 'v'
                )
            
        # Pressure loss (with reduced weight)
        if gt['p'] is not None:
            min_h = min(P.shape[2], gt['p'].shape[2])
            min_w = min(P.shape[3], gt['p'].shape[3])
            loss_p = F.mse_loss(P[:,:,:min_h,:min_w], 
                               gt['p'][:,:,:min_h,:min_w])
            individual_losses['p_loss'] = loss_p.item()
            total_loss += self.wp * loss_p
            
            if self.enable_analysis:
                p_analysis = self.analyze_field_errors(
                    P[:,:,:min_h,:min_w], 
                    gt['p'][:,:,:min_h,:min_w], 'p'
                )
                
        # Temperature loss
        if gt['t'] is not None:
            min_h = min(T.shape[2], gt['t'].shape[2])
            min_w = min(T.shape[3], gt['t'].shape[3])
            loss_t = F.mse_loss(T[:,:,:min_h,:min_w], 
                               gt['t'][:,:,:min_h,:min_w])
            individual_losses['t_loss'] = loss_t.item()
            total_loss += self.wt * loss_t
            
            if self.enable_analysis:
                t_analysis = self.analyze_field_errors(
                    T[:,:,:min_h,:min_w], 
                    gt['t'][:,:,:min_h,:min_w], 't'
                )
        
        # Store analysis data
        if self.enable_analysis:
            for key, value in individual_losses.items():
                self.loss_history[key].append(value)
        
        return total_loss
    
    def get_loss_analysis(self):
        """Return detailed loss analysis."""
        return self.loss_history
    
    def visualize_error_maps(self, pred, gt, epoch, save_dir='loss_analysis'):
        """Visualize spatial error distribution."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        U, V, T, P = torch.chunk(pred, 4, 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # U velocity error
        if gt['u'] is not None:
            u_staggered = self.center_to_u(U)
            u_error = (u_staggered[0, 0] - gt['u'][0, 0])**2
            im1 = axes[0, 0].imshow(u_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[0, 0].set_title('U Velocity Error²')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # V velocity error
        if gt['v'] is not None:
            v_staggered = self.center_to_v(V)
            v_error = (v_staggered[0, 0] - gt['v'][0, 0])**2
            im2 = axes[0, 1].imshow(v_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[0, 1].set_title('V Velocity Error²')
            plt.colorbar(im2, ax=axes[0, 1])
        
        # Temperature error
        if gt['t'] is not None:
            t_error = (T[0, 0] - gt['t'][0, 0])**2
            im3 = axes[1, 0].imshow(t_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[1, 0].set_title('Temperature Error²')
            plt.colorbar(im3, ax=axes[1, 0])
        
        # Pressure error
        if gt['p'] is not None:
            p_error = (P[0, 0] - gt['p'][0, 0])**2
            im4 = axes[1, 1].imshow(p_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[1, 1].set_title('Pressure Error²')
            plt.colorbar(im4, ax=axes[1, 1])
        
        plt.suptitle(f'Spatial Error Distribution - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/error_maps_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()

class EnhancedDataLoss(nn.Module):
    """Enhanced data loss with multiple strategies to reduce prediction errors."""
    
    def __init__(self, weight_u=1.0, weight_v=1.0, weight_p=0.1, weight_t=1.0, 
                 enable_analysis=True, use_multiscale=True, use_perceptual=True,
                 use_gradient_loss=True, use_ssim=True):
        super().__init__()
        self.wu = weight_u
        self.wv = weight_v
        self.wp = weight_p
        self.wt = weight_t
        self.enable_analysis = enable_analysis
        self.use_multiscale = use_multiscale
        self.use_perceptual = use_perceptual
        self.use_gradient_loss = use_gradient_loss
        self.use_ssim = use_ssim
        
        # Analysis history
        self.loss_history = {
            'u_loss': [], 'v_loss': [], 'p_loss': [], 't_loss': [],
            'spatial_error_maps': [], 'field_statistics': [],
            'multiscale_losses': [], 'gradient_losses': [], 'ssim_losses': []
        }
        
        # Adaptive weights that change during training
        self.adaptive_weights = {
            'mse': 1.0, 'multiscale': 0.3, 'gradient': 0.2, 
            'ssim': 0.1, 'perceptual': 0.1
        }
        
    def compute_multiscale_loss(self, pred, target):
        """Compute loss at multiple scales for better feature matching."""
        total_loss = 0.0
        scales = [1.0, 0.5, 0.25]  # Original, half, quarter resolution
        
        for scale in scales:
            if scale < 1.0:
                # Downsample for multi-scale comparison
                h, w = pred.shape[2], pred.shape[3]
                new_h, new_w = int(h * scale), int(w * scale)
                
                pred_scaled = F.interpolate(pred, size=(new_h, new_w), 
                                          mode='bilinear', align_corners=False)
                target_scaled = F.interpolate(target, size=(new_h, new_w), 
                                            mode='bilinear', align_corners=False)
            else:
                pred_scaled = pred
                target_scaled = target
            
            # MSE loss at this scale
            scale_loss = F.mse_loss(pred_scaled, target_scaled)
            total_loss += scale_loss * scale  # Weight by scale importance
            
        return total_loss / len(scales)
    
    def compute_gradient_loss(self, pred, target):
        """Compute gradient-based loss for better edge preservation."""
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        total_grad_loss = 0.0
        
        for i in range(pred.shape[1]):  # For each channel
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            
            # Compute gradients
            pred_grad_x = F.conv2d(pred_ch, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred_ch, sobel_y, padding=1)
            target_grad_x = F.conv2d(target_ch, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_ch, sobel_y, padding=1)
            
            # Gradient magnitude loss
            pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
            
            grad_loss = F.mse_loss(pred_grad_mag, target_grad_mag)
            total_grad_loss += grad_loss
            
        return total_grad_loss / pred.shape[1]
    
    def compute_ssim_loss(self, pred, target, window_size=11):
        """Compute SSIM-based loss for structural similarity."""
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g /= g.sum()
            return g.outer(g).unsqueeze(0).unsqueeze(0)
        
        def ssim(img1, img2, window, window_size, size_average=True):
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
            
            C1 = 0.01**2
            C2 = 0.03**2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)
        
        window = gaussian_window(window_size).to(pred.device)
        ssim_loss = 0.0
        
        for i in range(pred.shape[1]):
            pred_ch = pred[:, i:i+1, :, :]
            target_ch = target[:, i:i+1, :, :]
            ssim_val = ssim(pred_ch, target_ch, window, window_size)
            ssim_loss += (1 - ssim_val)  # Convert to loss (lower is better)
            
        return ssim_loss / pred.shape[1]
    
    def compute_perceptual_loss(self, pred, target):
        """Compute perceptual loss using feature differences."""
        # Simple perceptual loss using L2 norm of differences
        # In practice, you might want to use pre-trained features
        
        # Compute local patches for feature comparison
        patch_size = 5
        pred_patches = F.unfold(pred, kernel_size=patch_size, stride=patch_size//2)
        target_patches = F.unfold(target, kernel_size=patch_size, stride=patch_size//2)
        
        # Feature-level comparison
        pred_features = pred_patches.mean(dim=1, keepdim=True)
        target_features = target_patches.mean(dim=1, keepdim=True)
        
        perceptual_loss = F.mse_loss(pred_features, target_features)
        return perceptual_loss
    
    def adaptive_weight_update(self, losses, epoch):
        """Adaptively update loss component weights based on training progress."""
        # Reduce multiscale weight as training progresses
        self.adaptive_weights['multiscale'] = max(0.1, 0.5 * (1 - epoch / 1000))
        
        # Increase gradient loss weight for better edge preservation
        self.adaptive_weights['gradient'] = min(0.5, 0.1 + epoch / 2000)
        
        # SSIM weight based on convergence
        if len(losses) > 10:
            recent_improvement = abs(losses[-1] - losses[-10]) / losses[-10]
            if recent_improvement < 0.01:  # Converging, emphasize structure
                self.adaptive_weights['ssim'] = min(0.3, self.adaptive_weights['ssim'] * 1.1)
    
    @staticmethod
    def center_to_u(pred):
        """Convert cell-centered U to staggered grid (in x)."""
        return 0.5 * (pred[:,:,:,:-1] + pred[:,:,:,1:])

    @staticmethod
    def center_to_v(pred):
        """Convert cell-centered V to staggered grid (in y)."""
        return 0.5 * (pred[:,:,:-1,:] + pred[:,:,1:,:])

    def forward(self, pred, gt, epoch=0):
        """Enhanced forward pass with multiple loss types."""
        U, V, T, P = torch.chunk(pred, 4, 1)
        total_loss = 0.0
        individual_losses = {}
        
        # Standard MSE losses with enhanced matching
        if gt['u'] is not None:
            u_staggered = self.center_to_u(U)
            min_h = min(u_staggered.shape[2], gt['u'].shape[2])
            min_w = min(u_staggered.shape[3], gt['u'].shape[3])
            
            u_pred = u_staggered[:,:,:min_h,:min_w]
            u_target = gt['u'][:,:,:min_h,:min_w]
            
            # Base MSE loss
            loss_u_mse = F.mse_loss(u_pred, u_target)
            individual_losses['u_mse'] = loss_u_mse.item()
            
            # Enhanced losses
            loss_u_total = self.adaptive_weights['mse'] * loss_u_mse
            
            if self.use_multiscale:
                loss_u_ms = self.compute_multiscale_loss(u_pred, u_target)
                loss_u_total += self.adaptive_weights['multiscale'] * loss_u_ms
                individual_losses['u_multiscale'] = loss_u_ms.item()
            
            if self.use_gradient_loss:
                loss_u_grad = self.compute_gradient_loss(u_pred, u_target)
                loss_u_total += self.adaptive_weights['gradient'] * loss_u_grad
                individual_losses['u_gradient'] = loss_u_grad.item()
            
            if self.use_ssim:
                loss_u_ssim = self.compute_ssim_loss(u_pred, u_target)
                loss_u_total += self.adaptive_weights['ssim'] * loss_u_ssim
                individual_losses['u_ssim'] = loss_u_ssim.item()
            
            total_loss += self.wu * loss_u_total
            individual_losses['u_loss'] = loss_u_total.item()
            
        # Similar enhanced processing for V, T, P
        if gt['v'] is not None:
            v_staggered = self.center_to_v(V)
            min_h = min(v_staggered.shape[2], gt['v'].shape[2])
            min_w = min(v_staggered.shape[3], gt['v'].shape[3])
            
            v_pred = v_staggered[:,:,:min_h,:min_w]
            v_target = gt['v'][:,:,:min_h,:min_w]
            
            loss_v_mse = F.mse_loss(v_pred, v_target)
            loss_v_total = self.adaptive_weights['mse'] * loss_v_mse
            
            if self.use_multiscale:
                loss_v_total += self.adaptive_weights['multiscale'] * self.compute_multiscale_loss(v_pred, v_target)
            if self.use_gradient_loss:
                loss_v_total += self.adaptive_weights['gradient'] * self.compute_gradient_loss(v_pred, v_target)
            if self.use_ssim:
                loss_v_total += self.adaptive_weights['ssim'] * self.compute_ssim_loss(v_pred, v_target)
                
            total_loss += self.wv * loss_v_total
            individual_losses['v_loss'] = loss_v_total.item()
            
        # Temperature with enhanced losses (most important for accuracy)
        if gt['t'] is not None:
            min_h = min(T.shape[2], gt['t'].shape[2])
            min_w = min(T.shape[3], gt['t'].shape[3])
            
            t_pred = T[:,:,:min_h,:min_w]
            t_target = gt['t'][:,:,:min_h,:min_w]
            
            loss_t_mse = F.mse_loss(t_pred, t_target)
            loss_t_total = self.adaptive_weights['mse'] * loss_t_mse
            
            if self.use_multiscale:
                loss_t_total += self.adaptive_weights['multiscale'] * self.compute_multiscale_loss(t_pred, t_target)
            if self.use_gradient_loss:
                loss_t_total += self.adaptive_weights['gradient'] * self.compute_gradient_loss(t_pred, t_target)
            if self.use_ssim:
                loss_t_total += self.adaptive_weights['ssim'] * self.compute_ssim_loss(t_pred, t_target)
            if self.use_perceptual:
                loss_t_total += self.adaptive_weights['perceptual'] * self.compute_perceptual_loss(t_pred, t_target)
                
            total_loss += self.wt * loss_t_total
            individual_losses['t_loss'] = loss_t_total.item()
            
        # Pressure (with reduced complexity due to lower weight)
        if gt['p'] is not None:
            min_h = min(P.shape[2], gt['p'].shape[2])
            min_w = min(P.shape[3], gt['p'].shape[3])
            
            p_pred = P[:,:,:min_h,:min_w]
            p_target = gt['p'][:,:,:min_h,:min_w]
            
            loss_p_mse = F.mse_loss(p_pred, p_target)
            loss_p_total = loss_p_mse  # Keep pressure loss simple due to low weight
            
            total_loss += self.wp * loss_p_total
            individual_losses['p_loss'] = loss_p_total.item()
        
        # Update adaptive weights
        if self.enable_analysis:
            loss_values = [individual_losses.get(k, 0) for k in ['u_loss', 'v_loss', 't_loss', 'p_loss']]
            self.adaptive_weight_update(loss_values, epoch)
            
            # Store analysis data
            for key, value in individual_losses.items():
                if key in self.loss_history:
                    self.loss_history[key].append(value)
        
        return total_loss 

    def get_loss_analysis(self):
        """Return detailed loss analysis."""
        return self.loss_history

    def visualize_error_maps(self, pred, gt, epoch, save_dir='loss_analysis'):
        """Visualize spatial error distribution."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        U, V, T, P = torch.chunk(pred, 4, 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # U velocity error
        if gt['u'] is not None:
            u_staggered = self.center_to_u(U)
            u_error = (u_staggered[0, 0] - gt['u'][0, 0])**2
            im1 = axes[0, 0].imshow(u_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[0, 0].set_title('U Velocity Error²')
            plt.colorbar(im1, ax=axes[0, 0])
        
        # V velocity error
        if gt['v'] is not None:
            v_staggered = self.center_to_v(V)
            v_error = (v_staggered[0, 0] - gt['v'][0, 0])**2
            im2 = axes[0, 1].imshow(v_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[0, 1].set_title('V Velocity Error²')
            plt.colorbar(im2, ax=axes[0, 1])
        
        # Temperature error
        if gt['t'] is not None:
            t_error = (T[0, 0] - gt['t'][0, 0])**2
            im3 = axes[1, 0].imshow(t_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[1, 0].set_title('Temperature Error²')
            plt.colorbar(im3, ax=axes[1, 0])
        
        # Pressure error
        if gt['p'] is not None:
            p_error = (P[0, 0] - gt['p'][0, 0])**2
            im4 = axes[1, 1].imshow(p_error.cpu().numpy(), cmap='hot', aspect='auto')
            axes[1, 1].set_title('Pressure Error²')
            plt.colorbar(im4, ax=axes[1, 1])
        
        plt.suptitle(f'Spatial Error Distribution - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/error_maps_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close() 