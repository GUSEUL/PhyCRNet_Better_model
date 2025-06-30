"""
Accurate Physics Loss Implementation
Based on the provided PDE system for magnetohydrodynamic natural convection in porous media.

PDE System:
1. Continuity: ∂U/∂X + ∂V/∂Y = 0
2. X-momentum: ∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + Pr[∂²U/∂X² + ∂²U/∂Y²] - (Pr/Da)U
3. Y-momentum: ∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + Pr[∂²V/∂X² + ∂²V/∂Y²] + Ra·Pr·θ - Ha²·Pr·V - (Pr/Da)V
4. Energy: ∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (1 + 4Rd/3)[∂²θ/∂X² + ∂²θ/∂Y²] + Q·θ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AccuratePhysicsLoss(nn.Module):
    """
    Physics-informed loss based on the exact PDE system.
    Implements all terms from the governing equations.
    """
    
    def __init__(self, params, dt=0.0001, dx=1.0, dy=1.0, enable_analysis=True):
        super().__init__()
        
        # Physical parameters (use actual values)
        self.Pr = params['Pr']  # Prandtl number
        self.Ra = params['Ra']  # Rayleigh number  
        # self.Ha = params.get('Ha', 0.0)  # Hartmann number (magnetic field)
        # self.Da = params.get('Da', 1e6)  # Darcy number (porous media)
        # self.Rd = params.get('Rd', 1.8)  # Radiation parameter
        # self.Q = params.get('Q', 0.0)   # Heat source parameter
        self.Ha = params['Ha']  # Hartmann number (magnetic field)
        self.Da = params['Da'] # Darcy number (porous media)
        self.Rd = params['Rd']  # Radiation parameter
        self.Q = params['Q']   # Heat source parameter
        
        # Grid parameters
        self.dt = dt
        self.dx = dx
        self.dy = dy
        
        # Loss weighting
        self.w_continuity = 1.0
        self.w_momentum_x = 1.0
        self.w_momentum_y = 1.0
        self.w_energy = 1.0
        
        # Analysis
        self.enable_analysis = enable_analysis
        self.loss_history = {
            'continuity': [],
            'momentum_x': [],
            'momentum_y': [],
            'energy': [],
            'total': []
        }
        
        # Progressive training
        self.current_epoch = 0
        self.max_residual_scale = 1.0
        
        print(f"Accurate Physics Loss initialized:")
        print(f"   Pr={self.Pr:.3f}, Ra={self.Ra:.1e}, Ha={self.Ha:.1f}")
        print(f"   Da={self.Da:.1e}, Rd={self.Rd:.2f}, Q={self.Q:.3f}")
    
    def compute_derivatives(self, field):
        """
        Compute spatial derivatives using central differences.
        
        Args:
            field: [B, C, H, W] tensor
            
        Returns:
            dict with 'dx', 'dy', 'dxx', 'dyy', 'dxy' derivatives
        """
        # First derivatives (central difference)
        dfdx = torch.gradient(field, dim=-1)[0] / self.dx  # ∂f/∂x
        dfdy = torch.gradient(field, dim=-2)[0] / self.dy  # ∂f/∂y
        
        # Second derivatives
        d2fdx2 = torch.gradient(dfdx, dim=-1)[0] / self.dx  # ∂²f/∂x²
        d2fdy2 = torch.gradient(dfdy, dim=-2)[0] / self.dy  # ∂²f/∂y²
        
        # Mixed derivative (if needed)
        d2fdxy = torch.gradient(dfdy, dim=-1)[0] / self.dx  # ∂²f/∂x∂y
        
        return {
            'dx': dfdx,
            'dy': dfdy,
            'dxx': d2fdx2,
            'dyy': d2fdy2,
            'dxy': d2fdxy
        }
    
    def compute_time_derivative(self, f_now, f_next):
        """
        Compute time derivative using finite difference.
        
        Args:
            f_now: field at time t
            f_next: field at time t+dt
            
        Returns:
            ∂f/∂t
        """
        return (f_next - f_now) / self.dt
    
    def continuity_residual(self, U, V):
        """
        Continuity equation: ∂U/∂X + ∂V/∂Y = 0
        
        Args:
            U, V: velocity components [B, 1, H, W]
            
        Returns:
            residual tensor
        """
        U_derivs = self.compute_derivatives(U)
        V_derivs = self.compute_derivatives(V)
        
        # ∂U/∂X + ∂V/∂Y
        residual = U_derivs['dx'] + V_derivs['dy']
        
        return residual
    
    def momentum_x_residual(self, U_now, V_now, P_next, U_next):
        """
        X-momentum equation:
        ∂U/∂t + U∂U/∂X + V∂U/∂Y = -∂P/∂X + Pr[∂²U/∂X² + ∂²U/∂Y²] - (Pr/Da)U
        
        Args:
            U_now, V_now: velocity at time t
            P_next: pressure at time t+dt  
            U_next: U velocity at time t+dt
            
        Returns:
            residual tensor
        """
        # Time derivative: ∂U/∂t
        dUdt = self.compute_time_derivative(U_now, U_next)
        
        # Spatial derivatives of U
        U_derivs = self.compute_derivatives(U_next)
        P_derivs = self.compute_derivatives(P_next)
        
        # Convection terms: U∂U/∂X + V∂U/∂Y
        convection = U_next * U_derivs['dx'] + V_now * U_derivs['dy']
        
        # Pressure gradient: -∂P/∂X
        pressure_grad = -P_derivs['dx']
        
        # Viscous terms: Pr[∂²U/∂X² + ∂²U/∂Y²]
        viscous = self.Pr * (U_derivs['dxx'] + U_derivs['dyy'])
        
        # Porous media drag: -(Pr/Da)U
        porous_drag = -(self.Pr / self.Da) * U_next
        
        # Residual: LHS - RHS = 0
        residual = (dUdt + convection) - (pressure_grad + viscous + porous_drag)
        
        return residual
    
    def momentum_y_residual(self, U_now, V_now, P_next, V_next, theta_next):
        """
        Y-momentum equation:
        ∂V/∂t + U∂V/∂X + V∂V/∂Y = -∂P/∂Y + Pr[∂²V/∂X² + ∂²V/∂Y²] + Ra·Pr·θ - Ha²·Pr·V - (Pr/Da)V
        
        Args:
            U_now, V_now: velocity at time t
            P_next: pressure at time t+dt
            V_next: V velocity at time t+dt
            theta_next: temperature at time t+dt
            
        Returns:
            residual tensor
        """
        # Time derivative: ∂V/∂t
        dVdt = self.compute_time_derivative(V_now, V_next)
        
        # Spatial derivatives
        V_derivs = self.compute_derivatives(V_next)
        P_derivs = self.compute_derivatives(P_next)
        
        # Convection terms: U∂V/∂X + V∂V/∂Y
        convection = U_now * V_derivs['dx'] + V_next * V_derivs['dy']
        
        # Pressure gradient: -∂P/∂Y
        pressure_grad = -P_derivs['dy']
        
        # Viscous terms: Pr[∂²V/∂X² + ∂²V/∂Y²]
        viscous = self.Pr * (V_derivs['dxx'] + V_derivs['dyy'])
        
        # Buoyancy force: Ra·Pr·θ
        buoyancy = self.Ra * self.Pr * theta_next
        
        # Magnetic force: -Ha²·Pr·V
        magnetic = -(self.Ha**2) * self.Pr * V_next
        
        # Porous media drag: -(Pr/Da)V
        porous_drag = -(self.Pr / self.Da) * V_next
        
        # Residual: LHS - RHS = 0
        residual = (dVdt + convection) - (pressure_grad + viscous + buoyancy + magnetic + porous_drag)
        
        return residual
    
    def energy_residual(self, U_now, V_now, theta_now, theta_next):
        """
        Energy equation:
        ∂θ/∂t + U∂θ/∂X + V∂θ/∂Y = (1 + 4Rd/3)[∂²θ/∂X² + ∂²θ/∂Y²] + Q·θ
        
        Args:
            U_now, V_now: velocity at time t
            theta_now: temperature at time t
            theta_next: temperature at time t+dt
            
        Returns:
            residual tensor
        """
        # Time derivative: ∂θ/∂t
        dthetadt = self.compute_time_derivative(theta_now, theta_next)
        
        # Spatial derivatives of temperature
        theta_derivs = self.compute_derivatives(theta_next)
        
        # Convection terms: U∂θ/∂X + V∂θ/∂Y
        convection = U_now * theta_derivs['dx'] + V_now * theta_derivs['dy']
        
        # Diffusion with radiation: (1 + 4Rd/3)[∂²θ/∂X² + ∂²θ/∂Y²]
        diffusion_coeff = 1.0 + (4.0 * self.Rd) / 3.0
        diffusion = diffusion_coeff * (theta_derivs['dxx'] + theta_derivs['dyy'])
        
        # Heat source: Q·θ
        heat_source = self.Q * theta_next
        
        # Residual: LHS - RHS = 0
        residual = (dthetadt + convection) - (diffusion + heat_source)
        
        return residual
    
    def forward(self, f_now, f_next, validation_mode=False):
        """
        Compute physics-informed loss for all governing equations.
        
        Args:
            f_now: fields at time t [B, 4, H, W] (U, V, T, P)
            f_next: fields at time t+dt [B, 4, H, W] (U, V, T, P)
            validation_mode: if True, return detailed analysis
            
        Returns:
            total physics loss
        """
        if f_now.dim() != 4 or f_now.size(1) != 4:
            return torch.tensor(0.0, device=f_now.device, dtype=f_now.dtype)
        
        # Extract fields at current time
        U_now, V_now, T_now, P_now = torch.chunk(f_now, 4, 1)
        
        # Extract fields at next time
        U_next, V_next, T_next, P_next = torch.chunk(f_next, 4, 1)
        
        # Progressive scaling for training stability
        progress = min(self.current_epoch / 100.0, 1.0)
        base_scale = 1e-4 * progress  # Start very small
        
        try:
            # 1. Continuity equation residual
            continuity_res = self.continuity_residual(U_next, V_next)
            loss_continuity = torch.mean(continuity_res**2) * base_scale * self.w_continuity
            
            # 2. X-momentum equation residual
            momentum_x_res = self.momentum_x_residual(U_now, V_now, P_next, U_next)
            loss_momentum_x = torch.mean(momentum_x_res**2) * base_scale * self.w_momentum_x
            
            # 3. Y-momentum equation residual  
            momentum_y_res = self.momentum_y_residual(U_now, V_now, P_next, V_next, T_next)
            loss_momentum_y = torch.mean(momentum_y_res**2) * base_scale * self.w_momentum_y
            
            # 4. Energy equation residual
            energy_res = self.energy_residual(U_now, V_now, T_now, T_next)
            loss_energy = torch.mean(energy_res**2) * base_scale * self.w_energy
            
            # Total physics loss
            total_loss = loss_continuity + loss_momentum_x + loss_momentum_y + loss_energy
            
            # Safety clamp
            total_loss = torch.clamp(total_loss, min=1e-10, max=1.0)
            
            # Store analysis data
            if self.enable_analysis:
                self.loss_history['continuity'].append(loss_continuity.item())
                self.loss_history['momentum_x'].append(loss_momentum_x.item())
                self.loss_history['momentum_y'].append(loss_momentum_y.item())
                self.loss_history['energy'].append(loss_energy.item())
                self.loss_history['total'].append(total_loss.item())
            
            if validation_mode:
                return {
                    'total': total_loss,
                    'continuity': loss_continuity,
                    'momentum_x': loss_momentum_x,
                    'momentum_y': loss_momentum_y,
                    'energy': loss_energy,
                    'residuals': {
                        'continuity': continuity_res,
                        'momentum_x': momentum_x_res,
                        'momentum_y': momentum_y_res,
                        'energy': energy_res
                    }
                }
            
            return total_loss
            
        except Exception as e:
            print(f"Warning - Physics loss computation error: {e}")
            # Return small fallback value
            return torch.tensor(1e-6, device=f_now.device, dtype=f_now.dtype)
    
    def set_epoch(self, epoch):
        """Set current epoch for progressive training."""
        self.current_epoch = epoch
    
    def get_residual_statistics(self):
        """Get statistics about PDE residuals for analysis."""
        if not self.loss_history['total']:
            return None
        
        stats = {}
        for key, values in self.loss_history.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values[-10:]),  # Last 10 values
                    'std': np.std(values[-10:]),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats

# End of AccuratePhysicsLoss class and module 