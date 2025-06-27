"""
Loss functions for PhyCRNet.
Includes physics-based and data-driven losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMSELoss(nn.Module):
    """Simple MSE loss for basic training."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)

class FieldSpecificLoss(nn.Module):
    """Field-specific weighted MSE loss for different physical quantities."""
    
    def __init__(self, weight_u=1.0, weight_v=1.0, weight_p=0.1, weight_t=1.0):
        super().__init__()
        self.weight_u = weight_u
        self.weight_v = weight_v
        self.weight_p = weight_p
        self.weight_t = weight_t
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 4, H, W] predicted fields [U, V, T, P]
            target: [B, 4, H, W] target fields [U, V, T, P]
        """
        loss_u = self.mse(pred[:, 0], target[:, 0]) * self.weight_u
        loss_v = self.mse(pred[:, 1], target[:, 1]) * self.weight_v
        loss_t = self.mse(pred[:, 2], target[:, 2]) * self.weight_t
        loss_p = self.mse(pred[:, 3], target[:, 3]) * self.weight_p
        
        total_loss = loss_u + loss_v + loss_t + loss_p
        
        return total_loss

class PhysicsLoss(nn.Module):
    """Simple physics-based loss using basic PDE residuals."""
    
    def __init__(self, dx=1.0/42, params=None):
        super().__init__()
        self.dx = dx
        
        # Default physical parameters if not provided
        if params is None:
            self.Pr = 0.71
            self.Ra = 1e5
            self.Ha = 0.0
            self.Da = 1e-3
            self.Rd = 1.8
            self.Q = 0.0
        else:
            self.Pr = params.get('Pr', 0.71)
            self.Ra = params.get('Ra', 1e5)
            self.Ha = params.get('Ha', 0.0)
            self.Da = params.get('Da', 1e-3)
            self.Rd = params.get('Rd', 1.8)
            self.Q = params.get('Q', 0.0)
    
    def compute_gradients(self, field):
        """Compute spatial gradients using simple finite differences."""
        # Pad field for boundary conditions
        field_padded = F.pad(field, [1, 1, 1, 1], mode='replicate')
        
        # Compute gradients
        dfdx = (field_padded[:, :, :, 2:] - field_padded[:, :, :, :-2]) / (2 * self.dx)
        dfdy = (field_padded[:, :, 2:, :] - field_padded[:, :, :-2, :]) / (2 * self.dx)
        
        # Compute Laplacian
        d2fdx2 = (field_padded[:, :, :, 2:] - 2*field_padded[:, :, :, 1:-1] + field_padded[:, :, :, :-2]) / (self.dx**2)
        d2fdy2 = (field_padded[:, :, 2:, :] - 2*field_padded[:, :, 1:-1, :] + field_padded[:, :, :-2, :]) / (self.dx**2)
        laplacian = d2fdx2 + d2fdy2
        
        return dfdx, dfdy, laplacian
    
    def forward(self, f_now, f_next, dt=0.001):
        """
        Compute physics loss based on PDE residuals.
        
        Args:
            f_now: [B, 4, H, W] current state [U, V, T, P]
            f_next: [B, 4, H, W] next state [U, V, T, P]
            dt: time step
        """
        # Extract fields
        U = f_next[:, 0:1]  # U velocity
        V = f_next[:, 1:2]  # V velocity  
        T = f_next[:, 2:3]  # Temperature
        P = f_next[:, 3:4]  # Pressure
        
        # Compute spatial derivatives
        dUdx, dUdy, lapU = self.compute_gradients(U)
        dVdx, dVdy, lapV = self.compute_gradients(V)
        dTdx, dTdy, lapT = self.compute_gradients(T)
        dPdx, dPdy, _ = self.compute_gradients(P)
        
        # Time derivatives (finite difference)
        dUdt = (f_next[:, 0:1] - f_now[:, 0:1]) / dt
        dVdt = (f_next[:, 1:2] - f_now[:, 1:2]) / dt
        dTdt = (f_next[:, 2:3] - f_now[:, 2:3]) / dt
        
        # Continuity equation: ∇·u = 0
        continuity = dUdx + dVdy
        
        # X-momentum equation (simplified)
        x_momentum = dUdt + U*dUdx + V*dUdy + dPdx - (1/self.Pr) * lapU
        
        # Y-momentum equation (simplified)  
        y_momentum = dVdt + U*dVdx + V*dVdy + dPdy - (1/self.Pr) * lapV + self.Ra*T
        
        # Energy equation (simplified)
        energy = dTdt + U*dTdx + V*dTdy - lapT
        
        # Compute residual losses
        loss_continuity = torch.mean(continuity**2)
        loss_x_momentum = torch.mean(x_momentum**2)  
        loss_y_momentum = torch.mean(y_momentum**2)
        loss_energy = torch.mean(energy**2)
        
        # Total physics loss
        total_loss = loss_continuity + loss_x_momentum + loss_y_momentum + loss_energy
        
        return total_loss

class CombinedLoss(nn.Module):
    """Combined data and physics loss."""
    
    def __init__(self, data_weight=1.0, physics_weight=0.1, dx=1.0/42, params=None):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        
        self.data_loss = FieldSpecificLoss()
        self.physics_loss = PhysicsLoss(dx=dx, params=params)
    
    def forward(self, pred, target, f_now=None, dt=0.001):
        """
        Args:
            pred: [B, 4, H, W] predicted next state
            target: [B, 4, H, W] ground truth next state
            f_now: [B, 4, H, W] current state (for physics loss)
            dt: time step
        """
        # Data loss
        loss_data = self.data_loss(pred, target)
        
        # Physics loss
        if f_now is not None:
            loss_physics = self.physics_loss(f_now, pred, dt)
        else:
            loss_physics = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = self.data_weight * loss_data + self.physics_weight * loss_physics
        
        return total_loss 