"""
Neural network models for PhyCRNet.
Includes PhyCRNet and its components (ConvLSTM, ResBlock).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhyCRNet(nn.Module):
    """Physics-informed Convolutional-Recurrent Network."""
    
    def __init__(self, ch=4, hidden=192, upscale=1, dropout_rate=0.2):
        super().__init__()
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ConvLSTM layers
        self.conv_lstm = DeepConvLSTM(hidden, hidden, num_layers=3, kernel_size=5, padding=2)
        
        # Residual block
        self.residual_block = ResidualBlock(hidden, hidden)
        
        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(dropout_rate/2),
            nn.Conv2d(64, ch*(upscale**2), 3, padding=1),
            nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()
        )
        
        self._initialize_weights()
        self.up = upscale

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor [B×C×H×W]
            
        Returns:
            torch.Tensor: Output tensor [B×C×H×W]
        """
        # Encoding
        z = self.enc(x)                           # B×hidden×H×W
        
        # ConvLSTM processing
        z = z.unsqueeze(1)                        # B×1×hidden×H×W
        z, _ = self.conv_lstm(z)                  # B×1×hidden×H×W
        z = z.squeeze(1)                          # B×hidden×H×W
        
        # Residual processing
        z = self.residual_block(z)                # B×hidden×H×W
        
        # Decoding
        out = self.dec(z)                         # B×C×H×W
        return out

class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class DeepConvLSTM(nn.Module):
    """Multi-layer ConvLSTM with attention mechanism."""
    
    def __init__(self, in_channels, hidden_channels, num_layers=3, kernel_size=5, padding=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # ConvLSTM layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(ConvLSTM(in_channels, hidden_channels, kernel_size, padding))
            else:
                self.cells.append(ConvLSTM(hidden_channels, hidden_channels, kernel_size, padding))
        
        # Attention mechanism
        self.attention = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x, hidden_states=None):
        """Forward pass through all ConvLSTM layers.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_states (list): Initial hidden states for each layer
            
        Returns:
            tuple: (Output tensor, New hidden states)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
            
        output = x
        new_hidden_states = []
        
        # Process through each layer
        for i, cell in enumerate(self.cells):
            output, state = cell(output, hidden_states[i])
            new_hidden_states.append(state)
        
        # Apply attention to last time step
        last_output = output[:, -1]
        attention_weights = torch.sigmoid(self.attention(last_output))
        output_attended = last_output * attention_weights
        
        new_output = output.clone()
        new_output[:, -1] = output_attended
            
        return new_output, new_hidden_states

class ConvLSTM(nn.Module):
    """Enhanced Convolutional LSTM cell with peephole connections."""
    
    def __init__(self, in_channels, hidden_channels, kernel_size=5, padding=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Combined gates computation
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 
            hidden_channels * 4,  # 4 gates
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([hidden_channels, 42, 42])
        
        # Peephole connections
        self.w_ci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_cf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        self.w_co = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'w_c' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, hidden_state=None):
        """Forward pass with peephole connections and layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor [B×T×C×H×W]
            hidden_state (tuple): Previous (h, c) state
            
        Returns:
            tuple: (Output tensor, New state)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device)
        else:
            h, c = hidden_state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Concatenate input and hidden state
            combined = torch.cat([x_t, h], dim=1)
            gates = self.conv(combined)
            
            # Split into gates
            i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
            
            # Add peephole connections
            i = torch.sigmoid(i + self.w_ci * c)
            f = torch.sigmoid(f + self.w_cf * c)
            g = torch.tanh(g)
            
            # Update cell state
            c = f * c + i * g
            
            # Output gate with peephole
            o = torch.sigmoid(o + self.w_co * c)
            h = o * torch.tanh(c)
            
            # Apply layer normalization
            h = self.layer_norm(h)
            
            outputs.append(h.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, (h, c)

class HeavyPhyCRNet(nn.Module):
    """Heavy PhyCRNet with advanced features."""
    
    def __init__(self, num_layers=12, hidden_dim=256, num_heads=8, 
                 use_attention=True, use_skip_connections=True, use_se_blocks=True,
                 dropout_rate=0.1, use_spectral_norm=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.use_skip_connections = use_skip_connections
        self.use_se_blocks = use_se_blocks
        self.dropout_rate = dropout_rate
        
        # Input projection
        self.input_proj = nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1)
        
        # Enhanced ConvLSTM layers
        self.conv_lstm_layers = nn.ModuleList([
            EnhancedConvLSTMCell(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=5,
                use_attention=use_attention,
                num_heads=num_heads,
                use_se_block=use_se_blocks,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Skip connection projections
        if use_skip_connections:
            self.skip_projections = nn.ModuleList([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1) 
                for _ in range(num_layers // 2)
            ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_dim // 2, 4, kernel_size=3, padding=1)
        )
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            self._apply_spectral_norm()
        
        self._initialize_weights()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to conv layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        batch_size, channels, height, width = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Initialize hidden states
        hidden_states = [None] * self.num_layers
        
        # Store skip connections
        skip_connections = []
        
        # Process through ConvLSTM layers
        for i, conv_lstm in enumerate(self.conv_lstm_layers):
            # Add skip connection input
            if self.use_skip_connections and i > 0:
                skip_idx = min(i // 2, len(skip_connections) - 1)
                if skip_idx < len(skip_connections):
                    skip_input = self.skip_projections[skip_idx](skip_connections[skip_idx])
                    x = x + skip_input
            
            # ConvLSTM forward
            x, hidden_states[i] = conv_lstm(x.unsqueeze(1), hidden_states[i])
            x = x.squeeze(1)
            
            # Store for skip connections
            if self.use_skip_connections and i % 2 == 0:
                skip_connections.append(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x

class EnhancedConvLSTMCell(nn.Module):
    """Enhanced ConvLSTM cell with attention and SE blocks."""
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True,
                 use_attention=True, num_heads=8, use_se_block=True, dropout_rate=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.use_attention = use_attention
        self.use_se_block = use_se_block
        
        # Convolution for gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(hidden_dim, num_heads)
        
        # SE block
        if use_se_block:
            self.se_block = SEBlock(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, input_tensor, cur_state):
        """Forward pass."""
        batch_size, seq_len, _, h, w = input_tensor.size()
        
        if cur_state is None:
            cur_h = torch.zeros(batch_size, self.hidden_dim, h, w, device=input_tensor.device)
            cur_c = torch.zeros(batch_size, self.hidden_dim, h, w, device=input_tensor.device)
        else:
            cur_h, cur_c = cur_state
        
        output_inner = []
        
        for t in range(seq_len):
            # Concatenate input and hidden state
            combined = torch.cat([input_tensor[:, t], cur_h], dim=1)
            combined_conv = self.conv(combined)
            
            # Split into gates
            cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
            
            # Apply activations
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g)
            
            # Update cell state
            cur_c = f * cur_c + i * g
            
            # Apply attention to cell state
            if self.use_attention:
                cur_c = self.attention(cur_c)
            
            # Compute hidden state
            cur_h = o * torch.tanh(cur_c)
            
            # Apply SE block
            if self.use_se_block:
                cur_h = self.se_block(cur_h)
            
            # Apply dropout
            cur_h = self.dropout(cur_h)
            
            output_inner.append(cur_h.unsqueeze(1))
        
        layer_output = torch.cat(output_inner, dim=1)
        return layer_output, (cur_h, cur_c)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """Forward pass."""
        batch_size, channels, height, width = x.shape
        
        # Generate Q, K, V
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Compute attention
        attention_scores = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(v, attention_weights.transpose(-2, -1))
        attended = attended.view(batch_size, channels, height, width)
        
        # Output projection
        output = self.output_proj(attended)
        return output + x  # Residual connection

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass."""
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x) 