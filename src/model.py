"""
Module: model.py
Description: Dynamic Entity-Aware LSTM (EA-LSTM) for Multi-Horizon Streamflow Forecasting.

This model implements the EA-LSTM architecture proposed by Kratzert et al. (2019)
for rainfall-runoff modeling, adapted to predict multiple future lead times simultaneously.

Mathematical Formulations (Aligned with Code Variables):
1. Static Embedding Layer (Input Gate):
   static_input_gate = sigmoid(W_static * x_static + b_static)
2. Dynamic Recurrent Cell Updates:
   forget_gate = sigmoid(W_forget * dynamic_input_t + U_forget * hidden_state_prev + b_forget)
   cell_input  = tanh(W_cell * dynamic_input_t + U_cell * hidden_state_prev + b_cell)
   output_gate = sigmoid(W_output * dynamic_input_t + U_output * hidden_state_prev + b_output)
   cell_state_new   = forget_gate * cell_state_prev + static_input_gate * cell_input
   hidden_state_new = output_gate * tanh(cell_state_new)

Efficiency & Optimization Decisions:
- Gate Concatenation: Combined the linear projections for the dynamic gates (forget, cell, output) 
  into a single larger matrix multiplication layer to maximize GPU parallelism.
- Bias Consolidation: Bias is enabled only on the dynamic input projection layer 
  and disabled on the recurrent layer to eliminate redundant parameters.
- Dynamic Configuration: Grid/Layer shapes are derived at runtime directly from 
  the configuration file, ensuring a highly modular and robust architecture.
"""

import torch
import torch.nn as nn

# ==============================================================================
# 1. EA-LSTM Cell Implementation
# ==============================================================================
class EALSTMCell(nn.Module):
    """
    Entity-Aware LSTM cell. Uses a constant, externally provided static input gate
    to modulate the flow of time-dependent dynamic inputs into the cell memory.
    """
    def __init__(self, dynamic_size: int, hidden_size: int, initial_forget_bias: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size

        # Combined dynamic weights for efficiency (Forget Gate, Cell Input, Output Gate)
        self.weight_dynamic_inputs = nn.Linear(dynamic_size, 3 * hidden_size, bias=True)

        # Combined recurrent weights (Biases are omitted to avoid redundancy)
        self.weight_hidden_state = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        # Bias the forget gate toward "remember" at initialization (first chunk of gates_raw)
        with torch.no_grad():
            self.weight_dynamic_inputs.bias[:hidden_size] = initial_forget_bias

    def forward(self, dynamic_input_t: torch.Tensor, hidden_state_prev: torch.Tensor, 
                cell_state_prev: torch.Tensor, static_input_gate: torch.Tensor):
        """
        Executes a single hourly time step forward pass.
        
        Equations Processed:
          gates_raw = (W * dynamic_input_t + b) + (U * hidden_state_prev)
          forget_gate = sigmoid(forget_gate_raw)
          cell_input  = tanh(cell_input_raw)
          output_gate = sigmoid(output_gate_raw)
          cell_state_new   = forget_gate * cell_state_prev + static_input_gate * cell_input
          hidden_state_new = output_gate * tanh(cell_state_new)
        """
        # Linear projection step combining dynamic inputs and previous hidden state
        gates_raw = self.weight_dynamic_inputs(dynamic_input_t) + self.weight_hidden_state(hidden_state_prev)
        
        # Split the combined matrix into individual dynamic components
        forget_gate_raw, cell_input_raw, output_gate_raw = gates_raw.chunk(3, dim=-1)
        
        # Non-linear activation routing
        forget_gate = torch.sigmoid(forget_gate_raw)
        cell_input = torch.tanh(cell_input_raw)
        output_gate = torch.sigmoid(output_gate_raw)
        
        # Cell and hidden state structural updates (Modulated by the static gate)
        cell_state_new = forget_gate * cell_state_prev + static_input_gate * cell_input
        hidden_state_new = output_gate * torch.tanh(cell_state_new)
        
        return hidden_state_new, cell_state_new


# ==============================================================================
# 2. Main Model Wrapper Class
# ==============================================================================
class EALSTMModel(nn.Module):
    """
    Full EA-LSTM network wrapper managing the time-loop execution and multi-horizon prediction head.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Runtime-derived dimensions from configuration properties
        self.dynamic_size = len(config['dynamic_inputs'])
        self.static_size = len(config['static_attributes'])
        self.output_size = len(config['forecast_lead_times'])
        self.hidden_size = config['hidden_size']
        dropout_rate = config['output_dropout']
        
        # Static Embedding Layer (Computes the static input gate)
        statics_embedding_type = config.get('statics_embedding', {}).get('type')
        if statics_embedding_type == 'one_hot':
            self.static_gate_layer = nn.Sequential(
                nn.Linear(self.static_size, self.hidden_size),
                nn.Sigmoid()
            )
        else:
            print(f"[WARNING] No static embedding defined for statics_embedding.type='{statics_embedding_type}'. "
                  f"Static input gate disabled (model will run without entity-aware conditioning).")
            self.static_gate_layer = None

        self.cell = EALSTMCell(self.dynamic_size, self.hidden_size,
                               initial_forget_bias=config.get('initial_forget_bias', 0.0))
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Multi-Horizon Projection Head
        self.prediction_head = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x_dynamic: torch.Tensor, x_static: torch.Tensor):
        """
        Processes a full look-back sequence window to predict future streamflow targets.
        """
        batch_size, seq_length, _ = x_dynamic.size()
        device = x_dynamic.device
        
        # State Initialization (Recursive base-cases)
        hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        # Compute the catchment-specific static input gate exactly once per sequence
        if self.static_gate_layer is not None:
            static_input_gate = self.static_gate_layer(x_static)
        else:
            static_input_gate = torch.ones(batch_size, self.hidden_size, device=device)
        
        # Sequential Bounded Time Loop Execution
        for t in range(seq_length):
            dynamic_input_t = x_dynamic[:, t, :]
            hidden_state, cell_state = self.cell(dynamic_input_t, hidden_state, cell_state, static_input_gate)
            
        # Feature Extraction and Regularization
        hidden_state_dropped = self.dropout(hidden_state)
        
        # Direct Mapping to Multi-Lead Time Targets
        predictions = self.prediction_head(hidden_state_dropped)
        
        return predictions