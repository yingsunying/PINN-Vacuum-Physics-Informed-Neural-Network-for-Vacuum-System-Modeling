import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

V, S, Q_leak, dt, t_max = 10.0, 2.0, 0.1, 0.1, 5.0
time_steps = int(t_max / dt)
t_eval = np.linspace(0, t_max, time_steps)
P_true = np.zeros(time_steps)
P_true[0] = 1000.0
true_outgassing = lambda t: 5.0 * np.exp(-0.1 * t)

for i in range(1, time_steps):
    P_true[i] = P_true[i-1] + ((Q_leak + true_outgassing(t_eval[i-1]) - S * P_true[i-1]) / V) * dt
P_meas = P_true + np.random.normal(0, 0.5, size=time_steps)

t_min, t_max_val = t_eval.min(), t_eval.max()
t_scaled = (t_eval - t_min) / (t_max_val - t_min)
t_tensor = torch.tensor(t_scaled.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
Y_tensor = torch.tensor(P_meas.reshape(-1, 1), dtype=torch.float32)

class VacuumDualPINN(nn.Module):
    def __init__(self, p_init):
        super(VacuumDualPINN, self).__init__()
        self.p_net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.p_init = p_init
        
        self.q_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, t):
        P_pred = self.p_init + self.p_net(t) * 100.0
        
        Q_out_pred = torch.exp(self.q_net(t)) 
        return P_pred, Q_out_pred

model = VacuumDualPINN(p_init=Y_tensor[0].item())
optimizer = optim.Adam(model.parameters(), lr=0.002)

epochs = 6000
scale_factor = 0.02  

optimizer = optim.Adam([
    {'params': model.p_net.parameters(), 'lr': 0.001},
    {'params': model.q_net.parameters(), 'lr': 0.005} 
])

for epoch in range(epochs):
    optimizer.zero_grad()
    P_pred, Q_out_pred = model(t_tensor)
  
    loss_data = nn.MSELoss()(P_pred, Y_tensor)

    dP_dt_scaled = torch.autograd.grad(
        P_pred, t_tensor, grad_outputs=torch.ones_like(P_pred),
        create_graph=True, retain_graph=True
    )[0]
    dP_dt = dP_dt_scaled / (t_max_val - t_min)

    physics_residual = V * dP_dt + S * P_pred - Q_leak - Q_out_pred
    loss_physics_raw = nn.MSELoss()(physics_residual, torch.zeros_like(physics_residual))
    
    dQ_dt_scaled = torch.autograd.grad(
        Q_out_pred, t_tensor, grad_outputs=torch.ones_like(Q_out_pred),
        create_graph=True, retain_graph=True
    )[0]
    loss_q_smooth = torch.mean(torch.relu(dQ_dt_scaled)) 
    
    total_loss = loss_data + scale_factor * loss_physics_raw + 10.0 * loss_q_smooth
    
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.4f} | Data Loss: {loss_data.item():.4f} | Phys Loss: {loss_physics_raw.item():.4f} | Smooth Loss: {loss_q_smooth.item():.4f}")
            
model.eval()
with torch.no_grad():
    t_dense = np.linspace(0, t_max, time_steps * 2)
    t_dense_scaled = (t_dense - t_min) / (t_max_val - t_min)
    t_dense_tensor = torch.tensor(t_dense_scaled.reshape(-1, 1), dtype=torch.float32)
    
    final_pred_P, learned_Q_out = model(t_dense_tensor)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_eval, P_true, 'k-', label='True Physics', linewidth=2)
plt.scatter(t_eval, P_meas, color='gray', alpha=0.5, s=15, label='Measured Data')
plt.plot(t_dense, final_pred_P.numpy(), 'r--', label='PINN Smooth Path (Interpolated)', linewidth=2)
plt.xlabel('Time (s)'); plt.ylabel('Pressure (Pa)'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_eval, true_outgassing(t_eval), 'g-', label='True Hidden $Q_{out}$', linewidth=2)
plt.plot(t_dense, learned_Q_out.numpy().flatten(), 'b--', label='Neural Latent $Q_{out}$ (Continuous)', linewidth=2)
plt.xlabel('Time (s)'); plt.ylabel('Outgassing rate (Pa*L/s)'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()