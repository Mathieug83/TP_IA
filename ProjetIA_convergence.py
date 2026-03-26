import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Javascript, display


device = torch.device("cpu")


class FCN(torch.nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.hidden_layer1 = torch.nn.Linear(2, 100)
        self.hidden_layer2 = torch.nn.Linear(100, 10)
        self.output_layer = torch.nn.Linear(10, 1)

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        output = self.output_layer(layer2_out)
        return output


class forward_problem:
    def __init__(self, net, ic_weight=1.0, bc_weight=1.0, pde_weight=1.0, sensor_weight=0.0, sensors_x=None, reseau_reel=None):
        self.net = net
        self.ic_weight = ic_weight
        self.bc_weight = bc_weight
        self.pde_weight = pde_weight
        

        self.sensor_weight = sensor_weight
        self.sensors_x = sensors_x if sensors_x is not None else []
        self.reseau_reel = reseau_reel 
        

        self.liste_loss = []

    def heat_source(self, x, t):
        return torch.zeros_like(x)

    def advection(self, x, t, u, beta=3, uref=0):
        return beta * (u - uref)

    def conductivity(self, x):
        return torch.ones_like(x)

    def heat_capacity(self, x):
        return torch.ones_like(x)

    def ic(self, x):
        return torch.zeros_like(x)

    def bc(self, x, t):
        B = 1.0
        Tu = 0.1
        bc_values = torch.where(
            x == 0,
            B * (1 - torch.exp(-t/Tu)),
            torch.zeros_like(x)
        )
        bc_values = torch.where(
            x == 2,
            torch.zeros_like(x),
            bc_values
        )
        return bc_values

    def f(self, x, t):
        u = self.net(x, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        flux = 1.0 * self.conductivity(x) * u_x
        flux_x = torch.autograd.grad(flux, x, grad_outputs=torch.ones_like(flux), retain_graph=True, create_graph=True)[0]

        c = self.heat_capacity(x)
        beta = 3
        uref = 0

        residual = c * u_t - flux_x + self.advection(x, t, u, beta, uref)
        return residual

    def f_bc(self, x, t):
        u = self.net(x, t)
        return u - self.bc(x, t)

    def f_ic(self, x, t):
        u = self.net(x, t)
        return u - self.ic(x)

    def solve(self, N_iter=1000):
        mse_cost_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1.0e-2, weight_decay=1.0e-100)
        
        self.liste_loss = [] 

        if self.reseau_reel is not None:
            val_x = np.linspace(0, 2, 40)
            val_t = np.linspace(0, 1, 40)
            ms_val_x, ms_val_t = np.meshgrid(val_x, val_t)
            pt_val_x = Variable(torch.from_numpy(np.ravel(ms_val_x).reshape(-1, 1)).float(), requires_grad=False).to(device)
            pt_val_t = Variable(torch.from_numpy(np.ravel(ms_val_t).reshape(-1, 1)).float(), requires_grad=False).to(device)
            

            with torch.no_grad():
                u_truth_eval = self.reseau_reel(pt_val_x, pt_val_t)

        for epoch in range(N_iter):
            optimizer.zero_grad()


            pt_x_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)
            pt_t_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)

            pt_x_collocation_ic = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)
            pt_t_collocation_ic = Variable(torch.zeros_like(pt_x_collocation_ic), requires_grad=True).to(device)

            pt_x_collocation_bc = Variable((torch.tensor([[0.0], [2.0]])).float(), requires_grad=True).to(device)
            pt_t_collocation_bc = Variable(torch.Tensor(2, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)


            f_out = self.f(pt_x_collocation, pt_t_collocation)
            mse_f = mse_cost_function(f_out, torch.zeros_like(f_out))

            f_out_ic = self.f_ic(pt_x_collocation_ic, pt_t_collocation_ic)
            mse_ic = mse_cost_function(f_out_ic, torch.zeros_like(f_out_ic))

            f_out_bc = self.f_bc(pt_x_collocation_bc, pt_t_collocation_bc)
            mse_bc = mse_cost_function(f_out_bc, torch.zeros_like(f_out_bc))


            mse_sensors = 0.0
            active_sensor_weight = 0.0
            
            if self.sensor_weight > 0 and len(self.sensors_x) > 0 and self.reseau_reel is not None:
                active_sensor_weight = self.sensor_weight
                for sx in self.sensors_x:
                    pt_t_sens = Variable(torch.Tensor(50, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)
                    pt_x_sens = Variable(torch.full((50, 1), sx).float(), requires_grad=True).to(device)
                    
                    u_pred = self.net(pt_x_sens, pt_t_sens)
                    
                    with torch.no_grad():
                        u_true = self.reseau_reel(pt_x_sens, pt_t_sens)
                        
                    mse_sensors += mse_cost_function(u_pred, u_true)
                
                mse_sensors = mse_sensors / len(self.sensors_x)


            loss_train = (
                self.pde_weight * mse_f + 
                self.ic_weight * mse_ic + 
                self.bc_weight * mse_bc + 
                active_sensor_weight * mse_sensors
            )

            loss_train.backward()
            optimizer.step()
            

            if self.reseau_reel is not None:
                with torch.no_grad():

                    u_pred_eval = self.net(pt_val_x, pt_val_t)
                    loss_affichage = mse_cost_function(u_pred_eval, u_truth_eval).item()
            else:
 
                loss_affichage = loss_train.item()


            self.liste_loss.append(loss_affichage)

            # Affichage de suivi 
            if epoch == 0 or (epoch + 1) % (N_iter // 5) == 0 or epoch == N_iter - 1:
                with torch.no_grad():
                    if self.reseau_reel is not None:
                        print(f"Epoch {epoch+1:4d}/{N_iter} | MSE vs Truth: {loss_affichage:.5e} | (Loss Train: {loss_train.item():.5e})")
                    else:
                        print(f"Epoch {epoch+1:4d}/{N_iter} | Loss Train: {loss_train.item():.5e}")



net_truth = FCN().to(device)
prob_truth = forward_problem(net_truth, ic_weight=20, bc_weight=1, pde_weight=1.0)
prob_truth.solve(N_iter=20000)

net_early = FCN().to(device)
prob_early = forward_problem(net_early, ic_weight=20, bc_weight=1, pde_weight=1.0, reseau_reel=net_truth)
prob_early.solve(N_iter=5000)

capteurs_positions = [0.5, 1.5]
net_sensors = FCN().to(device)
prob_sensors = forward_problem(
    net_sensors, 
    ic_weight=20, 
    bc_weight=1,
    pde_weight=1,
    sensor_weight=50.0,
    sensors_x=capteurs_positions, 
    reseau_reel=net_truth
)
prob_sensors.solve(N_iter=5000)



x = np.arange(0, 2, 0.02)
t = np.arange(0, 1, 0.02)
ms_x, ms_t = np.meshgrid(x, t)
x_all = np.ravel(ms_x).reshape(-1, 1)
t_all = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x_all).float(), requires_grad=False).to(device)
pt_t = Variable(torch.from_numpy(t_all).float(), requires_grad=False).to(device)

with torch.no_grad():
    u_truth = net_truth(pt_x, pt_t).cpu().numpy().reshape(ms_x.shape)
    u_early = net_early(pt_x, pt_t).cpu().numpy().reshape(ms_x.shape)
    u_sensors = net_sensors(pt_x, pt_t).cpu().numpy().reshape(ms_x.shape)

fig3d = plt.figure(figsize=(20, 6))

ax1 = fig3d.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(ms_x, ms_t, u_truth, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig3d.colorbar(surf1, shrink=0.4, aspect=5, ax=ax1)
ax1.set_xlabel('x')
ax1.set_ylabel('t')
ax1.set_title('Vérité de terrain (5000 iters)')

ax2 = fig3d.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(ms_x, ms_t, u_early, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig3d.colorbar(surf2, shrink=0.4, aspect=5, ax=ax2)
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_title("Physique pure (2000 iters)")

ax3 = fig3d.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(ms_x, ms_t, u_sensors, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig3d.colorbar(surf3, shrink=0.4, aspect=5, ax=ax3)
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_title(f"Physique + Capteurs (2000 iters)")

plt.tight_layout()
plt.show()



def filtrage(points, window=20):
    if len(points) < window:
        return points
    box = np.ones(window) / window
    smoothed = np.convolve(points, box, mode='valid')
    return smoothed

fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

window_size = 20
loss_early_smooth = filtrage(prob_early.liste_loss, window=window_size)
loss_sensors_smooth = filtrage(prob_sensors.liste_loss, window=window_size)

epochs_smooth = np.arange(window_size - 1, len(prob_early.liste_loss))

ax_loss.plot(prob_early.liste_loss, color='blue', alpha=0.15, label='_nolegend_')
ax_loss.plot(epochs_smooth, loss_early_smooth, color='blue', linewidth=2, label='Physique Pure')

ax_loss.plot(prob_sensors.liste_loss, color='red', alpha=0.15, label='_nolegend_')
ax_loss.plot(epochs_smooth, loss_sensors_smooth, color='red', linewidth=2, label='Physique + Capteurs')

ax_loss.set_xlabel('Itérations (Epochs)')
ax_loss.set_ylabel('MSE avec Vérité de Terrain (Log)')
ax_loss.set_title("Comparaison : Écart global à la Vérité de Terrain")
ax_loss.set_yscale('log')
ax_loss.grid(True, which="both", ls="--", alpha=0.5)

ax_loss.legend(loc="upper right", fontsize=12)

plt.tight_layout()
plt.show()