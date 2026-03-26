import torch
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0))
else:
    print("Aucun GPU CUDA disponible")


def compare_npz(paths: list, labels: list = None):
    """
    Affiche les surfaces u(x,t) de plusieurs fichiers .npz côte à côte.
    
    Args:
        paths  : liste de chemins vers des fichiers .npz (contenant ms_x, ms_t, u)
        labels : liste de titres pour chaque subplot (optionnel)
    """
    if labels is None:
        labels = [p.replace('.npz', '') for p in paths]

    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             subplot_kw={'projection': '3d'})
    
    # Si un seul fichier, axes n'est pas une liste
    if n == 1:
        axes = [axes]

    for ax, path, label in zip(axes, paths, labels):
        ref = np.load(path)
        ms_x = ref['ms_x']
        ms_t = ref['ms_t']
        u    = ref['u']

        surf = ax.plot_surface(ms_x, ms_t, u, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        fig.colorbar(surf, ax=ax, shrink=0.4)
        ax.set_title(label)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')

        print(f"[{label}]  max={u.max():.4f}  min={u.min():.4f}  mean={u.mean():.4f}")

    # Synchronisation des vues 3D
    def on_move(event):
        if event.inaxes in axes:
            for ax in axes:
                if ax != event.inaxes:
                    ax.view_init(elev=event.inaxes.elev,
                                 azim=event.inaxes.azim)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.tight_layout()
    plt.show()

class forward_problem:
    def __init__(self, net):
        # Store the neural network as an attribute
        self.net = net

    def heat_source(self, x, t):
        # As specified in the problem, q(x,t) = 0 initially
        q = torch.zeros_like(x)
        return q

    def advection(self, x, t, u, beta=3, uref=0):
        # Advection term as specified: β(u - uref)
        q = beta * (u - uref)
        return q

    def conductivity(self, x):
        # Simplified constant conductivity initially
        # You can modify this later to be more complex
        k = torch.ones_like(x)
        return k

    def heat_capacity(self, x):
        # Default constant heat capacity
        c = torch.ones_like(x)
        return c

    def ic(self, x):
        # Initial conditions: u(x, t=0) = 0
        ic = torch.zeros_like(x)
        return ic

    def bc(self, x, t):
        # Boundary conditions:
        # u(x=0, t) = B(1 - exp(-t/Tu)), B=1, Tu=0.1
        # u(x=L, t) = 0, L=2
        B = 1.0
        Tu = 0.1

        # Create a tensor to store boundary conditions
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
    # Cette méthode calcule le résidu de l'équation de la chaleur:
    # ρc∂u/∂t = ∂/∂x(k∂u/∂x) - β(u-uref) + q
    # Le réseau est entraîné pour minimiser ce résidu
    def f(self, x, t):
        # Complete the residual calculation for the heat equation
        u = self.net(x, t)

        # Compute gradients
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        # Compute flux and flux_x
        flux = self.conductivity(x) * u_x
        flux_x = torch.autograd.grad(
            flux, x,
            grad_outputs=torch.ones_like(flux),
            retain_graph=True,
            create_graph=True
        )[0]

        # Heat equation residual
        c = self.heat_capacity(x)
        beta = 3  # Given in the problem description
        uref = 0  # Given in the problem description

        residual = c * u_t - flux_x + self.advection(x, t, u, beta, uref)

        return residual

    def f_bc(self, x, t):
        # Boundary condition residual
        u = self.net(x, t)
        residual = u - self.bc(x, t)
        return residual

    def f_ic(self, x, t):
        # Initial condition residual
        u = self.net(x, t)
        residual = u - self.ic(x)
        return residual

    def plot_forward(self, conductivity=None, heat_source=None):
        # Make sure to add the plotting method from the original code
        x = np.arange(0, 2, 0.02)
        t = np.arange(0, 1, 0.02)
        ms_x, ms_t = np.meshgrid(x, t)
        x_all = np.ravel(ms_x).reshape(-1, 1)
        t_all = np.ravel(ms_t).reshape(-1, 1)
        pt_x = Variable(torch.from_numpy(x_all).float(), requires_grad=False).to(device)
        pt_t = Variable(torch.from_numpy(t_all).float(), requires_grad=False).to(device)

        if heat_source is not None:
            pt_u = heat_source(pt_x, pt_t)
            u = pt_u.data.cpu().numpy()
            ms_u = u.reshape(ms_x.shape)

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.4, aspect=5)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
            ax.set_title('heat source q(x,t)')
            plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        pt_u = self.net(pt_x, pt_t)
        u = pt_u.data.cpu().numpy()
        ms_u = u.reshape(ms_x.shape)

        surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.4, aspect=5)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('temperature u(x,t)')
        plt.show()

        if conductivity is not None:
            u = conductivity(Variable(torch.from_numpy(x).float()))

            plt.plot(x, u, '+')
            plt.xlabel('x')
            plt.ylabel('k')
            plt.title('conductivity k(x)')
            plt.show()


    # Méthode principale d'entraînement du réseau
    # Utilise la méthode des résidus pour résoudre l'EDP:
    # 1. Échantillonne des points dans le domaine (points de collocation)
    # 2. Calcule le résidu de l'EDP en ces points
    # 3. Minimise ce résidu par descente de gradient
    def solve(self, N_iter=1000):
        # Add the solve method from the original code to the class
        net = self.net

        # Mean squared error loss function
        mse_cost_function = torch.nn.MSELoss()

        # Adam optimizer with learning rate and weight decay
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-2, weight_decay=1.0e-100)

        # Échantillonnage aléatoire dans le domaine de x pour les points de collocation
        pt_x_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)

        # Échantillonnage aléatoire dans le domaine de t pour les points de collocation
        pt_t_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)
    
        # Échantillonnage aléatoire dans le domaine de x pour les conditions initiales
        pt_x_collocation_ic = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)

        # t pour conditions initiales (toujours zéro)
        pt_t_collocation_ic = Variable(torch.zeros_like(pt_x_collocation_ic), requires_grad=True).to(device)

        # Génération des x pour les conditions aux bords (0 et 2)
        pt_x_collocation_bc = Variable(torch.tensor([[0.0], [2.0]]).float(), requires_grad=True).to(device)

        # Échantillonnage aléatoire dans le domaine de t pour les conditions aux bords
        pt_t_collocation_bc = Variable(torch.Tensor(2, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)

        # Indicateur pour la randomisation des points de collocation
        randomise_colloc = 1
            
        # Boucle d'entraînement
        for epoch in range(N_iter):
            # Réinitialiser les gradients à zéro
            optimizer.zero_grad()

            # Randomisation des points de collocation si activée
            if randomise_colloc == 1:
                pt_x_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)
                pt_t_collocation = Variable(torch.Tensor(100, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)

                pt_x_collocation_ic = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)
                pt_t_collocation_ic = Variable(torch.zeros_like(pt_x_collocation_ic), requires_grad=True).to(device)

                pt_x_collocation_bc = Variable(torch.tensor([[0.0], [2.0]]).float(), requires_grad=True).to(device)
                pt_t_collocation_bc = Variable(torch.Tensor(2, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)

            # Calcul de la perte basée sur l'EDP
            f_out = self.f(pt_x_collocation, pt_t_collocation)
            pt_all_zeros = Variable(torch.zeros_like(f_out), requires_grad=False).to(device)
            mse_f = mse_cost_function(f_out, pt_all_zeros)

            # Calcul de la perte pour les conditions initiales
            f_out_ic = self.f_ic(pt_x_collocation_ic, pt_t_collocation_ic)
            pt_all_zeros = Variable(torch.zeros_like(f_out_ic), requires_grad=False).to(device)
            mse_ic = mse_cost_function(f_out_ic, pt_all_zeros)

            # Calcul de la perte pour les conditions aux bords
            f_out_bc = self.f_bc(pt_x_collocation_bc, pt_t_collocation_bc)
            pt_all_zeros = Variable(torch.zeros_like(f_out_bc), requires_grad=False).to(device)
            mse_bc = mse_cost_function(f_out_bc, pt_all_zeros)

            # Calcul de la perte globale
            # Vous pouvez ajuster les coefficients si nécessaire
            loss = mse_f + mse_ic*20 + mse_bc

            # Rétropropagation
            loss.backward()

            # Mise à jour des paramètres
            optimizer.step()

            # Affichage périodique de la perte
            if epoch <= 10 or epoch == 25 or epoch == 50 or epoch % 100 == 0:
                with torch.autograd.no_grad():
                    print(epoch, "Training Loss:", loss.data)

            # Visualisation périodique
            if epoch == N_iter-1:
                self.plot_forward()
    def solve_with_sensors(self, N_iter=1000, ref_path='reference_clean.npz', P_bruit=0.01,dist_capteurs=1.5, critere_loss=1e-4):

        ref = np.load(ref_path)
        ms_x_ref = ref['ms_x'] 
        ms_t_ref = ref['ms_t']
        u_ref    = ref['u']

        x_vals = ms_x_ref[0, :]         
        idx_02 = np.argmin(np.abs(x_vals - 0.2))  
        idx_d = np.argmin(np.abs(x_vals - dist_capteurs)) 

        t_sensor   = ms_t_ref[:, 0].reshape(-1, 1)     
        u_sensor_02 = u_ref[:, idx_02].reshape(-1, 1)   
        u_sensor_d = u_ref[:, idx_d].reshape(-1, 1)     

        pt_t_s02 = Variable(torch.from_numpy(t_sensor).float(),    requires_grad=False).to(device)
        pt_x_s02 = Variable(torch.full_like(pt_t_s02, 0.2),        requires_grad=False).to(device)
        pt_u_s02 = Variable(torch.from_numpy(u_sensor_02).float()+torch.randn(*u_sensor_02.shape)*P_bruit, requires_grad=False).to(device)

        pt_t_sd = Variable(torch.from_numpy(t_sensor).float(),    requires_grad=False).to(device)
        pt_x_sd = Variable(torch.full_like(pt_t_sd, dist_capteurs),        requires_grad=False).to(device)
        pt_u_sd = Variable(torch.from_numpy(u_sensor_d).float()+torch.randn(*u_sensor_d.shape)*P_bruit, requires_grad=False).to(device)

        net = self.net
        mse = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-2, weight_decay=1.0e-100)

        for epoch in range(N_iter):
            optimizer.zero_grad()

            pt_x_colloc = Variable(torch.Tensor(100, 1).uniform_(0.0, 2.0), requires_grad=True).to(device)
            pt_t_colloc = Variable(torch.Tensor(100, 1).uniform_(0.0, 1.0), requires_grad=True).to(device)
            f_colloc = self.f(pt_x_colloc, pt_t_colloc)
            mse_f = mse(f_colloc, torch.zeros_like(f_colloc))

            pt_x_ic  = Variable(torch.Tensor(100,1).uniform_(0.0, 2.0), requires_grad=True).to(device)
            pt_t_ic  = Variable(torch.zeros(100,1), requires_grad=True).to(device)
            f_ic     = self.f_ic(pt_x_ic, pt_t_ic)
            mse_ic   = mse(f_ic, torch.zeros_like(f_ic))

            u_pred_02 = net(pt_x_s02, pt_t_s02)
            mse_s02   = mse(u_pred_02, pt_u_s02)
            pt_u_s02 = Variable(torch.from_numpy(u_sensor_02).float()+torch.randn(*u_sensor_02.shape)*P_bruit, requires_grad=False).to(device)

            u_pred_d = net(pt_x_sd, pt_t_sd)
            mse_d    = mse(u_pred_d, pt_u_sd)
            pt_u_sd = Variable(torch.from_numpy(u_sensor_d).float()+torch.randn(*u_sensor_d.shape)*P_bruit, requires_grad=False).to(device)

            loss = mse_ic + mse_s02 + mse_d + mse_f*0.01

            loss.backward()
            optimizer.step()
    
            if epoch <= 10 or epoch == 25 or epoch == 50 or epoch % 100 == 0:
                with torch.autograd.no_grad():
                    print(epoch, "Loss:", loss.data)
            if loss.data < critere_loss:
                print(f"Convergence atteinte à l'epoch {epoch} avec une perte de {loss.data:.6f}")
                break
        '''
            if epoch == N_iter - 1:
                self.plot_forward()
        '''
    def save_reference(self, path='reference_clean.npz'):
        x = np.arange(0, 2, 0.02)
        t = np.arange(0, 1, 0.02)
        ms_x, ms_t = np.meshgrid(x, t)
        x_all = torch.from_numpy(np.ravel(ms_x).reshape(-1, 1)).float().to(device)
        t_all = torch.from_numpy(np.ravel(ms_t).reshape(-1, 1)).float().to(device)

        with torch.no_grad():
            u = self.net(x_all, t_all).cpu().numpy()

        np.savez(path, ms_x=ms_x, ms_t=ms_t, u=u.reshape(ms_x.shape))
        torch.save(self.net.state_dict(), path.replace('.npz', '.pth'))
        print(f"Référence sauvegardée : {path}")
    
    def compare_with_reference(self, path='reference_clean.npz'):
        ref = np.load(path)
        ms_x, ms_t = ref['ms_x'], ref['ms_t']
        u_clean = ref['u']

        x_all = torch.from_numpy(np.ravel(ms_x).reshape(-1, 1)).float().to(device)
        t_all = torch.from_numpy(np.ravel(ms_t).reshape(-1, 1)).float().to(device)

        with torch.no_grad():
            u_noisy = self.net(x_all, t_all).cpu().numpy().reshape(ms_x.shape)

        diff = np.abs(u_noisy - u_clean)
        print(f"Erreur max   : {diff.max():.6f}")
        print(f"Erreur moyenne: {diff.mean():.6f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                                subplot_kw={'projection': '3d'})
        for ax, data, title in zip(axes,
            [u_clean, u_noisy, diff],
            ['Référence (clean)', 'Capteurs avec physique', 'Différence absolue']):
            surf = ax.plot_surface(ms_x, ms_t, data, cmap=cm.coolwarm)
            fig.colorbar(surf, ax=ax, shrink=0.4)
            ax.set_title(title)
            ax.set_xlabel('x')
            ax.set_ylabel('t')
        def on_move(event):
            if event.inaxes in axes:
                for ax in axes:
                    if ax != event.inaxes:
                        ax.view_init(
                            elev=event.inaxes.elev,
                            azim=event.inaxes.azim
                        )
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.tight_layout()
        plt.show()

class FCN(torch.nn.Module):
    def __init__(self):
      super(FCN, self).__init__()
      self.hidden_layer1 = torch.nn.Linear(2,100)
      self.hidden_layer2 = torch.nn.Linear(100,10)
      self.output_layer = torch.nn.Linear(10,1)

    def forward(self, x,t):
      inputs = torch.cat([x,t],axis=1) # on concatene pour avoir une seule variable
      layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
      layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
      output = self.output_layer(layer2_out) ## pour une regression, pas d'activation apres la derniere couche
      return output
    



distance_levels = np.linspace(0, 1.8, 20).tolist()
N_SWEEPS = 5
all_errors = np.zeros((N_SWEEPS, len(distance_levels)))

for sweep_idx in range(N_SWEEPS):
    print(f"\n{'='*50}")
    print(f"  SWEEP {sweep_idx + 1} / {N_SWEEPS}")
    print(f"{'='*50}")
    
    for i, dist_capteurs in enumerate(distance_levels):
        print(f"\n  === dist_capteurs = {dist_capteurs:.4f} ===")
        
        net = FCN().to(device)
        heat_equation = forward_problem(net)
        
        heat_equation.solve_with_sensors(N_iter=10000, dist_capteurs=dist_capteurs+0.02, critere_loss=1e-3)
        
        ref = np.load('reference_clean.npz')
        ms_x, ms_t = ref['ms_x'], ref['ms_t']
        u_clean = ref['u']
        
        x_all = torch.from_numpy(np.ravel(ms_x).reshape(-1, 1)).float().to(device)
        t_all = torch.from_numpy(np.ravel(ms_t).reshape(-1, 1)).float().to(device)
        
        with torch.no_grad():
            u_pred = net(x_all, t_all).cpu().numpy().reshape(ms_x.shape)
        
        err_mean = np.abs(u_pred - u_clean).mean()
        all_errors[sweep_idx, i] = err_mean
        print(f"  → Erreur moyenne : {err_mean:.6f}")

mean_errors = all_errors.mean(axis=0)
std_errors  = all_errors.std(axis=0)

plt.figure(figsize=(8, 5))

for sweep_idx in range(N_SWEEPS):
    plt.plot(distance_levels, all_errors[sweep_idx], 'o-',
             color='steelblue', linewidth=1, markersize=4, alpha=0.25)

plt.fill_between(distance_levels,
                 mean_errors - std_errors,
                 mean_errors + std_errors,
                 color='steelblue', alpha=0.2, label='± 1 std')
plt.plot(distance_levels, mean_errors, 'o-',
         color='steelblue', linewidth=2.5, markersize=7, label=f'Moyenne ({N_SWEEPS} sweeps)')

plt.xlabel('Distance entre les capteurs (x=0.2 et x=d)')
plt.ylabel('Erreur moyenne |u_pred - u_clean|')
plt.title('Erreur moyenne en fonction de la distance entre les capteurs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('erreur_vs_bruit.png', dpi=150)
plt.show()