#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def compute_ISI_strict(g002, g100, g101):
    x = g101 - g002
    y = g100 - g002
    gbar = (x + y) / 2.0
    numerator = (x - gbar) + (y - gbar)
    return np.sqrt(numerator) / abs(gbar)


def compute_delta_wulff(g002, g100, g101):
    delta_gamma_rel = abs(g101 - g100) / g100
    return 8.0 * delta_gamma_rel


def compute_nucleation_params(g002, g100, g101, ISI):
    gamma_avg = (g002 + g100 + g101) / 3.0
    gamma_eff = gamma_avg * (1.0 + 2.0 * ISI)
    r_crit = 2.0 * gamma_eff / Delta_mu
    Delta_G_star = (16.0 * np.pi / 3.0) * (gamma_eff ** 3) / (Delta_mu)
    n_seeds_float = 6.0 * (1.0 / gamma_eff) + 3.0
    n_seeds = max(3, int(n_seeds_float))
    seed_radius = max(1, int(r_crit / dx * 1.2))
    return {
        'gamma_eff': gamma_eff,
        'r_crit': r_crit,
        'Delta_G_star': Delta_G_star,
        'n_seeds': n_seeds,
        'seed_radius': seed_radius,
    }


def compute_pf_params(mol_name):
    delta = compute_delta_wulff(g002, g100, g101)
    gamma_avg = (g002 + g100 + g101) / 3.0
    nuc = compute_nucleation_params(g002, g100, g101, isi, dx=dx)
    noise_amp = 0.03 + 0.10 * isi
    return {
        'name': mol_name,
        'ISI': isi,
        'delta': delta,
        'noise_amp': noise_amp,
        'gamma_avg': gamma_avg,
        'gamma': {'002': g002, '100': g100, '101': g101},
        'ads': ads,
        'nucleation': nuc,
    }

class TorqueNonOverlappingPF:
    def __init__(self, mol_name):
        params = compute_pf_params(mol_name, dx=dx)
        nuc = params['nucleation']

        self.mol_name = mol_name
        self.ISI = params['ISI']
        self.delta = params['delta']
        self.noise_amp = params['noise_amp']
        self.gamma_avg = params['gamma_avg']
        self.gamma_dict = params['gamma']
        self.ads = params['ads']

        self.n_seeds = nuc['n_seeds']
        self.seed_radius = nuc['seed_radius']
        self.gamma_eff = nuc['gamma_eff']
        self.r_crit = nuc['r_crit']
        self.Delta_G_star = nuc['Delta_G_star']

        self.omega = 6
        self.nx, self.ny = nx, ny
        self.dx = dx
        self.L_sigma = L_sigma
        self.L_eta = L_eta
        self.D_ion = D_ion
        self.k_dep = k_dep

        self.L_tau = L_tau * (1.0 + 3.0 * self.delta)
        self.alpha_eta = alpha_eta
        self.lambda_torque = lambda_torque
        self.noise_eta = 0.03 + 0.04 * self.ISI

        self.kappa0 = 4.8 * self.gamma_avg
        self.W = 1.2 * self.gamma_avg
        self.drive_base = 0.55
        self.epsilon = np.sqrt(self.kappa0 / self.W)

        self._init_fields()
        self._init_concentration()
        self.time = 0.0
        self.history = {
            't': [], 'vol': [], 'rough': [], 'max_height': [], 'coverage': [],
            'c_avg': [], 'c_min': [],
            'eta_mean': [], 'eta_std': [], 'torque_mean': [], 'wulff_torque_mean': [],
            'couple_energy': []
        }

        dt_max = dx**2 / (4.0 * self.L_sigma * self.kappa0**2)
        dt_used = 0.003

    def _circular_diff(self, a, b):
        return np.mod(a - b + np.pi, 2*np.pi) - np.pi

    def _h_prime(self, xi):
        return 30.0 * xi**2 * (1.0 - xi)**2

    def _init_fields(self):
        self.xi = np.zeros((self.ny, self.nx))
        self.eta = np.zeros((self.ny, self.nx))

        margin = 15
        avail_width = self.nx - 2 * margin
        min_gap = 2
        max_seeds = max(1, int(avail_width / (2 * self.seed_radius + min_gap)) + 1)

        actual_n_seeds = min(self.n_seeds, max_seeds)

        self.actual_n_seeds = actual_n_seeds

        if actual_n_seeds > 1:
            x_positions = np.linspace(margin, self.nx - margin, actual_n_seeds, dtype=int)
        else:
            x_positions = np.array([self.nx // 2])

        self.seed_centers = []
        for idx, x_pos in enumerate(x_positions):
            y_max = min(2 + self.seed_radius // 2, self.ny - self.seed_radius - 1)
            y_pos = np.random.randint(2, max(3, y_max + 1))

            self.seed_centers.append((x_pos, y_pos))
            rr = self.seed_radius
            y, x = np.mgrid[:self.ny, :self.nx]
            mask = (x - x_pos)**2 + (y - y_pos)**2 <= rr**2
            self.xi[mask] = 0.9

            eta_seed = np.random.uniform(-np.pi, np.pi)
            self.eta[mask] = eta_seed

    def _init_concentration(self):
        self.c = np.ones((self.ny, self.nx))

    def _gradients(self, field):
        dx = self.dx
        fx = np.zeros_like(field)
        fy = np.zeros_like(field)
        fx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2*dx)
        fy[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2*dx)
        fx[:, 0] = 0.0
        fx[:, -1] = 0.0
        fy[0, :] = (field[1, :] - field[0, :]) / dx
        fy[-1, :] = (field[-1, :] - field[-2, :]) / dx
        return fx, fy

    def _gradients_circular(self, field):
        dx = self.dx
        fx = np.zeros_like(field)
        fy = np.zeros_like(field)
        fx[:, 1:-1] = self._circular_diff(field[:, 2:], field[:, :-2]) / (2*dx)
        fx[:, 0] = self._circular_diff(field[:, 1], field[:, 0]) / dx
        fx[:, -1] = self._circular_diff(field[:, -1], field[:, -2]) / dx
        fy[1:-1, :] = self._circular_diff(field[2:, :], field[:-2, :]) / (2*dx)
        fy[0, :] = self._circular_diff(field[1, :], field[0, :]) / dx
        fy[-1, :] = self._circular_diff(field[-1, :], field[-2, :]) / dx
        return fx, fy

    def _laplacian(self, field):
        return ndimage.laplace(field) / (self.dx ** 2)

    def _laplacian_circular(self, field):
        dx = self.dx
        lap = np.zeros_like(field)
        d2x = (self._circular_diff(field[:, 2:], field[:, 1:-1])
               - self._circular_diff(field[:, 1:-1], field[:, :-2]))
        d2y = (self._circular_diff(field[2:, :], field[1:-1, :])
               - self._circular_diff(field[1:-1, :], field[:-2, :]))
        lap[1:-1, 1:-1] = (d2x[1:-1, :] + d2y[:, 1:-1]) / dx**2 
        lap[0, :] = 2.0 * self._circular_diff(field[1, :], field[0, :]) / dx**2 
        lap[-1, :] = 2.0 * self._circular_diff(field[-1, :], field[-2, :]) / dx**2
        lap[:, 0] = 2.0 * self._circular_diff(field[:, 1], field[:, 0]) / dx**2
        lap[:, -1] = 2.0 * self._circular_diff(field[:, -1], field[:, -2]) / dx**2
        return lap

    def _kappa_fields(self):
        cos_term = (1.0 + self.delta * np.cos(self.omega * self.eta)) / (1.0 + self.delta)
        kappa_raw = self.kappa0 * (0.6 + 0.7 * cos_term)
        kappa = np.maximum(kappa_raw, 0.01 * self.kappa0)
        kappa_prime = -self.kappa0 * 0.8 * self.delta * self.omega * np.sin(self.omega * self.eta) / (1.0 + self.delta)
        return kappa, kappa_prime

    def _interface_normal(self):
        grad_x, grad_y = self._gradients(self.xi)
        theta = np.arctan2(grad_y, grad_x)
        return theta, grad_x, grad_y

    def _xi_chemical_potential(self, kappa, theta, eta):
        xi = self.xi
        gp = self.W * 2.0 * xi * (1.0 - xi) * (1.0 - 2.0 * xi)

        grad_x, grad_y = self._gradients(xi)
        flux_x = kappa**2 * grad_x
        flux_y = kappa**2 * grad_y

        dx = self.dx
        div = np.zeros_like(xi)
        div[:, 1:-1] = (flux_x[:, 2:] - flux_x[:, :-2]) / (2*dx)
        div[:, 0] = (flux_x[:, 1] - flux_x[:, 0]) / dx
        div[:, -1] = (flux_x[:, -1] - flux_x[:, -2]) / dx
        div[1:-1, :] += (flux_y[2:, :] - flux_y[:-2, :]) / (2*dx)
        div[0, :] += (flux_y[1, :] - flux_y[0, :]) / dx
        div[-1, :] += (flux_y[-1, :] - flux_y[-2, :]) / dx

        hp = self._h_prime(xi)
        delta_circ = self._circular_diff(theta, eta)
        couple_term = 0.5 * self.lambda_torque * hp * delta_circ**2

        return gp - div + couple_term

    def _torque_terms(self, theta, kappa, kappa_prime):
        xi = self.xi
        hp = self._h_prime(xi)

        delta_circ = self._circular_diff(theta, self.eta)
        torque_lock = self.lambda_torque * hp * delta_circ

        grad_x, grad_y = self._gradients(xi)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2)
        grad_norm_reg = np.minimum(grad_norm, 1.0 / self.epsilon)
        grad_norm_sq = grad_norm_reg**2
        torque_wulff = -kappa * kappa_prime * grad_norm_sq

        return torque_lock, torque_wulff

    def _concentration_step(self, dt):
        lap_c = self._laplacian(self.c)
        hp = self._h_prime(self.xi)
        reaction = self.k_dep * hp * self.c * self.drive_base
        self.c += dt * (self.D_ion * lap_c - reaction)
        self.c = np.clip(self.c, 0.05, 1.0)
        self.c[0:2, :] = 1.0

    def step(self, dt):
        self._concentration_step(dt)
        kappa, kappa_prime = self._kappa_fields()
        theta, _, _ = self._interface_normal()
        mu_xi = self._xi_chemical_potential(kappa, theta, self.eta)
        torque_lock, torque_wulff = self._torque_terms(theta, kappa, kappa_prime)

        lap_eta = self._laplacian_circular(self.eta)
        hp_eta = self._h_prime(self.xi)
        noise_eta = self.noise_eta * hp_eta * np.random.randn(*self.eta.shape)

        deta = (self.L_tau * (torque_lock + torque_wulff)
                + self.L_tau * self.alpha_eta * lap_eta
                + noise_eta)

        self.eta += dt * deta
        self.eta = np.mod(self.eta + np.pi, 2*np.pi) - np.pi

        hp_xi = self._h_prime(self.xi)
        noise_xi = self.noise_amp * hp_xi * np.random.randn(*self.xi.shape)

        drive_field = self.drive_base * self.c

        dxi = (-self.L_sigma * mu_xi
               + self.L_eta * hp_xi * drive_field
               + noise_xi)

        self.xi += dt * dxi
        self.xi = np.clip(self.xi, 0.0, 1.0)
        self.xi[0:2, :] = 1.0
        self.xi[-1, :] = 0.0

        self._last_torque_lock = torque_lock
        self._last_torque_wulff = torque_wulff

    def run(self, n_steps, dt, record_intervals):
        frames = {}
        eta_frames = {}
        for i in range(n_steps):
            self.step(dt)
            if i in record_intervals:
                frames[i] = self.xi.copy()
                eta_frames[i] = self.eta.copy()

                vol = np.mean(self.xi)
                surface = np.zeros(self.nx)
                for j in range(self.nx):
                    col = self.xi[:, j] > 0.5
                    if np.any(col):
                        surface[j] = np.max(np.where(col)[0]) * self.dx

                self.history['t'].append(i * dt)
                self.history['vol'].append(vol)
                self.history['rough'].append(np.std(surface))
                self.history['max_height'].append(np.max(surface))
                self.history['coverage'].append(np.mean(self.xi[2:, :] > 0.5))
                self.history['c_avg'].append(np.mean(self.c))
                self.history['c_min'].append(np.min(self.c))
                self.history['eta_mean'].append(np.mean(self.eta))
                self.history['eta_std'].append(np.std(self.eta))

                hp = self._h_prime(self.xi)
                theta, _, _ = self._interface_normal()
                delta_circ = self._circular_diff(theta, self.eta)
                couple_e = 0.5 * self.lambda_torque * np.mean(hp * delta_circ**2)
                self.history['couple_energy'].append(couple_e)

                interface_mask = (self.xi > 0.1) & (self.xi < 0.9)
                if np.any(interface_mask):
                    self.history['torque_mean'].append(np.mean(self._last_torque_lock[interface_mask]))
                    self.history['wulff_torque_mean'].append(np.mean(self._last_torque_wulff[interface_mask]))
                else:
                    self.history['torque_mean'].append(0.0)
                    self.history['wulff_torque_mean'].append(0.0)

        if (n_steps - 1) not in frames:
            frames[n_steps - 1] = self.xi.copy()
            eta_frames[n_steps - 1] = self.eta.copy()
        return frames, eta_frames

def plot_filmstrip(results, time_points, dx, nx, ny, save_path="torque_noseed_filmstrip.pdf"):
    n_rows = len(results)
    n_cols = len(time_points)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 5.0),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.12})
    time_labels = [f"t={t}dt" for t in time_points]
    width = nx * dx
    height = ny * dx
    extent = [0, width, 0, height]

    for row_idx, res in enumerate(results):
        for col_idx, t in enumerate(time_points):
            ax = axes[row_idx, col_idx]
            xi = res["frames"][t]
            im = ax.imshow(xi, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1,
                          extent=extent)
            ax.set_aspect('equal')
            if col_idx == 0:
                actual_seeds = res.get('actual_n_seeds', res['n_seeds'])
                ax.set_ylabel(
                    f"{res['name']}\n(ISI={res['ISI']:.3f}, δ={res['delta']:.3f}, seeds={actual_seeds})",
                    fontsize=8, fontweight='bold', rotation=0, ha='right', va='center', labelpad=120
                )
            else:
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(time_labels[col_idx], fontsize=8, fontweight='bold', pad=6)
            ax.set_xticks([])
            coverage = np.mean(xi[2:, :] > 0.5)
            ax.text(5, height*0.8, f"{coverage:.1%}", fontsize=8, color='black', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\xi$ (Zn Metal)", fontsize=8)
    plt.suptitle(
        "TORQUE: Non-overlapping Seeds + Y-dispersed Nucleation"
        fontsize=8, fontweight='bold', y=0.98
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_eta_overlay(results, time_points, dx, nx, ny, save_path="torque_eta_overlay.pdf"):
    n_rows = len(results)
    n_cols = len(time_points)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5, 5.0),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.12})
    time_labels = [f"t={t}dt" for t in time_points]
    width = nx * dx
    height = ny * dx
    extent = [0, width, 0, height]

    for row_idx, res in enumerate(results):
        for col_idx, t in enumerate(time_points):
            ax = axes[row_idx, col_idx]
            xi = res["frames"][t]
            eta = res["eta_frames"][t]
            im = ax.imshow(xi, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=1,
                          extent=extent, alpha=0.7)
            interface_mask = (xi > 0.2) & (xi < 0.8)
            y_idx, x_idx = np.where(interface_mask)
            if len(y_idx) > 0:
                step = max(1, len(y_idx) // 200)
                y_sub = y_idx[::step]
                x_sub = x_idx[::step]
                eta_sub = eta[y_sub, x_sub]
                u = np.cos(eta_sub)
                v = np.sin(eta_sub)
                ax.quiver(x_sub * dx, y_sub * dx, u, v,
                         color='blue', alpha=0.8, scale=20, width=0.003)
            ax.set_aspect('equal')
            if col_idx == 0:
                ax.set_ylabel(f"{res['name']}\n(ISI={res['ISI']:.3f})",
                             fontsize=8, fontweight='bold', rotation=0, ha='right', va='center', labelpad=80)
            else:
                ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(time_labels[col_idx], fontsize=8, fontweight='bold', pad=6)
            ax.set_xticks([])
    plt.suptitle("Orientation Field η (blue arrows) overlaid on ξ field", fontsize=8, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

