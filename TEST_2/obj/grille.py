"""
Version optimisée de grille.py — API identique à l'originale (drop-in).

Optimisations :
  1. SolveurPoisson :
     - noyau de Green précalculé UNE SEULE FOIS (avant : exp/sqrt sur la
       grille (2N)² à chaque appel de solve()),
     - rfft2/irfft2 (FFT réelle : ~2x moins de calcul que fft2 complexe),
     - multithread via scipy.fft (workers = tous les cœurs),
     - backend pyFFTW activé automatiquement s'il est installé,
     - buffer de padding réutilisé (zéro allocation par pas de temps).
  2. maj_rho : dépôt CIC par np.bincount (5-10x plus rapide que np.add.at),
     ou par boucle numba si numba est installé.
  3. cic_weights : indices clippés -> plus d'IndexError possible pour une
     particule exactement sur le bord +L (bug latent de la version originale).
  4. interpolation : gather vectorisé via indices aplatis (np.take).
"""

from numpy.fft import fftfreq  # compat éventuelle
import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.fft as sfft

_WORKERS = os.cpu_count() or 1

# ---- backend pyFFTW (optionnel, accélère encore les FFT) ----
try:
    import pyfftw
    import pyfftw.interfaces.scipy_fft as _pyfftw_backend
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60.0)
    sfft.set_global_backend(_pyfftw_backend)
    _FFT_BACKEND = "pyfftw"
except Exception:
    _FFT_BACKEND = "scipy (pocketfft)"

# ---- numba (optionnel) pour le dépôt CIC ----
try:
    from numba import njit
    _HAS_NUMBA = True

    @njit(cache=True, fastmath=True)
    def _deposit_cic_numba(rho, i0, j0, i1, j1, w00, w10, w01, w11, m):
        for p in range(i0.size):
            mp = m[p]
            rho[i0[p], j0[p]] += mp * w00[p]
            rho[i1[p], j0[p]] += mp * w10[p]
            rho[i0[p], j1[p]] += mp * w01[p]
            rho[i1[p], j1[p]] += mp * w11[p]
except Exception:
    _HAS_NUMBA = False


class Grille:
    def __init__(self, N: int, L: float, TN: bool, MN: bool, G: float, M_BH, soft):
        self.N = N
        self.L = L
        self.G = G
        self.TN = TN
        self.MN = MN

        self.dx = 2 * self.L / (self.N - 1)
        self.soft = soft
        self.soft_BH = self.soft

        x = (np.arange(self.N) + 0.5) * self.dx - self.L
        self.X, self.Y = np.meshgrid(x, x, indexing="ij")

        # rayon plan (disque mince)
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.soft**2)

        self.rho = np.zeros((self.N, self.N), dtype=float)
        self.phi = np.zeros((self.N, self.N), dtype=float)

        # Trou noir
        self.BH_step0 = None          # step de déclenchement (None = pas encore déclenché)
        self.N_grow = 10_000
        self.seuil_mode = 0.15        # amplitude du Mode 2 qui déclenche la croissance
        self.M_BH_final = M_BH
        self.M_BH = 0.0
        self.R_BH = np.sqrt(self.X**2 + self.Y**2 + self.soft_BH**2)

        # précalculs (évite de refaire 1/R et 1/R² à chaque maj_BH)
        self._inv_R_BH = 1.0 / self.R_BH
        self._inv_R_BH2 = self._inv_R_BH * self._inv_R_BH
        self._inv_R_BH3 = self._inv_R_BH2 * self._inv_R_BH

        self.phi_BH = -self.G * self.M_BH * self._inv_R_BH
        self.g_BH = -self.G * self.M_BH * self._inv_R_BH2

        # Disque / matière noire 2D mince
        self.R_MN = 55.0 #7.0
        self.rho_0_MN = 1e-5
        self.v_h = 0.40

        self.rho_MN = self.v_h**2 * (self.R_MN**2 - 2*self.R*self.R_MN) / (self.R**2 + self.R_MN**2)**2
        self.M_h = np.sum(self.rho_MN) * self.dx**2

        print(f"Masse de matière noire : {self.M_h}")
        print(f"Masse trou noir : {self.M_BH}")

        self.phi_MN = 0.5 * self.v_h**2 * np.log(self.R**2 + self.R_MN**2)
        self.g_MN = -self.v_h**2 * self.R / (self.R**2 + self.R_MN**2)
        self.gx_MN = self.g_MN * self.X / self.R
        self.gy_MN = self.g_MN * self.Y / self.R

        self.phi_0 = self.phi_BH * self.TN + self.phi_MN * self.MN

        # buffer interne pour le dépôt bincount
        self._rho_flat = np.zeros(self.N * self.N, dtype=float)

    def calc_rho_0(self, px_, py_):
        r = np.sqrt(px_**2 + py_**2)
        out = self.R_MN * self.rho_0_MN * self.G / ((1 + r / self.R_MN)**2)
        return out * self.MN

    def cic_weights(self, px, py):
        inv_dx = 1.0 / self.dx

        fx = (px + self.L) * inv_dx
        fy = (py + self.L) * inv_dx

        i = np.floor(fx).astype(np.int64)
        j = np.floor(fy).astype(np.int64)

        tx = fx - i
        ty = fy - j

        # masques de validité du voisin de droite (comme l'original)
        ok_i = (i + 1) < self.N
        ok_j = (j + 1) < self.N

        # clip de sécurité : une particule exactement en +L donnait i1 = N
        # -> IndexError dans la version originale. Le poids associé est nul,
        # donc clipper ne change pas la physique.
        i0 = np.clip(i, 0, self.N - 1)
        j0 = np.clip(j, 0, self.N - 1)
        i1 = np.minimum(i0 + 1, self.N - 1)
        j1 = np.minimum(j0 + 1, self.N - 1)

        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty) * ok_i
        w01 = (1 - tx) * ty * ok_j
        w11 = tx * ty * (ok_i & ok_j)

        return i0, j0, i1, j1, w00, w10, w01, w11

    def interpolation(self, A, x, y, dx, L, N):
        i0, j0, i1, j1, w00, w10, w01, w11 = self.cic_weights(x, y)

        Af = A.ravel()
        n = self.N
        return (w00 * Af.take(i0 * n + j0) +
                w10 * Af.take(i1 * n + j0) +
                w01 * Af.take(i0 * n + j1) +
                w11 * Af.take(i1 * n + j1))

    def _tsc_weight(self, u):
        a = np.abs(u)
        w = np.zeros_like(u)
        m1 = a < 0.5
        m2 = (a >= 0.5) & (a < 1.5)
        w[m1] = 0.75 - u[m1]**2
        w[m2] = 0.5 * (1.5 - a[m2])**2
        return w

    def M_BH_t(self, step) -> float:
        if self.BH_step0 is None:
            return 0.0
        s = step - self.BH_step0
        if s <= 0:
            return 0.0
        if s >= self.N_grow:
            return self.M_BH_final
        f = 0.5 * (1 - np.cos(np.pi * s / self.N_grow))
        return self.M_BH_final * f

    def maj_BH(self, M_BH) -> None:
        self.M_BH = M_BH
        c = -self.G * self.M_BH
        self.phi_BH = c * self._inv_R_BH
        self.g_BH = c * self._inv_R_BH2
        self.phi_0 = self.phi_BH * self.TN + self.phi_MN * self.MN

    def maj_rho(self, px, py, mass):
        """Projette les particules sur la grille (dépôt CIC).
        Identique numériquement à l'original, mais via bincount/numba :
        np.add.at est notoirement lent (boucle C non bufferisée)."""
        mask = (np.abs(px) <= self.L) & (np.abs(py) <= self.L)
        px = px[mask]
        py = py[mask]
        mass = mass[mask]

        i0, j0, i1, j1, w00, w10, w01, w11 = self.cic_weights(px, py)

        n = self.N
        if _HAS_NUMBA:
            self.rho.fill(0.0)
            _deposit_cic_numba(self.rho, i0, j0, i1, j1,
                               w00, w10, w01, w11, np.ascontiguousarray(mass))
        else:
            flat = np.concatenate((i0 * n + j0, i1 * n + j0,
                                   i0 * n + j1, i1 * n + j1))
            w = np.concatenate((mass * w00, mass * w10,
                                mass * w01, mass * w11))
            self.rho = np.bincount(flat, weights=w,
                                   minlength=n * n).reshape(n, n)

        self.rho /= self.dx**2

    def maj_phi(self, phi):
        self.phi = phi + self.phi_0

    def calcul_G(self, gx, gy):
        if self.TN or self.MN:
            if self.TN:
                c = -self.G * self.M_BH
                gx += c * self.X * self._inv_R_BH3
                gy += c * self.Y * self._inv_R_BH3
            if self.MN:
                gx += self.gx_MN
                gy += self.gy_MN
        return gx, gy

    def phi_ext_part(self, px, py):
        """Potentiel extérieur (TN + halo) évalué aux positions des particules.
        Analytique : pas d'interpolation, valable même hors grille."""
        phi = np.zeros_like(px)
        R2 = px**2 + py**2
        if self.TN:
            phi += -self.G * self.M_BH / np.sqrt(R2 + self.soft_BH**2)
        if self.MN:
            phi += 0.5 * self.v_h**2 * np.log(R2 + self.R_MN**2)
        return phi

    def viriel_ext_part(self, px, py):
        """Terme de Clausius r·g_ext par particule (pour le viriel)."""
        v = np.zeros_like(px)
        R2 = px**2 + py**2
        if self.TN:
            v += -self.G * self.M_BH / np.sqrt(R2 + self.soft_BH**2)  # 1/r : r·g = φ
        if self.MN:
            v += -self.v_h**2 * R2 / (R2 + self.R_MN**2)
        return v
    
    def ratio_W(self, px, py, mass, phi_self):
        """R_W = W_particules / W_grille, sur l'auto-gravité seule.
        Doit rester ~1 ; une dérive signale une incohérence CIC/interpolation.
        px, py, mass : le MÊME jeu de particules que celui passé à maj_rho."""
        mask = (np.abs(px) <= self.L) & (np.abs(py) <= self.L)
    
        W_grid = 0.5 * np.sum(self.rho * phi_self) * self.dx**2
    
        phi_i = self.interpolation(phi_self, px[mask], py[mask],
                                   self.dx, self.L, self.N)
        W_part = 0.5 * np.sum(mass[mask] * phi_i)
    
        return W_part / W_grid

class SolveurPoisson:
    """Solveur FFT optimisé.

    Le noyau de Green, les grilles k et les indices de crop sont précalculés
    une seule fois. solve() n'effectue plus que :
        1 rfft2 + 3 irfft2 + 3 produits spectraux.
    (au lieu de : reconstruction du noyau + 4 FFT complexes pleines).
    """

    def __init__(self, g: Grille):
        self.g = g
        self.Nfft = 2 * self.g.N

        Lfft = self.g.L
        d = 2 * Lfft / self.Nfft

        kx = sfft.fftfreq(self.Nfft, d=d) * 2 * np.pi          # axe complet
        ky = sfft.rfftfreq(self.Nfft, d=d) * 2 * np.pi         # demi-axe (rfft)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")  # (Nfft, Nfft//2+1)
        self.K2 = self.KX**2 + self.KY**2

        # ---- noyau de Green précalculé (gros gain : avant recalculé à chaque pas)
        k = np.sqrt(self.K2)
        k[0, 0] = 1.0
        self.Green = -2 * np.pi * self.g.G * np.exp(-k * self.g.soft) / k
        self.Green[0, 0] = 0.0

        # noyaux gradients précombinés : G_gx = -i kx * Green
        self._Kgx = -1j * self.KX * self.Green
        self._Kgy = -1j * self.KY * self.Green
        # mode de Nyquist annulé pour la dérivée impaire : équivalent exact
        # du np.real(ifft2(...)) de la version originale, qui supprimait
        # implicitement sa contribution imaginaire.
        self._Kgx[self.Nfft // 2, :] = 0.0
        self._Kgy[:, -1] = 0.0

        # ---- padding : buffer réutilisé + slices précalculées
        n = self.g.N
        self._pad = np.zeros((self.Nfft, self.Nfft), dtype=float)
        self._s_in = (self.Nfft - n) // 2                    # insertion
        s = self.Nfft // 2 - n // 2                          # crop (comme l'original)
        self._crop = (slice(s, s + n), slice(s, s + n))

        print(f"[SolveurPoisson] backend FFT : {_FFT_BACKEND}, "
              f"{_WORKERS} threads, rfft2 {self.Nfft}x{self.Nfft}")

    def cic_window(self):
        kx = self.KX * self.g.dx
        ky = self.KY * self.g.dx
        wx = np.sinc(kx / (2 * np.pi))**2
        wy = np.sinc(ky / (2 * np.pi))**2
        return wx * wy

    def solve(self, rho=None):
        if rho is None:
            rho = self.g.rho

        n = self.g.N
        s = self._s_in
        pad = self._pad
        pad.fill(0.0)
        pad[s:s + n, s:s + n] = rho

        # FFT réelle multithread
        rho_k = sfft.rfft2(pad, workers=_WORKERS)

        # produits spectraux avec noyaux précalculés
        phi_k = self.Green * rho_k

        crop = self._crop
        phi = sfft.irfft2(phi_k, s=pad.shape, workers=_WORKERS)[crop].copy()
        gx = sfft.irfft2(self._Kgx * rho_k, s=pad.shape, workers=_WORKERS)[crop].copy()
        gy = sfft.irfft2(self._Kgy * rho_k, s=pad.shape, workers=_WORKERS)[crop].copy()

        return phi, gx, gy
    
    
    def Epot(self, px, py, mass):
        phi_i = self.g.interpolation(self.g.phi, px, py,
                                     self.g.dx, self.g.L, self.g.N)
        phi_i = np.nan_to_num(phi_i, nan=0.0)
        return mass * phi_i


class BulbeDiagnostic:

    def __init__(self, grid, solver):
        self.g = grid
        self.s = solver

    def radial_profile(self, Phi):
        x = self.g.X
        y = self.g.Y
        r = np.sqrt(x**2 + y**2)

        gy, gx = np.gradient(-Phi, self.g.dx)
        gr = (gx * x + gy * y) / (r + 1e-10)

        bins = np.linspace(0, self.g.L, 50)
        centers = 0.5 * (bins[:-1] + bins[1:])

        # binning radial vectorisé (avant : boucle python sur les bins)
        idx = np.digitize(r.ravel(), bins) - 1
        valid = (idx >= 0) & (idx < len(centers))
        sums = np.bincount(idx[valid], weights=gr.ravel()[valid],
                           minlength=len(centers))
        counts = np.bincount(idx[valid], minlength=len(centers))
        g_r = np.divide(sums, counts, out=np.zeros_like(sums),
                        where=counts > 0)
        return centers, g_r

    def compute_fields(self):
        Phi_d = self.s.solve()
        Phi_h = self.g.phi_MN
        Phi_BH = self.g.phi_BH
        Phi_t = self.s.solve(self.g.rho)
        return Phi_BH, Phi_d, Phi_h, Phi_t

    def rotation_curve(self):
        gy, gx = np.gradient(-self.g.phi, self.g.dx)
        g_r = (gx * self.g.X + gy * self.g.Y) / (self.g.R + 1e-10)
        v_c = np.sqrt(np.maximum(self.g.R * g_r, 0))
        return self.g.R, v_c

    # ---- helper interne : binning radial vectorisé ----
    def _radial_bin(self, values_flat, r_flat, bins):
        nc = len(bins) - 1
        idx = np.digitize(r_flat, bins) - 1
        valid = (idx >= 0) & (idx < nc)
        sums = np.bincount(idx[valid], weights=values_flat[valid], minlength=nc)
        counts = np.bincount(idx[valid], minlength=nc)
        return np.divide(sums, counts, out=np.zeros(nc), where=counts > 0)

    def bulge_influence_radius(self):
        _, _, Phi_h, _ = self.compute_fields()

        r = np.sqrt(self.g.X**2 + self.g.Y**2)
        _, gx_d, gy_d = self.s.solve()

        def radial_component(gx, gy):
            return (gx * self.g.X + gy * self.g.Y) / (r + 1e-10)

        g_d = radial_component(gx_d, gy_d)
        g_h = radial_component(self.g.gx_MN, self.g.gy_MN)

        bins = np.linspace(0, self.g.L, 200)
        centers = 0.5 * (bins[:-1] + bins[1:])

        r_flat = r.ravel()
        gd_r = self._radial_bin(g_d.ravel(), r_flat, bins)
        gh_r = self._radial_bin(g_h.ravel(), r_flat, bins)
        gB_r = -self.g.G * self.g.M_BH / (self.g.soft_BH + centers)**2

        g_tot = gd_r + gh_r + gB_r + 1e-12
        eta_d = gd_r / g_tot
        eta_h = gh_r / g_tot
        eta_BH = gB_r / g_tot

        diff_bh = np.abs(eta_d - eta_h)
        r_dh = centers[np.argmin(diff_bh)]
        r_BH = centers[np.argmax(eta_BH < 0.5)]

        plt.semilogy(centers, np.abs(gd_r), label='disque')
        if self.g.TN:
            plt.semilogy(centers, np.abs(gB_r), label='BH')
        if self.g.MN:
            plt.semilogy(centers, np.abs(gh_r), label='Matière noire')
        plt.title('g')
        plt.legend()
        plt.show()

        return {
            "Rd/h_influence": r_dh,
            "R_BH_influence": r_BH
        }

    def bulge_influence_radius_BH(self):
        _, gx_d, gy_d = self.s.solve()

        def radial_component(gx, gy):
            return (gx * self.g.X + gy * self.g.Y) / (self.g.R + 1e-10)

        bins = np.linspace(0, self.g.L, self.g.N)
        centers = 0.5 * (bins[:-1] + bins[1:])

        g_d = radial_component(gx_d, gy_d)
        gB_r = -self.g.G * self.g.M_BH / (self.g.soft_BH + centers)**2

        gd_r = self._radial_bin(g_d.ravel(), self.g.R.ravel(), bins)

        g_tot = gd_r + gB_r + 1e-12
        eta_d = gd_r / g_tot
        eta_BH = gB_r / g_tot

        diff_bd = np.abs(eta_d - 0.5)
        r_d = centers[np.argmin(diff_bd)]
        r_BH = centers[np.argmax(eta_BH < 0.5)]

        plt.semilogy(centers, np.abs(gd_r), label='disque')
        if self.g.TN:
            plt.semilogy(centers, np.abs(gB_r), label='BH')
        plt.title('g')
        plt.legend()
        plt.show()

        return {
            "R_BH_influence": r_BH,
            "R_d_influence": r_d
        }
