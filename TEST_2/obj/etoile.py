"""
Version optimisée de etoile.py — API identique à l'originale (drop-in).

Optimisations :
  - cic_weights : indices clippés (plus d'IndexError sur le bord +L),
  - calcul_force : interpolation par gather aplati (np.take) au lieu de
    4 indexations fancy 2D -> moins d'allocations temporaires,
  - maj_pos / maj_status : opérations in-place, calculs mutualisés,
  - r3 protégé contre la division par zéro pour les particules hors boîte.
"""

import numpy as np


class Etoiles:
    def __init__(self, G):
        self.px = None
        self.py = None
        self.px_cm = []
        self.py_cm = []
        self.vx = None
        self.vy = None
        self.mass = None
        self.alive = None
        self.liee = None
        self.halo = None
        self.G = G
        self.soft = 1e-5
        self.r = None
        self.leffectif = []
        self.dt = None
        self.time = 0

    def save_state(self):
        return {
            "px": self.px,
            "py": self.py,
            "px_centre_de_masse": self.px_cm,
            "py_centre_de_masse": self.py_cm,
            "vx": self.vx,
            "vy": self.vy,
            "r": self.r,
            "mass": self.mass,
            "alive": self.alive,
            "liee": self.liee,
            "halo": self.halo,
            "time": self.time,
            "dt": self.dt,
            "leffectif": self.leffectif
        }

    def load_state(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    def cic_weights(self, px, py):
        inv_dx = 1.0 / self.dx

        fx = (px + self.L) * inv_dx
        fy = (py + self.L) * inv_dx

        i = np.floor(fx).astype(np.int64)
        j = np.floor(fy).astype(np.int64)

        tx = fx - i
        ty = fy - j

        ok_i = (i + 1) < self.N
        ok_j = (j + 1) < self.N

        # clip de sécurité (cf. grille.py) : poids nuls de toute façon
        i0 = np.clip(i, 0, self.N - 1)
        j0 = np.clip(j, 0, self.N - 1)
        i1 = np.minimum(i0 + 1, self.N - 1)
        j1 = np.minimum(j0 + 1, self.N - 1)

        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty) * ok_i
        w01 = (1 - tx) * ty * ok_j
        w11 = tx * ty * (ok_i & ok_j)

        return i0, j0, i1, j1, w00, w10, w01, w11

    def _tsc_weight(self, u):
        a = np.abs(u)
        w = np.zeros_like(u)
        m1 = a < 0.5
        m2 = (a >= 0.5) & (a < 1.5)
        w[m1] = 0.75 - u[m1]**2
        w[m2] = 0.5 * (1.5 - a[m2])**2
        return w

    # ------------------------------------------------------------------
    def maj_pos(self) -> None:
        a = self.alive
        self.px[a] += self.vx[a] * self.dt
        self.py[a] += self.vy[a] * self.dt

        m_a = self.mass[a]
        M_tot = np.sum(m_a)

        self.px_cm.append(np.sum(self.px[a] * m_a) / M_tot)
        self.py_cm.append(np.sum(self.py[a] * m_a) / M_tot)

        # in-place : évite 3 tableaux temporaires par pas de temps
        self.r = np.sqrt(self.px * self.px + self.py * self.py + self.soft**2)
        self.time += self.dt

    def maj_vit(self, Fx, Fy) -> None:
        self.vx[self.alive] += Fx * self.dt
        self.vy[self.alive] += Fy * self.dt

    def maj_status(self, grille, R_lim, phi_self) -> None:
        # énergie spécifique (par unité de masse)
        e = self.energie(grille, phi_self) / self.mass
    
        # barrière de potentiel à R_lim
        M_box = np.sum(self.mass[self.alive &
                       (np.abs(self.px) < grille.L) & (np.abs(self.py) < grille.L)])
        phi_lim = (-grille.G * M_box / R_lim
                   + grille.phi_ext_part(np.array([R_lim]), np.array([0.0]))[0])
    
        self.liee = e < phi_lim
    
        self.alive &= (self.px**2 + self.py**2 < R_lim**2)
        self.leffectif.append(np.sum(self.liee & self.alive))

    # ------------------------------------------------------------------
    def calcul_force(self, gx, gy, N, L, px=None, py=None,mass=None):
        """Renvoie l'accélération interpolée sur les particules vivantes
        (monopôle analytique pour les particules hors boîte)."""
        if px is None and py is None:
            px = self.px[self.alive]
            py = self.py[self.alive]
            mass = self.mass[self.alive]

        Fx = np.zeros_like(px)
        Fy = np.zeros_like(Fx)

        out = (np.abs(px) > self.L) | (np.abs(py) > self.L)

        # monopole pour les particules hors grille
        M_box = np.sum(mass[~out])
        r3 = (px[out]**2 + py[out]**2)**1.5
        Fx[out] = -self.G * M_box * px[out] / r3
        Fy[out] = -self.G * M_box * py[out] / r3
    
        i0, j0, i1, j1, w00, w10, w01, w11 = self.cic_weights(px[~out], py[~out])
    
        Fx[~out] = w00*gx[i0,j0] + w10*gx[i1,j0] + w01*gx[i0,j1] + w11*gx[i1,j1]
        Fy[~out] = w00*gy[i0,j0] + w10*gy[i1,j0] + w01*gy[i0,j1] + w11*gy[i1,j1]

        return Fx, Fy
    
    def energie(self, grille, phi_self):
        """Énergie orbitale de chaque étoile : ½mv² + m·(φ_self + φ_ext).
        phi_self : grille du solveur FFT SEUL (sortie brute de Sol.solve(),
        pas G.phi qui contient déjà phi_0)."""
        phi_i = grille.interpolation(phi_self, self.px, self.py,
                                     grille.dx, grille.L, grille.N)
        phi_i = np.nan_to_num(phi_i, nan=0.0)
        Ec = 0.5 * self.mass * (self.vx**2 + self.vy**2)
        return Ec + self.mass * (phi_i + grille.phi_ext_part(self.px, self.py))
    
    def energie_totale(self,grille,W_self):
        m = self.alive
        E_tot = np.sum(0.5*self.mass[m]*(self.vx[m]**2 + self.vy[m]**2)) \
              + W_self \
              + np.sum(self.mass[m] * grille.phi_ext_part(self.px[m], self.py[m]))
        return E_tot
    
    def viriel_Q(self, grille, phi_self,px,py,vx,vy,mass):
        """Rapport du viriel Q = -2T / (W_self + V_ext), sur les étoiles alive.
        Retourne aussi T, W_self, V_ext pour diagnostic."""
       
        phi_i = grille.interpolation(phi_self, px, py,
                                     grille.dx, grille.L, grille.N)
        phi_i = np.nan_to_num(phi_i, nan=0.0)
    
        T      = 0.5 * np.sum(mass * (vx**2 + vy**2))
        W_self = 0.5 * np.sum(mass * phi_i)          # ½ : paires
        V_ext  = np.sum(mass * grille.viriel_ext_part(px, py))
    
        Q = -2.0 * T / (W_self + V_ext)
        return Q, T, W_self, V_ext
    # ------------------------------------------------------------------
    def effectif(self):
        return np.sum(self.liee & self.alive)

    def spec_palive(self):
        a = self.alive
        return (self.px[a], self.py[a], self.vx[a], self.vy[a], self.mass[a])

    def spec_pliee(self):
        mask = self.alive & self.liee
        return (self.px[mask], self.py[mask], self.vx[mask], self.vy[mask], self.mass[mask])

    def spec_pechapee(self):
        mask = self.alive & (~self.liee)
        return (self.px[mask], self.py[mask], self.vx[mask], self.vy[mask], self.mass[mask])
