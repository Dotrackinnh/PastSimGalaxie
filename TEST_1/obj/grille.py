from numpy.fft import fft2, ifft2, fftfreq
import numpy as np
import matplotlib.pyplot as plt

class Grille:
    """Objet définissant la grille de la simulation ainsi que les objets
    analytiquement implémentés"""
    def __init__(self,N : int,L:float,TN:bool,MN:bool,G:float,M_BH,soft):
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
    
        # rayon plan
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.soft**2)
        
        self.rho=np.zeros((self.N,self.N),dtype=float)
        self.phi=np.zeros((self.N,self.N),dtype=float)
        
        ### Trou noir
        
        self.BH_step0 = None     # step de déclenchement (None = pas encore déclenché)
        self.N_grow = 2500
        self.seuil_mode = 0.15   # amplitude du Mode 2 qui déclenche la croissance
        self.M_BH_final = M_BH
        self.M_BH = 0.0
        self.R_BH = np.sqrt(self.X**2 + self.Y**2 + self.soft_BH**2)
        self.phi_BH = -self.G * self.M_BH / self.R_BH
        self.g_BH = - self.G*self.M_BH/self.R_BH**2
        
        ### Disque / matière noire 2D mince
        
        self.R_MN = 7.0
        self.rho_0_MN = 1e-5  # densité surfacique centrale
        self.v_h = 0.30       
        
        # densité surfacique
        #self.rho_MN = self.rho_0_MN * (self.R_MN**2) / (self.R**2 + self.R_MN**2)
        self.rho_MN = self.v_h**2 * (self.R_MN**2 - 2*self.R*self.R_MN) / (self.R**2 + self.R_MN**2)**2
        # masse totale approx (discrétisation)
        self.M_h = np.sum(self.rho_MN) * self.dx**2
    
        print(f"Masse de matière noire : {self.M_h}")
        print(f"Masse trou noir : {self.M_BH}")
    
        ### Potentiel disque 2D mince (pseudo-isotherme)
            
        self.phi_MN  =  0.5 * self.v_h**2 * np.log(self.R**2 + self.R_MN**2)
        self.g_MN    = -self.v_h**2 * self.R / (self.R**2 + self.R_MN**2)
        self.gx_MN   =  self.g_MN * self.X / self.R
        self.gy_MN   =  self.g_MN * self.Y / self.R
        
        ### Potentiel total
        
        self.phi_0 = self.phi_BH * self.TN + self.phi_MN * self.MN
        
        
    def calc_rho_0(self,px_,py_):
        r=np.sqrt(px_**2+py_**2)
        out = self.R_MN*self.rho_0_MN*self.G/((1+r/self.R_MN)**2) #Matière noire
        return out*self.MN
    
    def interpolation(self,A, x, y, dx, L, N):
        fx = (x + L) / dx
        fy = (y + L) / dx
    
        i = np.floor(fx).astype(int)
        j = np.floor(fy).astype(int)
    
        tx = fx - i
        ty = fy - j
    
        i0 = i #% self.N
        j0 = j #% self.N
        i1 = (i + 1) #% self.N
        j1 = (j + 1) #% self.N
    
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty) * (i0 + 1 < self.N)
        w01 = (1 - tx) * ty * (j0 + 1 < self.N)
        w11 = tx * ty * (i0 + 1 < self.N) * (j0 + 1 < self.N)
    
        return w00*A[i0, j0] + w10*A[i1, j0] +w01*A[i0, j1] + w11*A[i1, j1]
    
    def cic_weights(self, px, py):
        inv_dx = 1.0 / self.dx
    
        fx = (px + self.L) * inv_dx
        fy = (py + self.L) * inv_dx
    
        i = np.floor(fx).astype(int)
        j = np.floor(fy).astype(int)
    
        tx = fx - i
        ty = fy - j
    
        i0 = i #% self.N
        j0 = j #% self.N
        i1 = (i + 1) #% self.N
        j1 = (j + 1) #% self.N
    
        w00 = (1 - tx) * (1 - ty)
        w10 = tx * (1 - ty) * (i0 + 1 < self.N)
        w01 = (1 - tx) * ty * (j0 + 1 < self.N)
        w11 = tx * ty * (i0 + 1 < self.N) * (j0 + 1 < self.N)
    
        return i0, j0, i1, j1, w00, w10, w01, w11
    
    def _tsc_weight(self, u):
        """
        poids TSC 1D (B-spline quadratique)
        u = distance en unités de dx
        """
    
        a = np.abs(u)
        w = np.zeros_like(u)
    
        m1 = a < 0.5
        m2 = (a >= 0.5) & (a < 1.5)
    
        w[m1] = 0.75 - u[m1]**2
        w[m2] = 0.5 * (1.5 - a[m2])**2
    
        return w
    
    
    def M_BH_t(self, step) -> float:
        """Donne la valeur de la masse du trou noir à l'instant t"""
        if self.BH_step0 is None:
            return 0.0
        s = step - self.BH_step0
        if s <= 0:
            return 0.0
        if s >= self.N_grow:
            return self.M_BH_final
        f = 0.5 * (1 - np.cos(np.pi * s / self.N_grow))
        return self.M_BH_final * f
    
    def maj_BH(self,M_BH) -> None:
        """Met à jour les données du trou noir"""
        self.M_BH = M_BH
        self.phi_BH = -self.G * self.M_BH / self.R_BH
        self.g_BH   = -self.G * self.M_BH / self.R_BH**2
        self.phi_0 = self.phi_BH * self.TN + self.phi_MN * self.MN
    
    def maj_rho(self,px,py,mass):
        """init_rho : projette particules (px,py) avec masses sur la grille N x N.
        Retourne rh"o (masse par unité de surface").
        """
        mask = (abs(px) <= self.L) & (abs(py) <= self.L)
        px = px[mask]
        py = py[mask]
        mass = mass[mask]
        
        self.rho.fill(0.0)
    
        i0, j0, i1, j1, w00, w10, w01, w11 = self.cic_weights(px, py)
    
        np.add.at(self.rho, (i0, j0), mass * w00)
        np.add.at(self.rho, (i1, j0), mass * w10)
        np.add.at(self.rho, (i0, j1), mass * w01)
        np.add.at(self.rho, (i1, j1), mass * w11)

        # normalisation surface
        self.rho /= self.dx**2
        
    
    def maj_phi(self,phi):
        """Met à jour phi en ajoutant les potentiels analytique"""
        self.phi = phi + self.phi_0 
        
    def calcul_G(self, gx, gy):
        """Calcul l'accélération gravitationnelle en chaque noeud 
        de la grille"""
        # disque (FFT)
        if self.TN or self.MN:
            if self.TN:
                gx += -self.G * self.M_BH * self.X / (self.R_BH**3)
                gy += -self.G * self.M_BH * self.Y / (self.R_BH**3)
    
            if self.MN:
                gx += self.gx_MN
                gy += self.gy_MN
    
        return gx, gy
    
            
class SolveurPoisson:

    def __init__(self, g : Grille):
        self.g = g
        self.Nfft = 2 * self.g.N  

        # espace physique FFT cohérent
        Lfft = self.g.L

        k = fftfreq(self.Nfft, d=(2*Lfft/self.Nfft)) * 2*np.pi
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        kN = np.pi/self.g.dx
    
    def cic_window(self):
        kx = self.KX * self.g.dx
        ky = self.KY * self.g.dx

        wx = np.sinc(kx / (2*np.pi))**2
        wy = np.sinc(ky / (2*np.pi))**2

        return wx * wy

    def solve(self, rho=None):
        if rho is None:
            rho = self.g.rho
        
        #Padding
        rho_pad = np.zeros((self.Nfft, self.Nfft))
        n = self.g.N
        start = (self.Nfft - n) // 2
        rho_pad[start:start+n, start:start+n] = rho

        rho = rho_pad
        rho_k = fft2(rho)
        W = self.cic_window()
        k = np.sqrt(self.K2)
        k[0, 0] = 1.0

        Green = -2*np.pi*self.g.G *np.exp(-k*self.g.soft)/ k

        Green[0, 0] = 0.0
        
        phi_k = Green * rho_k


        ### RETOUR AU REEL
        phi_full = np.real(ifft2(phi_k))

        ### CROP
        s = self.Nfft//2 - self.g.N//2
        e = s + self.g.N

        phi = phi_full[s:e, s:e]
        
        gx_k = -1j * self.KX * phi_k
        gy_k = -1j * self.KY * phi_k

        gx_full = np.real(ifft2(gx_k))
        gy_full = np.real(ifft2(gy_k))

        gx = gx_full[s:e, s:e]
        gy = gy_full[s:e, s:e]

        return phi, gx, gy
        
    def Epot(self, px, py, mass):
        phi_i = self.g.interpolation(
            self.g.phi, px, py,
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
    
        # gradient radial approximatif
        gy, gx = np.gradient(-Phi, self.g.dx)
    
        gr = (gx * x + gy * y) / (r + 1e-10)
    
        # binning radial
        r_flat = r.flatten()
        gr_flat = gr.flatten()
    
        bins = np.linspace(0, self.g.L, 50)
        centers = 0.5 * (bins[:-1] + bins[1:])
    
        g_r = np.zeros(len(centers))
    
        for i in range(len(centers)):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
            if np.any(mask):
                g_r[i] = np.mean(gr_flat[mask])
    
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
    
    def bulge_influence_radius(self):

        _, _, Phi_h, _ = self.compute_fields()
    
        r = np.sqrt(self.g.X**2 + self.g.Y**2)
    
        # gradients des potentiels
        _,gx_d, gy_d = self.s.solve()
        # projections radiales
        def radial_component(gx, gy):
            return (gx * self.g.X + gy * self.g.Y) / (r + 1e-10)
    
    
        g_d  = radial_component(gx_d, gy_d)
        g_h  = radial_component(self.g.gx_MN, self.g.gy_MN)
    
        # binning radial
        bins = np.linspace(0, self.g.L, 200)
        centers = 0.5 * (bins[:-1] + bins[1:])
    
        r_flat = r.flatten()
        gd = g_d.flatten()
        gh = g_h.flatten()
    
        gd_r, gh_r = [], []
    
        for i in range(len(centers)):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
    
            if np.any(mask):
                gd_r.append(np.mean(gd[mask]))
                gh_r.append(np.mean(gh[mask]))
            else:
                gd_r.append(0)
                gh_r.append(0)
    
        gd_r = np.array(gd_r)
        gh_r = np.array(gh_r)
        gB_r = - self.g.G*self.g.M_BH/(self.g.soft_BH + centers)**2
    
        # normalisation de dominance locale
        g_tot = gd_r + gh_r + gB_r + 1e-12
    
        eta_d  = gd_r / g_tot
        eta_h  = gh_r / g_tot
        eta_BH = gB_r / g_tot
    
        # critères physiques utiles :
    
        # disque vs halo transition
        diff_bh = np.abs(eta_d - eta_h)
        r_dh = centers[np.argmin(diff_bh)]
    
        # BH domination radius (quasi arbitraire mais utile)
        r_BH = centers[np.argmax(eta_BH < 0.5)]
        
        plt.semilogy(centers, np.abs(gd_r),label='disque')
        if self.g.TN:
            plt.semilogy(centers, np.abs(gB_r),label='BH')
        if self.g.MN:
            plt.semilogy(centers, np.abs(gh_r),label='Matière noire')
        plt.title('g')
        plt.legend()
        plt.show()
        
        return {
            "Rd/h_influence" : r_dh,
            "R_BH_influence": r_BH
        }
    
    def bulge_influence_radius_BH(self):

        # gradients des potentiels
        _,gx_d,gy_d = self.s.solve()
    
        # projections radiales
        def radial_component(gx, gy):
            return (gx * self.g.X + gy * self.g.Y) / (self.g.R + 1e-10)
        # binning radial
        bins = np.linspace(0, self.g.L, self.g.N)
        centers = 0.5 * (bins[:-1] + bins[1:])
        
        g_d=radial_component(gx_d, gy_d)
        gB_r = - self.g.G*self.g.M_BH/(self.g.soft_BH + centers)**2
        
        
    
        r_flat = self.g.R.flatten()
        gd = g_d.flatten()
    
        gd_r = []
    
        for i in range(len(centers)):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i+1])
    
            if np.any(mask):
                gd_r.append(np.mean(gd[mask]))
            else:
                gd_r.append(0)
    
        gd_r = np.array(gd_r)
    
        # normalisation de dominance locale
        g_tot = gd_r + gB_r + 1e-12
    
        
        eta_d  = gd_r / g_tot
        eta_BH = gB_r / g_tot
    
        # critères physiques utiles :
    
        # influence bulbe vs disque
        diff_bd = np.abs(eta_d - 0.5)
        r_d = centers[np.argmin(diff_bd)]
    
        # BH domination radius (quasi arbitraire mais utile)
        r_BH = centers[np.argmax(eta_BH < 0.5)]
        
        plt.semilogy(centers, np.abs(gd_r),label='disque')
        if self.g.TN:
            plt.semilogy(centers, np.abs(gB_r),label='BH')
        plt.title('g')
        plt.legend()
        plt.show()
        
        # retour structuré (plus utile que un seul rayon)
        return {
            "R_BH_influence": r_BH,
            "R_d_influence": r_d
        }
            