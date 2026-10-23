import numpy as np


class Etoiles:
    def __init__(self,G):
        self.px = None
        self.py = None
        self.px_cm = []
        self.py_cm = []
        self.vx = None
        self.vy = None
        self.mass = None
        self.alive=None
        self.liee=None
        self.halo=None
        self.G=G
        self.soft=1e-5
        self.r=None
        self.leffectif=[]
        self.dt=None
        self.time=0
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
    
    def maj_pos(self) -> None:
        self.px[self.alive] += self.vx[self.alive] * self.dt
        self.py[self.alive] += self.vy[self.alive] * self.dt
    
        M_tot = np.sum(self.mass[self.alive])
    
        # Correction position CM
        self.px_cm.append(np.sum(self.px[self.alive] * self.mass[self.alive]) / M_tot)
        self.py_cm.append(np.sum(self.py[self.alive] * self.mass[self.alive]) / M_tot)
        
    
        self.r = np.sqrt(self.px**2 + self.py**2 + self.soft**2)
        self.time += self.dt
        
        
    def maj_vit(self,Fx,Fy) -> None:
        self.vx[self.alive] += Fx * self.dt
        self.vy[self.alive] += Fy * self.dt


    def maj_status(self,R_lim,L,Ep,phi,TN) -> None:
                
        # énergie cinétique
        v2 = self.vx**2 + self.vy**2

        E = 0.5 *self.mass* v2 + Ep
        self.liee = (E < np.max(phi))
    
        self.alive &= (self.px**2+self.py**2 < R_lim**2)
        self.leffectif.append(np.sum(self.liee & self.alive))
        

    def calcul_force(self, gx, gy, N, L,px=None,py=None):
        """ Bien qu'elle devait caculée la force pour éviter de multiplier puis diviser pas la 
            masse elle renvoie en réalité l'accélération."""
        if px is None and py is None:
            px=self.px[self.alive]
            py=self.py[self.alive]
            
        Fx = np.zeros_like(px)
        Fy = np.zeros_like(Fx)

    
        i0, j0, i1, j1, w00, w10, w01, w11 = self.cic_weights(px, py)

        out = (np.abs(px) > self.L) | (np.abs(py) > self.L)
        # monopôle pour les autres
        M_box = np.sum(self.mass[~out])
        r3 = (px[out]**2 + py[out]**2)**1.5
        Fx[out] = -self.G * M_box * px[out] / r3
        Fy[out] = -self.G * M_box * py[out] / r3
        
        
        Fx[~out] = (
            w00 * gx[i0, j0] +
            w10 * gx[i1, j0] +
            w01 * gx[i0, j1] +
            w11 * gx[i1, j1]
        )
    
        Fy[~out] = (
            w00 * gy[i0, j0] +
            w10 * gy[i1, j0] +
            w01 * gy[i0, j1] +
            w11 * gy[i1, j1]
        )
        
        Fx -= np.mean(Fx)
        Fy -= np.mean(Fy)
       
        return Fx, Fy
    
    def effectif(self):
        return sum(self.liee & self.alive)

    def spec_palive(self):
        return (self.px[self.alive],self.py[self.alive],self.vx[self.alive],self.vy[self.alive],self.mass[self.alive])
   
    def spec_pliee(self):
        mask=self.alive & self.liee
        return (self.px[mask],self.py[mask],self.vx[mask],self.vy[mask],self.mass[mask])
    
    def spec_pechapee(self):
        mask=self.alive & (1-self.liee)
        return (self.px[mask],self.py[mask],self.vx[mask],self.vy[mask],self.mass[mask])