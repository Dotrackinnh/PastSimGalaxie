import numpy as np


class Streams:
    def __init__(self,origin,u,width,Q,v,m):
        self.origin=origin
        self.u=u
        self.width=width
        self.Q=Q
        self.v=v
        self.m=m       
    
    min_m_part = 1e-8
    def _normalize_vec(self,vx, vy):
        norm = np.hypot(vx, vy)
        if norm == 0:
            return 1.0, 0.0
        return vx/norm, vy/norm
    
    def spawn(self, dt):
        """
        Retourne px_new, py_new, vx_new, vy_new, mass_new (numpy arrays)
        """
        
        # nombre moyen de particules à injecter ce pas
        mean_n = self.Q * dt / self.m
        if mean_n <= 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))

        # tirage aléatoire du nombre réel (stochastique)
        n_new = np.random.poisson(mean_n)
        
        if n_new == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,))

        # positions : le long de origin + petite dispersion transverse
        # on place les particules juste *à l'extérieur* du domaine si origin est sur le bord,
        # ou à origin si entry interne.
        ox, oy = self.origin
        ux,uy=self.u
        ux,uy=self._normalize_vec(ux,uy)
        # offset le long du stream pour étaler l'entrée selon x
        along_offsets = np.random.uniform(0.0, 0.5, size=n_new)  # petits offsets le long du flux
        perp = np.array([-uy, ux])  # vecteur perpendiculaire
        # transverse gaussian selon y
        perp_offsets = np.random.normal(scale=self.width, size=n_new)

        px_new = ox + along_offsets * ux + perp_offsets * perp[0]
        py_new = oy + along_offsets * uy + perp_offsets * perp[1]

        # vitesses : direction du flux plus dispersion
        vx_new = ux * self.v + np.random.randn(n_new) * 0.02 * self.v
        vy_new = uy * self.v + np.random.randn(n_new) * 0.02 * self.v

        mass_new = np.full(n_new, self.m, dtype=float)

        return px_new, py_new, vx_new, vy_new, mass_new
    
    def inject(self, px, py,r, vx, vy, alive, liee, mass, dt,soft,max_particles=200_000,mode="keep_newest"):
        """
        Injecte de nouvelles particules et retourne des arrays cohérents.
    
        Parameters
        ----------
        mode:
            - "keep_newest" : garde les particules les plus récentes (ton choix actuel)
            - "keep_oldest" : garde le système initial
        """

        # 1. spawn nouvelles particules
        pxn, pyn, vxn, vyn, mn = self.spawn(dt)
        rn=np.sqrt(pxn**2+pyn**2+soft**2)
        
        # rien à injecter
        if pxn is None or pxn.size == 0:
            return px, py, r, vx, vy, alive, liee, mass
    
        # 2. concaténation propre
        px = np.concatenate((px, pxn))
        py = np.concatenate((py, pyn))
        r = np.concatenate((r, rn))
        vx = np.concatenate((vx, vxn))
        vy = np.concatenate((vy, vyn))
        mass = np.concatenate((mass, mn))
    

        # 3. flags
        n_new = pxn.size
    
        alive = np.concatenate((
            alive,
            np.ones(n_new, dtype=bool)
        ))
    
        liee = np.concatenate((
            liee,
            np.ones(n_new, dtype=bool)   # ou False si tu veux les injecter "chaud"
        ))
    

        # 4. sécurité cohérence globale
        assert px.shape == py.shape == vx.shape == vy.shape == mass.shape
        assert alive.shape[0] == px.shape[0]
        assert liee.shape[0] == px.shape[0]
    

        # 5. gestion mémoire contrôlée
        N = px.size
    
        if N > max_particles:
    
            if mode == "keep_newest":
                keep = np.arange(N - max_particles, N)
    
            elif mode == "keep_oldest":
                keep = np.arange(0, max_particles)
    
            else:
                raise ValueError("mode must be 'keep_newest' or 'keep_oldest'")
    
            px = px[keep]
            py = py[keep]
            vx = vx[keep]
            vy = vy[keep]
            mass = mass[keep]
            alive = alive[keep]
            liee = liee[keep]
    
        return px, py, r, vx, vy, alive, liee, mass