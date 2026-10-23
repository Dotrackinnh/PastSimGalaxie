"""
Version optimisée de stream.py — API identique à l'originale (drop-in).

Optimisations :
  - générateur np.random.default_rng (PCG64) : plus rapide et de meilleure
    qualité statistique que l'API legacy np.random.*,
  - une seule concaténation groupée via np.concatenate sur tuples,
  - moins d'allocations intermédiaires dans spawn().
"""

import numpy as np

_rng = np.random.default_rng()


class Streams:
    def __init__(self, origin, u, width, Q, v, m,
                 M_clump=None, N_clump=200,r_clump=None,sigma_v_clump_frac=0.05,
        M_bulge=None,clump_period_steps=4_000):
        
        self.origin = origin
        self.u = u
        self.width = width
        self.Q = Q
        self.v = v
        self.m = m
        """
        Paramètres des amas
        -------------------
        M_clump : float | None
            Masse totale de l'amas en unités de simulation.
            None → fonctionnalité clump entièrement désactivée.
        N_clump : int
            Nombre de particules représentant l'amas (résolution).
            Masse par particule = M_clump / N_clump.
        r_clump : float | None
            Rayon (σ) gaussien perpendiculaire de l'amas.
            Défaut : 2 × width du stream.
        sigma_v_clump_frac : float
            Dispersion interne en fraction de v (défaut 5 %).
        M_bulge : float | None
            Seuil massique : premier clump quand Q·t = M_bulge,
            puis tous les M_bulge suivants.  None → déclencheur
            massique désactivé.
        clump_period_steps : int | None
            Période moyenne (steps) du déclencheur stochastique.
            None → déclencheur stochastique désactivé.
        """
        # --- amas ---
        self.M_clump            = M_clump if M_clump is not None else N_clump * m
        self.N_clump            = N_clump
        self.r_clump            = r_clump if r_clump is not None else 2.0 * width
        self.sigma_v_clump_frac = sigma_v_clump_frac
        self.M_bulge            = M_bulge
        self.clump_period_steps = clump_period_steps
 
        # --- état interne ---
        # Note : Streams n'est PAS sérialisé dans save_checkpoint ;
        # ces compteurs repartent de zéro à chaque reprise, ce qui est
        # cohérent (les déclencheurs recommencent leur compte).
        self._n_calls         = 0       # step courant (= nb d'appels à spawn)
        self._t_elapsed       = 0.0     # temps physique cumulé
        self.clump_log = []             # public : [(step, t, M_clump)]
 
    
    min_m_part = 1e-8

    def _normalize_vec(self, vx, vy):
        norm = np.hypot(vx, vy)
        if norm == 0:
            return 1.0, 0.0
        return vx / norm, vy / norm
    
    def _build_clump(self):
        """
        Génère les N_clump particules de l'amas compact à l'origine du stream.
 
        Distribution spatiale : blob gaussien légèrement aplati dans la
        direction longitudinale (σ_long = r_clump/2, σ_perp = r_clump).
        Vitesse : direction du stream avec dispersion interne sigma_v_clump_frac·v.
        """
        n   = self.N_clump
        m_p = self.M_clump / n          # masse par particule du clump
 
        ox, oy = self.origin
        ux, uy = self._normalize_vec(*self.u)
 
        along = _rng.normal(scale=self.r_clump * 0.5, size=n)  # longitudinal
        perp  = _rng.normal(scale=self.r_clump,        size=n)  # transverse
 
        px_c = ox + along * ux - perp * uy
        py_c = oy + along * uy + perp * ux
 
        sig_v = self.sigma_v_clump_frac * self.v if self.v > 0.0 else 0.0
        if sig_v > 0.0:
            vx_c = ux * self.v + _rng.normal(scale=sig_v, size=n)
            vy_c = uy * self.v + _rng.normal(scale=sig_v, size=n)
        else:
            vx_c = np.full(n, ux * self.v)
            vy_c = np.full(n, uy * self.v)
 
        return px_c, py_c, vx_c, vy_c, np.full(n, m_p, dtype=float)
    
    # ------------------------------------------------------------------
    def spawn(self, dt):
        """
        Retourne px_new, py_new, vx_new, vy_new, mass_new (numpy arrays).
 
        Contient le flux régulier (Poisson) + éventuellement les particules
        d'un amas si un déclencheur a tiré.  Les deux jeux sont concaténés
        avant le retour → inject() ne voit qu'un seul lot de nouvelles
        particules.
 
        Déclencheurs (logique OU, au plus 1 clump par step) :
          • Stochastique : Bernoulli(p = 1/clump_period_steps) chaque step.
          • Massique     : Q·t_cumulé ≥ _next_clump_mass.
        Quand le déclencheur massique tire, _next_clump_mass += M_bulge.
        """
        self._n_calls   += 1
        self._t_elapsed += dt
 
        parts = []      # liste de tuples (px, py, vx, vy, mass)
 
        # ---- flux régulier (identique à la version originale) ----------
        mean_n = self.Q * dt / self.m
        if mean_n > 0:
            n_new = _rng.poisson(mean_n)
            if n_new > 0:
                ox, oy = self.origin
                ux, uy = self._normalize_vec(*self.u)
 
                along_offsets = _rng.uniform(0.0, 0.5, size=n_new)
                perp_offsets  = _rng.normal(scale=self.width, size=n_new)
 
                px_new = ox + along_offsets * ux - perp_offsets * uy
                py_new = oy + along_offsets * uy + perp_offsets * ux
 
                sigma_v = 0.02 * self.v
                if sigma_v > 0:
                    vx_new = ux * self.v + _rng.normal(scale=sigma_v, size=n_new)
                    vy_new = uy * self.v + _rng.normal(scale=sigma_v, size=n_new)
                else:
                    vx_new = np.full(n_new, ux * self.v)
                    vy_new = np.full(n_new, uy * self.v)
 
                parts.append(
                    (px_new, py_new, vx_new, vy_new,
                     np.full(n_new, self.m, dtype=float))
                )
 
        # ---- déclencheurs de clump ------------------------------------
        if self.M_clump is not None:
            cause = None
 
            
            if (self.clump_period_steps
                    and _rng.random() < 1.0 / self.clump_period_steps):
                cause = "stochastic"
 
            if cause is not None:
                parts.append(self._build_clump())
                self.clump_log.append(
                    (self._n_calls, self._t_elapsed, cause, self.M_clump)
                )
                print(
                    f"[Clump] step={self._n_calls:6d}  t={self._t_elapsed:.3f}  "
                    f"M_clump={self.M_clump:.2e}  "
                )
 
        # ---- assemblage final -----------------------------------------
        if not parts:
            e = np.empty(0)
            return e, e, e, e, e
 
        if len(parts) == 1:
            return parts[0]
 
        # np.concatenate sur 2 lots (régulier + clump)
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(5))

    # ------------------------------------------------------------------
    def inject(self, px, py, r, vx, vy, alive, liee, mass, dt, soft,
               max_particles=200_000, mode="keep_newest"):
        """Injecte de nouvelles particules et retourne des arrays cohérents.

        mode :
            - "keep_newest" : garde les particules les plus récentes
            - "keep_oldest" : garde le système initial
        """
        # 1. spawn
        pxn, pyn, vxn, vyn, mn = self.spawn(dt)

        if pxn.size == 0:
            return px, py, r, vx, vy, alive, liee, mass

        rn = np.sqrt(pxn * pxn + pyn * pyn + soft * soft)
        n_new = pxn.size

        # 2. concaténation
        px = np.concatenate((px, pxn))
        py = np.concatenate((py, pyn))
        r = np.concatenate((r, rn))
        vx = np.concatenate((vx, vxn))
        vy = np.concatenate((vy, vyn))
        mass = np.concatenate((mass, mn))

        # 3. flags cohérents
        ones = np.ones(n_new, dtype=bool)
        alive = np.concatenate((alive, ones))
        liee = np.concatenate((liee, ones))

        # 4. cohérence globale
        assert px.shape == py.shape == vx.shape == vy.shape == mass.shape
        assert alive.shape[0] == px.shape[0]
        assert liee.shape[0] == px.shape[0]

        # 5. plafond mémoire
        N = px.size
        if N > max_particles:
            if mode == "keep_newest":
                sl = slice(N - max_particles, N)
            elif mode == "keep_oldest":
                sl = slice(0, max_particles)
            else:
                raise ValueError("mode must be 'keep_newest' or 'keep_oldest'")

            px = px[sl]
            py = py[sl]
            r = r[sl]
            vx = vx[sl]
            vy = vy[sl]
            mass = mass[sl]
            alive = alive[sl]
            liee = liee[sl]

        return px, py, r, vx, vy, alive, liee, mass
    
    def reset_clump_state(self, t0: float = 0.0, n0: int = 0) -> None:
        """
        Réinitialise les compteurs internes.
 
        À appeler après load_save() si l'on veut que les déclencheurs
        reprennent à zéro depuis le checkpoint (comportement par défaut
        puisque Streams n'est pas sérialisé), ou à t0/n0 pour recaler le
        déclencheur massique sur le temps physique déjà écoulé.
 
        Parameters
        ----------
        t0 : float
            Temps physique à injecter dans _t_elapsed (ex : E.time au
            moment du checkpoint).
        n0 : int
            Numéro de step de départ pour _n_calls.
        """
        self._n_calls         = n0
        self._t_elapsed       = t0
        self._next_clump_mass = (self.M_bulge
                                 if self.M_bulge is not None else np.inf)
        self.clump_log        = []
