import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pickle
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
from pathlib import Path

dossier_du_script = Path(__file__).resolve().parent
path = ".." / dossier_du_script
sys.path.append(path)
base = Path(path + 'MainTests')


from obj.grille import Grille, SolveurPoisson, BulbeDiagnostic
from obj.etoile import Etoiles
from obj.stream import Streams


# PARAMÈTRES
N = 512
L = 150.0
G0 = 1.0

v_circ = 0.1
random_dispersion = 0.05 * v_circ
disk_scale = 7.0

R_lim = 1.8 * L
M_BH = 2e-5


def save_checkpoint(filename, E, G, step, step_diag, Q_t, V_t, Lz_tot, E_tot, R_W, Mode):
    data = {
        "step": step,
        "step_diag" : step_diag,
        "E": E.save_state(),
        "Q_t": Q_t,
        "V_t": V_t,
        "Lz_tot": Lz_tot,
        "E_tot": E_tot,
        "R_W": R_W,
        "Mode": Mode,
        "BH_step0": G.BH_step0
    }
    tmp = filename + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, filename)

def load_save(name, output_dir="MainTests"):
    """
    Charge un checkpoint sauvegardé et reconstruit l'objet Etoiles + l'historique des diagnostics.

    Parameters
    ----------
    name : str
        Nom du dossier de la run (ex: "Test_2024-01-01 12_00_00")
    output_dir : str
        Dossier parent contenant les runs (par défaut "MainTests")

    Returns
    -------
    E : Etoiles
        L'objet étoiles restauré (positions, vitesses, masses, etc.)
    step : int
        Le step auquel le checkpoint a été sauvegardé
    Q_t, V_t, Lz_tot, E_tot, R_W : list
        Historique des diagnostics
    Mode : dict
        Historique des amplitudes des modes azimutaux
    """
    path = os.path.join(output_dir, name, "checkpoint.pkl")

    with open(path, "rb") as f:
        data = pickle.load(f)

    E = Etoiles(G0)  # G0 doit être défini/importé dans le contexte
    E.load_state(data["E"])

    step     = data["step"]
    step_diag= data["step_diag"]
    Q_t      = data["Q_t"]
    V_t      = data["V_t"]
    Lz_tot   = data["Lz_tot"]
    E_tot    = data["E_tot"]
    R_W      = data["R_W"]
    Mode     = data["Mode"]
    BH_step0 =data["BH_step0"]
    
    # Champs nécessaires au fonctionnement de E (non sauvegardés dans save_state)
    E.L  = L
    E.N  = N
    E.dx = 2 * L / (N - 1)

    return E, step, step_diag, Q_t, V_t, Lz_tot, E_tot, R_W, Mode,BH_step0

def init_affichage(N, L, px, py):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)

    M0 = np.zeros_like(px)
    scatter = ax1.scatter(px, py, c=M0, cmap='viridis', s=0.005)

    cg_marker = ax1.scatter(
        [0], [0], c='red', s=200, marker='+',
        linewidths=3, label='Centre gravitationnel', zorder=20
    )
    ax1.legend()
    ax1.set_xlim(-L, L)
    ax1.set_ylim(-L, L)
    ax1.set_title("Distribution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    cbar1 = fig.colorbar(scatter, ax=ax1)
    cbar1.set_label("Lz")

    im = ax2.imshow(np.zeros((N, N), dtype=float),origin='lower',
           norm=SymLogNorm(linthresh=1e-3),
           extent=[-L/4, L/4, -L/4, L/4], cmap='RdBu_r')
    
    ax2.set_title("Potentiel gravitationnel")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cbar2 = fig.colorbar(im, ax=ax2)
    cbar2.set_label("Potentiel")

    plt.ion()
    plt.show()

    return fig, ax1, ax2, scatter, cg_marker, im, False

def affichage(fig, ax1, ax2,
              scatter, cg_marker, im, dezoom,
              px, py, mass, Lz, phi,
              temps, afficher,
              name, step, save_interval,
              dx, output_dir):

    need_render = afficher or (step % save_interval == 0)
    if need_render:
        if not dezoom and step % 10 == 0:
            if np.any(np.abs(px) > L) or np.any(np.abs(py) > L):
                ax1.set_xlim(-1.5*L, 1.5*L)
                ax1.set_ylim(-1.5*L, 1.5*L)
                dezoom = True

        scatter.set_offsets(np.c_[px, py])
        scatter.set_array(Lz)

        if step % 10 == 0:
            scatter.vmin, scatter.vmax = np.percentile(Lz, [1, 99])
            norm = colors.SymLogNorm(linthresh=1e-4, vmin=scatter.vmin, vmax=scatter.vmax)
            scatter.set_norm(norm)

        xg = np.sum(mass*px) / np.sum(mass)
        yg = np.sum(mass*py) / np.sum(mass)
        cg_marker.set_offsets([[xg, yg]])
        ax1.set_title(f"Distribution t = {temps:.2f}")
        
        N_zoom = N // 8  # 64 pixels centraux
        phi_center = phi[N//2 - N_zoom : N//2 + N_zoom,
                         N//2 - N_zoom : N//2 + N_zoom]
        
        im.set_data(phi_center)
        im.set_clim(vmin=np.min(phi_center), vmax=np.max(phi_center))

        if afficher:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if step % save_interval == 0:
            fig.savefig(
                f"{output_dir}/{name}_{step:06d}.png",
                dpi=200, bbox_inches='tight'
            )


def simulation(Np=50_000, dt=1e-3, n_steps=100, TN=False, MN=False, Stream=False,data = None, afficher=False):

    # INITIALISATION
    np.random.seed(42)

    Np = (Np // 2) * 2

    G = Grille(N, L, TN, MN, G0, M_BH, 0.5 * 2 * L / N)
    dx = G.dx
    soft = G.soft
    Sol = SolveurPoisson(G)
    D = BulbeDiagnostic(G, Sol)

    
    E = Etoiles(G0)
    E.L = L
    E.N = N
    E.dx = dx
    if data is None:
        # SPLIT 3 COMPOSANTES
        N_bulge = int(0.0 * Np)
        N_disk  = int(0.1 * Np)
        N_halo  = Np - N_bulge - N_disk
    
        # BULBE (profil de Hernquist)
        a_h = 1.5
        r_max_b = 15.0
        r_b = []
        while len(r_b) < N_bulge:
            r_try = np.random.uniform(0, r_max_b, 10 * N_bulge)
            prob = r_try / (r_try + a_h)**3
            prob /= prob.max()
            accept = np.random.uniform(size=len(r_try)) < prob
            r_b.extend(r_try[accept].tolist())
        r_b = np.array(r_b[:N_bulge])
        N_bulge = len(r_b)
    
        if N_bulge > 0:
            theta_b = np.random.uniform(0, 2*np.pi, N_bulge)
            px_b = r_b * np.cos(theta_b)
            py_b = r_b * np.sin(theta_b)
            sigma_b_local = 0.3 * v_circ / np.sqrt(r_b / a_h + 1)
            vx_b = np.random.normal(0, sigma_b_local)
            vy_b = np.random.normal(0, sigma_b_local)
        else:
            px_b = py_b = vx_b = vy_b = np.array([])
    
        # DISQUE INSTABLE
        N2 = N_disk // 2
        r_min = 2.0
        r = np.random.exponential(scale=disk_scale, size=N2)
        r = r[r > r_min][:N2]
    
        theta = np.random.uniform(0, 2*np.pi, len(r))
        r_full = np.concatenate([r, r])
        N_disk = len(r_full)
        theta_full = np.concatenate([theta, theta + np.pi])
    
        px_d = r_full * np.cos(theta_full)
        py_d = r_full * np.sin(theta_full)
    
        # Vitesse circulaire locale depuis le champ gravitationnel réel
        E.px = px_d
        E.py = py_d
        E.mass = np.ones(N_disk, dtype=float) / (N_bulge + N_disk + N_halo)
        G.maj_rho(E.px, E.py, E.mass)
        phi, gx, gy = Sol.solve()
        gx, gy = G.calcul_G(gx, gy)
        gx_i = G.interpolation(gx, px_d, py_d, dx, L, N)
        gy_i = G.interpolation(gy, px_d, py_d, dx, L, N)
        g_r = (px_d * gx_i + py_d * gy_i) / r_full
        v_c_local = np.sqrt(np.maximum(-r_full * g_r, 0))
    
        plt.scatter(r_full, v_c_local, s=0.5)
        plt.title('v_c locale')
        plt.xlabel('r')
        plt.ylabel('v_c')
        plt.show()
    
        theta_full_dir = np.arctan2(py_d, px_d)
    
        sigma_d_local = 0.1 * v_c_local
        vx_d = -v_c_local * np.sin(theta_full_dir) + random_dispersion * np.random.normal(0, sigma_d_local)
        vy_d =  v_c_local * np.cos(theta_full_dir) + random_dispersion * np.random.normal(0, sigma_d_local)

        # HALO DIFFUS
        N2 = N_halo // 2
        N_halo = 2 * N2
    
        if N_halo > 0:
            r_core_h = 25.0
            r_h = r_core_h * np.tan(np.random.uniform(0, np.arctan(0.7*L / r_core_h), N2))
            theta_h_ang = np.random.uniform(0, 2*np.pi, N2)
            r_full_h = np.concatenate([r_h, r_h])
            theta_h = np.concatenate([theta_h_ang, theta_h_ang + np.pi])
            px_h = r_full_h * np.cos(theta_h)
            py_h = r_full_h * np.sin(theta_h)
            """
            vr_h = v_circ * (1 - np.exp(-r_full_h/(0.5*r_core_h)))
            vx_h = - vr_h * np.sin(theta_h) + random_dispersion*np.random.normal(size=N_halo)
            vy_h = vr_h * np.cos(theta_h) + random_dispersion*np.random.normal(size=N_halo)
            """
            vx_h = - v_circ * np.sin(theta_h) + random_dispersion*np.random.normal(size=N_halo)
            vy_h = v_circ * np.cos(theta_h) + random_dispersion*np.random.normal(size=N_halo)
            
            
        else:
            px_h = py_h = vx_h = vy_h = np.array([])
    
        # CONCATÉNATION
        Np = N_bulge + N_disk + N_halo
        E.leffectif = [Np]
    
        E.px = np.concatenate([px_b, px_d, px_h])
        E.py = np.concatenate([py_b, py_d, py_h])
        E.vx = np.concatenate([vx_b, vx_d, vx_h])
        E.vy = np.concatenate([vy_b, vy_d, vy_h])
    
        # Centrage en position et vitesse
        E.px -= np.mean(E.px)
        E.py -= np.mean(E.py)
        E.vx -= np.mean(E.vx)
        E.vy -= np.mean(E.vy)
    
        E.r     = np.sqrt(E.px**2 + E.py**2 + soft**2)
        E.alive = np.ones(Np, dtype=bool)
        E.liee  = np.ones(Np, dtype=bool)
        E.mass  = np.ones(Np) / Np
    
        E.vx += np.random.normal(size=Np) * random_dispersion * 0.1
        E.vy += np.random.normal(size=Np) * random_dispersion * 0.1

        # dt adaptatif
        v_char = np.percentile(np.sqrt(E.vx**2 + E.vy**2), 95)
        a_char = np.percentile(np.sqrt(gx**2 + gy**2), 95)
        dt_min = 0.05 * dx / v_char
        dt_max = 0.1 * min(np.sqrt(dx / a_char), dx / v_char)
        E.dt = np.clip(dt, dt_min, dt_max)
        
        G.maj_rho(E.px, E.py, E.mass)
        phi, _, _ = Sol.solve()
        Q, T, W_self, V_ext = E.viriel_Q(G, phi, E.px, E.py, E.vx, E.vy, E.mass)
        Q_t   = [abs(Q)]
        V_t   = [2*T + W_self + V_ext]   # → doit osciller autour de 0 à l'équilibre
            
        E_tot = [E.energie_totale(G,W_self)]
        R_W   = [G.ratio_W(E.px, E.py, E.mass, phi)]
        
        print(f"Q = {Q:.4f}")
        print(f"ratio FFT/grid = {R_W[0]:.4f}")
        
        m = np.linspace(1, 10, 10)
        Mode = {}
        for i in m:
            Mode[f'Mode {i}'] = [
                abs(np.sum(E.mass * np.exp(1j*i*np.arctan2(E.px, E.py)))) / np.sum(E.mass[E.alive])
            ]
        
        theta = np.arctan2(E.py, E.px)
        L_z0 = np.sum(E.mass * E.r * (-np.sin(theta)*E.vx + np.cos(theta)*E.vy))
        Lz_tot = [L_z0]
        
        params_dict = {
            "N": N,
            "L": L,
            "G0": G0,
            "soft": soft,
            "Nombre d'étoiles dans le bulbe": N_bulge,
            "Nombre d'étoiles dans le disque": N_disk,
            "Nombre d'étoiles dans le halo diffus": N_halo,
            "v_circ": v_circ,
            "random_dispersion": random_dispersion,
            "R_lim": R_lim,
            "M_BH": M_BH,
            "Np": Np,
            "dt": E.dt,
            "BH_step0": G.BH_step0,
            "n_steps": n_steps,
            "TN": TN,
            "MN": MN,
        }
        step0 = 0
        step_diag = [0]
        
    else:
        data_name = data
        E, step, step_diag, Q_t, V_t, Lz_tot, E_tot, R_W, Mode,G.BH_step0 = load_save(data)
        print(f"Reprise au step {step}, Q = {Q_t[-1]:.3f}, R_W = {R_W[-1]:.4f}")
        m = np.linspace(1, 10, 10)
        theta = np.arctan2(E.py, E.px)
        step0 = len(Q_t)
        
        params_dict = {
            "resumed_from": data_name,
            "step0": step0,
            "n_steps": n_steps,
            "TN": TN,
            "MN": MN,
            "BH_step0" : G.BH_step0,
            "M_BH_final": G.M_BH_final,
            "M_BH_current": G.M_BH,
            "v_h_current": G.v_h if MN else None,
            "Stream": Stream,
            "dt": E.dt,
            "Np": E.px.size,
        }

    # CHAMP INITIAL
    G.maj_rho(E.px, E.py, E.mass)
    phi, gx, gy = Sol.solve()
    gx, gy = G.calcul_G(gx, gy)
    G.maj_phi(phi)

    # Énergie et viriel initiaux
    
    # Calibrage du Bulbe / halo / trou noir
    if MN:
        r_inf = D.bulge_influence_radius()
    else:
        r_inf = D.bulge_influence_radius_BH()
    for k, val in r_inf.items():
        print(k, ":", val)

    S=[Streams((-L+2,-20), (1,0), 3, 5e-2/(Np*E.dt), 0.1, 1/Np)]#,Streams((-2+L,0), (-1,0), 10, 1e-1/(Np*E.dt), 0.2, 1/Np),
       #Streams((0,-L), (0,1), 25, 1e-1/(Np*E.dt), 0.1, 1/Np),Streams((0,L-2), (-1,0), 1e-3, 1e-1/(Np*E.dt), 0.05, 1/Np)]

    SI = int(input('Nombre de screen ?'))

    print('Création du dossier...')
    
    output_dir = os.path.join("MainTests", "Test_" + time.strftime("%Y-%m-%d %H_%M_%S"))
    fig, ax1, ax2, scatter, cg_marker, im, dezoom = init_affichage(N, L, E.px, E.py)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "parameters.txt")
    
    print('... dossier crée !')
        
    

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Simulation parameters ===\n\n")
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        for key, value in params_dict.items():
            f.write(f"{key} = {value}\n")
        f.write(f'Q = {Q_t[-1]}\n')
        f.write(f'Ratio FFT / grid : {R_W[-1]}\n')
        f.write(f'Masse halo : {G.M_h}\n')
        f.write(f'Masse Trou noir : {G.M_BH}\n')
        for k, val in r_inf.items():
            f.write(f"{k} = {val}\n")

    affichage(fig, ax1, ax2, scatter, cg_marker, im, dezoom,
              E.px, E.py, E.mass,
              E.mass * E.r * (-np.sin(theta)*E.vx + np.cos(theta)*E.vy),
              G.phi, 0, True, 'initial', 0, 1, dx, output_dir)
    
    
    # INTÉGRATION
    steps_fin = step0

    Fx, Fy = E.calcul_force(gx, gy, N, L)

    for step in range(step0,n_steps):

        # 1. Demi-kick initial
        E.maj_vit(0.5*Fx, 0.5*Fy)
        # 2. Drift
        E.maj_pos()
        # 3. Maj trou noir
        # déclenchement quand le mode m=2 atteint le seuil
        m2 = np.mean(Mode['Mode 2.0'][-50:])
        if TN and G.BH_step0 is None and m2 > G.seuil_mode and step > 500:
            G.BH_step0 = step
            print(f"Croissance du TN déclenchée au step {step}")
        
        # croissance (uniquement pendant la rampe, pour éviter des maj inutiles)
        if TN and G.BH_step0 is not None and step <= G.BH_step0 + G.N_grow:
            G.maj_BH(G.M_BH_t(step))
        
        if Stream:
            for s in S:
                E.px, E.py, E.r, E.vx, E.vy, E.alive, E.liee, E.mass = s.inject(E.px, E.py, E.r, E.vx, E.vy, E.alive, E.liee, E.mass,E.dt,G.soft)
                
        E.maj_status(G,R_lim,phi)
        steps_fin += 1
        if np.sum(E.alive) < 0.01 * Np:
            print("Galaxie dissoute")
            break

        X, Y, VX, VY, M = E.spec_palive()
        R = E.r[E.alive]
        
        # 4. Densité
        G.maj_rho(X, Y, M)
        # 5. Poisson
        phi, gx, gy = Sol.solve()
        G.maj_phi(phi)
        gx, gy = G.calcul_G(gx, gy)
        # 6. Forces interpolées
        Fx, Fy = E.calcul_force(gx, gy, N, L,X,Y,M)
        # 7. Demi-kick final
        E.maj_vit(0.5*Fx, 0.5*Fy)

        theta = np.arctan2(Y, X)
        
        Lz = M * R * (-np.sin(theta)*VX + np.cos(theta)*VY)
        
        affichage(fig, ax1, ax2, scatter, cg_marker, im, dezoom,
                  X, Y, M, Lz,
                  G.phi, E.time, False, f'{step}', step, max(1, (n_steps-step0)//SI),
                  dx, output_dir)
        
        if step%10 == 0:
            step_diag.append(step)
            Q, T, W_self, V_ext = E.viriel_Q(G, phi,X,Y,VX,VY,M)
            
            V_t.append(2*T + W_self + V_ext) 
            Q_t.append(abs(Q))
            
            E_tot.append(E.energie_totale(G, W_self))
            Lz_tot.append(np.sum(Lz))
            
            mask = E.liee & (E.r < 40)
            X_,Y_,M_ = E.px[mask],E.py[mask],E.mass[mask]
            theta_ = np.arctan2(Y_, X_)
            Z_ = np.exp(1j*theta_)
            Z = Z_
            M_mode_tot = np.sum(M_)
            for i in m:
                Mode[f'Mode {i}'].append(abs(np.sum(M_ * Z)) / M_mode_tot)
                Z*=Z_
                
            R_W.append(G.ratio_W(X, Y, M, phi))
        
        if step % 5000 == 0 and step > 0:

            save_checkpoint(os.path.join(output_dir, "checkpoint.pkl"),
                E,G,step,step_diag,Q_t,V_t,Lz_tot,E_tot,R_W,Mode)
            print(f"Checkpoint sauvegardé (step {step})")
        print(f'{100*(step+1)/n_steps:.3f} %')

    # AFFICHAGE FINAL
    X, Y, VX, VY, M = E.spec_palive()
    G.maj_rho(X, Y, M)
    phi, gx, gy = Sol.solve()
    G.maj_phi(phi)

    theta = np.arctan2(Y, X)
    affichage(fig, ax1, ax2, scatter, cg_marker, im, dezoom,
              X, Y, M,
              M * np.sqrt(X**2 + Y**2) * (-np.sin(theta)*VX + np.cos(theta)*VY),
              G.phi, E.time, True, 'final', step, 1, dx, output_dir)

    print(f'Durée : {E.time}')
    if Stream:
        for s in S:
            print(s.clump_log)

    # DIAGNOSTICS
    r     = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    v     = np.sqrt(VX**2 + VY**2)
    v_th  = -np.sin(theta)*VX + np.cos(theta)*VY

    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.plot(r, v, '.', ms=0.5, color='b')
    plt.xlabel("r")
    plt.ylabel("|v|")

    plt.subplot(2, 3, 2)
    plt.hist(v, bins=50)
    plt.xlabel("v")

    plt.subplot(2, 3, 3)
    plt.hist(v_th, bins=50)
    plt.xlabel("v_th")

    plt.subplot(2, 3, 4)
    plt.plot(E.leffectif)
    plt.xlabel("step")
    plt.ylabel("effectifs")

    plt.subplot(2, 3, 5)
    plt.plot(step_diag,E_tot, label='Energie', color='b')
    plt.plot(step_diag,Lz_tot - Lz_tot[0], label='dLz', color='r')    
    plt.plot(step_diag,R_W,color='g')
    plt.xlabel("step")
    plt.ylabel("Energie")
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(step_diag,Q_t, label='Q', color='r')
    plt.plot(step_diag,V_t, label='V', color='b')
    plt.xlabel("step")
    plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Diagnostics.png"), dpi=250)
    plt.show()

    print(f'Etoiles restantes : {100*np.sum(E.liee)/Np:.1f} %')
    N_half = np.sum(E.liee * E.alive)//2
    
    R_ = np.sort(E.r)
    R_25 = np.max(R_[:N_half//2])
    R_50 = np.max(R_[:N_half])
    print(f'Rayon 25% : {R_25:.1f}')
    print(f'Rayon 50% : {R_50:.1f}')
    for i in m:
        plt.plot(step_diag,Mode[f'Mode {i}'], label=f'Mode {i}')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Modes.png"), dpi=250)
    plt.show()

    if MN:
        r_inf = D.bulge_influence_radius()
    else:
        r_inf = D.bulge_influence_radius_BH()
    plt.savefig(os.path.join(output_dir, "g_final.png"), dpi=250)

    save_checkpoint(os.path.join(output_dir, "checkpoint.pkl"),
                    E,G, steps_fin, step_diag, Q_t, V_t, Lz_tot, E_tot, R_W, Mode)
    print(f'Checkpoint sauvegardé dans {output_dir}')
