import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import matplotlib.colors as colors
import pickle
from matplotlib.colors import SymLogNorm


sys.path.append(r'C:\Users\verre\OneDrive\Desktop\Project\NextTIPE\TEST_1')
    
from obj.grille import Grille,SolveurPoisson,BulbeDiagnostic
from obj.etoile import Etoiles
from obj.stream import Streams

### PARAMÈTRES

N = 512     #Paramètres de la grille
L = 150.0   #Taille de la zone d'étude
G0 = 1.0    #Constante gravitationnel unviersel

v_circ = 0.
random_dispersion = 0.05 * v_circ
disk_scale=7.0

R_lim = 1.8 * L  # rayon limite de la galaxie
M_BH = 1e-3      # masse du trou noir central en masse solaire

def init_affichage(N, L, px, py):
    """Cette fonction a pour utilité de définir l'affichage pour être mis à jour
       lors de de chaque affichage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)

    ### PARTICULES
    M0 = np.zeros_like(px)
    scatter = ax1.scatter(px,py,c=M0,cmap='viridis',s=0.05)
    
    ### CENTRE DE GRAVITÉ
    cg_marker = ax1.scatter([0], [0],c='red',s=200,marker='+',linewidths=3,
                            label='Centre gravitationnel',zorder=20)

    ax1.legend()

    ax1.set_xlim(-L, L)
    ax1.set_ylim(-L, L)

    ax1.set_title("Distribution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    cbar1 = fig.colorbar(scatter, ax=ax1)
    cbar1.set_label("Lz")

    ### POTENTIEL
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

def affichage(fig, ax1, ax2,scatter, cg_marker, im, dezoom,
              px, py, mass,Lz, phi,temps, afficher,
              name, step, save_interval,dx, output_dir):
    """Cette fonction met a jour l'affichage qui a été initialisé """
    
    need_render = afficher or (step % save_interval == 0)
    if need_render:
        
        ### PARTICULES
        if not dezoom and step%10 == 0:
            if np.any(np.abs(px) > L) or np.any(np.abs(py) > L):
                ax1.set_xlim(-1.5*L, 1.5*L)
                ax1.set_ylim(-1.5*L, 1.5*L)
                dezoom = True
            
        scatter.set_offsets(np.c_[px, py])
        scatter.set_array(Lz)
        
        if step%10 == 0:
            scatter.vmin, scatter.vmax = np.percentile(Lz, [1, 99]) 
            norm = colors.SymLogNorm(linthresh=1e-4, vmin=scatter.vmin, vmax=scatter.vmax)
            scatter.set_norm(norm)
        
        ### CENTRE DE GRAVITÉ
            
        xg = np.sum(mass*px)/np.sum(mass)
        yg = np.sum(mass*py)/np.sum(mass)
    
        cg_marker.set_offsets([[xg, yg]])
    
        ax1.set_title(f"Distribution t = {temps:.2f}")
    
        ### POTENTIEL
        
        N_zoom = N // 8  # 64 pixels centraux
        phi_center = phi[N//2 - N_zoom : N//2 + N_zoom,
                         N//2 - N_zoom : N//2 + N_zoom]
        im.set_data(phi_center)
        
        if step%10==0:
            im.vmin, im.vmax = np.percentile(phi, [1, 80])
            vmin_phi, vmax_phi = np.percentile(phi_center, [1, 99])
            norm = colors.SymLogNorm(linthresh=abs(vmin_phi)*0.01 + 1e-6,
                         vmin=vmin_phi, vmax=vmax_phi)
            im.vmin, im.vmax = vmin_phi, vmax_phi
            
            im.set_norm(norm)

        ### REDESSINER
        
        if afficher:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
    
        ### SAUVEGARDE
        
        if step % save_interval == 0:
            fig.savefig(f"{output_dir}/{name}_{step:06d}.png",
                        dpi=200,bbox_inches='tight')

def save_checkpoint(filename,E,G,step,Q_t,V_t,Lz_tot,E_tot,R_W,Mode):
    """Cette fonction est en charge d'enregistrer les données d'intérêt dans un fichier 
    pour recommencer une simulation ou l'arrêter sans perdre les données"""
    data = {
        "step": step,
        "E": E.save_state(),
        "Q_t": Q_t,
        "V_t": V_t,
        "Lz_tot": Lz_tot,
        "E_tot" : E_tot,
        "R_W": R_W,
        "Mode": Mode,
        "BH_step0": G.BH_step0
    }

    tmp = filename + ".tmp"

    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp, filename)

def simulation(Np = 50_000,dt = 1e-3,n_steps = 100,TN=False,MN=False,Stream=False,data=None,afficher=False):
    
    ### INITIALISATION
    np.random.seed(42)
    
    
    dx = 2 * L / (N-1)
    soft = 0.5 * dx
    
    G = Grille(N, L,TN, MN, G0, M_BH, soft)
    Sol = SolveurPoisson(G)
    D = BulbeDiagnostic(G, Sol)
    
    if not data is None:
        with open(f"MainTests\\{data}\\checkpoint.pkl", "rb") as f:
            data = pickle.load(f)

        E = Etoiles(G0)
        E.load_state(data["E"])
        
        step0 = data["step"]
        
        Q_t = data["Q_t"]
        V_t = data["V_t"]
        Lz_tot = data["Lz_tot"]
        E_tot = data["E_tot"]
        R_W = data["R_W"]
        Mode = data["Mode"]
        G.BH_step0 = data["BH_step0"]
        E.L=L
        E.N=N
        E.dx=dx
    else:
        E = Etoiles(G0)
        step0=-1
        E.L=L
        E.N=N
        E.dx=dx
        
        # nombre pair
        Np = (Np // 2) * 2
        
        E.alive=np.ones((Np,),dtype=bool)
        E.liee=np.ones((Np,),dtype=bool)
        E.halo = np.zeros((Np,),dtype = bool)
        
        
        N_bulge = int(0.0 * Np)
        N_disk  = int(1 * Np)
        N_halo  = Np - N_bulge - N_disk
        
        ###  BULBE STABLE 
        
        # Profil de Hernquist : rho ∝ 1/(r*(r+a)^3)
        a_h = 1.5  # rayon d'échelle
        r_max_b = 15.0
        
        # tirage par rejet
        r_b = []
        while len(r_b) < N_bulge:
            r_try = np.random.uniform(0, r_max_b, 10*N_bulge)
            prob   = r_try / (r_try + a_h)**3  # ∝ rho*r (jacobien 2D)
            prob  /= prob.max()
            accept = np.random.uniform(size=len(r_try)) < prob
            r_b.extend(r_try[accept].tolist())
        r_b = np.array(r_b[:N_bulge])
        N_bulge = len(r_b)
        theta_b = np.random.uniform(0, 2*np.pi, N_bulge)
        
        px_b = r_b * np.cos(theta_b)
        py_b = r_b * np.sin(theta_b)
        
        # dispersion isotrope locale : sigma ∝ 1/sqrt(r+a)
        sigma_b_local = 0.3 * v_circ / np.sqrt(r_b/a_h + 1)
        vx_b = np.random.normal(0, sigma_b_local)
        vy_b = np.random.normal(0, sigma_b_local)
          
        ###  DISQUE INSTABLE 
        
        N2 = N_disk // 2
        N_disk=2*N2
        
        r_min = 2.0   # évite r < soft
        r = np.random.exponential(scale=disk_scale, size=N2)
        r = r[r > r_min][:N2]  # rejet des r trop petits
        
        theta = np.random.uniform(0, 2*np.pi, len(r))
        
        r_full = np.concatenate([r, r])
        N_disk = len(r_full)
        theta_full = np.concatenate([theta, theta + np.pi])
        
        px_d = r_full * np.cos(theta_full)
        py_d = r_full * np.sin(theta_full)
        E.px=px_d
        E.py=py_d
        E.mass=np.ones((N_disk,),dtype=float)/(N_bulge+N_disk+N_halo)
        
        G.maj_rho(E.px, E.py, E.mass)
        phi,gx,gy=Sol.solve()
        gx, gy = G.calcul_G(gx, gy)  # champ calculé
        gx_i = G.interpolation(gx,px_d,py_d,dx,L,N)
        gy_i = G.interpolation(gy,px_d,py_d,dx,L,N)
        g_r = (px_d*gx_i + py_d*gy_i)/r_full
        v_c_local = np.sqrt(np.maximum(-r_full * g_r, 0))  # vitesse circulaire locale
        plt.scatter(r_full,v_c_local)
        plt.title('v_c')
        plt.show()
        theta_full_dir = np.arctan2(py_d, px_d)
        vx_d = -v_c_local * np.sin(theta_full_dir)
        vy_d =  v_c_local * np.cos(theta_full_dir)
        
        ###  HALO DIFFUS
        
        N2 = N_halo // 2
        N_halo=2*N2
        
        # Halo : profil isotherme tronqué, raccordé au disque
        r_core_h = 15.0
        r_h = r_core_h * np.tan(np.random.uniform(0, np.arctan(0.7*L/r_core_h), N2))
        theta = np.random.uniform(0, 2*np.pi, N2)
        
        r_full = np.concatenate([r_h, r_h])
        theta_h = np.concatenate([theta, theta + np.pi])
        
        px_h = r_full * np.cos(theta_h)
        py_h = r_full * np.sin(theta_h)
        
        
        vx_h = 0.2*v_circ * ((-py_h/r_full) + 0.2*np.random.normal(size=N_halo))
        vy_h = 0.2*v_circ * ((px_h/r_full) + 0.2*np.random.normal(size=N_halo))
        
        ### CONCATÉNATION
        
        Np = N_bulge + N_disk + N_halo
        E.leffectif=[Np]
        
        E.px = np.concatenate([px_b, px_d, px_h])
        E.py = np.concatenate([py_b, py_d, py_h])
        
        E.vx = np.concatenate([vx_b, vx_d, vx_h])
        E.vy = np.concatenate([vy_b, vy_d, vy_h])
        
        E.px -= np.mean(E.px)
        E.py -= np.mean(E.py)
        
        E.vx -= np.mean(E.vx)
        E.vy -= np.mean(E.vy)
        
        # soft 
        E.r = np.sqrt(E.px**2 + E.py**2 + soft**2)
        
        # alive / liee
        E.alive = np.ones_like(E.px, dtype=bool)
        E.liee = np.ones_like(E.px, dtype=bool)
        
        # masses
        E.mass = np.ones_like(E.px) / Np
        
        ### AJUSTEMENT FINAL VITESSES
        
        E.vx += np.random.normal(size=Np)*random_dispersion*0.1
        E.vy += np.random.normal(size=Np)*random_dispersion*0.1
        
        Q_t=[]
        V_t=[]
        R_W=[]
        E_tot=[]
        
        ### MODES
        m=np.linspace(1,10,10)
        Mode = {}
        for i in m:
            Mode[f'Mode {i}'] = [abs(np.sum(E.mass*np.exp(1j*i*np.arctan2(E.px,E.py))))/np.sum(E.mass[E.alive])]
        
        ### Lz
        theta = np.arctan2(E.py, E.px)
        L_z0=np.sum(E.mass*E.r*(-np.sin(theta)*E.vx + np.cos(theta)*E.vy))
        Lz_tot=[L_z0]
   
    ### CHAMP INITIAL
    
    G.maj_rho(E.px, E.py, E.mass)
    phi,gx,gy=Sol.solve()
    gx,gy = G.calcul_G(gx, gy)
    
    if data is None:
        ### dt ADAPTATIF
        
        v_char = np.percentile(np.sqrt(E.vx**2 + E.vy**2), 95)
        a_char = np.percentile(np.sqrt(gx**2 + gy**2), 95)
        dt_min = 0.05 * dx / v_char
        dt_max = 0.1 * min(np.sqrt(dx/a_char) , dx / v_char)
        
        E.dt = np.clip(dt, dt_min, dt_max)  
        
    G.maj_phi(phi)
    
    ### Q 
    phi_interp = G.interpolation(phi, E.px, E.py, dx, L, N)
    W_brut = 0.5 * np.sum(E.mass * phi_interp)
    T = 0.5 * np.sum(E.mass * (E.vx**2 + E.vy**2))
    
    W_grid = 0.5 * np.sum(G.rho * G.phi * G.dx**2)

    ### ENERGIE POTENTIELLE PARTICULES
    W_fft = 0.5 * np.sum(Sol.Epot(E.px[E.alive], E.py[E.alive], E.mass[E.alive]))
   
    ### VIRIEL
    Q = -2 * T / W_brut
    print(f"T = {T:.4e}  W_brut = {W_brut:.4e}  max(phi) = {np.max(phi):.4e}")
    print(f"Q = {Q:.4f}")
    
    print(f"max(phi) = {np.max(phi):.4e}")  # doit être ~0 ou très petit négatif
    print(f"W_brut = {0.5*np.sum(E.mass * G.interpolation(phi, E.px, E.py, dx, L, N)):.4e}")
    print(f"W_shifté = {0.5*np.sum(Sol.Epot(E.px, E.py, E.mass)):.4e}")
    print("ratio =", W_fft/W_grid)
    
    
    
    ### CALIBRAGE DU BULBE,HALO,TROU NOIR
    if MN:
        r_inf = D.bulge_influence_radius()
    else:
        r_inf = D.bulge_influence_radius_BH()
    for k, v in r_inf.items():
        print(k, ":", v)
    
    Q_t.append(Q)
    V_t.append(2*T+W_fft)
    
    r_bins = np.linspace(0, 20, 100)
    phi_rad = [np.mean(phi[np.abs(G.R - r) < 0.5*dx]) for r in r_bins]
    plt.semilogy(r_bins, -np.array(phi_rad))
    plt.title('Phi')
    plt.show()
    
    SI = int(input('Nombre de screen ?'))
    
    output_dir = os.path.join("MainTests", "Test_"+f'{time.strftime("%Y-%m-%d %H_%M_%S")}')
    fig, ax1, ax2, scatter,cg_marker, im,dezoom = init_affichage(N,L,E.px,E.py)
    os.makedirs(output_dir,exist_ok=True)
    file_path = os.path.join(output_dir, "parameters.txt")
    
    if data is None:
        params_dict = {
        "N": N,
        "L": L,
        "G0": G0,
        "soft": soft,
        "Nombre d'étoiles dans le bulbe" : N_bulge,
        "Nombre d'étoiles dans le disque" : N_disk,
        "Nombre d'étoiles dans le halo diffus" : N_halo,
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
        "Stream":Stream}
    else:    
        theta = np.arctan2(E.py,E.px)
        m=np.linspace(1,10,10)
        params_dict = {'data' : data }
        
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Simulation parameters ===\n\n")
        f.write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        for key, value in params_dict.items():
            f.write(f"{key} = {value}\n")
        f.write(f'Q = {Q}\n')
        f.write(f'Ratio FFT / grid : {W_fft/W_grid}\n')
        f.write(f'Masse halo : {G.M_h}\n')
        f.write(f'Masse Trou noir : {G.M_BH}\n')
        for k, v in r_inf.items():
            f.write(f"{k} = {v}\n")
            
    affichage(fig, ax1, ax2, scatter, cg_marker,im,dezoom, E.px, E.py, E.mass,E.mass*E.r*(-np.sin(theta)*E.vx + np.cos(theta)*E.vy), G.phi, 0, True, 'initial', 0, 1, dx, output_dir)
    
    S=[Streams((-L,0), (1,1), 4, 2e-2/(Np*E.dt), 0.2*v_circ, 1/Np),Streams((L,0), (-1,-1), 4, 2e-2/(Np*E.dt), 0.2*v_circ, 1/Np)]
    
    ### INTÉGRATION
    steps_fin=0
    
    E0=W_fft + T
    E_tot.append(np.sum(E0))
      
    gx,gy=G.calcul_G(gx,gy)
    Fx, Fy = E.calcul_force(gx, gy, N, L)
    
    Fx_tot,Fy_tot = [np.sum(Fx)],[np.sum(Fy)]
    Fxg_tot,Fyg_tot = [np.sum(gx*G.rho)],[np.sum(gy*G.rho)]
    
    try:
        for step in range(step0 + 1,n_steps):
            
            #demi-kick initial
            E.maj_vit(0.5*Fx,0.5*Fy)
            #drift
            E.maj_pos()
            
            ### MAJ TN
            
            # déclenchement quand le mode m=2 atteint le seuil
            m2 = np.mean(Mode['Mode 2.0'][-50:])
            if TN and G.BH_step0 is None and m2 > G.seuil_mode and step > 500:
                G.BH_step0 = step
                print(f"Croissance du TN déclenchée au step {step}")
            
            # croissance (uniquement pendant la rampe, pour éviter des maj inutiles)
            if TN and G.BH_step0 is not None and step <= G.BH_step0 + G.N_grow:
                G.maj_BH(G.M_BH_t(step))
            #On ajoute les particules provenant des streams
            if Stream:
                for s in S:
                    E.px, E.py,E.r, E.vx, E.vy, E.alive,E.liee, E.mass = s.inject(E.px, E.py,E.r, E.vx, E.vy, E.alive,E.liee, E.mass, E.dt,soft)
            
            
            E.maj_status(R_lim,L,Sol.Epot(E.px, E.py, E.mass),G.phi,TN) 
            steps_fin+=1
            if np.sum(E.alive) < 0.01*Np:
                print("Galaxie dissoute")
                break
        
            X,Y,VX,VY,M =E.spec_palive()
            
        
            G.maj_rho(X, Y, M)
            
            phi,gx,gy = Sol.solve() #induit par les étoiles
            
            G.maj_phi(phi)
            Fxg, Fyg = G.calcul_G(gx,gy)
            
            Fx, Fy = E.calcul_force(Fxg, Fyg, N, L)
            Fx_tot.append(np.sum(Fx))
            Fy_tot.append(np.sum(Fy))
            Fxg_tot.append(np.sum(gx*G.rho))
            Fyg_tot.append(np.sum(gy*G.rho))
            
            #Demi-kick final
            E.maj_vit(0.5*Fx,0.5*Fy)
            
            theta = np.arctan2(E.py,E.px)
            Lz=E.mass*E.r*(-np.sin(theta)*E.vx + np.cos(theta)*E.vy)
            affichage(fig, ax1, ax2, scatter,cg_marker, im, dezoom,X, Y, E.mass[E.alive],Lz[E.alive], phi, E.time, False, f'{step}', step, max(1,n_steps//SI), dx, output_dir) #Le 30.0 coreespond à un rayon dynamique
            
            E_= Sol.Epot(E.px, E.py, E.mass)+0.5*E.mass*(E.vx**2+E.vy**2)
            E_tot.append(np.sum(E_[E.alive]))
            
            Lz_tot.append(np.sum(Lz[E.alive]))
            
            T=0.5*np.sum(E.mass[E.alive]*(E.vx[E.alive]**2+E.vy[E.alive]**2))
            W_grid = np.sum(G.rho * (0.5*phi + G.phi_0) * G.dx**2)
            W_fft=0.5*np.sum(Sol.Epot(E.px[E.alive], E.py[E.alive], E.mass[E.alive]))
            
            Q=2*T/W_fft
            
            Q_t.append(abs(Q))
            V_t.append(2*T + W_fft)
            
            ### MODES
            for i in m:
                Mode[f'Mode {i}'].append(abs(np.sum(E.mass[E.liee]*np.exp(1j*i*np.arctan2(E.px[E.liee],E.py[E.liee]))))/np.sum(E.mass[E.liee]))
    
            ### RAPPORT R_W
            
            # énergie potentielle grille
            W_grid = 0.5 * np.sum(G.rho * G.phi * G.dx**2)
    
            # énergie potentielle particules
            W_fft = 0.5 * np.sum(Sol.Epot(E.px[E.alive], E.py[E.alive], E.mass[E.alive]))
            
            R_W.append(W_fft/W_grid)
            
            print(str(100*(step+1)/n_steps)+' %')
            steps_fin+=1
            
            if step % 500 == 0 and step > 0:

                save_checkpoint(os.path.join(output_dir, "checkpoint.pkl"),
                    E,G,step,Q_t,V_t,Lz_tot,E_tot,R_W,Mode)
                
                phi_rad = [np.mean(phi[np.abs(G.R - r) < 0.5*dx]) for r in r_bins]
                plt.semilogy(r_bins, -np.array(phi_rad))
                plt.title(f'Phi à {step}')
                plt.show()
            
                print(f"Checkpoint sauvegardé (step {step})")
    except KeyboardInterrupt:
        print("\nInterruption utilisateur")

        save_checkpoint(
            os.path.join(output_dir, "checkpoint.pkl"),
            E,G,step,Q_t,V_t,Lz_tot,E_tot,R_W,Mode)
    
        print("Checkpoint final sauvegardé")


    ### AFFICHAGE FINAL
    
    X,Y,VX,VY,M =E.spec_palive()
    
    G.maj_rho(X, Y, M)
    phi,gx,gy = Sol.solve() #induit par les étoiles
    G.maj_phi(phi)
    
    theta = np.arctan2(Y,X)
    affichage(fig, ax1, ax2, scatter, cg_marker,im,dezoom, X, Y, M,
              M*np.sqrt(X**2+Y**2)*(-np.sin(theta)*VX + np.cos(theta)*VY),
              G.phi, E.time, True, 'final', step, 1, dx, output_dir)
    
    print(f'Durée : {E.time}')
    
    ### DIAGNOSTICS
    
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    v = np.sqrt(VX**2 + VY**2)
    v_th = -np.sin(theta)*VX + np.cos(theta)*VY
    
    plt.figure(figsize=(18,10))
    
    # 1. vitesse radiale
    plt.subplot(2,3,1)
    plt.plot(r[r<L], v[r<L], '.', ms=0.5,color='b')
    plt.xlabel("r")
    plt.ylabel("|v|")
    
    
    # 2. distribution vitesses
    plt.subplot(2,3,2)
    plt.hist(np.log10(v + 1e-12), bins=50)
    plt.xlabel("log10(v)")
    

    # 3. vitesse tangentielle
    plt.subplot(2,3,3)
    plt.hist(v_th, bins=50)
    plt.xlabel("v_th")
    
    
    # 4. évolution effectifs
    plt.subplot(2,3,4)
    plt.plot(E.leffectif)
    plt.xlabel("step")
    plt.ylabel("effectifs")
    
  
    # 5. évolution énergie/Lz
    plt.subplot(2,3,5)
    plt.plot(E_tot,label='Energie',color='b')
    plt.plot(Lz_tot,label='Lz',color='r')
    plt.plot(R_W,color='g')
    plt.xlabel("step")
    plt.ylabel("Energie")
    plt.grid()
    
    
    # 6. Viriel
    plt.subplot(2,3,6)
    plt.plot(Q_t,label='Q',color='r')
    plt.plot(V_t,label='V',color='b')
    plt.xlabel("step")
    plt.grid()
    
    plt.tight_layout()
            
    plt.savefig(os.path.join(output_dir, "Diagnostics.png"), dpi=250)    

    plt.show()
    
    print(f'Etoiles restantes : {100*E.effectif()/Np} %')
    
    for i in m:    
        plt.plot(Mode[f'Mode {i}'],label=f'Mode {i}')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Modes.png"), dpi=250)
    plt.show()     
    
    if MN:
        r_inf = D.bulge_influence_radius()
    else:
        r_inf = D.bulge_influence_radius_BH()
    
    plt.savefig(os.path.join(output_dir, "g_final.png"), dpi=250)  
    
    plt.plot(Fx_tot,label='Fx_tot')
    plt.plot(Fy_tot,label='Fy_tot')
    plt.legend()
    plt.show()
    
    plt.plot(Fxg_tot,label='Fxg_tot')
    plt.plot(Fyg_tot,label='Fyg_tot')
    plt.legend()
    plt.show()