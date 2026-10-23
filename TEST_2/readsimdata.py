# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 05:55:23 2026

@author: verre
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import pickle
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
from pathlib import Path


sys.path.append(r"C:\Users\verre\OneDrive\Desktop\Project\NextTIPE\TEST_2")
base = Path(r'C:\Users\verre\OneDrive\Desktop\Project\NextTIPE\TEST_2\MainTests')


from obj.grille import Grille, SolveurPoisson, BulbeDiagnostic
from obj.etoile import Etoiles
from obj.stream import Streams

N = 512
L = 150.0
G0 = 1.0

v_circ = 0.1
random_dispersion = 0.05 * v_circ
disk_scale = 7.0

R_lim = 1.8 * L
M_BH = 2e-5


path = os.path.join('MainTests\Test_2026-08-17 01_22_00', "checkpoint.pkl")

with open(path, "rb") as f:
    data = pickle.load(f)

E = Etoiles(G0)  # G0 doit être défini/importé dans le contexte
E.load_state(data["E"])
Np = E.leffectif[0]
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


X, Y, VX, VY, M = E.spec_palive()

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
plt.show()

print(f'Etoiles restantes : {100*np.sum(E.liee)/Np:.1f} %')
N_half = np.sum(E.liee * E.alive)//2

R_ = np.sort(E.r)
R_25 = np.max(R_[:N_half//2])
R_50 = np.max(R_[:N_half])
print(f'Rayon 25% : {R_25:.1f}')
print(f'Rayon 50% : {R_50:.1f}')

m = np.linspace(1, 10, 10)
for i in m:
    plt.plot(step_diag,Mode[f'Mode {i}'], label=f'Mode {i}')
plt.grid()
plt.legend()
plt.show()
