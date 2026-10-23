# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:12:00 2026

@author: Utilisateur
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import matplotlib.colors as colors
import pickle


L=150
m=np.linspace(1,10,10)

sys.path.append(r'C:\Users\Utilisateur\Desktop\Test_numba')
from pathlib import Path
import pickle
from obj.etoile import Etoiles
base = Path(r'C:\Users\Utilisateur\Desktop\Test_numba')
file = base / "MainTests" / "Test_2026-06-27 17_15_04" / "checkpoint.pkl"

print(file.exists())  # debug important

with file.open("rb") as f:
    data = pickle.load(f)

E = Etoiles(1)
E.load_state(data["E"])
step0 = data["step"]
m=np.linspace(1,10,10)
step_diag = data["step_diag"]
Q_t = data["Q_t"]
V_t = data["V_t"]
Lz_tot = data["Lz_tot"]
E_tot = data["E_tot"]
R_W = data["R_W"]
Mode = data["Mode"]
BH_step0 = data["BH_step0"]


print(f"Diagnositc de l'étape {step_diag[-1]} :")
print(f"Croissance du TN déclenchée au step {BH_step0}")
print(f"Cohérence particules / grille : {R_W[-1]:.5f}")
print(f"Q : {Q_t[-1]:.3f}")
print(f"Variation de Q : {Q_t[-1]-Q_t[0]:.3f}")
print(f"Variation de population : {E.leffectif[-1]-E.leffectif[0]} | {100 * (E.leffectif[-1]-E.leffectif[0])/E.leffectif[0]} %")

N_half = np.sum(E.liee * E.alive)//2

R_ = np.sort(E.r)
R_25 = np.max(R_[:N_half//2])
R_50 = np.max(R_[:N_half])
print(f'Rayon 25% : {R_25:.1f}')
print(f'Rayon 50% : {R_50:.1f}')

# ========================
# DIAGNOSTICS
# ========================
X,Y,VX,VY,M =E.spec_palive()

r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
v = np.sqrt(VX**2 + VY**2)
v_th = -np.sin(theta)*VX + np.cos(theta)*VY

plt.figure(figsize=(18,10))

# ------------------------
# 1. vitesse radiale
# ------------------------
#mask_ = E.alive * E.liee * (1-E.halo) #appartenant à la galaxie
plt.subplot(2,3,1)
R=50

plt.plot(r[r<R], v[r<R], '.', ms=0.5,color='b')
#plt.plot(np.sqrt(E.px**2+E.py**2)[mask_] ,v[mask_],'.',ms='0.5',color='r')
plt.xlabel("r")
plt.ylabel("|v|")

# ------------------------
# 2. distribution vitesses
# ------------------------
plt.subplot(2,3,2)
plt.hist(v, bins=100)
plt.xlabel("v")

# ------------------------
# 3. vitesse tangentielle
# ------------------------
plt.subplot(2,3,3)
plt.hist(v_th, bins=100)
plt.xlabel("v_th")

# ------------------------
# 4. évolution effectifs
# ------------------------
plt.subplot(2,3,4)
plt.plot(E.leffectif)
plt.xlabel("step")
plt.ylabel("effectifs")

# ------------------------
# 5. évolution énergie/Lz
# ------------------------
plt.subplot(2,3,5)
plt.plot(step_diag,E_tot,label='Energie',color='b')
plt.plot(step_diag,(Lz_tot-Lz_tot[0])/Lz_tot[0],label='Lz',color='r')
plt.plot(step_diag,R_W,color='g')
plt.xlabel("step")
plt.ylabel("Energie")
plt.grid()

# ------------------------
# 6. Viriel
# ------------------------
plt.subplot(2,3,6)
plt.plot(step_diag,Q_t,label='Q',color='r')
plt.plot(step_diag,V_t,label='V',color='b')
plt.xlabel("step")
plt.grid()

plt.tight_layout()
           

plt.show()


for i in m:    
    plt.plot(step_diag,Mode[f'Mode {i}'],label=f'Mode {i}')
plt.grid()
plt.legend()
plt.show()
