"""
Diagnostic : fraction de particules de stream "capturables" (E<0)
en fonction de width et v, pour une géométrie radiale
(u pointant vers le centre, origin à distance D du centre).

Approche analytique simplifiée (pas besoin de tourner la grille FFT) :

 - origin à distance D du centre, u = direction radiale vers le centre
 - perp_offset ~ N(0, width)  ->  paramètre d'impact b = |perp_offset|
 - r_spawn = sqrt(D^2 + perp_offset^2)   (toujours >= D, donc >= L si D=L)
 - phi_self(r) ~ -G0*M_tot/r   (approx monopole, valable pour r >= L)
 - phi_ext(r)  : TN (trou noir) + halo logarithmique si activés
 - E = 0.5*v^2 + phi_self(r) + phi_ext(r)
 - particule "capturable" si E < 0

Sert à dégrossir le choix de (width, v, D, TN/MN) AVANT de relancer
une simulation longue. Pour une estimation plus précise, remplacer
phi_self(r) par une interpolation du vrai champ G.phi (en dehors
de la grille -> retomber sur le monopole de toute façon).
"""

import numpy as np
import matplotlib.pyplot as plt

# ===================== Paramètres physiques =====================
G0    = 1.0
M_tot = 1.0          # masse totale du disque (normalisée à 1, cf E.mass.sum()==1)
L     = 150.0
D     = L            # distance origine -> centre du domaine (par défaut = L)
R_lim = 1.8*L

# Composantes externes (mettre TN=True / MN=True si tu testes avec)
TN = False
MN = True
M_BH    = 1e-3
soft_BH = 0.5 * 2*L/512
v_h     = 0.30
R_MN    = 7.0


def phi_ext(r):
    phi = np.zeros_like(r)
    if TN:
        phi += -G0 * M_BH / np.sqrt(r**2 + soft_BH**2)
    if MN:
        phi += 0.5 * v_h**2 * np.log(r**2 + R_MN**2)
    return phi


def phi_self(r):
    return -G0 * M_tot / r


# ===================== Échantillonnage =====================
N_samples  = 200_000
widths     = np.array([2, 5, 10, 20, 40, 80, 120])
# v=0.6 = valeur actuelle (0.2*v_circ avec v_circ=3) ; les autres pour comparaison
velocities = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 1.0])

frac_captured = np.zeros((len(widths), len(velocities)))

for iw, width in enumerate(widths):
    perp = np.random.normal(0, width, N_samples)
    r    = np.sqrt(D**2 + perp**2)
    phi  = phi_self(r) + phi_ext(r)
    phi_lim = phi_self(R_lim) + phi_ext(R_lim)
    for iv, v in enumerate(velocities):
        E = 0.5 * v**2 + phi
        frac_captured[iw, iv] = np.mean(E < phi_lim)

# ===================== Carte width x v =====================
plt.figure(figsize=(8, 6))
im = plt.imshow(frac_captured, origin='lower', aspect='auto',
                 extent=[velocities.min(), velocities.max(),
                         widths.min(), widths.max()],
                 cmap='viridis')
plt.colorbar(im, label='fraction E<0 (capturable)')
plt.xlabel('v (vitesse du stream)')
plt.ylabel('width (= dispersion de b)')
plt.title(f'Fraction capturée — D = {D}, TN={TN}, MN={MN}')
plt.tight_layout()
plt.savefig('capture_fraction.png', dpi=150)
plt.show()

# ===================== Distributions b et r pour un width donné =====================
width_test = 40
perp = np.random.normal(0, width_test, N_samples)
r = np.sqrt(D**2 + perp**2)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(np.abs(perp), bins=80)
plt.xlabel("paramètre d'impact b")
plt.ylabel("compte")

plt.subplot(1, 2, 2)
plt.hist(r, bins=80)
plt.xlabel("r_spawn")
plt.ylabel("compte")

plt.suptitle(f'width = {width_test}, D = {D}')
plt.tight_layout()
plt.savefig('b_r_distribution.png', dpi=150)
plt.show()

print("Fraction capturée (lignes = widths, colonnes = velocities) :")
print("widths     =", widths)
print("velocities =", velocities)
print(frac_captured)
