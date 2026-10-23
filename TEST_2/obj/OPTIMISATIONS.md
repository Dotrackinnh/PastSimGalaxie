# Optimisation des modules (drop-in, API inchangée)

Les trois fichiers remplacent directement les originaux : `main.py` n'a pas besoin
d'être modifié. Validé numériquement contre les versions originales (écarts à la
précision machine, ~1e-16) et sur 20 pas complets d'intégration leapfrog avec
injection de streams.

## Gains mesurés (N = 512, 50 000 particules, 1 cœur)

| Étape            | Avant     | Après    | Gain  |
|------------------|-----------|----------|-------|
| `maj_rho` (CIC)  | ~9 ms     | ~2,8 ms  | x3,2  |
| `solve` (FFT)    | ~290 ms   | ~70 ms   | x4,1  |
| `calcul_force`   | ~9 ms     | ~5,5 ms  | x1,6  |
| **Total / pas**  | ~310 ms   | ~79 ms   | **x4** |

Sur une machine multi-cœurs le gain sur `solve` sera encore plus grand
(`workers = os.cpu_count()`).

## grille.py — SolveurPoisson

- **Noyau de Green précalculé une seule fois** dans `__init__` (avant : `sqrt`,
  `exp` et division sur la grille (2N)² = 1024² à chaque `solve()`). Les noyaux
  de gradient `-i·k·Green` sont aussi précombinés.
- **FFT réelle** (`rfft2`/`irfft2` au lieu de `fft2`/`ifft2`) : la densité est
  réelle, on ne calcule que la moitié du spectre → ~2x moins de travail.
- **Multithread** : `scipy.fft` avec `workers=os.cpu_count()`.
- **Backend pyFFTW** activé automatiquement s'il est installé
  (`pip install pyfftw`), avec cache de plans. Repli silencieux sur
  scipy/pocketfft sinon.
- **Buffer de padding réutilisé** et slices de crop précalculées : zéro
  allocation de grille (2N)² par pas.
- Modes de Nyquist annulés dans les noyaux de dérivée : strictement équivalent
  au `np.real(ifft2(...))` original (vérifié : écart 4e-16).

## grille.py — Grille

- **`maj_rho` : dépôt CIC via `np.bincount`** sur indices aplatis (ou boucle
  `numba @njit` si numba est installé : `pip install numba`). `np.add.at` est
  notoirement lent car non bufferisé.
- `interpolation` : gather vectorisé `A.ravel().take(i*N+j)` au lieu de 4
  indexations fancy 2D.
- `maj_BH` : `1/R_BH`, `1/R_BH²`, `1/R_BH³` précalculés (appelé 2500 fois
  pendant la croissance du trou noir).
- **Correction d'un bug latent** : une particule exactement en `x = +L` donnait
  `i1 = N` → `IndexError` dans `np.add.at`/l'indexation. Les indices sont
  maintenant clippés (le poids associé est nul, la physique est inchangée).

## etoile.py

- `calcul_force` : interpolation par gather aplati ; les poids CIC ne sont
  calculés que pour les particules dans la boîte (l'original les calculait
  pour toutes, y compris hors boîte, avec des indices potentiellement hors
  limites — et l'affectation `Fx[~out] = ...` avait une incohérence de taille
  dès qu'une particule sortait de la boîte).
- `r3` protégé contre la division par zéro pour le monopôle hors boîte.
- `maj_pos`/`maj_status` : carrés via `x*x` (plus rapide que `x**2`),
  opérations mutualisées.
- `effectif()` : `np.sum` au lieu du `sum` Python.

## stream.py

- Générateur `np.random.default_rng()` (PCG64) : plus rapide que l'API legacy
  `np.random.*` et statistiquement meilleur.
- Moins d'allocations intermédiaires dans `spawn` et `inject`.

## Pistes suivantes (si besoin d'aller plus loin)

1. **Mutualiser les poids CIC** entre `maj_rho` et `calcul_force` : les deux
   travaillent sur les mêmes particules vivantes au même pas → on peut calculer
   les poids une fois et les passer aux deux (encore ~3-4 ms/pas).
2. `float32` pour les particules (positions/vitesses) : x2 sur la bande
   passante mémoire, précision largement suffisante pour ce type de simulation.
3. Décimer l'affichage : `scatter.set_offsets` sur 50k points coûte cher ;
   tracer 1 particule sur 5 suffit visuellement.
4. Si numba est installé, le dépôt CIC est automatiquement compilé ; on peut
   aussi compiler le gather de `calcul_force` de la même façon.
