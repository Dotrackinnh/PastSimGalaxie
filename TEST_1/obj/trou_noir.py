import numpy as np
from obj.grille import Grille

class Trou_noir:

    def __init__(self,g : Grille,M):
        self.g=g
        self.m=M
        self.R=np.sqrt(self.g.X**2+self.g.Y**2+0.01*self.g.soft**2)
        
        self.phi_B=-self.g.G*self.m/self.g.R

    def calcul_forceB(self,X,Y,mass):
        r_=(X**2+Y**2+self.soft**2)**1.5
        Fx = -self.G * self.m * mass * X / r_
        Fy = -self.G * self.m * mass * Y / r_
        return Fx,Fy
    