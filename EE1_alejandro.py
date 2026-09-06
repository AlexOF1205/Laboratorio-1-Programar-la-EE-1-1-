import numpy as np
import matplotlib.pyplot as plt

class EE1mas1:
    def __init__(self, fitness_fn, dim, lim_min, lim_max, sigma, generaciones, seed=None):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.sigma = sigma
        self.generaciones = generaciones
        self.padre = None
        self.aptitud_padre = None
        self.historial = []
        self.rng = np.random.default_rng(seed)

    def inicializar(self):
        self.padre = self.rng.uniform(self.lim_min, self.lim_max, size = self.dim) 
        self.aptitud_padre = self.fitness_fn(self.padre)

    def mutar(self, x):
        ruido = self.rng.normal(0,self.sigma, size = self.dim)
        hijo = x + ruido
        hijo = np.clip(hijo, self.lim_min, self.lim_max)
        return hijo

    def ejecutar(self):
        self.inicializar()
        for i in range(self.generaciones):
            hijo = self.mutar(self.padre)
            aptitud_hijo = self.fitness_fn(hijo)
            if aptitud_hijo <= self.aptitud_padre:
                self.padre = hijo
                self.aptitud_padre = aptitud_hijo
            self.historial.append(self.aptitud_padre)
        return self.padre, self.aptitud_padre, self.historial

def rastrigin(x, A = 10):
    n = len(x)
    return A*n + np.sum(x**2-A*np.cos(2*np.pi*x))


class Lanzamiento:
    def __init__(self, g = 9.81, x_muro = 40, h_muro = 15, x_blanco = 80, theta_min = 10, theta_max = 80, v_min = 5, v_max = 50):
        self.g = g
        self.x_muro = x_muro
        self.h_muro = h_muro
        self.x_blanco = x_blanco
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.v_min = v_min
        self.v_max = v_max

    def altura_en(self, x_pos, theta, v):
        theta_rad = np.deg2rad(theta)
        vx = v*np.cos(theta_rad)
        vy = v*np.sin(theta_rad)
        t = x_pos/vx
        y = vy*t - 0.5*self.g*t**2
        return y
    
    def alcance(self, theta, v):
        theta_rad = np.deg2rad(theta)
        R = v**2 * np.sin(2*theta_rad)/self.g
        return R

    def aptitud(self, x):
        theta = x[0]
        v = x[1]
        R = self.alcance(theta, v)
        error = abs(R - self.x_blanco)
        if self.altura_en(self.x_muro, theta, v) < self.h_muro:
            error = error + 1000
        return error
        

lanzamiento = Lanzamiento()
ee = EE1mas1(
    fitness_fn = lanzamiento.aptitud,
    dim = 2,
    lim_min = [lanzamiento.theta_min, lanzamiento.v_min],
    lim_max = [lanzamiento.theta_max, lanzamiento.v_max],
    sigma = 0.5,
    generaciones = 1000
)

mejor_x, mejor_aptitud, historial = ee.ejecutar()
print("Mejor x encontrado: ", mejor_x)
print("Mejor aptitud:", mejor_aptitud) 

plt.plot(historial)
plt.xlabel("Generación")
plt.ylabel("Aptitud (Error al blanco)")
plt.title("Curva de convergencia - EE(1+1) en el lanzamiento")
plt.show()


''' MAIN

ee_rastrigin = EE1mas1(
    fitness_fn = rastrigin,
    dim = 2,
    lim_min = -5.12,
    lim_max = 5.12,
    sigma = 0.5,
    generaciones = 2000,
    seed = 42
)

mejor_x, mejor_aptitud, historial = ee_rastrigin.ejecutar()
print("Mejor x encontrado: ", mejor_x)
print("Mejor aptitud:", mejor_aptitud) 

plt.plot(historial)
plt.xlabel("Generación")
plt.ylabel("Aptitud (Rastrigin)")
plt.title("Curva de convergencia - EE(1+1) en Rastrigin")
plt.show()'''
# IRONEDIT:1788653576:ux23ii012:a4e3847bcdda45c923f688ae3834e1b6c2b224458f45afd426b9c9103a62e45e
