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
plt.show()
# IRONEDIT:1788648447:ux23ii012:f10efbbcb3c05342fcdded47a62e64cc46c472c06f00b23e6aa7e000394de9ea
