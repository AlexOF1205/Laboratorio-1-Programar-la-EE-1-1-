import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EE1mas1:
    def __init__(self, fitness_fn, dim, lim_min, lim_max, sigma, generaciones, seed=60):
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.sigma = sigma
        self.generaciones = generaciones
        self.padre = None
        self.aptitud_padre = None
        self.historial = []
        self.historial_x = []
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
            self.historial_x.append(self.padre.copy())
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

    def graficar(self, theta, v):
        R = self.alcance(theta, v)
        xs = np.linspace(0, R, 200)
        ys = self.altura_en(xs, theta, v)

        fig, ax = plt.subplots(figsize = (8,5))
        ax.plot(xs, ys, color = 'royalblue', linewidth = 2, label = 'Trayectoria')
        
        # Muro
        ax.bar(self.x_muro, self.h_muro, width = 2, color = 'dimgray', label = 'Muro')
        
        # Cañon
        ax.scatter(0, 0, s = 200, marker = '^', color = 'black', label = 'Cañon', zorder = 5)

        # Blanco
        ax.scatter(self.x_blanco, 0, s = 200, marker = '*', color = 'gold', edgecolor = 'orange', label = 'Blanco', zorder = 5)

        # Final
        ax.axhline(0, color = 'saddlebrown', linewidth = 2)
        ax.set_xlabel("Distancia (m)")
        ax.set_ylabel("Altura (m)")
        ax.set_title(f"Lanzamiento: theta={theta:.1f}°, v={v:.1f} m/s")
        ax.legend()
        ax.grid(alpha = 0.3)
        plt.show()

def animar_lanzamiento(lanzamiento, frames_x, frames_aptitud):
    fig, ax = plt.subplots(figsize = (9,5.5))
        
    ax.bar(lanzamiento.x_muro, lanzamiento.h_muro, width = 2, color = 'dimgray', label = 'Muro', zorder = 3)
    ax.scatter(0, 0, s = 250, marker = '^', color = 'black', label = 'Cañon', zorder = 5)
    ax.scatter(lanzamiento.x_blanco, 0, s = 250, marker = '*', color = 'gold', edgecolor = 'orange', label = 'Blanco', zorder = 5)
    ax.axhline(0, color = 'saddlebrown', linewidth = 2, zorder = 2)

    ax.set_xlim(-5, max(lanzamiento.x_blanco, lanzamiento.x_muro) + 15)
    ax.set_ylim(0,60)
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Altura (m)")
    ax.grid(alpha = 0.3)

    linea_trayectoria, = ax.plot([], [], color = 'royalblue', linewidth = 2.5, label = 'Trayectoria', zorder = 4)
    punto_proyectil, = ax.plot([], [], 'o', color = 'crimson', markersize = 10, zorder = 6)
    texto_info = ax.text(0.02, 0.95, '', transform = ax.transAxes, fontsize = 11, verticalalignment = 'top', bbox = dict(boxstyle = 'round', facecolor = 'white', alpha = 0.8))
    ax.legend(loc = 'upper right')

    def actualizar(frame_idx): 
        theta, v = frames_x[frame_idx]
        aptitud = frames_aptitud[frame_idx]
        generacion = frame_idx * 10

        R = lanzamiento.alcance(theta, v)
        xs = np.linspace(0, R, 100)
        ys = lanzamiento.altura_en(xs, theta, v)

        linea_trayectoria.set_data(xs, ys)
        punto_proyectil.set_data([xs[-1]], [max(ys[-1], 0)])
        texto_info.set_text(f"Generación: {generacion}\ntheta = {theta:.1f}°  v = {v:.1f} m/s\nAptitud:  {aptitud:.3f}")

        return linea_trayectoria, punto_proyectil, texto_info
    
    anim = FuncAnimation(fig, actualizar, frames=len(frames_x), interval = 150, blit = True, repeat = False)
    plt.show()
    return anim


lanzamiento = Lanzamiento()
ee = EE1mas1(
    fitness_fn = lanzamiento.aptitud,
    dim = 2,
    lim_min = [lanzamiento.theta_min, lanzamiento.v_min],
    lim_max = [lanzamiento.theta_max, lanzamiento.v_max],
    sigma = 0.7,
    generaciones = 1010
)



mejor_x, mejor_aptitud, historial = ee.ejecutar()
print("Mejor x encontrado: ", mejor_x)
print("Mejor aptitud:", mejor_aptitud) 

# lanzamiento.graficar(mejor_x[0], mejor_x[1])

frames_x = ee.historial_x[::10]
frames_aptitud = ee.historial[::10]
anim = animar_lanzamiento(lanzamiento, frames_x, frames_aptitud)


''' MAIN

plt.plot(historial)
plt.xlabel("Generación")
plt.ylabel("Aptitud (Error al blanco)")
plt.title("Curva de convergencia - EE(1+1) en el lanzamiento")
plt.show()

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
# IRONEDIT:1788658854:ux23ii012:c65a050f19a917175af3640563a32c5bc97489d19d4b6b95c5e557ef8af5d961
