import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt


class FieldDisplay:
    def __init__(self, maxSize_m, dx, y_min, y_max, probePos, sourcePos):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line = self.ax.plot(np.arange(0, maxSize_m, dx), [0] * int(maxSize_m / dx))[0]
        self.ax.plot(probePos, 0, 'xr')
        self.ax.plot(sourcePos, 0, 'ok')
        self.ax.set_xlim(0, maxSize_m)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('x, м')
        self.ax.set_ylabel('$E_z$, В/м')
        self.ax.grid()

    def updateData(self, data, t):
        self.line.set_ydata(data)
        self.ax.set_title('t = {:.4f} нc'.format(t * 1e9))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def off(self):
        plt.ioff()


class Probe:
    def __init__(self, probePos, Nt, dt):
        self.Nt = Nt
        self.dt = dt
        self.probePos = probePos
        self.t = 0
        self.E = np.zeros(self.Nt)

    def addData(self, data):
        self.E[self.t] = data[self.probePos]
        self.t += 1


def showProbeSignal(probe):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(0, probe.Nt * probe.dt, probe.dt), probe.E)
    ax[0].set_xlabel('t, c')
    ax[0].set_ylabel('$E_z$, B/м')
    ax[0].set_xlim(0, probe.Nt * probe.dt)
    ax[0].grid()
    sp = np.abs(fft(probe.E))
    sp = fftshift(sp)
    df = 1 / (probe.Nt * probe.dt)
    freq = np.arange(-probe.Nt * df / 2, probe.Nt * df / 2, df)
    ax[1].plot(freq, sp / max(sp))
    ax[1].set_xlabel('f, Гц')
    ax[1].set_ylabel('|S|/|$S_{max}$|')
    ax[1].set_xlim(0, 10e9)
    ax[1].grid()
    plt.subplots_adjust(wspace=0.4)
    plt.show()


eps = 4
W0 = 120 * np.pi
Nt = 1750
Nx = 1500
c = 299792458
maxSize_m = 1.5
dx = maxSize_m / Nx
maxSize = int(maxSize_m / dx)
probePos = maxSize_m * 1 / 5
sourcePos = maxSize_m / 2
pp = int(probePos / dx)
sp = int(sourcePos / dx)
Sc = 1
dt = dx * np.sqrt(eps) * Sc / c
probe = Probe(pp, Nt, dt)
display = FieldDisplay(maxSize_m, dx, -1.5, 1.5, probePos, sourcePos)
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)
Ez_old = 0
Sc1 = Sc / np.sqrt(eps)
k = (Sc1 - 1) / (Sc1 + 1)
k1 = -1 / (1 / Sc1 + 2 + Sc1)
k2 = 1 / Sc1 - 2 + Sc1
k3 = 2 * (Sc1 - 1 / Sc1)
k4 = 4 * (1 / Sc1 + Sc1)
Ez_oldL1 = np.zeros(3)
Ez_oldL2 = np.zeros(3)
A_0 = 10
A_max = 10
f_0 = 5.5e9
DeltaF = 5e9
w_g = 2 * np.sqrt(np.log(A_max)) / (np.pi * DeltaF) / dt
d_g = w_g * np.sqrt(np.log(A_0))
Nl = c / f_0 / dx / np.sqrt(eps)
for q in range(1, Nt):
    Hy[1:] = Hy[1:] + (Ez[:-1] - Ez[1:]) * Sc / W0
    Ez[:-1] = Ez[:-1] + (Hy[:-1] - Hy[1:]) * Sc * W0 / eps
    Ez[sp] += np.sin(2 * np.pi / Nl * Sc * q) * np.exp(-((q - d_g) / w_g) ** 2)
    Ez[-1] = Ez_old + k * (Ez[-2] - Ez[-1])
    Ez_old = Ez[-2]
    Ez[0] = (k1 * (k2 * (Ez[2] + Ez_oldL2[0]) + k3 * (Ez_oldL1[0] + Ez_oldL1[2] - Ez[1] - Ez_oldL2[1]) - k4 * Ez_oldL1[
        1]) - Ez_oldL2[2])
    Ez_oldL2[:] = Ez_oldL1[:]
    Ez_oldL1[:] = Ez[:3]
    probe.addData(Ez)
    if q % 20 == 0:
        display.updateData(Ez, q * dt)

display.off()
showProbeSignal(probe)
