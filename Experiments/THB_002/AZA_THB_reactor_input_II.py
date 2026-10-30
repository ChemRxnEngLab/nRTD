import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
from scipy.constants import R
from scipy.integrate import solve_ivp

# Import custom modules
from AZA_THB_solver import *
from AZA_THB_tools import *

if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')


class Reactor_input:


    def __init__(self, x_i_RTD, t_RTD, t_lin, cycle_time=240.0, plateau_window=5.0):
        self.x_i_RTD = x_i_RTD
        self.t_RTD = t_RTD
        self.t_lin = t_lin
        self.cycle_time = float(cycle_time)
        self.plateau_window = float(plateau_window)   # Mittelungsfenster der Plateaus, s

    def extract_half_periods(self):
        """Reihenfolge der Laeufe in der nRTD-Datei: 4i = Ar auf, 1+4i = He ab,
        2+4i = Ar ab, 3+4i = He auf (an den Rohdaten geprueft)."""
        x_i_RTD = self.x_i_RTD
        n_periods = x_i_RTD.shape[0] // 4
        x_Ar_bu, x_Ar_bt, x_He_bu, x_He_bt = [], [], [], []

        for i in range(n_periods):
            x_Ar_bu.append(x_i_RTD[    4 * i, 0, ...])
            x_He_bt.append(x_i_RTD[1 + 4 * i, 0, ...])
            x_Ar_bt.append(x_i_RTD[2 + 4 * i, 0, ...])
            x_He_bu.append(x_i_RTD[3 + 4 * i, 0, ...])

        return np.array(x_Ar_bu), np.array(x_Ar_bt), np.array(x_He_bu), np.array(x_He_bt)

    def averaging_half_periods(self):
        x_Ar_bu, x_Ar_bt, x_He_bu, x_He_bt = self.extract_half_periods()
        return (np.average(x_Ar_bu, axis=0), np.average(x_Ar_bt, axis=0),
                np.average(x_He_bu, axis=0), np.average(x_He_bt, axis=0))

    def stacking(self):
        a_bu, a_bt, h_bu, h_bt = self.averaging_half_periods()
        x_Ar_avg = np.hstack((a_bu, a_bt[1:]))
        x_He_avg = np.hstack((h_bu, h_bt[1:]))
        return x_Ar_avg, x_He_avg

    def interpolating(self, t_new=None, plot=False):
        """Gemittelte Ar-/He-Antwort auf ein beliebiges Zeitgitter interpolieren."""
        if t_new is None:
            t_new = self.t_lin

        x_Ar_avg, x_He_avg = self.stacking()
        t_old = np.linspace(0, self.cycle_time, x_Ar_avg.size)

        x_Ar_int = interpolation_function(t_new, t_old, x_Ar_avg)
        x_He_int = interpolation_function(t_new, t_old, x_He_avg)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(t_new, x_Ar_int, label='Ar')
            ax.plot(t_new, x_He_int, label='He')
            ax.plot(t_new, x_Ar_int + x_He_int, label='Ar + He')
            ax.set_xlabel('t / s'); ax.set_ylabel('$x_i$ / 1'); ax.legend()
            plt.show()

        return x_Ar_int, x_He_int

    def _cumulative(self, t_grid, Ar_int):
        """F_Ar aus der Ar-Antwort, Plateaus ueber ein Zeitfenster gemittelt."""
        dt = t_grid[1] - t_grid[0]
        n_win = max(1, int(round(self.plateau_window / dt)))

        peak_idx = int(np.argmax(Ar_int))
        x_qss_up = np.mean(Ar_int[max(0, peak_idx - n_win + 1): peak_idx + 1])
        x_qss_down = np.mean(Ar_int[-n_win:])

        return (Ar_int - x_qss_down) / (x_qss_up - x_qss_down)

    def F_Ar(self, plot=False):
        """F_Ar auf t_lin (Verhalten wie bisher, nur gridunabhaengiges Plateaufenster)."""
        Ar_int = self.interpolating(self.t_lin, plot=plot)[0]
        return self._cumulative(self.t_lin, Ar_int)

    def F_Ar_on_grid(self, dt=0.25, plot=False):
        """F_Ar auf dem RTD-Gitter. Rueckgabe: (t_grid, F_Ar).

        dt muss dem Zeitschritt der Kerne entsprechen, damit die ganze
        Faltungskette ohne Zwischeninterpolation auf einem Gitter laeuft.
        """
        t_grid = np.arange(0, self.cycle_time + dt / 2, dt)
        Ar_int = self.interpolating(t_grid, plot=plot)[0]
        return t_grid, self._cumulative(t_grid, Ar_int)


    # ------------------------------------------------------------------ Variante A
    def F_dosing_step(self, t_E1, E1, dt=0.25, bu_fraction=0.5, lead_in=1.0, plot=False):
        """F fuer das Bett aus dem DOSIERSPRUNG, gefaltet mit E1 (Einlassleitung).

        Warum nicht die gemessene Einlassantwort: die Messung am Reaktoreinlass
        entsteht durch die MS-Entnahme, enthaelt also die Kapillare (E4). Wird sie
        als Eintrittssignal benutzt und der Austritt anschliessend mit E3 (x) E4
        gefaltet, wirkt E4 zweimal - der Modellverzug ist um tau(E4) = 11.9 s zu
        gross und die Flanken um ~35 % zu breit. Das Bett darf nur mit dem
        sehen, was tatsaechlich vor ihm liegt: Dosiersprung (x) E1.

        Das Dosiersignal in der nRTD-Datei ist ein exaktes Rechteck (an allen vier
        Konfigurationen, beiden Richtungen und allen Wiederholungen geprueft:
        Abweichung 0.00 %) mit dem Sprung bei lead_in = 1.0 s. Diese Vorlaufzeit
        gehoert zur Zeitbasis, auf der die Kerne identifiziert wurden, und muss
        deshalb mitgenommen werden.

        Args:
            t_E1, E1: Zeitachse und Kern der Einlassleitung (ts[0], Es[0])
            dt: Zeitschritt der Kerne
            bu_fraction: Anteil der BU-Phase an der Periode
            lead_in: Zeitpunkt des Dosiersprungs innerhalb der Periode, s

        Returns:
            (t_grid, F) mit F = 1 in der BU-Phase, 0 in der BT-Phase
        """
        t_grid = np.arange(0, self.cycle_time + dt / 2, dt)

        i_on = int(round(lead_in / dt))
        i_off = int(round((lead_in + bu_fraction * self.cycle_time) / dt))

        F_step = np.zeros(t_grid.size)
        F_step[i_on:i_off] = 1.0
        F_step[-1] = F_step[0]                      # Periode schliessen

        E1 = np.asarray(E1, dtype=float)
        E1 = E1 / np.trapezoid(E1, np.asarray(t_E1, dtype=float))

        F = self.convolve_periodic(F_step, E1, dt)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(t_grid, F_step, label='Dosiersprung', linestyle='--')
            ax.plot(t_grid, F, label='$F$ am Betteintritt = Dosiersprung $\\otimes$ E1')
            ax.set_xlabel('t / s'); ax.set_ylabel('$F$ / 1'); ax.legend()
            plt.show()

        return t_grid, F

    @staticmethod
    def convolve_periodic(one_period, kernel, dt):
        """Periodische Faltung ueber Kachelung (identisch zu Output.convolve_periodic)."""
        sig = np.atleast_2d(np.asarray(one_period, dtype=float))
        n = sig.shape[1]

        n_rep = int(np.ceil(kernel.size / (n - 1))) + 4
        tiled = np.hstack([sig[:, :-1]] * (n_rep - 1) + [sig])
        mid = (n_rep // 2) * (n - 1)

        out = np.empty_like(sig)
        for i in range(sig.shape[0]):
            out[i] = (np.convolve(tiled[i], kernel) * dt)[mid:mid + n]

        return out.reshape(np.shape(one_period))


display('-AZA_THB_reactor_input- imported at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))