"""
===============================================================================
 AZA_THB_reactor_output  --  Digitaler Zwilling: Kinetik + Transport
===============================================================================

WAS DIESES MODUL TUT
--------------------
Es verbindet das kinetische Reaktormodell (ideal durchmischt, geloest in
AZA_THB_solver) mit dem realen Transportverhalten der Anlage (beschrieben durch
die nRTD-Kerne). Ergebnis ist das, was das Massenspektrometer messen wuerde -
also direkt vergleichbar mit den Experimentdaten.

Signalweg:

    n_dot_func(t)        vorgegebener Reaktoreintritt (aus Reactor_input)
          |
          v
    kinetische ODE       AZA_THB_solver.solver -> idealer Reaktor
          |
          v
    STUFE E2             Reaktor-VWZV: korrigiert die Idealannahme um
          |              Rueckvermischung und Dispersion im Festbett
          v
    + Matrixverduennung  H2 / Ne werden HINTER dem Reaktor zudosiert
          |
          v
    eine eingeschwungene Periode ausschneiden
          |
          v
    STUFE E3 (x) E4      Auslassleitung + MS-Kapillare
          |
          v
    n_dot_i(t)           was das MS sieht


WO DIE KERNE HERKOMMEN
----------------------
E1 (Einlassleitung), E2 (Reaktor), E3 (Auslassleitung), E4 (Kapillare) stammen
aus dem nRTD-Training. E1 gehoert NICHT in dieses Modul: die Einlassleitung wirkt
VOR dem Bett und steckt bereits im Eintrittssignal n_dot_func, das
Reactor_input.F_dosing_step() als Dosiersprung (x) E1 liefert. Hier werden nur
E2, E3 und E4 angewandt.

HAEUFIGSTER FEHLER an dieser Stelle: als Eintrittssignal die GEMESSENE Antwort am
Reaktoreinlass zu verwenden. Die enthaelt E1 UND E4, denn jede Messung geht durch
die MS-Kapillare. Faltet man den Austritt anschliessend mit E3 (x) E4, wirkt E4
zweimal - der Modellverzug ist um tau(E4) zu gross und die Flanken sind deutlich
zu breit.


ZWEI VARIANTEN DER FALTUNG
--------------------------
    Output_II      faltet die SPEZIESMOLENSTROEME   (dieses Modul)
    Output_III  faltet die MOLENBRUECHE          (AZA_THB_reactor_output_III.py)

Beide liefern identische Ergebnisse, solange der Gesamtmolenstrom zeitlich
konstant ist - also in allen Tracerlaeufen. Sie unterscheiden sich erst, wenn die
Reaktion die Molzahl aendert (CO2 + 4 H2 -> CH4 + 2 H2O, also 5 mol -> 3 mol).
Die Begruendung beider Wege steht ausfuehrlich im Kopf von Output_III.


AENDERUNGEN GEGENUEBER DER URSPRUNGSFASSUNG (Hygiene-Paket)
-----------------------------------------------------------
  1. Alle Kerne werden auf Integral 1 normiert (Rohwerte lagen bei 0.994..1.003).
     Ueber eine mehrstufige Kette multiplizieren sich diese Fehler auf.
  2. Die ganze Kette laeuft auf dem RTD-Gitter (dt der Kerne, 0.25 s), nicht auf
     t_lin (0.475 s). Auf t_lin wird erst ganz am Ende interpoliert - vorher
     wurde die Aufloesung genau dort halbiert, wo die Flanken entstehen.
  3. Stufe E2 faltet die Speziesmolenstroeme. Vorher wurden Molenbrueche gefaltet
     und der Gesamtstrom separat um tau(E2) VERSCHOBEN. Eine Verschiebung ist
     aber keine Faltung: sie erhaelt den Mittelwert und wirft die gesamte Streuung
     weg. Zaehler und Nenner erfuhren also unterschiedliche Operationen.
  4. Stufe E3 (x) E4 faltet eine volle Periode PERIODISCH (Kachelung, mittlerer
     Zyklus). Vorher wurde jede Halbperiode einzeln mit ihrem konstanten Randwert
     gepaddet und danach zusammengenaeht. E_tot ist rund 55 s lang bei 120 s
     Halbperiode - die Vorgeschichte ist also keineswegs vernachlaessigbar, und
     die Naht erzeugt zusaetzlich einen Sprung.
  5. cycle_time / bu_fraction sind Parameter. Vorher standen 120 / 240 / size/4
     hartkodiert im Code - Erbstuecke der alten 480-s-Konvention, die nach der
     Umstellung auf 240 s alle an der falschen Stelle sassen.
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import glob
from scipy.constants import R
from scipy.integrate import solve_ivp

# Import custom modules
from AZA_THB_solver import *      # solver, n_dot_out, surf_start, support_start
from AZA_THB_tools import *       # interpolation_function, Speziesindizes, p_STP, T_STP

if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')


class Output:
    """Digitaler Zwilling: Kinetik + Transport, Faltung der SPEZIESMOLENSTROEME.

    Typischer Aufruf aus dem Notebook:

        out = Output(ts=CNN_ts, Es=CNN_Es, n_dot_func=n_dot_func, t_eval=t_eval,
                     y_init=y_init, V_R=V_R, n_max=n_max, nx_max=nx_max,
                     p=p, T=T, surf_start=surf_start, n_dot_in_bu=n_dot_in_bu,
                     t_lin=t_lin, ni_out_cycle=ni_out_cycle,
                     ni_trans_cycle=ni_trans_cycle, ni_RTD_cycle=ni_RTD_cycle,
                     number_of_periods=7, switch_delay=38.0,
                     cycle_time=240.0, bu_fraction=0.5,
                     V_dot_H2_matrix=0.0, V_dot_Ne_matrix=0.0)
        res = out.convolution_function(k, T)

    WICHTIG bei Laeufen OHNE Matrixdosierung: V_dot_H2_matrix=0.0 setzen. Der
    Default 250.0 stammt aus den alten Messreihen; bleibt er stehen, addiert das
    Modell 250 nmL/min H2, die es gar nicht gibt, und verfaelscht alle
    Molenbrueche und den Gesamtstrom.
    """

    def __init__(self, ts, Es, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T,
                 surf_start, n_dot_in_bu, t_lin, ni_out_cycle, ni_trans_cycle, ni_RTD_cycle,
                 number_of_periods, switch_delay,
                 cycle_time=240.0, bu_fraction=0.5, periods_from_end=3,
                 V_dot_H2_matrix=250.0, V_dot_Ne_matrix=0.0,
                 normalize_E_tot=True):
        # --- Transport ---
        self.ts = ts                          # Zeitachsen der Kerne [t1, t2, t3, t4]
        self.Es = Es                          # Kerne [E1 Einlass, E2 Reaktor, E3 Auslass, E4 Kapillare]

        # --- Kinetik / ODE ---
        self.n_dot_func = n_dot_func          # Funktion t -> Reaktoreintritt je Spezies, mol s-1
        self.t_eval = t_eval                  # Auswertezeitpunkte des ODE-Loesers, s
        self.y_init = y_init                  # Anfangsbedingung (Molenbrueche + Bedeckungen)
        self.V_R = V_R                        # Reaktorvolumen, m3
        self.n_max = n_max                    # Adsorptionsplaetze Aktivmaterial, mol
        self.nx_max = nx_max                  # Adsorptionsplaetze Support, mol
        self.p = p                            # Reaktordruck, Pa
        self.T = T                            # Reaktortemperatur, K
        self.surf_start = surf_start          # Index der ersten Oberflaechenspezies im Zustandsvektor
        self.n_dot_in_bu = n_dot_in_bu        # stationaerer Eintritt in der BU-Phase, mol s-1

        # --- Experimentdaten (fuer den Optimizer, hier nur durchgereicht) ---
        self.t_lin = t_lin                    # experimentelles Zeitgitter, s
        self.ni_out_cycle = ni_out_cycle      # gemessene Molenstroeme
        self.ni_trans_cycle = ni_trans_cycle  # gemessene transiente Anteile
        self.ni_RTD_cycle = ni_RTD_cycle      # gemessene RTD-Antwort

        # --- Zyklusgeometrie ---
        self.number_of_periods = number_of_periods     # wie viele Perioden simuliert werden
        self.periods_from_end = int(periods_from_end)  # welche davon ausgeschnitten wird
                                                       # (3 = drittletzte; laesst Abstand zum Rand,
                                                       #  damit der Periodizitaetstest greift)
        self.cycle_time = float(cycle_time)   # Periodendauer BU+BT, s
        self.bu_fraction = float(bu_fraction) # Anteil der BU-Phase (0.5 = symmetrisch)
        self.switch_delay = switch_delay      # Zeitversatz Modell <-> Experiment, s

        # --- Anlagenperipherie ---
        self.V_dot_H2_matrix = V_dot_H2_matrix  # Matrixdosierung H2 hinter dem Reaktor, nmL min-1
        self.V_dot_Ne_matrix = V_dot_Ne_matrix  # Matrixdosierung Ne hinter dem Reaktor, nmL min-1

        # --- Numerik ---
        self.normalize_E_tot = normalize_E_tot  # E3(x)E4 nachnormieren, siehe convolution_function

    # =========================================================================
    #  KERNE
    # =========================================================================
    def extract_RTD(self):
        """Kerne pruefen, normieren und zurueckgeben.

        Drei Pruefungen:

        1. GLEICHMAESSIGES RASTER je Kern. np.convolve(...) * dt setzt einen
           konstanten Zeitschritt voraus - bei ungleichmaessigem Raster waere der
           Faktor dt schlicht falsch.

        2. GEMEINSAMES dt ueber alle Kerne. Sonst laesst sich die Kette gar nicht
           bilden. Wird als Fehler geworfen, nicht nur gewarnt.

        3. INTEGRAL = 1. Eine Verweilzeitdichte ist eine Wahrscheinlichkeitsdichte.
           Ist das Integral != 1, skaliert jede Faltung die Amplitude mit diesem
           Faktor, und ueber die Kette MULTIPLIZIEREN sich die Abweichungen. Die
           Rohkerne aus dem Training liegen typisch 0.2..0.6 % daneben.

        Nebeneffekt: setzt self.dt_rtd (wird in convolution_function gebraucht).

        Returns:
            t_1, t_2, t_3, t_4, E1, E2, E3, E4   (Reihenfolge historisch gewachsen)
        """
        t_out, E_out, dts = [], [], []

        for t, E in zip(self.ts, self.Es):
            t = np.asarray(t, dtype=float)
            E = np.asarray(E, dtype=float)

            d = np.diff(t)
            if not np.allclose(d, d[0], rtol=1e-9, atol=1e-12):
                raise ValueError('Zeitachse eines RTD-Kerns ist nicht gleichmaessig!')

            dts.append(d[0])
            t_out.append(t)
            E_out.append(E / np.trapezoid(E, t))   # Normierung auf Integral 1, vorher auskommentiert

        if not np.allclose(dts, dts[0], rtol=1e-9, atol=1e-12):
            raise ValueError(f'Zeitschritte der RTDs sind nicht gleich: {dts}')

        self.dt_rtd = float(dts[0])

        return (*t_out, *E_out)

    # =========================================================================
    #  FALTUNGEN
    # =========================================================================
    @staticmethod
    def convolve_causal(signal, kernel, dt):
        """Faltung mit stationaerer Vorgeschichte - fuer NICHT-periodische Signale.

        Eine Faltung braucht Information ueber die Zeit VOR dem Signalanfang.
        np.convolve nimmt dort stillschweigend Null an - die Antwort startet dann
        bei 0 statt beim Anfangsplateau. Bei einem Absprung (Signal faellt von
        hoch auf niedrig) verzeichnet das die Kurve vollstaendig; beim Nachpruefen
        der Kerne ergaben sich so Fehler von 60 bis ueber 1000 %.

        Hier wird der ANFANGSWERT jeder Zeile ueber die volle Kernlaenge nach
        links fortgesetzt. Das entspricht der Annahme, das System habe vor
        Simulationsbeginn stationaer auf diesem Wert gestanden - fuer die
        ODE-Loesung, die aus einem definierten Anfangszustand startet, genau
        richtig.

        Fuer periodische Signale ist convolve_periodic() vorzuziehen: exakt statt
        naeherungsweise.

        Args:
            signal: 1D (n,) oder 2D (n_spezies, n)
            kernel: auf Integral 1 normierter Kern
            dt: Zeitschritt, identisch fuer Signal und Kern
        """
        sig = np.atleast_2d(np.asarray(signal, dtype=float))
        L = kernel.size
        n = sig.shape[1]
        out = np.empty_like(sig)

        for i in range(sig.shape[0]):
            # links um die volle Kernlaenge mit dem Anfangswert verlaengern ...
            padded = np.hstack((np.full(L, sig[i, 0]), sig[i]))
            # ... falten und genau das Fenster herausschneiden, das dem
            # Originalsignal entspricht
            out[i] = (np.convolve(padded, kernel) * dt)[L:L + n]

        return out.reshape(np.shape(signal))

    @staticmethod
    def convolve_periodic(one_period, kernel, dt):
        """Faltung eines PERIODISCHEN Signals ueber Kachelung - exakt.

        Bei einem zyklischen Prozess ist die Vorgeschichte einer Periode die
        vorhergehende Periode. Genau das wird hier abgebildet: das Signal so oft
        aneinanderhaengen, dass der Kern vollstaendig hineinpasst, einmal
        durchfalten, und den MITTLEREN (eingeschwungenen) Zyklus herausschneiden.

        Der Vorgaenger-Ansatz - jede Halbperiode einzeln mit ihrem Randwert
        paddieren und die Ergebnisse zusammennaehen - unterstellt dagegen, das
        System haette vor jeder Halbperiode ewig stillgestanden. Bei E_tot mit
        rund 55 s Laenge und 120 s Halbperiode ist das falsch, und die Naht
        erzeugt zusaetzlich einen Sprung.

        one_period MUSS geschlossen sein: erste Spalte = letzte Spalte, also
        t = 0 und t = cycle_time sind derselbe Zustand. Die Kachelung nutzt
        deshalb sig[:, :-1], sonst waere der Nahtpunkt doppelt.

        Args:
            one_period: 1D (n,) oder 2D (n_spezies, n), GESCHLOSSENE Periode
            kernel: auf Integral 1 normierter Kern
            dt: Zeitschritt
        """
        sig = np.atleast_2d(np.asarray(one_period, dtype=float))
        n = sig.shape[1]

        n_rep = int(np.ceil(kernel.size / (n - 1))) + 4   # +4 = Sicherheitsreserve
        tiled = np.hstack([sig[:, :-1]] * (n_rep - 1) + [sig])
        mid = (n_rep // 2) * (n - 1)                      # Start des mittleren Zyklus

        out = np.empty_like(sig)
        for i in range(sig.shape[0]):
            out[i] = (np.convolve(tiled[i], kernel) * dt)[mid:mid + n]

        return out.reshape(np.shape(one_period))

    # =========================================================================
    #  ALTLASTEN - nicht mehr im Faltungsweg
    # =========================================================================
    def padding_function(self, kernel, signal, type='None'):
        """VERALTET. Ersetzt durch convolve_causal / convolve_periodic.

        Paddet ein Signal beidseitig mit seinen Randwerten - also genau die
        Annahme "davor stand alles still", die bei periodischen Signalen falsch
        ist. Bleibt nur erhalten, um Altstaende nachrechnen zu koennen.
        """
        if type == 'E':
            pad = np.zeros(len(kernel) - 1)
            return np.hstack((pad, signal, pad))

        n_rows = signal.shape[0]
        before = signal[:, 0].repeat(len(kernel) - 1).reshape(n_rows, len(kernel) - 1)
        after = signal[:, -1].repeat(len(kernel) - 1).reshape(n_rows, len(kernel) - 1)
        return np.hstack((before, signal, after))

    def splitting_function(self, solutions, time_per_period, iter, bu_width_fraction=0.5):
        """VERALTET. Lieferte die Schnittindizes fuer BU- und BT-Halbperiode.

        Wird nicht mehr gebraucht: es wird nur noch EINE volle Periode
        ausgeschnitten (ein argmax auf t_uni), nicht zwei Haelften. Ausserdem
        arbeitet die Funktion auf solutions.t, die Indizes werden aber auf dem
        gleichmaessigen RTD-Gitter gebraucht.
        """
        x_span = solutions.t
        n = self.number_of_periods
        index_bu = np.where(x_span >= (n - 1 - iter) * time_per_period)
        index_bt = np.where(x_span >= (n - (1 - bu_width_fraction) - iter) * time_per_period)
        return np.array((index_bu[0][0], index_bt[0][0]))

    # =========================================================================
    #  HAUPTROUTINE
    # =========================================================================
    def convolution_function(self, k, T):
        """Kompletter Durchlauf: Kinetik -> E2 -> Verduennung -> E3(x)E4 -> MS.

        Args:
            k: Vektor der kinetischen Konstanten
            T: Temperatur, K

        Returns (dict):
            n_dot_theo_int   theoretische Molenstroeme auf t_lin (reiner Transport)
            n_dot_trans_int  transienter Anteil = tatsaechlich minus theoretisch
                             -> das eigentliche Signal von Adsorption/Reaktion
            n_dot_i_int      Speziesmolenstroeme auf t_lin  <- Hauptergebnis
            x_g, theta_s, theta_x  Zustaende im idealen Reaktor (vor jeder Faltung)
            C_ads, O_ads     Kohlenstoff-/Sauerstoffbilanz
            diagnostics      alle Zwischenstaende, siehe unten
        """
        t_1, t_2, t_3, t_4, E_1, E_2, E_3, E_4 = self.extract_RTD()
        dt = self.dt_rtd

        # lokale Kopien, damit die Formeln unten lesbar bleiben
        n_dot_func = self.n_dot_func
        t_eval = self.t_eval
        y_init = self.y_init
        V_R = self.V_R
        n_max = self.n_max
        nx_max = self.nx_max
        p = self.p
        T = self.T
        surf_start = self.surf_start
        n_dot_in_bu = self.n_dot_in_bu
        t_lin = self.t_lin
        switch_delay = self.switch_delay
        cycle_time = self.cycle_time
        bu_fraction = self.bu_fraction

        # ------------------------------------------------ 1. kinetische ODE
        # Idealer, perfekt durchmischter Reaktor. Alle Abweichungen davon
        # (Rueckvermischung, Dispersion im Bett) korrigiert anschliessend E2.
        solutions = solver(k, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T)

        x_g = solutions.y[:surf_start, :]                    # Gasphasen-Molenbrueche
        theta_s = solutions.y[surf_start:support_start, :]   # Bedeckung Aktivmaterial
        theta_x = solutions.y[support_start:, :]             # Bedeckung Support

        # Gesamtmolenstrom am Bettausgang. NICHT konstant: die Reaktion aendert
        # die Molzahl (CO2 + 4 H2 -> CH4 + 2 H2O, also 5 mol -> 3 mol).
        n_dot_reactor = n_dot_out(solutions, k, p, T, sum(n_dot_in_bu), V_R)

        # Speziesmolenstroeme, mol s-1. DAS ist die Groesse, die hier gefaltet
        # wird - sie ist extensiv und wird transportiert. Ein Molenbruch ist
        # dagegen ein Verhaeltnis zweier transportierter Groessen; die Variante,
        # die trotzdem Molenbrueche faltet, steht in Output_III.
        n_dot_species_pre_RTD = x_g * n_dot_reactor[np.newaxis, :]

        # ------------------------------------------------ 2. auf das RTD-Gitter
        # Der ODE-Loeser gibt seine eigenen Zeitstempel aus. Fuer die Faltung
        # braucht es ein gleichmaessiges Raster mit exakt dem dt der Kerne.
        t_grid = solutions.t
        n_uni = int(round((t_grid[-1] - t_grid[0]) / dt)) + 1
        t_uni = t_grid[0] + np.arange(n_uni) * dt
        n_dot_uni = interpolation_function(t_uni, t_grid, n_dot_species_pre_RTD)

        # ------------------------------------------------ 3. STUFE E2 (Reaktor)
        # Nicht-periodisch, weil die ODE-Loesung ueber mehrere Perioden laeuft und
        # am Anfang noch einschwingt -> convolve_causal.
        n_dot_after_E2 = self.convolve_causal(n_dot_uni, E_2, dt)

        # Gesamtstrom und Molenbrueche folgen aus den Speziesstroemen, nicht
        # umgekehrt. Sum(x_i) = 1 ist damit automatisch erfuellt.
        n_dot_tot_after_E2 = n_dot_after_E2.sum(axis=0)
        x_after_E2 = n_dot_after_E2 / n_dot_tot_after_E2[np.newaxis, :]

        # ------------------------------------------------ 4. Matrixverduennung
        # H2 und Ne werden HINTER dem Reaktor zudosiert, gehen also nicht durch
        # E2. Umrechnung nmL/min -> mol/s ueber das ideale Gasgesetz bei
        # Normbedingungen (p_STP, T_STP aus AZA_THB_tools).
        # ACHTUNG: bei Laeufen ohne Matrix beide Volumenstroeme auf 0.0 setzen!
        n_dot_H2_mat = self.V_dot_H2_matrix * 1e-6 / 60 * p_STP / (R * T_STP)
        n_dot_Ne_mat = self.V_dot_Ne_matrix * 1e-6 / 60 * p_STP / (R * T_STP)

        n_dot_after_matrix = n_dot_after_E2.copy()
        n_dot_after_matrix[H2] += n_dot_H2_mat
        n_dot_after_matrix[Ne] += n_dot_Ne_mat

        n_dot_tot_after_matrix = n_dot_after_matrix.sum(axis=0)
        solutions_x_i = n_dot_after_matrix / n_dot_tot_after_matrix[np.newaxis, :]

        # ------------------------------------------------ 5. Periode ausschneiden
        # Ab hier ist das Signal periodisch, also kann exakt gefaltet werden.
        # Ausgeschnitten wird die (number_of_periods - periods_from_end)-te
        # Periode; periods_from_end=3 laesst hinten genug Abstand zum Rand.
        n_per = int(round(cycle_time / dt)) + 1
        t_start = (self.number_of_periods - self.periods_from_end) * cycle_time
        i0 = int(np.argmax(t_uni >= t_start + t_uni[0]))

        if i0 + n_per > n_uni:
            raise ValueError('Zu wenig Perioden simuliert, um eine volle Periode '
                             'auszuschneiden - number_of_periods oder periods_from_end anpassen.')

        period = n_dot_after_matrix[:, i0:i0 + n_per]

        # Periodizitaetstest: Vergleich mit der VORHERGEHENDEN Periode. Prueft
        # genau die Annahme, die die Kachelung macht. Ist der Wert gross (> ~1 %),
        # ist die Loesung noch nicht eingeschwungen und number_of_periods muss
        # hoch. Bei Kinetiklaeufen mit langsamer Bedeckungsdynamik ist das der
        # Wert, den man im Auge behalten muss.
        i_prev = i0 - (n_per - 1)
        if i_prev >= 0:
            prev = n_dot_after_matrix[:, i_prev:i_prev + n_per]
            scale = np.max(np.abs(period), axis=1)
            scale[scale == 0] = 1.0                       # Division durch 0 vermeiden
            periodicity_error = float(np.max(np.abs(period - prev) / scale[:, np.newaxis]))
        else:
            periodicity_error = np.nan

        # Periode schliessen: t = 0 und t = cycle_time sind derselbe Zustand.
        # convolve_periodic setzt das voraus.
        period = period.copy()
        period[:, -1] = period[:, 0]

        # ------------------------- STUFE E3 (x) E4, periodisch gefaltet
        E_tot = np.convolve(E_3, E_4) * dt
        t_tot = np.arange(E_tot.size) * dt
        E_tot = E_tot / np.trapezoid(E_tot, t_tot)              # <- Normierung

        n_dot_i = self.convolve_periodic(period, E_tot, dt)
        t_after_convolution = np.arange(n_per) * dt

        n_dot_tot_final = n_dot_i.sum(axis=0)
        x_final = n_dot_i / n_dot_tot_final[np.newaxis, :]

        # ------------------------------------------------ 7. theoretisch / transient
        # theoretical_molar_flow_rates skaliert die Ar-Sprungantwort auf jede
        # Spezies -> das Signal, das reiner Transport OHNE Reaktion ergaebe.
        # Die Differenz dazu (transient) ist das eigentliche Nutzsignal.
        # ind_final_bu = Index des BU/BT-Umschaltpunkts auf dem Faltungsgitter.
        ind_final_bu = int(round(cycle_time * bu_fraction / dt)) + 1
        n_dot_theo = theoretical_molar_flow_rates(n_dot_i, ind_final_bu)
        n_dot_trans = transient_molar_flow_rates(n_dot_i, ind_final_bu)

        # ------------------------------------------------ 8. auf das Experimentgitter
        # Erst jetzt - vorher lief alles auf dem feineren RTD-Gitter.
        n_dot_theo_int = interpolation_function(t_lin, t_after_convolution, n_dot_theo)
        n_dot_trans_int = interpolation_function(t_lin, t_after_convolution, n_dot_trans)
        n_dot_i_int = interpolation_function(t_lin, t_after_convolution, n_dot_i)

        # ------------------------------------------------ 9. C-/O-Bilanz
        # Integriert den transienten Anteil ab dem BU/BT-Umschaltpunkt.
        # WAR: bu_start_ind = int(t_lin.size / 4) + delay. Das Viertel stammt aus
        # der alten 480-s-Konvention und liegt bei 240 s Periode bei 60 s statt
        # bei 120 s - also mitten in der BU-Phase statt am Umschaltpunkt.
        delay = np.where(t_lin > switch_delay)[0][0]
        dt_lin = t_lin[1] - t_lin[0]
        bu_start_ind = min(delay + int(round(cycle_time * bu_fraction / dt_lin)), t_lin.size - 2)

        C_ads = np.trapezoid(n_dot_trans_int[CH4, bu_start_ind:] + n_dot_trans_int[CO2, bu_start_ind:],
                             t_lin[bu_start_ind:])
        O_ads = np.trapezoid(n_dot_trans_int[H2O, bu_start_ind:] + 2 * n_dot_trans_int[CO2, bu_start_ind:],
                             t_lin[bu_start_ind:])

        # ------------------------------------------------ Diagnose
        # Alle Zwischenstaende, um den Signalweg stufenweise nachvollziehen zu
        # koennen - der wichtigste Hebel bei der Fehlersuche.
        diagnostics = {
            't_reactor_grid': solutions.t,
            't_uniform_grid': t_uni,
            'n_dot_species_pre_RTD': n_dot_species_pre_RTD,   # Bettausgang, vor jeder Faltung
            'x_after_E2': x_after_E2,                          # Molenbrueche nach E2 (vor Verduennung)
            'n_dot_tot_after_E2': n_dot_tot_after_E2,          # Gesamtstrom nach E2, aus Summe der Spezies
            'n_dot_after_E2': n_dot_after_E2,                  # Speziesstroeme nach E2
            'n_dot_after_matrix': n_dot_after_matrix,          # nach E2 + Matrixverduennung
            'n_dot_tot_after_matrix': n_dot_tot_after_matrix,
            'solutions_x_i': solutions_x_i,
            'n_dot_tot_final': n_dot_tot_final,
            'x_final': x_final,
            't_after_convolution': t_after_convolution,
            'E_tot': E_tot,
            't_tot': t_tot,
            'tau_E': [float(np.trapezoid(t * E, t)) for t, E in
                      zip((t_1, t_2, t_3, t_4), (E_1, E_2, E_3, E_4))],
            'periodicity_error': periodicity_error,            # 0 = perfekt eingeschwungen
            'ind_final_bu': ind_final_bu,
            'bu_start_ind': bu_start_ind,
        }

        return {
            'n_dot_theo_int': n_dot_theo_int,
            'n_dot_trans_int': n_dot_trans_int,
            'n_dot_i_int': n_dot_i_int,
            'x_g': x_g,
            'theta_s': theta_s,
            'theta_x': theta_x,
            'C_ads': C_ads,
            'O_ads': O_ads,
            'diagnostics': diagnostics
        }


display('-AZA_THB_reactor_output- imported at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))