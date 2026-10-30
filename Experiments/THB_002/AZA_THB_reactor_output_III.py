"""
===============================================================================
 AZA_THB_reactor_output_III  --  Variante: Faltung der MOLENBRUECHE
===============================================================================

WOFUER DIESE VARIANTE
---------------------
Output (in AZA_THB_reactor_output.py) faltet die SPEZIESMOLENSTROEME.
Output_III faltet stattdessen die MOLENBRUECHE - also so, wie die Kerne im
nRTD-Training auch identifiziert wurden.

Die Klasse erbt alles von Output und ueberschreibt NUR convolution_function.
Bewusst als Unterklasse und nicht als Kopie des ganzen Moduls: fast identische
Dateien mit je eigenen Altlasten waren in diesem Projekt schon mehrfach die
Ursache dafuer, dass eine Korrektur nur in einer der beiden Fassungen ankam.


DIE BEIDEN SICHTWEISEN
----------------------
Argument FUER Molenbrueche (diese Datei):
    Das nRTD-Training hat die Kerne auf Molenbruch-Signalen bestimmt:
    x_out = x_in (x) E. Wendet man sie wieder auf Molenbrueche an, ist die
    Anwendung konsistent zur Identifikation. Auch die klassische
    VWZV-Definition ist ueber eine intensive Groesse formuliert
    (c_aus(t) = Integral c_ein(t-t') E(t') dt').

Argument FUER Molenstroeme (AZA_THB_reactor_output.py):
    Eine Verweilzeitverteilung beschreibt, wie lange FLUIDELEMENTE unterwegs
    sind. Was am Austritt zur Zeit t ankommt, ist die Mischung aus Elementen
    verschiedener Eintrittszeiten:

        n_dot_i,aus(t)   = Integral n_dot_i,ein(t-t') E(t') dt'
        n_dot_tot,aus(t) = Integral n_dot_tot,ein(t-t') E(t') dt'
        x_i,aus(t)       = n_dot_i,aus(t) / n_dot_tot,aus(t)

    Streng genommen faltet also der Molenstrom linear, und der Molenbruch ist
    der QUOTIENT zweier gefalteter Groessen. Nur wenn n_dot_tot zeitlich
    konstant ist, kuerzt es sich heraus und beide Wege sind identisch.

WANN ES EINEN UNTERSCHIED MACHT
    In allen Tracerlaeufen: gar nicht. Dort ist n_dot_tot konstant, und genau
    deshalb sind die trainierten Kerne fuer beide Wege gueltig - das Training
    selbst kann die Frage nicht entscheiden.

    Bei der Methanisierung: schon. CO2 + 4 H2 -> CH4 + 2 H2O bedeutet
    5 mol -> 3 mol. Waehrend des Transienten aendert sich der Umsatz und damit
    n_dot_tot, und die beiden Formulierungen laufen auseinander.

EMPFEHLUNG
    Beide rechnen und die Differenz als Modellunsicherheit ausweisen - sie ist
    eine ehrliche Abschaetzung dessen, was die Formulierung selbst offen laesst.
    Vergleich siehe compare_formulations() am Ende dieser Datei.


WAS GENAU ANDERS GERECHNET WIRD
-------------------------------
    Stufe E2:      x_i und n_dot_tot werden GETRENNT mit E2 gefaltet,
                   danach n_dot_i = x_i * n_dot_tot
    Stufe E3(x)E4: dasselbe mit E_tot

WICHTIG - Unterschied zur urspruenglichen Fassung des Projekts: dort wurden die
Molenbrueche gefaltet, der Gesamtstrom aber nur um tau(E2) VERSCHOBEN. Eine
Verschiebung ist keine Faltung: sie erhaelt den Mittelwert und wirft die
gesamte Streuung weg. Zaehler und Nenner erfuhren dadurch unterschiedliche
Operationen, und die Kette war nicht mehr linear. Hier werden beide mit
DEMSELBEN Kern gefaltet - das ist die saubere Molenbruch-Formulierung.

Angenehmer Nebeneffekt: Sum(x_i) bleibt automatisch 1. Die Faltung ist linear
und E ist auf 1 normiert, also gilt Sum_i (x_i (x) E) = (Sum_i x_i) (x) E
= 1 (x) E = 1. Die im Altcode auskommentierte col_sum-Renormierung war nur ein
Pflaster fuer die damals unnormierten Kerne.
===============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.constants import R

from AZA_THB_solver import *
from AZA_THB_tools import *
from AZA_THB_reactor_output import Output

if 'ICIWstyle' in plt.style.available:
    plt.style.use('ICIWstyle')


class Output_III(Output):
    """Wie Output, faltet aber Molenbrueche statt Speziesmolenstroeme.

    Signatur, Parameter und Rueckgabewerte sind identisch zu Output - die
    Klassen sind gegeneinander austauschbar:

        out = Output_III(ts=CNN_ts, Es=CNN_Es, ..., V_dot_H2_matrix=0.0)
        res = out.convolution_function(k, T)

    Zusaetzlich in diagnostics: 'formulation' = 'mole_fraction'.
    """

    def convolution_function(self, k, T):
        """Kompletter Durchlauf, Molenbruch-Formulierung.

        Ablauf identisch zu Output.convolution_function; abweichend sind nur die
        beiden Faltungsstufen (mit "ABWEICHUNG" markiert).
        """
        t_1, t_2, t_3, t_4, E_1, E_2, E_3, E_4 = self.extract_RTD()
        dt = self.dt_rtd

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
        solutions = solver(k, n_dot_func, t_eval, y_init, V_R, n_max, nx_max, p, T)

        x_g = solutions.y[:surf_start, :]                    # Gasphasen-Molenbrueche
        theta_s = solutions.y[surf_start:support_start, :]
        theta_x = solutions.y[support_start:, :]

        n_dot_reactor = n_dot_out(solutions, k, p, T, sum(n_dot_in_bu), V_R)
        n_dot_species_pre_RTD = x_g * n_dot_reactor[np.newaxis, :]

        # ------------------------------------------------ 2. auf das RTD-Gitter
        t_grid = solutions.t
        n_uni = int(round((t_grid[-1] - t_grid[0]) / dt)) + 1
        t_uni = t_grid[0] + np.arange(n_uni) * dt

        x_g_uni = interpolation_function(t_uni, t_grid, x_g)              # intensiv
        n_dot_tot_uni = np.interp(t_uni, t_grid, n_dot_reactor)           # extensiv

        # ---------------------------------- 3. STUFE E2 (Reaktor)   ABWEICHUNG
        # Molenbrueche UND Gesamtstrom werden getrennt, aber mit DEMSELBEN Kern
        # gefaltet. Nur so bleibt die Beschreibung in sich konsistent - eine
        # blosse Zeitverschiebung des Gesamtstroms (wie im Altcode) waere keine
        # Faltung und wuerde die Dispersion des Nenners verschlucken.
        x_after_E2 = self.convolve_causal(x_g_uni, E_2, dt)
        n_dot_tot_after_E2 = self.convolve_causal(n_dot_tot_uni, E_2, dt)

        # Sum(x_i) bleibt hier automatisch 1 (Linearitaet + normierter Kern);
        # die Zeile ist nur eine numerische Absicherung.
        x_after_E2 = x_after_E2 / x_after_E2.sum(axis=0)[np.newaxis, :]

        # Speziesstroeme fuer die Matrixverduennung: die Zumischung ist eine
        # MENGENartige Operation und laesst sich nicht an Molenbruechen ausfuehren.
        n_dot_after_E2 = x_after_E2 * n_dot_tot_after_E2[np.newaxis, :]

        # ------------------------------------------------ 4. Matrixverduennung
        n_dot_H2_mat = self.V_dot_H2_matrix * 1e-6 / 60 * p_STP / (R * T_STP)
        n_dot_Ne_mat = self.V_dot_Ne_matrix * 1e-6 / 60 * p_STP / (R * T_STP)

        n_dot_after_matrix = n_dot_after_E2.copy()
        n_dot_after_matrix[H2] += n_dot_H2_mat
        n_dot_after_matrix[Ne] += n_dot_Ne_mat

        n_dot_tot_after_matrix = n_dot_tot_after_E2 + n_dot_H2_mat + n_dot_Ne_mat
        solutions_x_i = n_dot_after_matrix / n_dot_tot_after_matrix[np.newaxis, :]

        # ------------------------------------------------ 5. Periode ausschneiden
        n_per = int(round(cycle_time / dt)) + 1
        t_start = (self.number_of_periods - self.periods_from_end) * cycle_time
        i0 = int(np.argmax(t_uni >= t_start + t_uni[0]))

        if i0 + n_per > n_uni:
            raise ValueError('Zu wenig Perioden simuliert, um eine volle Periode '
                             'auszuschneiden - number_of_periods oder '
                             'periods_from_end anpassen.')

        period_x = solutions_x_i[:, i0:i0 + n_per].copy()
        period_n = n_dot_tot_after_matrix[i0:i0 + n_per].copy()

        # Periodizitaetstest auf den Speziesstroemen, damit der Wert direkt mit
        # dem aus Output vergleichbar bleibt.
        i_prev = i0 - (n_per - 1)
        if i_prev >= 0:
            cur = n_dot_after_matrix[:, i0:i0 + n_per]
            prev = n_dot_after_matrix[:, i_prev:i_prev + n_per]
            scale = np.max(np.abs(cur), axis=1)
            scale[scale == 0] = 1.0
            periodicity_error = float(np.max(np.abs(cur - prev) / scale[:, np.newaxis]))
        else:
            periodicity_error = np.nan

        # Perioden schliessen (Voraussetzung fuer convolve_periodic)
        period_x[:, -1] = period_x[:, 0]
        period_n[-1] = period_n[0]

        # ------------------------- 6. STUFE E3 (x) E4               ABWEICHUNG
        E_tot = np.convolve(E_3, E_4) * dt
        t_tot = np.arange(E_tot.size) * dt
        if self.normalize_E_tot:
            E_tot = E_tot / np.trapezoid(E_tot, t_tot)

        x_final = self.convolve_periodic(period_x, E_tot, dt)
        n_dot_tot_final = self.convolve_periodic(period_n, E_tot, dt)

        x_final = x_final / x_final.sum(axis=0)[np.newaxis, :]   # numerische Absicherung
        n_dot_i = x_final * n_dot_tot_final[np.newaxis, :]

        t_after_convolution = np.arange(n_per) * dt

        # ------------------------------------------------ 7. theoretisch / transient
        ind_final_bu = int(round(cycle_time * bu_fraction / dt)) + 1
        n_dot_theo = theoretical_molar_flow_rates(n_dot_i, ind_final_bu)
        n_dot_trans = transient_molar_flow_rates(n_dot_i, ind_final_bu)

        # ------------------------------------------------ 8. auf das Experimentgitter
        n_dot_theo_int = interpolation_function(t_lin, t_after_convolution, n_dot_theo)
        n_dot_trans_int = interpolation_function(t_lin, t_after_convolution, n_dot_trans)
        n_dot_i_int = interpolation_function(t_lin, t_after_convolution, n_dot_i)

        # ------------------------------------------------ 9. C-/O-Bilanz
        delay = np.where(t_lin > switch_delay)[0][0]
        dt_lin = t_lin[1] - t_lin[0]
        bu_start_ind = min(delay + int(round(cycle_time * bu_fraction / dt_lin)),
                           t_lin.size - 2)

        C_ads = np.trapezoid(n_dot_trans_int[CH4, bu_start_ind:]
                             + n_dot_trans_int[CO2, bu_start_ind:],
                             t_lin[bu_start_ind:])
        O_ads = np.trapezoid(n_dot_trans_int[H2O, bu_start_ind:]
                             + 2 * n_dot_trans_int[CO2, bu_start_ind:],
                             t_lin[bu_start_ind:])

        diagnostics = {
            'formulation':           'mole_fraction',      # <- Unterschied zu Output
            't_reactor_grid':        solutions.t,
            't_uniform_grid':        t_uni,
            'n_dot_species_pre_RTD': n_dot_species_pre_RTD,
            'x_after_E2':            x_after_E2,
            'n_dot_tot_after_E2':    n_dot_tot_after_E2,
            'n_dot_after_E2':        n_dot_after_E2,
            'n_dot_after_matrix':    n_dot_after_matrix,
            'n_dot_tot_after_matrix': n_dot_tot_after_matrix,
            'solutions_x_i':         solutions_x_i,
            'n_dot_tot_final':       n_dot_tot_final,
            'x_final':               x_final,
            't_after_convolution':   t_after_convolution,
            'E_tot':                 E_tot,
            't_tot':                 t_tot,
            'tau_E': [float(np.trapezoid(t * E, t)) for t, E in
                      zip((t_1, t_2, t_3, t_4), (E_1, E_2, E_3, E_4))],
            'periodicity_error':     periodicity_error,
            'ind_final_bu':          ind_final_bu,
            'bu_start_ind':          bu_start_ind,
        }

        return {
            'n_dot_theo_int':  n_dot_theo_int,
            'n_dot_trans_int': n_dot_trans_int,
            'n_dot_i_int':     n_dot_i_int,
            'x_g':             x_g,
            'theta_s':         theta_s,
            'theta_x':         theta_x,
            'C_ads':           C_ads,
            'O_ads':           O_ads,
            'diagnostics':     diagnostics,
        }


def compare_formulations(res_flow, res_frac, t_lin, species=None, names=None):
    """Vergleicht die Ergebnisse beider Formulierungen Spezies fuer Spezies.

    Die Differenz ist KEIN Fehler einer der beiden Varianten, sondern die
    Unsicherheit, die die Formulierung selbst offen laesst. Sie waechst mit der
    Molzahlaenderung durch die Reaktion und ist bei inerten Tracern exakt null.
    Ein guter Wert, um ihn in der Arbeit als Modellunsicherheit anzugeben.

    Args:
        res_flow: Rueckgabe von Output.convolution_function
        res_frac: Rueckgabe von Output_III.convolution_function
        t_lin: Zeitgitter
        species: Liste von Speziesindizes, None = alle mit Signal
        names: Speziesnamen, None = comp aus AZA_THB_tools

    Returns:
        dict {Speziesindex: max. relative Abweichung}
    """
    a = res_flow['n_dot_i_int']
    b = res_frac['n_dot_i_int']
    names = comp if names is None else names

    if species is None:
        species = [i for i in range(a.shape[0])
                   if max(np.abs(a[i]).max(), np.abs(b[i]).max()) > 1e-14]

    print('Vergleich der Formulierungen (Molenstrom- gegen Molenbruch-Faltung):')
    print(f'{"Spezies":10s} {"max |Strom|":>13s} {"max Diff":>13s} {"rel":>9s} {"bei t":>9s}')

    out = {}
    for i in species:
        scale = max(np.abs(a[i]).max(), 1e-30)
        d = np.abs(a[i] - b[i])
        rel = d.max() / scale
        out[i] = float(rel)
        print(f'{names[i]:10s} {np.abs(a[i]).max():13.4e} {d.max():13.4e} '
              f'{rel * 100:8.3f} % {t_lin[int(np.argmax(d))]:8.1f}s')

    worst = max(out.values()) if out else 0.0
    print(f'\n  groesste Abweichung: {worst * 100:.3f} %')
    if worst < 1e-6:
        print('  -> praktisch identisch. Der Gesamtmolenstrom ist zeitlich konstant')
        print('     (kein Umsatz bzw. keine Molzahlaenderung), damit sind beide')
        print('     Formulierungen mathematisch aequivalent.')
    else:
        print('  -> Die Molzahlaenderung durch die Reaktion wirkt sich aus. Der Wert')
        print('     ist als Modellunsicherheit der Transportbeschreibung zu lesen.')
    return out


display('-AZA_THB_reactor_output_III- imported at',
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"))