#NMRSI - Ignacio Lembo Ferrari - 02/09/2024

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

sns.set_theme(context='paper')
sns.set_style("whitegrid")

#############################################################################
#GRAFICA DE DATOS
#############################################################################

def plot_nogse_vs_x_data(ax, nroi, x, f, error, tnogse, g, n, slic, color):
    #ax.plot(x, f, "-o", markersize=7, linewidth = 2, color=color, label=nroi)
    ax.errorbar(x, f, yerr=error, fmt='o-', markersize=3, linewidth=2, capsize=5, label=nroi)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.legend(title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"$T_\mathrm{{NOGSE}}$ = {tnogse} ms | $G$ = {g} mT/m | $N$ = {n} | Slice = {slic}", fontsize=15)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def plot_A0_vs_x_data(ax, nroi, x, f, error, tnogse, g, n, slic, color):
    #ax.plot(x, f, "--", linewidth = 2, color=color, label=f"A0 - {nroi} - ({np.round(np.mean(f),2)} $\pm$ {np.round(np.std(f),2)})")
    ax.errorbar(x, f, yerr=error, fmt='o-', markersize=3, linewidth=2, capsize=5,  label=f"A0 - {nroi} - ({np.round(np.mean(f),2)} $\pm$ {np.round(np.std(f),2)})")
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.legend(title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"$T_\mathrm{{NOGSE}}$ = {tnogse} ms | $G$ = {g} mT/m | $N$ = {n} | Slice = {slic}", fontsize=15)
    #plt.tight_layout()
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_generated(ax, modelo, label, label_title, x, f, tnogse, g, n):
    ax.plot(x, f, linewidth = 2, label=label)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.legend(title=label_title, title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{modelo} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $G$ = {g} mT/m | $N$ = {n}", fontsize=15)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptG(ax, nroi, x, f, tnogse, g, n, num_grad, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=f"{num_grad} = {g}")
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27) 
    ax.legend(title='Gradiente [mT/m]', title_fontsize=15, fontsize=15, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{nroi} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | Slice = {slic}", fontsize=15)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_pttc(ax, nroi, x, f, tnogse, g, n, tc, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=f"$t_c$ = {tc}")
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27) 
    ax.legend(title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{nroi} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | Slice = {slic}", fontsize=15)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptN(ax, nroi, x, f, tnogse, g, n, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=n)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$ normalizada", fontsize=27)
    ax.legend(title='$N$', title_fontsize=12, fontsize=12, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title("{} | $T_\mathrm{{NOGSE}}$ = {} ms | $G$ = {} | Slice = {}".format(nroi, tnogse, g, slic), fontsize=18)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_data_ptTNOGSE(ax, nroi, x, f, tnogse, n, slic, color):
    ax.plot(x, f, "-o", markersize=7, linewidth = 2, color = color, label=tnogse)
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=18, color='black')
    ax.tick_params(axis='y', labelsize=18, color='black')
    title = ax.set_title(f"{nroi} | $N$ = {n} | Slice = {slic}", fontsize=15)
    #ax.set_xlim(0.5, 10.75)

def plot_nogse_vs_x_fit(ax, nroi, modelo, x, x_fit, f, fit, tnogse, g, n, num_grad, slic, label, color):
    ax.plot(x, f, "o", markersize=7, linewidth=2, color = color)
    ax.plot(x_fit, fit, linewidth=2, label = label, color = color)
    ax.legend(title_fontsize=15, fontsize=15, loc='best')
    ax.set_xlabel("Tiempo de modulación x [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{modelo} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | {num_grad} = {g} mT/m | $N$ = {n} | slice = {slic} ", fontsize=15)

def plot_nogse_vs_x_restdist_ptTNOGSE(ax, nroi, modelo, x, x_fit, f, fit, tnogse, g, n, slic, color, label):
    ax.plot(x, f, "o", markersize=7, linewidth = 2, color = color)
    ax.plot(x_fit, fit, linewidth=2, color = color, label = label)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"$N$ = {n} | slice = {slic}", fontsize=15)

def plot_nogse_vs_x_fit_ptG(ax, nroi, modelo, x, x_fit, f, fit, tnogse, n, slic, color, label):
    ax.plot(x, f, "o", markersize=7, linewidth = 2, color = color)
    ax.plot(x_fit, fit, linewidth=2, color = color, label = label)
    ax.legend(title='Gradiente [mT/m]', title_fontsize=15, fontsize=15, loc='upper right')
    ax.set_xlabel("Tiempo de modulación $x$ [ms]", fontsize=27)
    ax.set_ylabel("Señal $\mathrm{NOGSE}$", fontsize=27)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{nroi} | {modelo} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic} ", fontsize=15)

def plot_contrast_vs_g_data(ax, nroi, g_contrast, f, error, tnogse, n, slic, color):
    ax.errorbar(g_contrast, f, yerr=error, fmt='o-', markersize=3, linewidth=2, label=nroi, capsize=5)
    #ax.plot(g_contrast, f, 'o-', markersize=7, linewidth=2, color=color, label=nroi)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    ax.legend(title='ROI', title_fontsize=15, fontsize=15, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x', rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    #ax.axvline(x=55, color='k', linestyle='--')
    #ax.axvline(x=110, color='k', linestyle='--')
    #ax.axvline(x=190, color='k', linestyle='--')
    #ax.axvline(x=550, color='k', linestyle='--')
    #ax.axvline(x=800, color='k', linestyle='--')
    title = ax.set_title(f"$T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic} ", fontsize=15)
    #plt.tight_layout()    
    #ax.set_xlim(0.5, 10.75)

def plot_contrast_vs_g_ptTNOGSE(ax, nroi, modelo, g_contrast, f, tnogse, n, slic, color):
    ax.plot(g_contrast, f, "-", markersize=7, linewidth = 2, color=color, label= tnogse)
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    ax.legend(title='$T_\mathrm{{NOGSE}}$ [ms]', title_fontsize=15, fontsize=15, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{modelo} | $N$ = {n} | Slice = {slic}", fontsize=15)
    #ax.set_xlim(-10, 1200)

def plot_contrast_vs_g_fit(ax, nroi, modelo, g, g_fit, f, fit, tnogse, n, slic, color):
    ax.plot(g, f, "o", markersize=7, linewidth=2, color = color)
    ax.plot(g_fit, fit, linewidth=2, label = nroi, color = color)
    ax.legend(title_fontsize=15, fontsize=15, loc='best')
    ax.set_xlabel("Intensidad de gradiente $g$ [mT/m]", fontsize=27)
    ax.set_ylabel("Contraste $\mathrm{NOGSE}$ $\Delta M$", fontsize=27)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"$T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic}", fontsize=15)

def plot_contrast_rest_mixto_levs(ax, nroi, modelo, g_contrast, roi, T_nogse, n, t_c_int_fit, t_c_ext_fit, alpha_fit, M0_int, M0_ext, D0_int, D0_ext):
    if(nroi == "ROI1"):
        color = "green"
    if(nroi == "ROI2"):
        color = "darkorange"
    if(nroi == "ROIw"):
        color = "blue"
    g_contrast_fit = np.linspace(0, np.max(g_contrast), num=1000)
    fit = delta_M_intra_extra(T_nogse, g_contrast_fit, n, t_c_int_fit, t_c_ext_fit, alpha_fit, M0_int, M0_ext, D0_int, D0_ext)
    ax.plot(g_contrast, roi, 'o', color = color, linewidth=2)
    ax.plot(g_contrast_fit, fit, '-', color = color, linewidth=2, label="$\\tau_{c,int}$ = " + str(round(t_c_int_fit,4)) + " - $\\tau_{c,ext}$ = " + str(round(t_c_ext_fit,4)) + " - $\\alpha = $" + str(round(alpha_fit,4)))
    ax.legend(title_fontsize=15, fontsize=15, loc='upper right')
    ax.set_xlabel("Intensidad de gradiente g [mT/m]", fontsize=16)
    ax.set_ylabel("Contraste $\Delta M_\mathrm{NOGSE}$ [u.a.]", fontsize=16)
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title("{} | Modelo: {} | $T_\mathrm{{NOGSE}}$ = {} ms  |  $N$ = {} ".format(nroi, modelo, T_nogse, int(n)), fontsize=18)

def plot_lognorm_dist(ax, nroi, tnogse, n, l_c, l_c_mode, sigma, slic, color):
    dist = lognormal_mode(l_c, sigma, l_c_mode)
    l_c_median = l_c_mode*np.exp((sigma**2))
    l_c_mid = l_c_mode*np.exp((sigma**2)/2)
    plt.axvline(x=l_c_mode, color=color, linestyle='-', label = "Moda") 
    plt.axvline(x=l_c_median, color=color, linestyle='--', label = "Mediana") 
    plt.axvline(x=l_c_mid, color=color, linestyle='-.', label = "Media") 
    #ax.plot(l_c, dist, "-", color=color, linewidth = 2, label = nroi)
    ax.set_xlabel("Longitud de correlación $l_c$ [$\mu$m]", fontsize=27)
    #ax.set_xlabel("Diámetro de célula $d$ [$\mu$m]", fontsize=27)
    ax.set_ylabel("P($l_c$)", fontsize=27)
    #ax.set_ylabel("P($d$)", fontsize=27)
    ax.legend(title_fontsize=15, fontsize=15, loc='upper right')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{nroi} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic} ", fontsize=15)
    ax.fill_between(l_c, dist, color=color, alpha=0.3)
    #ax.set_xlim(0.5, 10.75)

def plot_lognorm_dist_ptG(ax, nroi, tnogse, n, l_c, dist, slic, color, label):
    ax.plot(l_c, dist, "-", color=color, linewidth = 2, label = label)
    ax.set_xlabel("Longitud de correlación $l_c$ [$\mu$m]", fontsize=27)
    ax.set_ylabel("P($l_c$) normalizada", fontsize=27)
    ax.legend(title='Gradiente [mT/m]', title_fontsize=15, fontsize=15, loc='best')
    ax.tick_params(direction='in', top=True, right=True, left=True, bottom=True)
    ax.tick_params(axis='x',rotation=0, labelsize=16, color='black')
    ax.tick_params(axis='y', labelsize=16, color='black')
    title = ax.set_title(f"{nroi} | $T_\mathrm{{NOGSE}}$ = {tnogse} ms | $N$ = {n} | slice = {slic} ", fontsize=15)
    ax.fill_between(l_c, dist, color=color, alpha=0.3)
    #ax.set_xlim(0.5, 10.75)

def plot_results_brute(result, best_vals=True, varlabels=None, output=True):
    
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars, figsize=(11,7))

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parámetro vs chi2 en el caso de un solo parámetro
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=7)
                axes.set_ylabel(r'$\chi**{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parámetro vs chi2 arriba
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi**{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parámetro vs chi2 a la izquierda
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple(a for a in range(npars) if a != i)
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r'$\chi**{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots de todas las combinaciones de dos parámetros
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple(a for a in range(npars) if a not in (i, j))
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 7)
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 3)
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output, bbox_inches="tight", dpi=500) # 
        plt.close(_fig)

#############################################################################
#AJUSTE DE DATOS
#############################################################################
 
def M_nogse_free(TE, G, N, x, M0, D0):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)


    y = TE - (N-1) * x

    return M0*np.exp(-1.0/12 * g**2 * G**2 * D0 * ((N-1) * x**3 + y**3))

def M_nogse_free_offset(TE, G, N, x, M0, D0, C):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)


    y = TE - (N-1) * x

    return M0*np.exp(-1.0/12 * g**2 * G**2 * D0 * ((N-1) * x**3 + y**3)) + C

def M_nogse_rest(TE, G, N, x, t_c, M0, D0):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    bSE=g*G*np.sqrt(D0*t_c)

    return M0 * np.exp(-bSE ** 2 * t_c ** 2 * (4 * np.exp(-y / t_c / 2) - np.exp(-y / t_c) - 3 + y / t_c)) * np.exp(-bSE ** 2 * t_c ** 2 * ((N - 1) * x / t_c + (-1) ** (N - 1) * np.exp(-(N - 1) * x / t_c) + 1 - 2 * N - 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2 + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) * ((N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + N - 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2)) * np.exp(2 * t_c ** 2 * ((np.exp((-y + 2 * x) / t_c / 2) + np.exp((x - 2 * y) / t_c / 2) - np.exp((x - y) / t_c) / 2 - np.exp(-y / t_c) / 2 + np.exp(x / t_c / 2) + np.exp(-y / t_c / 2) - np.exp(x / t_c) / 2 - 0.1e1 / 0.2e1) * (-1) ** (2 * N) + 2 * (-1) ** (1 + N) * np.exp(-(2 * N * x - 3 * x + y) / t_c / 2) + (np.exp(((3 - 2 * N) * x - 2 * y) / t_c / 2) - np.exp((-N * x + 2 * x - y) / t_c) / 2 + np.exp(-(2 * N * x - 4 * x + y) / t_c / 2) + np.exp(-(2 * N * x - 2 * x + y) / t_c / 2) - np.exp((-N * x + x - y) / t_c) / 2 + np.exp(-x * (-3 + 2 * N) / t_c / 2) - np.exp(-x * (N - 2) / t_c) / 2 - np.exp(-(N - 1) * x / t_c) / 2) * (-1) ** N + 2 * (-1) ** (1 + 2 * N) * np.exp((x - y) / t_c / 2)) * bSE ** 2 / (np.exp(x / t_c) + 1))

def M_nogse_rest_offset(TE, G, N, x, t_c, M0, D0, C):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    bSE=g*G*np.sqrt(D0*t_c)

    return C + M0 * np.exp(-bSE ** 2 * t_c ** 2 * (4 * np.exp(-y / t_c / 2) - np.exp(-y / t_c) - 3 + y / t_c)) * np.exp(-bSE ** 2 * t_c ** 2 * ((N - 1) * x / t_c + (-1) ** (N - 1) * np.exp(-(N - 1) * x / t_c) + 1 - 2 * N - 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1) / 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) + 4 * (-np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1))) ** (N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2 + 4 * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) * ((N - 1) * np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + N - 2) / (np.exp(-(N - 1) * x / t_c) ** (1 / (N - 1)) + 1) ** 2)) * np.exp(2 * t_c ** 2 * ((np.exp((-y + 2 * x) / t_c / 2) + np.exp((x - 2 * y) / t_c / 2) - np.exp((x - y) / t_c) / 2 - np.exp(-y / t_c) / 2 + np.exp(x / t_c / 2) + np.exp(-y / t_c / 2) - np.exp(x / t_c) / 2 - 0.1e1 / 0.2e1) * (-1) ** (2 * N) + 2 * (-1) ** (1 + N) * np.exp(-(2 * N * x - 3 * x + y) / t_c / 2) + (np.exp(((3 - 2 * N) * x - 2 * y) / t_c / 2) - np.exp((-N * x + 2 * x - y) / t_c) / 2 + np.exp(-(2 * N * x - 4 * x + y) / t_c / 2) + np.exp(-(2 * N * x - 2 * x + y) / t_c / 2) - np.exp((-N * x + x - y) / t_c) / 2 + np.exp(-x * (-3 + 2 * N) / t_c / 2) - np.exp(-x * (N - 2) / t_c) / 2 - np.exp(-(N - 1) * x / t_c) / 2) * (-1) ** N + 2 * (-1) ** (1 + 2 * N) * np.exp((x - y) / t_c / 2)) * bSE ** 2 / (np.exp(x / t_c) + 1))

def M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0): #alpha es 1/alpha
    return M0 * M_nogse_free(TE, G, N, x, 1, alpha*D0) * M_nogse_rest(TE, G, N, x, t_c, 1, (1-alpha)*D0)

def M_nogse_mixto_offset(TE, G, N, x, t_c, alpha, M0, D0, C): #alpha es 1/alpha
    return M0 * M_nogse_mixto(TE, G, N, x, t_c, alpha, 1, D0) + C

def M_nogse_rest_free_offset(TE, G, N, x, t_c, alpha, D0, A, B, C): #alpha es 1/alpha
    return A * M_nogse_rest(TE, G, N, x, t_c, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, alpha*D0) + C 

def M_nogse_tort_rest_free(TE, G, N, x, t_c, alpha, D0, A, B, C): #alpha es 1/alpha
    return A * M_nogse_rest(TE, G, N, x, t_c, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, alpha*D0) + C * M_nogse_free(TE, G, N, x, 1, D0)

def M_nogse_mixto_free_offset(TE, G, N, x, t_c, alpha, D0, A, B, C): #alpha es 1/alpha
    return A * M_nogse_mixto(TE, G, N, x, t_c, alpha, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, D0) + C

def M_nogse_mixto_riciannoise(TE, G, N, x, t_c, alpha, M0, D0, C): #alpha es 1/alpha
    return np.sqrt( ( M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0) )**2  + (C/M0)**2) + (C/M0)

def M_nogse_mixtooffset_riciannoise(TE, G, N, x, t_c, alpha, M0, D0, C, B): #alpha es 1/alpha
    return np.sqrt( ( M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0) + B )**2  + (C/M0)**2) + (C/M0)

def M_nogse_mixto_offsetort(TE, G, N, x, t_c, alpha, M0, D0, C): #alpha es 1/alpha
    return M_nogse_mixto(TE, G, N, x, t_c, alpha, M0, D0) + C*M_nogse_free(TE, G, N, x, 1, D0*alpha)

def lognormal_mode(l_c, sigma, l_c_mode):
    #l_c_mid = l_c_median*np.exp((sigma**2)/2)
    l_c_mid = l_c_mode*np.exp(sigma**2)
    return (1/(l_c*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(l_c)- np.log(l_c_mid))**2 / (2*sigma**2))

def lognormal_median(l_c, sigma, l_c_median):
    l_c_mid = l_c_median*np.exp((sigma**2)/2)
    # l_c_mid = l_c_mode*np.exp(sigma**2)
    return (1/(l_c*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(l_c)- np.log(l_c_mid))**2 / (2*sigma**2))

def M_mixtoint_medio(TE, G, N, x, t_c, alpha, D0):  # alpha es 1/alpha

    # tc = np.linspace(0.5, 100, 1000)
    # dM = fit_contrast_vs_g_rest(TE, G, N, tc, 1, D0)
    # dM_final = dM[-1]
    
    # # Identificar el índice de estabilización
    # idx_stabilize_array = np.where(np.abs(dM - dM_final)/dM_final >= 0.025)[0]
    # if idx_stabilize_array.size > 0:
    #     idx_stabilize = idx_stabilize_array[-1]
    #     tc_interval = tc[:idx_stabilize + 1]
    # else:
    #     tc_interval = tc  # Usar todo el rango de tc si no se encuentra un punto de estabilización

    #######################################################################################################

    tc_interval = np.linspace(0.2, t_c, 1000)

    #######################################################################################################

    # Calcular la integral para cada valor en x
    integrales = []
    for xi in x:
        M_rest = M_nogse_mixto(TE, G, N, xi, tc_interval, alpha, 1, D0)
        dM_rest_medio = np.trapz(M_rest, tc_interval) / (tc_interval[-1])
        integrales.append(dM_rest_medio)

    return np.array(integrales)

def M_nogse_mixtoint_free_offset_amp(TE, G, N, x, t_c, alpha, D0, A, B, C, amp): #alpha es 1/alpha
    return amp*(A * M_mixtoint_medio(TE, G, N, x, t_c, alpha, D0) + B * M_nogse_free(TE, G, N, x, 1, D0) + C)

def fit_contrast_vs_g_free(TE, G, N, alpha, M0, D0): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, TE/N, M0, alpha*D0) - M_nogse_free(TE, G, N, 0.5, M0, alpha*D0)

def fit_contrast_vs_g_rest(TE, G, N, t_c, M0, D0):
    return M_nogse_rest(TE, G, N, TE/N, t_c, M0, D0) - M_nogse_rest(TE, G, N, 0.5, t_c, M0, D0)

def fit_contrast_vs_g_mixto(TE, G, N, tc, alpha, M0, D0):
    return M_nogse_mixto(TE, G, N, TE/N, tc, alpha, M0, D0) - M_nogse_mixto(TE, G, N, 0.5, tc, alpha, M0, D0) 

def fit_contrast_vs_g_ad(Lc, Ld, n, alpha, D0): #Estan invertidos Lc y Ld y alpha es 1/alpha
    gamma = 267.52218744
    return -np.exp(D0**3*((-0.08333333333333333*alpha*Lc**6)/D0**3 - ((1 - alpha)*Ld**4*(Lc**2/D0 + ((-3 - np.e**(-Lc**2/Ld**2) + 4/np.e**(Lc**2/(2.*Ld**2)))*Ld**2)/D0))/D0**2)) + np.exp(D0**3*((2*(-1)**n*(1 - alpha)*(-3.*(-1)**n - 1/(2.*np.e**(Lc**2/Ld**2)) - (0.5*(-1)**n)/np.e**(Lc**2/(Ld**2*n)) + (2.*(-1)**n)/np.e**(Lc**2/(2.*Ld**2*n)) + 2.*(-1)**n*np.e**(Lc**2/(2.*Ld**2*n)) - 0.5*(-1)**n*np.e**(Lc**2/(Ld**2*n)) + 2*np.e**((Lc**2*(3 - 2*n))/(2.*Ld**2*n)) - 1/(2.*np.e**((Lc**2*(-2 + n))/(Ld**2*n))) + 2*np.e**((D0*(Lc**2/D0 - (2*Lc**2*n)/D0))/(2.*Ld**2*n)) - 3*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))*Ld**6)/(D0**3*(1 + np.e**(Lc**2/(Ld**2*n)))) - (0.08333333333333333*alpha*Lc**6)/(D0**3*n**2) - ((-1 + alpha)*Ld**4*(-((np.e**(Lc**2/(Ld**2*n))*Lc**2)/D0) + ((1 - 4*np.e**(Lc**2/(2.*Ld**2*n)) + 3*np.e**(Lc**2/(Ld**2*n)))*Ld**2*n)/D0))/(D0**2*np.e**(Lc**2/(Ld**2*n))*n) - ((1 - alpha)*Ld**6*(1 + (-1)**(1 + n)*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)) - (4*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2 + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2.*(-1 + n))))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2 - 2*n))*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (Lc**2*(-1 + n))/(Ld**2*n) - 2*n + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-2 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-1 + n) + n))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2))/D0**3))


def fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode, sigma, alpha, M0, D0):

    n = 100 # Cambiar esto no altera demasiado el ajuste y aumentar n produce que el fitting sea mucho más lento
    lmax = 60 #um 

    lcs = np.linspace(0.6, lmax, n) # El mínimo es sensible a Tnogse y G. En general, menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(lcs, sigma, lc_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))

    for lc, w in zip(lcs, weights):
        E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

    return E

def M_nogse_mixtodist_riciannoise(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C): #alpha es 1/alpha
    return np.sqrt( ( fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode, sigma, alpha, M0, D0) )**2  + (C/M0)**2) + (C/M0)

def fit_nogse_vs_x_mixtodistmedian(TE, G, N, x, lc_median, sigma, alpha, M0, D0):

    n = 1000 # Cambiar esto no altera demasiado el ajuste y aumentar n produce que el fitting sea mucho más lento
    lmax = 100 #um 

    lcs = np.linspace(1.0, lmax, n) # El mínimo es sensible a Tnogse y G. En general, menos que 0.5 hace que diverja el ajuste
    weights = lognormal_median(lcs, sigma, lc_median)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))

    for lc, w in zip(lcs, weights):
        E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

    return E

def fit_nogse_vs_x_mixtodistmode_offset(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C):

    n = 100
    lmax = 50 #um 

    lcs = np.linspace(0.6, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(lcs, sigma, lc_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))

    for lc, w in zip(lcs, weights):
        E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

    return E + C

def fit_nogse_vs_x_mixtodistmode_riciannoise(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C):

    n = 1000
    lmax = 40 #um 

    lcs = np.linspace(0.3, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(lcs, sigma, lc_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))
    #Necesito un array de longitud x con el valor de C
    Carr = np.zeros(len(x))
    Carr = Carr + C

    for lc, w in zip(lcs, weights):
        E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

    return np.sqrt(E**2 + Carr**2)

def fit_nogse_vs_x_restdistmode(TE, G, N, x, lc_mode, sigma, M0, D0):

    if sigma<0:
         return 1e20

    n = 100
    lmax = 40 #um 

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(l_cs, sigma, lc_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(x))

    for l_c, w in zip(l_cs, weights):
        E = E + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0*1e12), M0, D0)*w

    return E

def fit_nogse_vs_x_restdistmode_restdistmode(TE, G, N, x, lc_mode_1, sigma_1, lc_mode_2, sigma_2, M0_1, M0_2, D0_1, D0_2):
    #sigma = 0.06416131084794455
    #l_cmid = 7.3*10**-6

    if sigma_1<0 or sigma_2<0:
        return 1e20

    n = 100
    lmax = 100 #um esto es hasta un tau_c de 135ms

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights_1 = lognormal_mode(l_cs, sigma_1, lc_mode_1)
    weights_1 = weights_1/np.sum(weights_1)

    weights_2 = lognormal_mode(l_cs, sigma_2, lc_mode_2)
    weights_2 = weights_2/np.sum(weights_2)

    E = np.zeros(len(x))

    for l_c, w1, w2 in zip(l_cs, weights_1, weights_2):
        E = E + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0_1*1e12), M0_1, D0_1)*w1 + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0_2*1e12), M0_2, D0_2)*w2

    return E

def fit_nogse_vs_x_free_rest(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, M0_2, D0_2): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_rest(TE, G, N, x, tc_2, M0_2, D0_2)

def fit_nogse_vs_x_free_mixto(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2)

def fit_nogse_vs_x_free_mixto_offset(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2) + C 

def fit_nogse_vs_x_mixto_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
    return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2)

def fit_nogse_vs_x_mixto_mixto_offset(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
    return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2) + C

def fit_nogse_vs_x_mixto_mixtodist(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
    return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

def fit_nogse_vs_x_mixtodist_mixtodist(TE, G, N, x, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
    return fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

def fit_nogse_vs_x_free_mixtodist(TE, G, N, x, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

def fit_nogse_vs_x_free_mixtodist_offset(TE, G, N, x, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) + C


def fit_contrast_vs_g_restdistmode(TE, G, N, l_c_mode, sigma, M0, D0):
    n = 100
    lmax = 10

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(l_cs, sigma, l_c_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(G))

    for l_c, w in zip(l_cs, weights):
        E = E + fit_contrast_vs_g_rest(TE, G, N, (l_c**2)/(2*D0*1e12) , M0, D0)*w

    return E

def fit_contrast_vs_g_mixto_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2):
    return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) 

def fit_contrast_vs_g_free_mixto(TE, G, N, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2):
    return fit_contrast_vs_g_free(TE, G, N, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) 

def fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode, sigma, alpha, M0, D0):

    if sigma<0:
        return 1e20
    
    n = 1000
    lmax = 120

    l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
    weights = lognormal_mode(l_cs, sigma, lc_mode)
    weights = weights/np.sum(weights)

    E = np.zeros(len(G))

    for l_c, w in zip(l_cs, weights):
        E = E + fit_contrast_vs_g_mixto(TE, G, N, (l_c**2)/(2*D0*1e12), alpha, M0, D0)*w

    return E

def fit_contrast_vs_g_mixto_mixtodist(TE, G, N, tc_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
    return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

def fit_contrast_vs_g_mixtodist_mixtodist(TE, G, N, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
    return fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

def fit_contrast_vs_g_free_mixtodist(TE, G, N, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
    return fit_contrast_vs_g_free(TE, G, N, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

def fit_contrast_vs_g_mixto_mixto_free(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, alpha_3, M0_3, D0_3):
    return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) + fit_contrast_vs_g_free(TE, G, N, alpha_3, M0_3, D0_3)

def fit_contrast_vs_g_mixto_mixto_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, tc_3, alpha_3, M0_3, D0_3):
    return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) + fit_contrast_vs_g_mixto(TE, G, N, tc_3, alpha_3, M0_3, D0_3)

######################################################################## BORRAR ######################################################################

def delta_M_intra_extra(TE, G, N, t_c_int, t_c_ext, alpha, M0_int, D0_int, D0_ext):
    return fit_contrast_vs_g_rest(TE, G, N, t_c_int, M0_int, D0_int) + fit_contrast_vs_g_mixto(TE, G, N, t_c_ext, alpha, 1 - M0_int, D0_ext)

def delta_M_rest_dist(t_c_mid, sigma, TE, G, N, M0, D0):
    n = 100
    lmax = 120

    t_cs = np.linspace(0.5, lmax, n)
    weights = lognormal_mode(t_cs, sigma, t_c_mid)
    weights = weights/np.sum(weights)

    out = np.zeros(len(G))

    for t_c, w in zip(t_cs, weights):
        out = out + fit_contrast_vs_g_rest(TE, G, N, t_c, M0, D0)*w

    return M0*out

#def delta_M_intra_extra_dist(TE, G, N, t_c_mid_int, t_c_mid_ext, sigma_int, sigma_ext, alpha, M0_int, M0_ext, D0_int, D0_ext):
    return delta_M_rest_dist(t_c_mid_int, sigma_int, TE, G, N, M0_int, D0_int) + delta_M_mixto_dist(t_c_mid_ext, sigma_ext, TE, G, N, alpha, M0_ext, D0_ext)

def delta_M_mixto_bimodal(t_c_mid_1, t_c_mid_2, sigma_1, sigma_2, p, TE, G, N, alpha, M0):
    n = 100
    lmax = 120

    t_cs = np.linspace(0.5, lmax, n)
    weights = p*lognormal_mode(t_cs, sigma_1, t_c_mid_1) + (1-p)*lognormal_mode(t_cs, sigma_2, t_c_mid_2)
    weights = weights/np.sum(weights)

    out = np.zeros(len(G))

    for t_c, w in zip(t_cs, weights):
        out = out + fit_contrast_vs_g_mixto(TE, G, N, t_c, alpha, 1, D0=2.3*10**-12)*w

    return M0*out