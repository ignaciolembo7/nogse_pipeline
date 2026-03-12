#NMRSI - Ignacio Lembo Ferrari - 02/09/2024

import numpy as np
import seaborn as sns

sns.set_theme(context='paper')
sns.set_style("whitegrid")
 
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

def M_nogse_rest(TE, G, N, x, tc, M0, D0):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    bSE=g*G*np.sqrt(D0*tc)

    return M0 * np.exp(-bSE ** 2 * tc ** 2 * (4 * np.exp(-y / tc / 2) - np.exp(-y / tc) - 3 + y / tc)) * np.exp(-bSE ** 2 * tc ** 2 * ((N - 1) * x / tc + (-1) ** (N - 1) * np.exp(-(N - 1) * x / tc) + 1 - 2 * N - 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1) / 2) * (-np.exp(-(N - 1) * x / tc) ** (1 / (N - 1))) ** (N - 1) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) + 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1) / 2) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) + 4 * (-np.exp(-(N - 1) * x / tc) ** (1 / (N - 1))) ** (N - 1) * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) ** 2 + 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) * ((N - 1) * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + N - 2) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) ** 2)) * np.exp(2 * tc ** 2 * ((np.exp((-y + 2 * x) / tc / 2) + np.exp((x - 2 * y) / tc / 2) - np.exp((x - y) / tc) / 2 - np.exp(-y / tc) / 2 + np.exp(x / tc / 2) + np.exp(-y / tc / 2) - np.exp(x / tc) / 2 - 0.1e1 / 0.2e1) * (-1) ** (2 * N) + 2 * (-1) ** (1 + N) * np.exp(-(2 * N * x - 3 * x + y) / tc / 2) + (np.exp(((3 - 2 * N) * x - 2 * y) / tc / 2) - np.exp((-N * x + 2 * x - y) / tc) / 2 + np.exp(-(2 * N * x - 4 * x + y) / tc / 2) + np.exp(-(2 * N * x - 2 * x + y) / tc / 2) - np.exp((-N * x + x - y) / tc) / 2 + np.exp(-x * (-3 + 2 * N) / tc / 2) - np.exp(-x * (N - 2) / tc) / 2 - np.exp(-(N - 1) * x / tc) / 2) * (-1) ** N + 2 * (-1) ** (1 + 2 * N) * np.exp((x - y) / tc / 2)) * bSE ** 2 / (np.exp(x / tc) + 1))

def M_nogse_rest_offset(TE, G, N, x, tc, M0, D0, C):

    g = 267.52218744 # ms**-1 mT**-1
    #[D0] = m2/ms

    x = np.array(x)
    TE = np.array(TE)
    N = np.array(N)
    G = np.array(G)

    y = TE - (N-1) * x

    bSE=g*G*np.sqrt(D0*tc)

    return C + M0 * np.exp(-bSE ** 2 * tc ** 2 * (4 * np.exp(-y / tc / 2) - np.exp(-y / tc) - 3 + y / tc)) * np.exp(-bSE ** 2 * tc ** 2 * ((N - 1) * x / tc + (-1) ** (N - 1) * np.exp(-(N - 1) * x / tc) + 1 - 2 * N - 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1) / 2) * (-np.exp(-(N - 1) * x / tc) ** (1 / (N - 1))) ** (N - 1) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) + 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1) / 2) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) + 4 * (-np.exp(-(N - 1) * x / tc) ** (1 / (N - 1))) ** (N - 1) * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) ** 2 + 4 * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) * ((N - 1) * np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + N - 2) / (np.exp(-(N - 1) * x / tc) ** (1 / (N - 1)) + 1) ** 2)) * np.exp(2 * tc ** 2 * ((np.exp((-y + 2 * x) / tc / 2) + np.exp((x - 2 * y) / tc / 2) - np.exp((x - y) / tc) / 2 - np.exp(-y / tc) / 2 + np.exp(x / tc / 2) + np.exp(-y / tc / 2) - np.exp(x / tc) / 2 - 0.1e1 / 0.2e1) * (-1) ** (2 * N) + 2 * (-1) ** (1 + N) * np.exp(-(2 * N * x - 3 * x + y) / tc / 2) + (np.exp(((3 - 2 * N) * x - 2 * y) / tc / 2) - np.exp((-N * x + 2 * x - y) / tc) / 2 + np.exp(-(2 * N * x - 4 * x + y) / tc / 2) + np.exp(-(2 * N * x - 2 * x + y) / tc / 2) - np.exp((-N * x + x - y) / tc) / 2 + np.exp(-x * (-3 + 2 * N) / tc / 2) - np.exp(-x * (N - 2) / tc) / 2 - np.exp(-(N - 1) * x / tc) / 2) * (-1) ** N + 2 * (-1) ** (1 + 2 * N) * np.exp((x - y) / tc / 2)) * bSE ** 2 / (np.exp(x / tc) + 1))

def M_nogse_mixed(TE, G, N, x, tc, alpha, M0, D0): #alpha es 1/alpha
    return M0 * M_nogse_free(TE, G, N, x, 1, alpha*D0) * M_nogse_rest(TE, G, N, x, tc, 1, (1-alpha)*D0)

def M_nogse_mixto_offset(TE, G, N, x, tc, alpha, M0, D0, C): #alpha es 1/alpha
    return M0 * M_nogse_mixed(TE, G, N, x, tc, alpha, 1, D0) + C

def NOGSE_contrast_vs_g_free(TE, G, N, M0, D0):
    return M_nogse_free(TE, G, N, TE/N, M0, D0) - M_nogse_free(TE, G, N, 0, M0, D0)

def NOGSE_contrast_vs_g_rest(TE, G, N, tc, M0, D0):
    return M_nogse_rest(TE, G, N, TE/N, tc, M0, D0) - M_nogse_rest(TE, G, N, 0, tc, M0, D0)

def NOGSE_contrast_vs_g_tort(TE, G, N, alpha, M0, D0): #alpha es 1/alpha
    return M_nogse_free(TE, G, N, TE/N, M0, alpha*D0) - M_nogse_free(TE, G, N, 0, M0, alpha*D0)

def NOGSE_contrast_vs_g_mixed(TE, G, N, tc, alpha, M0, D0):
    return M_nogse_mixed(TE, G, N, TE/N, tc, alpha, M0, D0) - M_nogse_mixed(TE, G, N, 0, tc, alpha, M0, D0) 

def NOGSE_contrast_vs_ad(Lc, Ld, n, alpha, D0): #Estan invertidos Lc y Ld y alpha es 1/alpha
    gamma = 267.52218744
    return -np.exp(D0**3*((-0.08333333333333333*alpha*Lc**6)/D0**3 - ((1 - alpha)*Ld**4*(Lc**2/D0 + ((-3 - np.e**(-Lc**2/Ld**2) + 4/np.e**(Lc**2/(2.*Ld**2)))*Ld**2)/D0))/D0**2)) + np.exp(D0**3*((2*(-1)**n*(1 - alpha)*(-3.*(-1)**n - 1/(2.*np.e**(Lc**2/Ld**2)) - (0.5*(-1)**n)/np.e**(Lc**2/(Ld**2*n)) + (2.*(-1)**n)/np.e**(Lc**2/(2.*Ld**2*n)) + 2.*(-1)**n*np.e**(Lc**2/(2.*Ld**2*n)) - 0.5*(-1)**n*np.e**(Lc**2/(Ld**2*n)) + 2*np.e**((Lc**2*(3 - 2*n))/(2.*Ld**2*n)) - 1/(2.*np.e**((Lc**2*(-2 + n))/(Ld**2*n))) + 2*np.e**((D0*(Lc**2/D0 - (2*Lc**2*n)/D0))/(2.*Ld**2*n)) - 3*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))*Ld**6)/(D0**3*(1 + np.e**(Lc**2/(Ld**2*n)))) - (0.08333333333333333*alpha*Lc**6)/(D0**3*n**2) - ((-1 + alpha)*Ld**4*(-((np.e**(Lc**2/(Ld**2*n))*Lc**2)/D0) + ((1 - 4*np.e**(Lc**2/(2.*Ld**2*n)) + 3*np.e**(Lc**2/(Ld**2*n)))*Ld**2*n)/D0))/(D0**2*np.e**(Lc**2/(Ld**2*n))*n) - ((1 - alpha)*Ld**6*(1 + (-1)**(1 + n)*np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)) - (4*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2 + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2.*(-1 + n))))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(2 - 2*n))*(-(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**n)/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))) + (Lc**2*(-1 + n))/(Ld**2*n) - 2*n + (4*(np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-2 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n))*(-1 + n) + n))/(1 + (np.e**((D0*(Lc**2/D0 - (Lc**2*n)/D0))/(Ld**2*n)))**(1/(-1 + n)))**2))/D0**3))

###############

def OGSE_contrast_vs_g_free(TE, G1, G2, N1, N2, M0, D0):
    return M_nogse_free(TE, G1, N1, TE/N1, M0, D0) - M_nogse_free(TE, G2, N2, TE/N2, M0, D0)

def OGSE_contrast_vs_g_rest(TE, G1, G2, N1, N2, tc, M0, D0):
    return M_nogse_rest(TE, G1, N1, TE/N1, tc, M0, D0) - M_nogse_rest(TE, G2, N2, TE/N2, tc, M0, D0)

def OGSE_contrast_vs_g_tort(TE, G1, G2, N1, N2, alpha, M0, D0):
    return M_nogse_free(TE, G1, N1, TE/N1, M0, alpha*D0) - M_nogse_free(TE, G2, N2, TE/N2, M0, alpha*D0)

def OGSE_contrast_vs_g_mixed(TE, G1, G2, N1, N2, tc, alpha, M0, D0):
    return M_nogse_mixed(TE, G1, N1, TE/N1, tc, alpha, M0, D0) - M_nogse_mixed(TE, G2, N2, TE/N2, tc, alpha, M0, D0)

######################

def PGSE_vs_bvalue_exp(bvalue, M0, D0):
    return M0 * np.exp(-bvalue * D0)







# def delta_ogse_mixed_Deff(TE, G, tc, Deff, D0, N1, N2):
#     return M_nogse_mixed(TE, G, N1, TE/N1, tc, Deff/D0, 1, D0) - M_nogse_mixed(TE, G, N2, TE/N2, tc, Deff/D0, 1, D0)

####################################

# def M_nogse_rest_free_offset(TE, G, N, x, tc, alpha, D0, A, B, C): #alpha es 1/alpha
#     return A * M_nogse_rest(TE, G, N, x, tc, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, alpha*D0) + C 

# def M_nogse_tort_rest_free(TE, G, N, x, tc, alpha, D0, A, B, C): #alpha es 1/alpha
#     return A * M_nogse_rest(TE, G, N, x, tc, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, alpha*D0) + C * M_nogse_free(TE, G, N, x, 1, D0)

# def M_nogse_mixto_free_offset(TE, G, N, x, tc, alpha, D0, A, B, C): #alpha es 1/alpha
#     return A * M_nogse_mixto(TE, G, N, x, tc, alpha, 1, D0) + B * M_nogse_free(TE, G, N, x, 1, D0) + C

# def M_nogse_mixto_riciannoise(TE, G, N, x, tc, alpha, M0, D0, C): #alpha es 1/alpha
#     return np.sqrt( ( M_nogse_mixto(TE, G, N, x, tc, alpha, M0, D0) )**2  + (C/M0)**2) + (C/M0)

# def M_nogse_mixtooffset_riciannoise(TE, G, N, x, tc, alpha, M0, D0, C, B): #alpha es 1/alpha
#     return np.sqrt( ( M_nogse_mixto(TE, G, N, x, tc, alpha, M0, D0) + B )**2  + (C/M0)**2) + (C/M0)

# def M_nogse_mixto_offsetort(TE, G, N, x, tc, alpha, M0, D0, C): #alpha es 1/alpha
#     return M_nogse_mixto(TE, G, N, x, tc, alpha, M0, D0) + C*M_nogse_free(TE, G, N, x, 1, D0*alpha)

# def lognormal_mode(l_c, sigma, l_c_mode):
#     #l_c_mid = l_c_median*np.exp((sigma**2)/2)
#     l_c_mid = l_c_mode*np.exp(sigma**2)
#     return (1/(l_c*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(l_c)- np.log(l_c_mid))**2 / (2*sigma**2))

# def lognormal_median(l_c, sigma, l_c_median):
#     l_c_mid = l_c_median*np.exp((sigma**2)/2)
#     # l_c_mid = l_c_mode*np.exp(sigma**2)
#     return (1/(l_c*sigma*np.sqrt(2*np.pi))) * np.exp(-(np.log(l_c)- np.log(l_c_mid))**2 / (2*sigma**2))

# def M_mixtoint_medio(TE, G, N, x, tc, alpha, D0):  # alpha es 1/alpha

#     # tc = np.linspace(0.5, 100, 1000)
#     # dM = fitcontrast_vs_g_rest(TE, G, N, tc, 1, D0)
#     # dM_final = dM[-1]
    
#     # # Identificar el índice de estabilización
#     # idx_stabilize_array = np.where(np.abs(dM - dM_final)/dM_final >= 0.025)[0]
#     # if idx_stabilize_array.size > 0:
#     #     idx_stabilize = idx_stabilize_array[-1]
#     #     tc_interval = tc[:idx_stabilize + 1]
#     # else:
#     #     tc_interval = tc  # Usar todo el rango de tc si no se encuentra un punto de estabilización

#     #######################################################################################################

#     tc_interval = np.linspace(0.2, tc, 1000)

#     #######################################################################################################

#     # Calcular la integral para cada valor en x
#     integrales = []
#     for xi in x:
#         M_rest = M_nogse_mixto(TE, G, N, xi, tc_interval, alpha, 1, D0)
#         dM_rest_medio = np.trapz(M_rest, tc_interval) / (tc_interval[-1])
#         integrales.append(dM_rest_medio)

#     return np.array(integrales)

# def M_nogse_mixtoint_free_offset_amp(TE, G, N, x, tc, alpha, D0, A, B, C, amp): #alpha es 1/alpha
#     return amp*(A * M_mixtoint_medio(TE, G, N, x, tc, alpha, D0) + B * M_nogse_free(TE, G, N, x, 1, D0) + C)


# #############################################################################
# #AJUSTE DE DATOS
# #############################################################################

# def fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode, sigma, alpha, M0, D0):

#     n = 100 # Cambiar esto no altera demasiado el ajuste y aumentar n produce que el fitting sea mucho más lento
#     lmax = 60 #um 

#     lcs = np.linspace(0.6, lmax, n) # El mínimo es sensible a Tnogse y G. En general, menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(lcs, sigma, lc_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(x))

#     for lc, w in zip(lcs, weights):
#         E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

#     return E

# def M_nogse_mixtodist_riciannoise(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C): #alpha es 1/alpha
#     return np.sqrt( ( fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode, sigma, alpha, M0, D0) )**2  + (C/M0)**2) + (C/M0)

# def fit_nogse_vs_x_mixtodistmedian(TE, G, N, x, lc_median, sigma, alpha, M0, D0):

#     n = 1000 # Cambiar esto no altera demasiado el ajuste y aumentar n produce que el fitting sea mucho más lento
#     lmax = 100 #um 

#     lcs = np.linspace(1.0, lmax, n) # El mínimo es sensible a Tnogse y G. En general, menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_median(lcs, sigma, lc_median)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(x))

#     for lc, w in zip(lcs, weights):
#         E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

#     return E

# def fit_nogse_vs_x_mixtodistmode_offset(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C):

#     n = 100
#     lmax = 50 #um 

#     lcs = np.linspace(0.6, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(lcs, sigma, lc_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(x))

#     for lc, w in zip(lcs, weights):
#         E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

#     return E + C

# def fit_nogse_vs_x_mixtodistmode_riciannoise(TE, G, N, x, lc_mode, sigma, alpha, M0, D0, C):

#     n = 1000
#     lmax = 40 #um 

#     lcs = np.linspace(0.3, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(lcs, sigma, lc_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(x))
#     #Necesito un array de longitud x con el valor de C
#     Carr = np.zeros(len(x))
#     Carr = Carr + C

#     for lc, w in zip(lcs, weights):
#         E = E + M_nogse_mixto(TE, G, N, x, (lc**2)/(2*D0*1e12), alpha, M0, D0)*w

#     return np.sqrt(E**2 + Carr**2)

# def fit_nogse_vs_x_restdistmode(TE, G, N, x, lc_mode, sigma, M0, D0):

#     if sigma<0:
#          return 1e20

#     n = 100
#     lmax = 40 #um 

#     l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(l_cs, sigma, lc_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(x))

#     for l_c, w in zip(l_cs, weights):
#         E = E + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0*1e12), M0, D0)*w

#     return E

# def fit_nogse_vs_x_restdistmode_restdistmode(TE, G, N, x, lc_mode_1, sigma_1, lc_mode_2, sigma_2, M0_1, M0_2, D0_1, D0_2):
#     #sigma = 0.06416131084794455
#     #l_cmid = 7.3*10**-6

#     if sigma_1<0 or sigma_2<0:
#         return 1e20

#     n = 100
#     lmax = 100 #um esto es hasta un tau_c de 135ms

#     l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights_1 = lognormal_mode(l_cs, sigma_1, lc_mode_1)
#     weights_1 = weights_1/np.sum(weights_1)

#     weights_2 = lognormal_mode(l_cs, sigma_2, lc_mode_2)
#     weights_2 = weights_2/np.sum(weights_2)

#     E = np.zeros(len(x))

#     for l_c, w1, w2 in zip(l_cs, weights_1, weights_2):
#         E = E + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0_1*1e12), M0_1, D0_1)*w1 + M_nogse_rest(TE, G, N, x, (l_c**2)/(2*D0_2*1e12), M0_2, D0_2)*w2

#     return E

# def fit_nogse_vs_x_free_rest(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, M0_2, D0_2): #alpha es 1/alpha
#     return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_rest(TE, G, N, x, tc_2, M0_2, D0_2)

# def fit_nogse_vs_x_free_mixto(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
#     return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2)

# def fit_nogse_vs_x_free_mixto_offset(TE, G, N, x, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
#     return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2) + C 

# def fit_nogse_vs_x_mixto_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
#     return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2)

# def fit_nogse_vs_x_mixto_mixto_offset(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
#     return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + M_nogse_mixto(TE, G, N, x, tc_2, alpha_2, M0_2, D0_2) + C

# def fit_nogse_vs_x_mixto_mixtodist(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
#     return M_nogse_mixto(TE, G, N, x, tc_1, alpha_1, M0_1, D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

# def fit_nogse_vs_x_mixtodist_mixtodist(TE, G, N, x, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
#     return fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

# def fit_nogse_vs_x_free_mixtodist(TE, G, N, x, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2): #alpha es 1/alpha
#     return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2)

# def fit_nogse_vs_x_free_mixtodist_offset(TE, G, N, x, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2, C): #alpha es 1/alpha
#     return M_nogse_free(TE, G, N, x, M0_1, alpha_1*D0_1) + fit_nogse_vs_x_mixtodistmode(TE, G, N, x, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) + C

# def fit_contrast_vs_g_restdistmode(TE, G, N, l_c_mode, sigma, M0, D0):
#     n = 100
#     lmax = 10

#     l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(l_cs, sigma, l_c_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(G))

#     for l_c, w in zip(l_cs, weights):
#         E = E + fit_contrast_vs_g_rest(TE, G, N, (l_c**2)/(2*D0*1e12) , M0, D0)*w

#     return E

# def fit_contrast_vs_g_mixto_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2):
#     return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) 

# def fit_contrast_vs_g_free_mixto(TE, G, N, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2):
#     return fit_contrast_vs_g_free(TE, G, N, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) 

# def fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode, sigma, alpha, M0, D0):

#     if sigma<0:
#         return 1e20
    
#     n = 1000
#     lmax = 120

#     l_cs = np.linspace(0.5, lmax, n) #menos que 0.5 hace que diverja el ajuste
#     weights = lognormal_mode(l_cs, sigma, lc_mode)
#     weights = weights/np.sum(weights)

#     E = np.zeros(len(G))

#     for l_c, w in zip(l_cs, weights):
#         E = E + fit_contrast_vs_g_mixto(TE, G, N, (l_c**2)/(2*D0*1e12), alpha, M0, D0)*w

#     return E

# def fit_contrast_vs_g_mixto_mixtodist(TE, G, N, tc_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
#     return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

# def fit_contrast_vs_g_mixtodist_mixtodist(TE, G, N, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
#     return fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_1, sigma_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

# def fit_contrast_vs_g_free_mixtodist(TE, G, N, alpha_1, M0_1, D0_1, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2):
#     return fit_contrast_vs_g_free(TE, G, N, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixtodist(TE, G, N, lc_mode_2, sigma_2, alpha_2, M0_2, D0_2) 

# def fit_contrast_vs_g_mixto_mixto_free(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, alpha_3, M0_3, D0_3):
    # return contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) + contrast_vs_g_free(TE, G, N, alpha_3, M0_3, D0_3)

# def fit_contrast_vs_g_mixto_mixto_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1, tc_2, alpha_2, M0_2, D0_2, tc_3, alpha_3, M0_3, D0_3):
#     return fit_contrast_vs_g_mixto(TE, G, N, tc_1, alpha_1, M0_1, D0_1) + fit_contrast_vs_g_mixto(TE, G, N, tc_2, alpha_2, M0_2, D0_2) + fit_contrast_vs_g_mixto(TE, G, N, tc_3, alpha_3, M0_3, D0_3)
