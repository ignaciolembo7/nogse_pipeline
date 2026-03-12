#NMRSI - Ignacio Lembo Ferrari - 02/09/2024

import numpy as np
from brukerapi.dataset import Dataset as ds
import seaborn as sns

sns.set_theme(context='paper')
sns.set_style("whitegrid")

#############################################################################
#PROCESAMIENTO DE IMAGENES 
#############################################################################

def nogse_image_params(method_path):
    with open(method_path) as file:
        txt = file.read()

        start_idx = txt.find("NSegments")
        end_idx = txt.find("##", start_idx)
        Nsegments = float(txt[start_idx + len("Nsegments="):end_idx])

        start_idx = txt.find("NAverages")
        end_idx = txt.find("##", start_idx)
        NAverages = float(txt[start_idx + len("NAverages="):end_idx])

        start_idx = txt.find("NRepetitions")
        end_idx = txt.find("$$", start_idx)
        NRepetitions = float(txt[start_idx + len("NRepetitions="):end_idx])

        start_idx = txt.find("DummyScans")
        end_idx = txt.find("##", start_idx)
        DummyScans = float(txt[start_idx + len("DummyScans="):end_idx])

        start_idx = txt.find("DummyScansDur")
        end_idx = txt.find("$$", start_idx)
        DummyScansDur = float(txt[start_idx + len("DummyScansDur="):end_idx])
        
        start_idx = txt.find("EffSWh")
        end_idx = txt.find("##", start_idx)
        EffSWh = float(txt[start_idx + len("EffSWh="):end_idx])

        start_idx = txt.find("ScanTime=")
        end_idx = txt.find("##", start_idx)
        ScanTime = float(txt[start_idx + len("ScanTime="):end_idx])
        import datetime
        delta = datetime.timedelta(seconds=ScanTime/1000)
        minutos = delta.seconds // 60
        segundos = delta.seconds % 60
        ScanTime = str(minutos) + " min " + str(segundos) + " s"

        start_idx = txt.find("DwUsedSliceThick")
        end_idx = txt.find("##", start_idx)
        DwUsedSliceThick = float(txt[start_idx + len("DwUsedSliceThick="):end_idx]) 

        PVM_Fov = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_Fov=" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_Fov.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        PVM_Fov = str(PVM_Fov[0]) + " mm" + " x " + str(PVM_Fov[1]) + " mm"

        PVM_SpatResol = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_SpatResol" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_SpatResol.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

        PVM_SpatResol = str(PVM_SpatResol[0]*1000) + " um" + " x " + str(PVM_SpatResol[1]*1000) + " um"

        PVM_Matrix = []
        with open(method_path, 'r') as archivo:
        # Establece una bandera para identificar cuándo debes leer los valores
            leyendo_valores = False

            # Lee el archivo línea por línea
            for linea in archivo:
                # Busca la línea que contiene "Matrix"
                if "PVM_Matrix" in linea:
                    # Activa la bandera para comenzar a leer los valores
                    leyendo_valores = True
                elif leyendo_valores:
                    # Extrae los valores de la línea (elimina espacios en blanco)
                    valores_str = linea.strip().split()
                    
                    # Verifica si la línea contiene solo números flotantes
                    if all(valor.replace(".", "", 1).isdigit() or (valor[0] == '-' and valor[1:].replace(".", "", 1).isdigit()) for valor in valores_str):
                        # Convierte los valores a números flotantes y agrégalos al vector
                        PVM_Matrix.extend([float(valor) for valor in valores_str])
                    else:
                        # Si la línea no contiene números flotantes, detén la lectura
                        break

    return {"Nsegments": Nsegments, "NAverages": NAverages, "NRepetitions": NRepetitions, "DummyScans": DummyScans, "DummyScansDur": DummyScansDur, "ScanTime": ScanTime, "EffSWh": EffSWh, "DwUsedSliceThick": DwUsedSliceThick, "Img size": PVM_Matrix,  "PVM_Fov": PVM_Fov, "PVM_SpatResol": PVM_SpatResol}

def nogse_params(method_path):
    with open(method_path) as file:
        txt = file.read()

        start_idx = txt.find("Tnogse")
        end_idx = txt.find("##", start_idx)
        t_nogse = float(txt[start_idx + len("Tnogse="):end_idx])

        start_idx = txt.find("RampGradStr")
        end_idx = txt.find("$$", start_idx)
        ramp_grad_str = float(txt[start_idx + len("RampGradStr="):end_idx])

        start_idx = txt.find("RampGradN")
        end_idx = txt.find("##", start_idx)
        ramp_grad_N = float(txt[start_idx + len("RampGradN="):end_idx])

        start_idx = txt.find("RampGradX")
        end_idx = txt.find("##", start_idx)
        ramp_grad_x = float(txt[start_idx + len("RampGradX="):end_idx])

        start_idx = txt.find("EchoTime")
        end_idx = txt.find("##", start_idx)
        EchoTime = float(txt[start_idx + len("EchoTime="):end_idx])

        return {"t_nogse": t_nogse, "ramp_grad_str": ramp_grad_str, "ramp_grad_N": ramp_grad_N, "ramp_grad_x": ramp_grad_x, "EchoTime": EchoTime}

def generate_contrast_vs_g_roi_A0unico(image_paths, method_paths, mask, slic):
    
    experiments = []
    params = []
    f_hahn = []
    f_cpmg = [] 
    f_matrix = []
    error_matrix = []
    f = []
    error = []
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        experiments.append(ims[:,:,slic,0])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 

    M_matrix = np.array(experiments)

    for i in range(len(M_matrix)):
        roi = np.zeros_like(M_matrix[i])
        roi[mask == 255] = M_matrix[i][mask == 255]
        f_matrix.append(np.mean(roi[roi != 0]))
        error_matrix.append(np.std(roi[roi != 0]))

    N = len(f_matrix) 
    middle_idx = int(N/2) 
    f_cpmg = f_matrix[middle_idx:] 
    f_hahn = f_matrix[:middle_idx] 
    error_cpmg = error_matrix[middle_idx:]
    error_hahn = error_matrix[:middle_idx]
    g_contrast = g[:middle_idx] 
    g_contrast_check = g[middle_idx:] 
    f = np.array(f_cpmg) - np.array(f_hahn)
    error = np.sqrt(np.array(error_cpmg)**2 + np.array(error_hahn)**2)

    #vectores_combinados = zip(g_contrast, f)
    #vectores_ordenados = sorted(vectores_combinados, key=lambda x: x[0])
    #g_contrast, f = zip(*vectores_ordenados)

    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n",T_nogse)
    print("g_contrast",g_contrast)
    print("g_contrast_check",g_contrast_check)
    print("f:\n",f)
    print("x:\n",x)
    print("N:\n",n)
    print("TE:\n",TE)

    return T_nogse[0], g_contrast, int(n[0]), f, error, f_hahn, error_hahn, f_cpmg, error_cpmg

def generate_NOGSE_vs_x_hist(image_paths, method_paths, mask, slic):

    experiments = []
    A0s = []
    params = []
    f = []
    error = []
    pixel_values_in_roi = []  # Lista para almacenar valores de píxeles en la ROI
    
    for image_path, method_path in zip(image_paths, method_paths):
        ims = ds(image_path).data
        A0s.append(ims[:, :, slic, 0]) 
        experiments.append(ims[:, :, slic, 1])
        param_dict = nogse_params(method_path)
        param_list = list(param_dict.values())
        params.append(param_list)
                
    T_nogse, g, n, x, TE = np.array(params).T 
    print(f"NOGSE parameters for the {len(experiments)} experiments:\n")
    print("T_nogse:\n", T_nogse)
    print("g:\n", g)
    print("x:\n", x)
    print("N:\n", n)
    print("TE:\n", TE)

    M_matrix = np.array(experiments)
    A0_matrix = np.array(A0s)
    E_matrix = M_matrix # / A0_matrix

    for i in range(len(E_matrix)):
        roi = np.zeros_like(E_matrix[i])
        roi[mask == 255] = E_matrix[i][mask == 255]
        f.append(np.mean(roi[roi != 0]))
        error.append(np.std(roi[roi != 0]))
        
        # Guardar valores de píxeles en la ROI para el histograma
        pixel_values_in_roi.extend(roi[roi != 0])
    
    return T_nogse[0], g[0], x, int(n[0]), pixel_values_in_roi
