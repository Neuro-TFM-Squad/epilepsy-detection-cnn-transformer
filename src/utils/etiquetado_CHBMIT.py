import os
import re
import mne
import pandas as pd
import numpy as np

def procesar_todo_chb_mit(root_dir):
    datos_completos = []
    
    # Recorrer carpetas de pacientes (chb01, chb02...)
    for paciente_folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, paciente_folder)
        if not os.path.isdir(folder_path): continue
        
        summary_file = os.path.join(folder_path, f"{paciente_folder}-summary.txt")
        if not os.path.exists(summary_file): continue
        
        print(f"Procesando Paciente: {paciente_folder}...")
        
        # Leer el resumen del paciente una sola vez
        with open(summary_file, 'r', errors='ignore') as f:
            summary_text = f.read()
            
        # Buscar todos los bloques de archivos EDF en el texto
        files_info = re.findall(r"File Name: (.*?\.edf).*?Number of Seizures in File: (\d+)", 
                                summary_text, re.DOTALL)
        
        for edf_name, num_seizures in files_info:
            num_seizures = int(num_seizures)
            
            # Extraer tiempos de crisis si existen
            # Buscamos el bloque de texto específico de este archivo EDF
            start_pos = summary_text.find(edf_name)
            end_pos = summary_text.find("File Name:", start_pos + 1)
            file_block = summary_text[start_pos:end_pos]
            
            seizures = []
            if num_seizures > 0:
                starts = re.findall(r"Seizure (?:\d+ )?Start Time: (\d+) seconds", file_block)
                ends = re.findall(r"Seizure (?:\d+ )?End Time: (\d+) seconds", file_block)
                seizures = list(zip(map(int, starts), map(int, ends)))
            
            # Aquí ya tienes: Nombre de archivo y lista de (inicio, fin) de crisis
            # Siguiente paso: Guardar esto en un CSV maestro o procesar el EDF
            datos_completos.append({
                'paciente': paciente_folder,
                'archivo': edf_name,
                'crisis': seizures,
                'total_crisis': num_seizures
            })
            
    return pd.DataFrame(datos_completos)

# --- EJECUCIÓN ---
df_index = procesar_todo_chb_mit(r"D:\TFM\epilepsy-detection-cnn-transformer\data\CHB-MIT\chb-mit-scalp-eeg-database-1.0.0")
df_index.to_csv("src/utils/metadata_chbmit_etiquetado.csv", index=False)