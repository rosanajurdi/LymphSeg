
import os
import json
from pathlib import Path
from typing import List

def Compute_Extract_Hospitals(path):
    json_paths: List[Path] = list(path.rglob('*json'))
    hospitals = []
    Manufacturer = []
    MFS = []
    Hospitals = []
    for patient_path in json_paths:
        try:
            js_file = json.load(open(patient_path))
            MFS.append(js_file['MagneticFieldStrength'])
            Manufacturer.append(js_file['Manufacturer'])
            Hospitals.append(js_file["InstitutionName"])
        except:
            try:
                Hospitals.append(js_file["InstitutionAddress"])
            except:
                print(js_file)
            pass

Compute_Extract_Hospitals(Path('/Users/rosana.eljurdi/Datasets/Lymphoma'))