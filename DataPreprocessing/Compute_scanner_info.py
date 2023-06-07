
import os
import json
from pathlib import Path
from typing import List

import pandas as pd


def Compute_Extract_Hospitals(path):

    json_paths: List[Path] = list(path.rglob('*json'))
    source_data_json : List[Path] = list(path.rglob('sourcedata/*/ses-M000/anat/*/*.json'))
    source_data_json2: List[Path] = list(path.rglob('sourcedata/*/ses-M000/anat/*/*/*.json'))

    json_paths =list( set(json_paths) - set(source_data_json) - set(source_data_json2))


    PATIENTS_LIST = []
    Manufacturer_LIST = []
    MFS_LIST = []
    Hospitals = []
    j = 0
    for i, patient_path in enumerate(json_paths):

        try:

            js_file = json.load(open(patient_path))
            patient_name = patient_path.stem.split('_ses')[0]
            MFS = js_file['MagneticFieldStrength']
            Manufacturer = js_file['Manufacturer']

            PATIENTS_LIST.append(patient_name)
            Manufacturer_LIST.append(Manufacturer)
            MFS_LIST.append(MFS)
            try:
                Hospitals.append(js_file['InstitutionName'])
            except:
                print(js_file)
                Hospitals.append('n/a')

        except:
            j = j +1
            print(j)
            print(patient_path)

    metadata_acquisition = {'subject': PATIENTS_LIST,
                            'magnetic-field-strength': MFS_LIST,
                            'manufacturer': Manufacturer_LIST,
                            'hospitals': Hospitals}


    metadata = pd.DataFrame.from_dict(data = metadata_acquisition)
    metadata.to_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/Magnetic_field_strength.csv")

    df_RES = pd.read_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/resolution.csv")
    df_RES['subject'] = df_RES['subject'].str.split('_ses').str[0]

    df_RES.to_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/resolution.csv")



Compute_Extract_Hospitals(Path('/Users/rosana.eljurdi/Datasets/Lymphoma'))