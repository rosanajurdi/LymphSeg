
import nibabel as nb
import pandas as pd
import json
import pydicom
import datetime
import os
import nibabel as nb
import json
import pydicom
import datetime
import os
import pathlib

def specify_partition(row:pd.DataFrame)-> str:
    # d1 : clean and yellow MRIs
    # d2 yellow with artefacts
    # d3 orange without artefacts (clean)
    # d4 orange with artefacts and red (else)
    is_yellow = row['color'] == 'yellow'
    is_clean = row['artefacts'] == 'clean'
    if is_clean and is_yellow:
        return 'D1'
    elif is_yellow and not is_clean:
        return "D2"
    elif not is_yellow and is_clean:
        return "D3"
    else:
        return 'D4'



sourcedata_path = "/Users/rosana.eljurdi/Datasets/Lymphoma/sourcedata"
# extract acquisition time !
data_dir = "/Users/rosana.eljurdi/Datasets/Lymphoma/sourcedata"

l1_metadata = (
    pd.read_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/Lymphoma_annotation_1st_group.csv")
    .set_axis(labels=['patient_id','artefacts','lymphome_type', 'isotropic','description','extra',  'color'], axis="columns")
    .drop(labels=['extra'], axis='columns')
    .set_index(keys='patient_id')
)

l2_metadata = (
    pd.read_excel("/Users/rosana.eljurdi/Datasets/Lymphoma/description/Lymphoma-40-Patients.xlsx")
    .set_axis(labels=['patient_id','artefacts','lymphome_type', 'isotropic','description',  'color'], axis='columns')
    .set_index(keys='patient_id')
)


age_sex_data = (
    pd.read_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/age_sex_info.csv")
    .set_axis(labels=['i', 'patient_id' ,'age', 'gender'], axis='columns')
    .drop(labels=['i'], axis='columns')
    .set_index(keys='patient_id')
)

acquisition = (
    pd.read_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/description/MRI_acquisition_date.csv")
    .set_axis(labels=['i', 'patient_id', 'acquisition-date'], axis='columns')
    .drop(labels=['i'], axis='columns')
    .set_index(keys='patient_id')
)
raw_metadata = pd.concat([l1_metadata, l2_metadata]).join([age_sex_data, acquisition]).replace({'000Y':'n/a'})

# Iterate over all subject directories
subject_ids = list(
    map(
        lambda path: int(path.name.split('-')[-1]),
        pathlib.Path(sourcedata_path).glob('sub-*')
    )
)

metadata = (
    raw_metadata.loc[subject_ids]
    .rename_axis(index={'patient_id':'participant_id'})
)
metadata = metadata.reset_index()
metadata['participant_id'] = "sub-" + metadata['participant_id'].astype(str)
metadata = metadata.set_index('participant_id')

metadata['partition'] = metadata.apply(specify_partition, axis='columns')

metadata.to_csv("/Users/rosana.eljurdi/Datasets/Lymphoma/participants_mni.tsv",sep='\t', na_rep="n/a")
