
from circor.parameters.params import BUCKET_NAME
import os
from circor.preprocessing.preprocessing_csv import select_patients
import subprocess

def download_bucket_objects(bucket_name, blob_path, local_path):
    # blob path is bucket folder name


    #creating folder locally
    if not os.path.exists(f'{local_path}'):
        os.makedirs(f'{local_path}')

    #downloading data to local folder
    command = "gsutil -m cp -r gs://{bucketname}/{blobpath} {localpath}".format(bucketname = bucket_name,
                                                                          blobpath = blob_path,
                                                                          localpath = local_path
                                                                          )
    os.system(command)

    return command

def download_to_local():

    if not os.path.exists(f'raw_data'):
        os.makedirs(f'raw_data')

    #download tsv files
    download_bucket_objects(bucket_name=BUCKET_NAME,
                            blob_path= 'tsv_raw',
                            local_path = f'raw_data'
                            )

    print("\n✅ TSV loaded!")

    #download csv files
    source_csv = f'gs://{BUCKET_NAME}/training_data_new.csv'
    local_csv = f'raw_data'
    command_csv = f'gsutil cp {source_csv} {local_csv} '

    subprocess.run(command_csv, shell=True)

    csv_filepath = f'raw_data/training_data_new.csv'

    print("\n✅ CSV loaded!")

    #download audio files
    download_bucket_objects(bucket_name=BUCKET_NAME,
                            blob_path= 'audio_raw',
                            local_path = f'raw_data'
                            )

    print("\n✅ Audio loaded!")

    return csv_filepath
