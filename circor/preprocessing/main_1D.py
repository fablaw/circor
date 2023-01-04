
from circor.parameters.params import BUCKET_NAME
import os
from circor.preprocessing.preprocessing_csv import select_patients

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
    #local path for tsv file
    local_tsv=download_bucket_objects(bucket_name=BUCKET_NAME,
                            blob_path= 'tsv_raw',
                            local_path = f'circor/raw_data'
                            )

    tsv_filepath=local_tsv + '/tsv_raw'

    print("\n✅ TSV loaded!")

    #local path for csv file
    csv_filepath=select_patients()

    #local path for npy file
    local_audio=download_bucket_objects(bucket_name=BUCKET_NAME,
                            blob_path= 'audio_raw',
                            local_path = f'circor/raw_data'
                            )
    npy_filepath=local_audio + '/audio_raw'

    print("\n✅ Audio loaded!")

    return print("\n✅ Data successfully loaded!")
