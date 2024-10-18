'''
 # @ Author: Abhinanda Ranjit Punnakkal
 # @ Create Time: 2024-10-18 10:58:10
 # @ Modified by: Abhinanda Ranjit Punnakkal
 # @ Modified time: 2024-10-18 10:58:27
 # @ Description: Code to unpack data instances from the zipped files downloaded from the Dataverse
 url: https://dataverse.
 '''

import os
import shutil
import glob
from tqdm import tqdm

def data_unpack(data_path, output_path):
    '''
    Unpack the data from the data_path to the output_path
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    zipped_files = glob.glob(data_path + '/*.zip')
    for data_path in tqdm(zipped_files):
        file_name = os.path.basename(data_path).split('.')[0]
        shutil.unpack_archive(data_path, os.path.join(output_path, file_name))

    # return output_path

if __name__ == "__main__":
    data_path = 'data_zip'      # Path to the zipped data instances downloded from the Dataverse
    output_path = 'data'        # Path to store the unzipped data instances
    data_unpack(data_path, output_path)