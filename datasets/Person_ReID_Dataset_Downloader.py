import os
import sys
import zipfile
import shutil
import argparse
import requests
from tqdm import tqdm

dataset_list = [
'CUHK03',
'Market-1501',
'DukeMTMC-reID',
'MSMT17_V2',
]

def Person_ReID_Dataset_Downloader(save_dir, dataset_name):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_dir_exist = os.path.join(save_dir , dataset_name)

    if dataset_name in dataset_list:
        if not os.path.exists(save_dir_exist):
            temp_dir = os.path.join(save_dir , 'temp')

            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            destination = os.path.join(temp_dir , dataset_name+'.zip')
        
            url = "https://linshanify.synology.me/Datasets/Person_ReID/{}.zip".format(dataset_name)

            filesize = int(requests.head(url).headers["Content-Length"])
            filename = os.path.basename(url)

            chunk_size = 1024

            with requests.get(url, stream=True) as r, open(destination, "wb") as f, tqdm(
                unit="B",  # unit string to be displayed.
                unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
                unit_divisor=1024,  # is used when unit_scale is true
                total=filesize,  # the total iteration.
                file=sys.stdout,  # default goes to stderr, this is the display on console.
                desc=filename  # prefix to be displayed on progress bar.
            ) as progress:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    # download the file chunk by chunk
                    datasize = f.write(chunk)
                    # on each chunk update the progress bar.
                    progress.update(datasize)

            zip_ref = zipfile.ZipFile(destination)
            print("Extracting %s" % dataset_name)
            zip_ref.extractall(save_dir)
            zip_ref.close()
            shutil.rmtree(temp_dir)
            print("Done")
        else:
            print("Dataset Check Success: %s exists!" %dataset_name)
    else:
        print("Dataset %s is not supported" %dataset_name)

#For United Testing and External Use
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Name and Dataset Directory')
    parser.add_argument(dest="save_dir", action="store", default="/Datasets/",help="")
    parser.add_argument(dest="dataset_name", action="store", default="Market-1501",type=str,help="")
    args = parser.parse_args() 
    Person_ReID_Dataset_Downloader(args.save_dir,args.dataset_name)
