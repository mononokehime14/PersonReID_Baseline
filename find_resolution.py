import os

import matplotlib.pyplot as plt

from statistics import median
from tqdm import tqdm
from PIL import Image

"""
The purpose of this script is to calculate the median resolution of images in a directory. The median image size is done by multiplying the height and width of the image together.
You can add your own directories in the dataset_subdirs below.

There are multiple picture sizes which can yield the same width*height value hence there may be more than 1 median value.
"""
def main():
    BASE_DIR = '/home'
    dataset_subdirs = {
                        'NTUWhite':'GTA/NTUOutdoor_Color_V2/white',
                        'NTUBlack':'GTA/NTUOutdoor_Color_V2/black',
                        'Market':'GTA/Market-1501',
                        'Duke':'GTA/DukeMTMC-reID',
                        'whitereid':'GTA/White-reID',
                        'blackreid':'GTA/Black-reID',
                        'MSMT17_V2':'syazwan/Person_ReID_Baseline/datasets/MSMT17_V2',
                        'Celeb':'GTA/Celeb-reID'
}
                

    for d_key,dataset in dataset_subdirs.items():
        print('\nCurrently computing for', d_key, '( ͡° ͜ʖ ͡°)')

        #Precomputing files count
        pic_size = {}
        files, f_count = walkdir(os.path.join(BASE_DIR,dataset))

        #Computing for real
        for file in tqdm(files, total=f_count, desc=f'{d_key} dataset', leave=False):
            with Image.open(file) as img:
                width,height = img.size
                if (width,height) not in pic_size:
                    pic_size[(width,height)] = [width*height]
                else:
                    pic_size[(width,height)].append(width*height)
        
        # pic_size.values() returns a list of lists. Join all the lists together and find the median resolution.
        # I know this may not be the best solution, alternatively can change it to calculate by median height or width.
        pic_sizes = [indiv for grp in pic_size.values() for indiv in grp]
        median_val = median(pic_sizes)
        for p_key,v in pic_size.items():
            if median_val in v:
                print(f"The median size for {d_key} dataset is {p_key} (╯ ͠° ͟ʖ ͡°)╯┻━┻")


def walkdir(folder):

    #Walk through every file in a directory
    file_list = []
    file_count = 0
    for root, subdir, files in os.walk(folder):
        for file in files:
            if file.endswith('.jpg'):
                file_list.append(os.path.abspath(os.path.join(root,file)))
                file_count += 1
    return(file_list,file_count)

if __name__ == '__main__':
    main()
