# encoding: utf-8
"""
ROSE Lab's GTA ReID Dataset, v3 alpha. This is the train-full version, meaning that there is no separation between
training set, query set and gallery set. The training set consists of all annotated images available. This is to
facillitate cross domain experiments.
"""
import os
import glob
import re
import numpy as np
from .BaseDataset import BaseImageDataset


class GTA_v3_full(BaseImageDataset):
    """
    GTA

    Dataset statistics:
    # identities: 910 (WIP)
    # images: 524160 
    """
    dataset_dir = 'GTAv3_uniform'

    def __init__(self, cfg, verbose=True, relabel=False, **kwargs):
        super(GTA_v3_full, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir) #/home/GTA, #GTAv3
        self.train_list = os.path.join(self.dataset_dir, 'train.txt')
        self.query_list = os.path.join(self.dataset_dir, 'query_1.txt')
        self.gallery_list = os.path.join(self.dataset_dir, 'gallery_1.txt')
        print(self.dataset_dir)

        if not os.path.exists(os.path.join(self.dataset_dir, 'train.txt')):
            print('Labels have not been generated.')
            print('Labels will be generated into dir: {}'.format(self.dataset_dir))
            self.generate_label()

        self._check_before_run()
        self.cam_dict = {
            'person': 0,
            'surveillance': 1,
            'drone': 2
        }
        train = self._process_list(self.train_list, relabel=True)
        gallery = self._process_list(self.gallery_list, relabel=False, camid=0)
        query = self._process_list(self.query_list, relabel=False, camid=1)

        if verbose:
            print("=> GTA_v3 FULL Loaded. If you did not intend for the training set to consist of the full images, ABORT NOW!")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def generate_label(self): 
        save_dir = self.dataset_dir
        f = open(os.path.join(save_dir, 'train.txt'), 'w')
        if not os.path.exists(self.dataset_dir): 
            print('Dataset not downloaded') 
            # Implement Person_ReID_Dataset_Downloader.py here
        for root, subdir, files in os.walk(self.dataset_dir):
            ped_list = []
            ped_list_s = []
            if root.split('/')[-1].__contains__('skip'):
                continue
            for _file in files: 
                if _file.endswith('.txt'): #skip the txt files
                    continue
                elif _file.endswith('.xml'): #skip the xml files
                    continue
                if not re.search(r'[a-z]+$', root): #[a-z] ->matches any character btw brackets. +$ -> string MUST end with letter
                    #So if the file ends with a number e.g. 355, 374, etc, go onto the next iteration of the loop.
                    continue
                #if _file.__contains__('carpark') or _file.__contains__('greenery') or _file.__contains__('indoorcarpark'):
                #    continue
                # Try person view only?
                if _file.__contains__('person'):
                # if not _file.__contains__('drone'):
                    # Skip yaw angles 90, 135, 180 and 225
                    #if int(_file.split('_')[5]) in [90, 135, 180, 225]:
                    #    continue
                    ped_list.append(os.path.join(''.join(root.split('/')[-2:-1]), _file.split('_')[0], _file))
                if _file.__contains__('surveillance'):
                    ped_list_s.append(os.path.join(''.join(root.split('/')[-2:-1]), _file.split('_')[0], _file))
            ped_list = np.array(ped_list)
            ped_list_s = np.array(ped_list_s)
            if ped_list.size == 0: 
                continue
            if ped_list_s.size == 0: 
                continue
            ped_list = np.random.choice(ped_list, 50)
#            ped_list_s = np.random.choice(ped_list_s, 100)
            for ped in ped_list:
                print(ped, file=f)
#            for ped in ped_list_s: 
#                print(ped, file=
        f.close()
        #open(os.path.join(save_dir, 'query.txt'), 'w').close()
        #open(os.path.join(save_dir, 'gallery.txt'), 'w').close() # For now, these are empty

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not os.path.exists(self.query_list):
            raise RuntimeError("'{}' is not available".format(self.query_list))
        if not os.path.exists(self.gallery_list):
            raise RuntimeError("'{}' is not available".format(self.gallery_list))

    def _process_list(self, list_path, relabel=False, camid=-1):
        with open(list_path, 'r') as f: 
            img_paths = [l.strip() for l in f.readlines()]
        dataset = [] 
        if relabel: 
            pid_container = set()
            for img_path in img_paths: 
                #example img_path: 99/subway/subway_RAIN_13_walk_person_99_3.jpg
                pid = int(img_path.split('/')[-3]) 
                #pid = 99 for the above example
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            pid = int(img_path.split('/')[-3])
            filename = '/'.join(img_path.split('/')[-3:])
            camid = self.cam_dict[str(filename.split('_')[4])]
            #example img_path: fbi_CLOUDS_13_walk_surveillance_288_8.jpg
            #                   0     1   2   3         4       5  6
            #for surveillance, this returns 1
            if camid == 2: 
                continue
            if relabel: pid = pid2label[pid]
            assert 0 <= camid <= 1 # No drone
            #assert 0 <= pid <= 1208, print(pid)
            dataset.append((os.path.join(self.dataset_dir, filename), int(pid), int(camid))) # camid :: person = 0, surveillance = 1, drone = 2
        print(len(dataset))
        return dataset