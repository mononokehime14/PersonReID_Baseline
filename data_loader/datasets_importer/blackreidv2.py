# encoding: utf-8
import os
import glob
import re
from .BaseDataset import BaseImageDataset

class BlackReID_FullBlack(BaseImageDataset):
    """
    Only extract PIDs wearing black clothes

    Dataset statistics:    
    # identities: 492
    # query images: 490
    # gallery images: 3674  
    """

    dataset_dir = 'Black-reID'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(BlackReID_FullBlack, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.train_list = os.path.join(self.dataset_dir, 'train.txt')
        self.query_list = os.path.join(self.dataset_dir, 'query.txt')
        self.gallery_list = os.path.join(self.dataset_dir, 'gallery.txt')

        self._check_before_run()

        train = self._process_list(self.train_list, relabel=True)
        query = self._process_list(self.query_list, relabel=False)
        gallery = self._process_list(self.gallery_list, relabel=False)

        if verbose:
            print("=> Black-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


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

    def _process_list(self, list_path, relabel=False):
        with open(list_path, 'r') as f:
            img_paths = [l.strip() for l in f.readlines()]
        pattern = re.compile(r'(.*\d.*)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = pattern.search(os.path.basename(img_path)).groups()
            pid = pid.split('/')[-1]
            print(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pids, camid = pattern.search(os.path.basename(img_path)).groups()
            pids = pids.split('/')[-1]
            pid = pids            
            if relabel: 
                pid = pid2label[pids]
            else:
                if pids.startswith('b'):
                    pid = pids.split('_')[1]
                else:
                    pid = pids
            pid = int(pid)
            camid = int(camid)            
            dataset.append((os.path.join(self.dataset_dir, img_path), pid, camid))

        return dataset