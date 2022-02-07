# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import glob
import re
from .BaseDataset import BaseImageDataset


class Full_Market1501(BaseImageDataset):
    """
    Market1501
    Train model using full dataset

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(Full_Market1501, self).__init__()
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)        

        self._check_before_run()

        train = self._process_dir(self.dataset_dir, relabel=True)
        query = []
        gallery = []

        self.train = train 
        self.query = query
        self.gallery = gallery
        
        if verbose:
            print("=> Full Market1501 Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        
    def _process_dir(self, dir_path, relabel=False):
        subdirs = ['bounding_box_train', 'query', 'bounding_box_test']
        img_paths = []
        for subdir in subdirs:
            img_paths += glob.glob(os.path.join(dir_path, subdir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
