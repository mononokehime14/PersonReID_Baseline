from torch.utils import data
# from reid.utils.data import transforms as T
from PIL import Image
import os
import glob
import re
from torchvision import transforms as T
from .BaseDataset import BaseImageDataset

class Celeb(BaseImageDataset):
    dataset_dir = 'Celeb-reID'  
    def __init__(self, cfg, verbose=True, **kwargs):
        super(Celeb, self).__init__()       
        self.dataset_dir = os.path.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)        
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'gallery')
        self.pids = []

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Celeb-reID loaded")
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
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))     
        pattern = re.compile(r'([-\d]+)_(\d)')        
        
        i = 0
        all_pids = {}
        dataset = []
        for img_path in img_paths:
            #fname = os.path.basename(img_path)
            pid, cam = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored

            #self.fnames.append(fpath)
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            cam -= 1
            pid = all_pids[pid]
            self.pids.append(pid)
            i = i+1
            dataset.append((img_path, pid, cam))

        return dataset
