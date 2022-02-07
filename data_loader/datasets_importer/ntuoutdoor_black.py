import os 
import glob
from .BaseDataset import BaseImageDataset

class NTUOutdoor_black(BaseImageDataset):
    
    dataset_dir = "NTUOutdoor_Color/black"

    def __init__(self, cfg, verbose=True, **kwargs):
        super(NTUOutdoor_black, self).__init__()
        self.dataset_dir = '/home/GTA/NTUOutdoor_Color/black'
        #self.dataset_dir = os.path.join(cfg.DATASETS.DATASETS_ROOT, self.dataset_dir)
        self.train_list = os.path.join(self.dataset_dir, 'train.txt')
        self.query_list = os.path.join(self.dataset_dir, 'query.txt')
        self.gallery_list = os.path.join(self.dataset_dir, 'gallery.txt')
        required_files = [
            self.dataset_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
        ]
        self._check_before_run()

        train = self.process_list(self.train_list, relabel=True)
        query = self.process_list(self.query_list, relabel=False)
        gallery = self.process_list(self.gallery_list, relabel=False) 

        if verbose:
            print("=> NTUOutdoor/black Loaded")
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

    def process_list(self, list_path, relabel=False):
        with open(list_path, 'r') as f:
            img_paths = [l.strip() for l in f.readlines()]

        pid_container = set()
        for img_path in img_paths:
            pid = int(img_path.split('_')[4])            
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pids = int(img_path.split('_')[4])    
            camid = int(img_path.split('_')[0])
            pid = pids
            if relabel:
                pid = pid2label[pids]     
            img_path = os.path.join(self.dataset_dir, img_path)         
            dataset.append((img_path, pid, camid))

        return dataset
