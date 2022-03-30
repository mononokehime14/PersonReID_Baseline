# encoding: utf-8
import os.path as osp
import glob
import re
from .BaseDataset import BaseImageDataset


class MSMT17_V1(BaseImageDataset):
    """
    MSMT17

    Dataset statistics:
    # appearance: 805
    # images: 20,411 (train) + 3,443 (query) + 21,542 (gallery)
    """
    dataset_dir = 'MSMT17_V1'

    def __init__(self, cfg, verbose=True, **kwargs):
        super(MSMT17_V1, self).__init__()
        self.dataset_dir = osp.join(cfg.DATASETS.STORE_DIR, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17_V1 Loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        dataset = []

        for img_path in img_paths:
            temp = img_path.split('/')[-1].split('.')[0]
            pid, camid, img_id = temp.split('_')
            pid = int(pid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #     assert idx == pid, "See code comment for explanation"
        for img_path in img_paths:
            temp = img_path.split('/')[-1].split('.')[0]
            pid, camid, img_id = temp.split('_')
            pid = int(pid)
            if relabel: pid = pid2label[pid]
            camid = int(camid[1:])
            dataset.append((img_path, pid, camid))
        return dataset
