import os
from torch.utils.data import Dataset
from PIL import Image

class CelebADataset(Dataset):
    """ A Dataset class for celeba dataset.  
    """
    
    def __init__(
            self,
            root_dir=None,
            partition_list_file=None,  # Should be relative to root dir
            split='train'):
        super(CelebADataset, self).__init__()
        if not os.path.exists(root_dir):
            raise FileNotFoundError
        self.root_dir = root_dir

        if split not in ["train", "val", "test"]:
            raise ValueError
        self.split = split

        if partition_list_file is None:
            raise ValueError
        partition_list_file = os.path.join(root_dir, partition_list_file)
        if  not os.path.exists(partition_list_file):
            raise FileNotFoundError
        with open(partition_list_file) as pfile:                
            fytpe = 3
            if split == 'train':
                ftype = 0
            elif split == 'val':
                ftype = 1
            elif split == 'test':
                ftype = 2
            self.files = [x.strip().split(' ')[0] for x in pfile
                          if int(x.strip().split(' ')[1]) == ftype]

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, 'img_align_celeba', self.files[index])
        img = Image.open(fname)
        return img

    def __len__(self):
        return len(self.files)
