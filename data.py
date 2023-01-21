import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
import zipfile
import io

class ZipDataset(Dataset):
    def __init__(self, zip_path, ds_descr_path, transform, cache_into_memory=True, **kwargs):
        self.transform = transform
        if cache_into_memory:
            f = open(str(zip_path), 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(str(zip_path), 'r')
        
        self._pathes = []
        self._labels = []
        with open(ds_descr_path, 'r') as f:
            for l in tqdm(f, desc='Read dataset list'):
                s = l.split()
                if len(s) != 2:
                    print('WARNING: skip line in dataset', l)
                    continue
                path, label = s
                label = int(label)
                self._pathes.append(path)
                self._labels.append(label)
        self._labels = np.array(self._labels)
        self.num_classes = max(self._labels) + 1
        self.root = zip_path
        self.img_size = 64

    def __getitem__(self, index):
        buf = self.zip_file.read(name=self._pathes[index])
        sample = self.transform(Image.open(io.BytesIO(buf)))
        label = torch.tensor(self._labels[index], dtype=torch.long)
        
        out = {'input': sample, 
               'target': label}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[index]
        return out

    def __len__(self):
        return len(self._pathes)

    @property
    def targets(self):
        return self._labels.copy()

class ImageNet(ZipDataset):
    def __init__(self, path, load_train=True, train_part=True, **kwargs):
        root_path = os.path.join(path, 'imagenet.zip')
        ds_descr_path = os.path.join(path,  'dataset_list_train.txt' if train_part else 'dataset_list_test.txt')
        super(ImageNet, self).__init__(root_path, ds_descr_path, cache_into_memory=True, **kwargs)

class SmallImageNet(ZipDataset):
    def __init__(self, path, load_train=True, train_part=True, **kwargs):
        root_path = os.path.join(path, 'imagenet.zip')
        ds_descr_path = os.path.join(path,  'dataset_list_small_train.txt' if train_part else 'dataset_list_small_test.txt')
        super(SmallImageNet, self).__init__(root_path, ds_descr_path, cache_into_memory=True, **kwargs)

class TestNewClassesImageNet(ZipDataset):
    def __init__(self, path, load_train=True, train_part=True, **kwargs):
        root_path = os.path.join(path, 'imagenet.zip')
        ds_descr_path = os.path.join(
            path,  
            'dataset_list_test_new_classes_train.txt' if train_part else 'dataset_list_test_new_classes_test.txt'
        )
        super(TestNewClassesImageNet, self).__init__(root_path, ds_descr_path, cache_into_memory=True, **kwargs)

class ListDataset(Dataset):
    def __init__(self, ds_descr_path, transform, tpr_list_name, issame_file_name, **kwargs):
        self.ds_descr_path = ds_descr_path
        self.transform = transform
        self.tpr_idx_list = np.load(tpr_list_name)
        self.issame = np.load(issame_file_name)
        self._pathes = list()
        self._labels = []
        with open(ds_descr_path, 'r') as f:
            for l in tqdm(f, desc='Read dataset list'):
                s = l.split()
                if len(s) != 2:
                    print('WARNING: skip line in dataset', l)
                    continue
                path, label = s
                label = int(label)
                self._pathes.append(path)
                self._labels.append(label)
        self._labels = np.array(self._labels)
        self.num_classes = max(self._labels) + 1
        self.__create_tpr_carray()

    def __getitem__(self, index):
        label = torch.tensor(self._labels[index], dtype=torch.long)
        sample = Image.open(self._pathes[index])
        sample = self.transform(sample)

        out = {'input': sample, 
               'target': label}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[index]
        return out

    def __len__(self):
        return len(self._pathes)

    @property
    def targets(self):
        return self._labels.copy()
    
    def __create_tpr_carray(self):
        self.carray = []
        
        for i in self.tpr_idx_list:
            self.carray.append(self[i]['input'])
        
        self.carray = torch.stack(self.carray)


class AGEDB(ListDataset):
    def __init__(self, path, load_train=True, train_part=True, **kwargs):
        if train_part:
            ds_descr_path = '../arcface_ensembles/dataset_folder/dataset_list_train.txt'
            tpr_list_name = '../arcface_ensembles/dataset_folder/dataset_list_train_tpr_idx_list.npy'
            issame_file_name = '../arcface_ensembles/dataset_folder/dataset_list_train_tpr_issame.npy'
        else:
            ds_descr_path = '../arcface_ensembles/dataset_folder/dataset_list_test.txt'
            tpr_list_name = '../arcface_ensembles/dataset_folder/dataset_list_test_tpr_idx_list.npy'
            issame_file_name = '../arcface_ensembles/dataset_folder/dataset_list_test_tpr_issame.npy'
        
        super(AGEDB, self).__init__(
            ds_descr_path = ds_descr_path,
            tpr_list_name = tpr_list_name, 
            issame_file_name = issame_file_name,
            **kwargs
        )
        self.img_size = 64

class TestNewClassesAGEDB(ListDataset):
    def __init__(self, path, load_train=True, train_part=True, **kwargs):
        if train_part:
            ds_descr_path = '../arcface_ensembles/dataset_folder/dataset_list_train_uniq_classes.txt'
            tpr_list_name = '../arcface_ensembles/dataset_folder/dataset_list_train_uniq_classes_tpr_idx_list.npy'
            issame_file_name = '../arcface_ensembles/dataset_folder/dataset_list_train_uniq_classes_tpr_issame.npy'
        else:
            ds_descr_path = '../arcface_ensembles/dataset_folder/dataset_list_test_uniq_classes.txt'
            tpr_list_name = '../arcface_ensembles/dataset_folder/dataset_list_test_uniq_classes_tpr_idx_list.npy'
            issame_file_name = '../arcface_ensembles/dataset_folder/dataset_list_test_uniq_classes_tpr_issame.npy'
        
        super(TestNewClassesAGEDB, self).__init__(
            ds_descr_path = ds_descr_path,
            tpr_list_name = tpr_list_name, 
            issame_file_name = issame_file_name,
            **kwargs
        )
        self.img_size = 64


class bootstrapped_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, 
                 load_train: bool=True, train_part: bool=True, noisy_data: bool=False, 
                 **kwargs):
        super().__init__(*args, train=load_train, **kwargs)
        if noisy_data and load_train:
            noised_targets = np.load(os.path.join(args[0], 'noisy_p_02_target.npy')).tolist()
            self.targets = noised_targets
        
        if train_part:
            print(f"Using train ({len(self.data) - test_size})")
            self.data = self.data if test_size == 0 else self.data[:-test_size]
            self.targets = self.targets if test_size == 0 else self.targets[:-test_size]
        else:
            print(f"Using validation ({test_size})")
            self.train = False
            self.data = self.data if test_size == 0 else self.data[-test_size:]
            self.targets = self.targets if test_size == 0 else self.targets[-test_size:]

        self.targets = np.array(self.targets)


        if use_bootstrapping:
            self.idxs = torch.randint(len(self.data), (len(self.data),))
        else:
            self.idxs = torch.arange(len(self.data))
        
        self.num_classes = max(self.targets) + 1
        self.img_size = 32
    
    def __getitem__(self, idx):
        a, b = super().__getitem__(self.idxs[idx])
        out = {'input': a, 
               'target': b}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[self.idxs[idx]]
        return out


class CIFAR10_fixed_aug(torchvision.datasets.CIFAR10):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, 
                 load_train: bool=True, train_part: bool=True, noisy_data: bool=False, 
                 **kwargs):
        super().__init__(*args, train=load_train, **kwargs)
        if noisy_data and load_train:
            noised_targets = np.load(os.path.join(args[0], 'noisy_p_02_target.npy')).tolist()
            self.targets = noised_targets
        
        if train_part:
            print(f"Using train ({len(self.data) - test_size})")
            self.data = self.data if test_size == 0 else self.data[:-test_size]
            self.targets = self.targets if test_size == 0 else self.targets[:-test_size]
        else:
            print(f"Using validation ({test_size})")
            self.train = False
            self.data = self.data if test_size == 0 else self.data[-test_size:]
            self.targets = self.targets if test_size == 0 else self.targets[-test_size:]

        self.num_classes = max(self.targets) + 1
    
    @property
    def ds_scale(self):
        return 30
    
    def __getitem__(self, idx):
        base_idx = idx // 30
        img, label = super().__getitem__(base_idx)
        
        ntrasform = idx % 30
        
        if ntrasform % 2 == 1:
            img = transforms.functional.hflip(img)

        if ntrasform % 5 == 0:
            (dx, dy) = (4, 4)
        elif ntrasform % 5 == 1:
            (dx, dy) = (0, 0)
        elif ntrasform % 5 == 2:
            (dx, dy) = (8, 8)
        elif ntrasform % 5 == 3:
            (dx, dy) = (8, 0)
        else:
            (dx, dy) = (0, 8)
        
        img = transforms.functional.crop(
            transforms.Pad(padding=4, padding_mode='reflect')(img), dx, dy, 32, 32
        )
        
        if ntrasform % 3 == 0:
            angle = 0
        elif ntrasform % 3 == 1:
            angle = -20
        else:
            angle = 20
        img = transforms.functional.rotate(img, angle=angle)

        out = {'input': img, 
               'target': label}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[idx]
        return out

    def __len__(self):
        return super().__len__() * self.ds_scale
    

class bootstrapped_CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, 
                 load_train: bool=True, train_part: bool=True, noisy_data: bool=False, 
                 **kwargs):
        super().__init__(*args, train=load_train, **kwargs)

        if noisy_data and load_train:
            noised_targets = np.load(os.path.join(args[0], 'noisy_p_02_target_100.npy')).tolist()
            self.targets = noised_targets
        
        if train_part:
            print(f"Using train ({len(self.data) - test_size})")
            self.data = self.data if test_size == 0 else self.data[:-test_size]
            self.targets = self.targets if test_size == 0 else self.targets[:-test_size]
        else:
            print(f"Using validation ({test_size})")
            self.train = False
            self.data = self.data if test_size == 0 else self.data[-test_size:]
            self.targets = self.targets if test_size == 0 else self.targets[-test_size:]

        if use_bootstrapping:
            self.idxs = torch.randint(len(self.data), (len(self.data),))
        else:
            self.idxs = torch.arange(len(self.data))
        
        self.num_classes = max(self.targets) + 1
    
    def __getitem__(self, idx):
        a, b = super().__getitem__(self.idxs[idx])
        out = {'input': a, 
               'target': b}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[self.idxs[idx]]
        return out


class SVHN_dataset(torchvision.datasets.SVHN):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, 
                 load_train: bool=True, train_part: bool=True, noisy_data: bool=False, 
                 **kwargs):
        super().__init__(*args, split='train' if load_train else 'test', **kwargs)

        if train_part:
            print(f"Using train ({len(self.data) - test_size})")
            self.data = self.data if test_size == 0 else self.data[:-test_size]
            self.labels = self.labels if test_size == 0 else self.labels[:-test_size]
        else:
            print(f"Using validation ({test_size})")
            self.train = False
            self.data = self.data if test_size == 0 else self.data[-test_size:]
            self.labels = self.labels if test_size == 0 else self.labels[-test_size:]

        if use_bootstrapping:
            self.idxs = torch.randint(len(self.data), (len(self.data),))
        else:
            self.idxs = torch.arange(len(self.data))
        
        self.num_classes = max(self.labels) + 1
    
    def __getitem__(self, idx):
        a, b = super().__getitem__(self.idxs[idx])
        out = {'input': a, 
               'target': b}
        
        if hasattr(self, 'gap_size'):
            out['gap'] = self.gap_size[self.idxs[idx]]
        return out


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
        
        class ResNet_9:
            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                transforms.RandomHorizontalFlip(), 
                transforms.RandomRotation(20), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        class ResNet_9_noDA:
            train = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        class VGG_noDA:

            train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet_noDA:

            train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

        class ResNet_9_Faces:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            test = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        class ResNet_9_Faces_noDA:
            train = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            test = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    CIFAR10_fixed_aug       = CIFAR10
    CIFAR100                = CIFAR10
    CIFAR10_fixed_aug       = CIFAR10
    SVHN                    = CIFAR10
    AGEDB                   = CIFAR10
    TestNewClassesAGEDB     = CIFAR10
    ImageNet                = CIFAR10
    SmallImageNet           = CIFAR10
    TestNewClassesImageNet  = CIFAR10
    
def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            use_bootstrapping=False, shuffle_train=True, noisy_data = False):
    
    if dataset == 'CIFAR10':
        ds = bootstrapped_CIFAR10
    elif dataset == 'CIFAR10_fixed_aug':
        ds = CIFAR10_fixed_aug
    elif dataset == 'CIFAR100':
        ds = bootstrapped_CIFAR100
    elif dataset == 'SVHN':
        ds = SVHN_dataset
    elif dataset == 'AGEDB':
        ds = AGEDB
    elif dataset == 'TestNewClassesAGEDB':
        ds = TestNewClassesAGEDB
    elif dataset == 'ImageNet':
        ds = ImageNet
    elif dataset == 'SmallImageNet':
        ds = SmallImageNet
    elif dataset == 'TestNewClassesImageNet':
        ds = TestNewClassesImageNet
    else:
        ds = getattr(torchvision.datasets, dataset)
    
    if 'ImageNet' in dataset:
        path = os.path.join(path, 'imagenet')
    else:
        path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        train_set = ds(path, 
                       test_size=0, 
                       load_train=True, 
                       train_part=True, 
                       use_bootstrapping=use_bootstrapping, 
                       download=True, 
                       transform=transform.train, 
                       noisy_data = noisy_data)
        test_set = ds(path, 
                      test_size=0, 
                      load_train=False, 
                      train_part=True, 
                      use_bootstrapping=False, 
                      download=False, 
                      transform=transform.test)
    else:
        train_set = ds(path, 
                       load_train=True, 
                       train_part=True, 
                       use_bootstrapping=use_bootstrapping, 
                       download=True, 
                       transform=transform.train, 
                       noisy_data = noisy_data)
        test_set = ds(path, 
                      load_train=True, 
                      train_part=False, 
                      use_bootstrapping=False, 
                      download=False, 
                      transform=transform.test)

    return {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, train_set.num_classes

def loaders_three_dataset(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            use_bootstrapping=False, shuffle_train=True, noisy_data = False):
    
    
    if dataset == 'CIFAR10':
        ds = bootstrapped_CIFAR10
    elif dataset == 'CIFAR10_fixed_aug':
        ds = CIFAR10_fixed_aug
    elif dataset == 'CIFAR100':
        ds = bootstrapped_CIFAR100
    elif dataset == 'SVHN':
        ds = SVHN_dataset
    elif dataset == 'AGEDB':
        ds = AGEDB
    elif dataset == 'TestNewClassesAGEDB':
        ds = TestNewClassesAGEDB
    elif dataset == 'ImageNet':
        ds = ImageNet
    elif dataset == 'SmallImageNet':
        ds = SmallImageNet
    elif dataset == 'TestNewClassesImageNet':
        ds = TestNewClassesImageNet
    else:
        ds = getattr(torchvision.datasets, dataset)
    
    if 'ImageNet' in dataset:
        path = os.path.join(path, 'imagenet')
    else:
        path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        train_set = ds(path, 
                       test_size=0, 
                       load_train=True, 
                       train_part=True, 
                       use_bootstrapping=use_bootstrapping, 
                       download=True, 
                       transform=transform.train, 
                       noisy_data = noisy_data)
        test_set = ds(path, 
                      test_size=0, 
                      load_train=False, 
                      train_part=True, 
                      use_bootstrapping=False, 
                      download=False, 
                      transform=transform.test)
    else:
        train_set = ds(path, 
                       load_train=True, 
                       train_part=True, 
                       use_bootstrapping=use_bootstrapping, 
                       download=True, 
                       transform=transform.train, 
                       noisy_data = noisy_data)
        test_set = ds(path, 
                      load_train=True, 
                      train_part=False, 
                      use_bootstrapping=False, 
                      download=False, 
                      transform=transform.test)

    tr = torch.utils.data.DataLoader(
       train_set,
       batch_size=batch_size,
       shuffle=shuffle_train,
       num_workers=num_workers,
       pin_memory=True
           )
    te = torch.utils.data.DataLoader(
               test_set,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=True
           )
    
    valid_set_size_1 = int(len(te.dataset) * 0.7)
    valid_set_size_2 = len(te.dataset) - valid_set_size_1
    test_set_1, test_set_2 = torch.utils.data.random_split(te.dataset, [valid_set_size_1, valid_set_size_2])
    set_ = torch.utils.data.ConcatDataset([tr.dataset, test_set_1])
    
    
    return {
               'train': torch.utils.data.DataLoader(
                   set_,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test': torch.utils.data.DataLoader(
                   test_set_2,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }, train_set.num_classes
