import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


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

    CIFAR100 = CIFAR10
    SVHN = CIFAR10


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            use_bootstrapping=False, shuffle_train=True, noisy_data = False):
    if dataset == 'CIFAR10':
        ds = bootstrapped_CIFAR10
    elif dataset == 'CIFAR100':
        ds = bootstrapped_CIFAR100
    elif dataset == 'SVHN':
        ds = SVHN_dataset
    else:
        ds = getattr(torchvision.datasets, dataset)
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
