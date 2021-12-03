import os
import torch
import torchvision
import torchvision.transforms as transforms


class bootstrapped_CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, load_train: bool=True, train_part: bool=True, noisy_data: bool=False, **kwargs):
        super().__init__(*args, train=load_train, **kwargs)
        if noisy_data:
            noised_idxs = torch.randperm(len(self.data))[:int(len(self.data) * 0.2)]
            self.targets = torch.LongTensor(self.targets)
            self.targets[noised_idxs] = torch.randint(0, 10, noised_idxs.size())
            self.targets = self.targets.tolist()
        
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
    
    def __getitem__(self, idx):
        return super().__getitem__(self.idxs[idx])

class bootstrapped_CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, test_size: int=5000, use_bootstrapping: bool=False, load_train: bool=True, train_part: bool=True, noisy_data: bool=False, **kwargs):
        super().__init__(*args, train=load_train, **kwargs)

        if noisy_data:
            noised_idxs = torch.randperm(len(self.data))[:int(len(self.data) * 0.2)]
            self.targets = torch.LongTensor(self.targets)
            self.targets[noised_idxs] = torch.randint(0, 100, noised_idxs.size())
            self.targets = self.targets.tolist()
        
        if train_part:
            print(f"Using train ({len(self.data) - test_size})")
            self.data = self.data[:-test_size]
            self.targets = self.targets[:-test_size]
        else:
            print(f"Using validation ({test_size})")
            self.train = False
            self.data = self.data[-test_size:]
            self.targets = self.targets[-test_size:]

        if use_bootstrapping:
            self.idxs = torch.randint(len(self.data), (len(self.data),))
        else:
            self.idxs = torch.arange(len(self.data))
    
        self.noisy_data = noisy_data
        if noisy_data:
            self.noised_idxs = torch.randperm(len(self.data))[:int(len(self.data) * 0.2)]
            self.noised_labels = torch.randint(0, 100, self.noised_idxs.size(0))        
    
    def __getitem__(self, idx):
        return super().__getitem__(self.idxs[idx])

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


def loaders(dataset, path, batch_size, num_workers, transform_name, use_test=False,
            use_bootstrapping=False, shuffle_train=True, noisy_data = False):
    if dataset == 'CIFAR10':
        ds = bootstrapped_CIFAR10
    elif dataset == 'CIFAR100':
        ds = bootstrapped_CIFAR100
    else:
        ds = getattr(torchvision.datasets, dataset)
    path = os.path.join(path, dataset.lower())
    transform = getattr(getattr(Transforms, dataset), transform_name)

    if use_test:
        print('You are going to run models on the test set. Are you sure?')
        train_set = ds(path, test_size=0, load_train=True, train_part=True, use_bootstrapping=use_bootstrapping, download=True, transform=transform.train, noisy_data = noisy_data)
        test_set = ds(path, test_size=0, load_train=False, train_part=True, use_bootstrapping=False, download=False, transform=transform.test)
    else:
        train_set = ds(path, load_train=True, train_part=True, use_bootstrapping=use_bootstrapping, download=True, transform=transform.train, noisy_data = noisy_data)
        test_set = ds(path, load_train=True, train_part=False, use_bootstrapping=False, download=False, transform=transform.test)

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
           }, max(train_set.targets) + 1
