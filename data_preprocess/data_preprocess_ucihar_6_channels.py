'''
Data Pre-processing on UCIHAR dataset.

'''

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pickle as cp
from data_preprocess.data_preprocess_utils import get_sample_weights, train_test_val_split
from data_preprocess.base_loader import base_loader

def format_data_x(datafile):
    """
    Carga los archivos de señales (lista de paths) y forma X con shape (N, 128, 6).

    Importante:
    - En el dataset UCI-HAR cada archivo tiene shape (N, 128).
    - Al apilar 6 archivos, x_data queda (N, 6*128).
    - Luego se reordena a (N, 128, 6).
    """
    x_data = None
    for item in datafile:
        item_data = np.loadtxt(item, dtype=float)  # shape: (N, 128)

        if x_data is None:
            x_data = np.zeros((len(item_data), 1), dtype=float)

        x_data = np.hstack((x_data, item_data))

    x_data = x_data[:, 1:]
    print('x_data.shape:', x_data.shape)  # (N, 6*128)

    X = None
    for i in range(len(x_data)):
        row = np.asarray(x_data[i, :], dtype=float)  # (6*128,)
        row = row.reshape(6, 128).T                  # (128, 6)

        if X is None:
            X = np.zeros((len(x_data), 128, 6), dtype=float)

        X[i] = row

    print('X.shape:', X.shape)  # (N, 128, 6)
    return X


def format_data_y(datafile):
    data = np.loadtxt(datafile, dtype=int) - 1
    return data

def load_domain_data(domain_idx):
    """Load data from the specific domain (subject) with index domain_idx.
    Devuelve X, y, d para el sujeto.
    """
    # NUEVA carpeta para no sobrescribir la versión de 9 canales
    data_dir = './data/ucihar2/'
    saved_filename = 'ucihar2_domain_' + domain_idx + '_wd_6ch.data'  # cache separado

    if os.path.isfile(os.path.join(data_dir, saved_filename)) == True:
        data = np.load(os.path.join(data_dir, saved_filename), allow_pickle=True)
        X = data[0][0]
        y = data[0][1]
        d = data[0][2]
    else:
        if os.path.isdir(data_dir) == False:
            os.makedirs(data_dir)

        str_folder = './data/UCI HAR Dataset/'

        # SOLO 6 señales: body_gyro (3) + total_acc (3)
        INPUT_SIGNAL_TYPES = [
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt'
                           for item in INPUT_SIGNAL_TYPES]
        str_test_files  = [str_folder + 'test/'  + 'Inertial Signals/' + item + 'test.txt'
                           for item in INPUT_SIGNAL_TYPES]

        str_train_y = str_folder + 'train/y_train.txt'
        str_test_y  = str_folder + 'test/y_test.txt'

        str_train_id = str_folder + 'train/subject_train.txt'
        str_test_id  = str_folder + 'test/subject_test.txt'

        X_train = format_data_x(str_train_files)  # (N_train, 128, 6)
        X_test  = format_data_x(str_test_files)   # (N_test, 128, 6)

        Y_train = format_data_y(str_train_y)
        Y_test  = format_data_y(str_test_y)

        id_train = format_data_y(str_train_id)
        id_test  = format_data_y(str_test_id)

        X_all  = np.concatenate((X_train, X_test), axis=0)
        y_all  = np.concatenate((Y_train, Y_test), axis=0)
        id_all = np.concatenate((id_train, id_test), axis=0)

        print('\nProcessing domain {0} files...\n'.format(domain_idx))
        target_idx = np.where(id_all == int(domain_idx))

        X = X_all[target_idx]
        y = y_all[target_idx]
        d = np.full(y.shape, int(domain_idx), dtype=int)

        print('\nProcessing domain {0} files | X: {1} y: {2} d:{3} \n'.format(
            domain_idx, X.shape, y.shape, d.shape
        ))

        obj = [(X, y, d)]
        f = open(os.path.join(data_dir, saved_filename), 'wb')
        cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()

    return X, y, d

class data_loader_ucihar(base_loader):
    def __init__(self, samples, labels, domains, t):
        super(data_loader_ucihar, self).__init__(samples, labels, domains)
        self.T = t

    def __getitem__(self, index):
        sample, target, domain = self.samples[index], self.labels[index], self.domains[index]
        sample = self.T(sample)
        return np.squeeze(np.transpose(sample, (1, 0, 2))), target, domain


def prep_domains_ucihar_subject(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['0', '1', '2', '3', '4']
    source_domain_list.remove(args.target_domain)

    # source domain data prep
    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        # n_channel = 6, H:1, W:128
        x = np.transpose(x.reshape((-1, 1, 128, 6)), (0, 2, 1, 3))
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Normalize para 6 canales
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1))
    ])

    data_set = data_loader_ucihar(x_win_all, y_win_all, d_win_all, transform)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    # target domain data prep
    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 128, 6)), (0, 2, 1, 3))
    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_ucihar(x, y, d, transform)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    print('target_loader batch: ', len(target_loader))
    return source_loaders, None, target_loader

def prep_domains_ucihar_subject_large(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                          '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
    source_domain_list.remove(args.target_domain)

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    for source_domain in source_domain_list:
        print('source_domain:', source_domain)
        x, y, d = load_domain_data(source_domain)

        x = np.transpose(x.reshape((-1, 1, 128, 6)), (0, 2, 1, 3))
        print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

        x_win_all = np.concatenate((x_win_all, x), axis=0) if x_win_all.size else x
        y_win_all = np.concatenate((y_win_all, y), axis=0) if y_win_all.size else y
        d_win_all = np.concatenate((d_win_all, d), axis=0) if d_win_all.size else d

    unique_y, counts_y = np.unique(y_win_all, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_all, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1))
    ])

    data_set = data_loader_ucihar(x_win_all, y_win_all, d_win_all, transform)
    source_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)
    print('source_loader batch: ', len(source_loader))
    source_loaders = [source_loader]

    print('target_domain:', args.target_domain)
    x, y, d = load_domain_data(args.target_domain)

    x = np.transpose(x.reshape((-1, 1, 128, 6)), (0, 2, 1, 3))
    print(" ..after sliding window: inputs {0}, targets {1}".format(x.shape, y.shape))

    data_set = data_loader_ucihar(x, y, d, transform)
    target_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)
    print('target_loader batch: ', len(target_loader))

    return source_loaders, None, target_loader

def prep_domains_ucihar_random(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    source_domain_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                          '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']

    x_win_all, y_win_all, d_win_all = np.array([]), np.array([]), np.array([])
    n_train, n_test, split_ratio = [], 0, 0.0

    for source_domain in source_domain_list:
        x_win, y_win, d_win = load_domain_data(source_domain)

        # n_channel = 6, H:1, W:128
        x_win = np.transpose(x_win.reshape((-1, 1, 128, 6)), (0, 2, 1, 3))

        x_win_all = np.concatenate((x_win_all, x_win), axis=0) if x_win_all.size else x_win
        y_win_all = np.concatenate((y_win_all, y_win), axis=0) if y_win_all.size else y_win
        d_win_all = np.concatenate((d_win_all, d_win), axis=0) if d_win_all.size else d_win
        n_train.append(x_win.shape[0])

    x_win_train, x_win_val, x_win_test, \
    y_win_train, y_win_val, y_win_test, \
    d_win_train, d_win_val, d_win_test = train_test_val_split(
        x_win_all, y_win_all, d_win_all, split_ratio=args.split_ratio
    )

    unique_y, counts_y = np.unique(y_win_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_y, counts_y)))
    weights = 100.0 / torch.Tensor(counts_y)
    print('weights of sampler: ', weights)
    weights = weights.double()

    sample_weights = get_sample_weights(y_win_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0, 0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1, 1))
    ])

    train_set_r = data_loader_ucihar(x_win_train, y_win_train, d_win_train, transform)
    train_loader_r = DataLoader(train_set_r, batch_size=args.batch_size, shuffle=False, drop_last=True, sampler=sampler)

    val_set_r = data_loader_ucihar(x_win_val, y_win_val, d_win_val, transform)
    val_loader_r = DataLoader(val_set_r, batch_size=args.batch_size, shuffle=False)

    test_set_r = data_loader_ucihar(x_win_test, y_win_test, d_win_test, transform)
    test_loader_r = DataLoader(test_set_r, batch_size=args.batch_size, shuffle=False)

    return [train_loader_r], val_loader_r, test_loader_r


def prep_ucihar(args, SLIDING_WINDOW_LEN=0, SLIDING_WINDOW_STEP=0):
    if args.cases == 'random':
        return prep_domains_ucihar_random(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject':
        return prep_domains_ucihar_subject(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == 'subject_large':
        return prep_domains_ucihar_subject_large(args, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    elif args.cases == '':
        pass
    else:
        return 'Error! Unknown args.cases!\n'

