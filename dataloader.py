from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class NGs(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NGs.mat')['Y'].astype(np.int32).reshape(500, )
        x = scipy.io.loadmat(path + 'NGs.mat')['X']
        self.V1 = x[0][0].astype(np.float32)
        self.V2 = x[1][0].astype(np.float32)
        self.V3 = x[2][0].astype(np.float32)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(2000)
        x2 = self.V2[idx].reshape(2000)
        x3 = self.V3[idx].reshape(2000)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class synthetic3d(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'synthetic3d.mat')['Y'].astype(np.int32).reshape(600, )
        x = scipy.io.loadmat(path + 'synthetic3d.mat')['X']
        self.V1 = x[0][0].astype(np.float32)
        self.V2 = x[1][0].astype(np.float32)
        self.V3 = x[2][0].astype(np.float32)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(3)
        x2 = self.V2[idx].reshape(3)
        x3 = self.V3[idx].reshape(3)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class NoisyMNIST(Dataset):
    def __init__(self, path):
        # 读取MATLAB文件
        mat_data = scipy.io.loadmat(path + 'NoisyMNIST.mat')
        # 输出MATLAB文件中的键（变量名）
        print(mat_data.keys())
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['trainLabel'].astype(np.int32).reshape(50000, )
        self.V1 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        data_size = 50000
        class_num = 10
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "NGs":
        dataset = NGs('./data/')
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == "synthetic3d":
        dataset = synthetic3d('./data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
