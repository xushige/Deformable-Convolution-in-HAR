import torch.utils.data as d
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from model import *
import os

dataset_dict = {'PAMAP2': 2e-4,
                'UCI': 2e-4,
                'UNIMIB': 1e-4,
                'WISDM': 5e-5,
                'USC_HAD': 1e-4,
                'OPPO': 2e-4
                }

ylim_dict = {'PAMAP2': (0, 1),
             'UCI': (0.5, 1),
             'UNIMIB': (0.3, 0.8),
             'WISDM': (0.8, 1),
             'USC_HAD': (0.5, 1),
             'OPPO': (0.5, 0.9)
             }

def adjust_learning_rate(optimizer, epoch, init_LR):
    if epoch in range(50):
        lr = init_LR
    elif epoch in range(50, 90):
        lr = init_LR / 2
    elif epoch in range(90, 120):
        lr = init_LR / 4
    else:
        lr = init_LR / 8
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(filename):
    fig_list = []
    acc_list = []
    loss_list = []
    X_train = torch.from_numpy(np.load('../public_dataset/' + filename + '/x_train.npy')).float()
    X_test = torch.from_numpy(np.load('../public_dataset/' + filename + '/x_test.npy')).float()
    Y_train = torch.from_numpy(np.load('../public_dataset/' + filename + '/y_train.npy')).long()
    Y_test = torch.from_numpy(np.load('../public_dataset/' + filename + '/y_test.npy')).long()

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    print(X_train.shape, X_test.shape)

    train_data = d.TensorDataset(X_train, Y_train)
    test_data = d.TensorDataset(X_test, Y_test)

    category = len(Counter(Y_train.tolist()))
    last_div_num = X_train.size(3) // 16
    sec_last_div_num = X_train.size(2) // 16 + 1
    if X_train.size(2) % 16 == 0:
        sec_last_div_num -= 1
    if last_div_num == 0:
        last_div_num = 1
    elif X_train.size(3) % 16 != 0:
        last_div_num += 1
    sec_last_div_num = sec_last_div_num // 4



    net_dict = {CNN(seclast_div=sec_last_div_num, last_div=last_div_num, classes=category, ifdeform=False): 'cnn_base',
                #SKcnn(sec_div_num=sec_last_div_num, last_div_num=last_div_num, classes=category): 'cnn_SK:',
                CNN(seclast_div=sec_last_div_num, last_div=last_div_num, classes=category, ifdeform=True, df=True): 'cnn_DFC',

                Resnet(seclast_div=sec_last_div_num, last_div=last_div_num, classes=category): 'Res_base',
                #SKNet(sec=sec_last_div_num, last=last_div_num, class_num=category): 'Res_SK:',
                Deform_resnet(seclast_div=sec_last_div_num, last_div=last_div_num, classes=category, df=True): 'Res_DFC'
                }

    EP = 150
    B_S = 128
    loss_fn = nn.CrossEntropyLoss()

    train_loader = d.DataLoader(train_data, batch_size=B_S, shuffle=True)
    test_loader = d.DataLoader(test_data, batch_size=B_S, shuffle=True)

    def start(NET):
        net = NET.cuda()
        net = nn.DataParallel(net).cuda()
        LR = dataset_dict[filename]

        optimizer = torch.optim.AdamW(net.parameters(), lr=LR)

        view = []
        loss_view = []
        for i in range(EP):
            adjust_learning_rate(optimizer, i, LR)
            cor = 0
            for data, label in train_loader:
                data, label = data.cuda(), label.cuda()
                net.train()
                out = net(data)
                loss = loss_fn(out, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for data, label in test_loader:
                data, label = data.cuda(), label.cuda()
                net.eval()
                out = net(data)
                _, pre = torch.max(out, 1)
                cor += (pre == label).sum()
            acc = cor.cpu().numpy() / len(test_data)
            view.append(acc)
            print('epoch:%d, loss:%f, acc:%f' % (i, loss, acc))
        avgacc = str(np.mean(view[-20:]))[:6]
        acc_list.append(avgacc)
        fig_list.append(view)


    for each_net in net_dict.keys():
        start(each_net)
    for i, each in enumerate(net_dict.values()):
        plt.plot(fig_list[i], label=each)
    plt.title('Comparisons On ' + filename + ' Dataset')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(ylim_dict[filename][0], ylim_dict[filename][1])
    plt.legend()
    plt.savefig(filename+'.png')
    plt.show()

    print('=====================================================================\n%s\n=====================================================================' % (acc_list))

for eachdataset in dataset_dict.keys():
    main(eachdataset)



