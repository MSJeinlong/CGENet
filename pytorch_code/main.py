#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse
import pickle
import time

from model import *
from utils import Data, split_validation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/Tmall')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--device', default='cuda', help='device: cpu or cuda')
parser.add_argument('--k', type=int, default=10, help='The numbers of Neighbor for session node')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gmlp_layers', type=int, default=1)
parser.add_argument('--gnn_layers', type=int, default=3)
parser.add_argument('--max_len', type=int, default=70)
parser.add_argument('--layer_norm_eps', type=float, default=1e-8)
# parser.add_argument('--topK', type=int, default=20, help='The number of candidate items for recommendation')
parser.add_argument('--no_hn', action='store_true', help='without highway network')
parser.add_argument('--no_gmlp', action='store_true', help='without gating multilayer perceptron')
parser.add_argument('--no_sca', action='store_true', help='without session context aware GAT')
parser.add_argument('--use_san', action='store_true', help='use SAN instead of gMLP')
parser.add_argument('--aggregation', default='sum',
                    help='The aggregation operations of session representation'
                         ' and session context information')
opt = parser.parse_args()


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    init_seed(2023)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
        opt.max_len = 70
        data_idx = 0
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
        opt.max_len = 70
        data_idx = 0
    elif opt.dataset == 'retailrocket':
        n_node = 48990
        # n_node = 36969
        opt.max_len = 50
        data_idx = 1
    elif opt.dataset == "Tmall":
        n_node = 40728
        opt.max_len = 30
        data_idx = 0
    elif opt.dataset == 'Nowplaying':
        # opt.max_len = 25
        n_node = 60417
        data_idx = 0
    else:
        n_node = 310
        data_idx = 0

    if opt.device == 'cuda':
        model = trans_to_cuda(SessionGraph(opt, n_node))
    else:
        model = trans_to_cpu(SessionGraph(opt, n_node))
    print(opt)
    print(model)

    # 打印和统计整体模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    _dict = {}
    for _, param in enumerate(model.named_parameters()):
        # print(param[0])
        # print(param[1])
        total_params = param[1].numel()
        # print(f'{total_params:,} total parameters.')
        k = param[0].split('.')[0]
        if k in _dict.keys():
            _dict[k] += total_params
        else:
            _dict[k] = 0
            _dict[k] += total_params
        # print('----------------')
    for k, v in _dict.items():
        print(k)
        print(v)
        print("%3.3fM parameters" % (v / (1024 * 1024)))
        print('--------')

    # 加载训练数据
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    # 根据opt.validation决定是否划分验证集
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion, opt.dataset)
        test_data = valid_data
        train_data = Data(train_data, train_len=opt.max_len)
        test_data = Data(test_data, train_len=opt.max_len)
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
        train_data = Data(train_data, train_len=opt.max_len, idx=data_idx)
        test_data = Data(test_data, train_len=opt.max_len, idx=data_idx)
    # train_data = Data(train_data, shuffle=True, seq_len=opt.max_len)
    # test_data = Data(test_data, shuffle=False, seq_len=opt.max_len)
    # print("train_labels = " + str(train_data[data_idx + 1]))
    # print("test_labels = " + str(test_data[data_idx + 1]))
    # print("len_data = " + str(train_data.len_data))

    # 生成train数据会话长度图片
    # train_x = list(set(train_data.len_data))
    # train_y = []
    # for i in train_x:
    #     train_y.append(train_data.len_data.count(i))
    # plt.bar(train_x, train_y, align='center')
    # plt.title('Figure')
    # plt.ylabel('Count')
    # plt.xlabel('Length')
    # plt.savefig("./" + opt.dataset + "-train-len3.png")
    #
    # # 生成test数据会话长度图片
    # test_x = list(set(test_data.len_data))
    # test_y = []
    # for i in test_x:
    #     test_y.append(test_data.len_data.count(i))
    # plt.bar(test_x, test_y, align='center')
    # plt.title('Figure')
    # plt.ylabel('Count')
    # plt.xlabel('Length')
    # plt.savefig("./" + opt.dataset + "-test-len3.png")

    start = time.time()
    best_result = [0, 0]
    best_result_10 = [0, 0]
    best_epoch = [0, 0]
    best_epoch_10 = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        # 在会话推荐中，hit = Recall = Precision, MRR与NDCG相似
        hit, mrr, hit_10, mrr_10 = train_test(model, train_data, test_data)
        flag = 0
        # 记录最好的结果
        if hit > best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            if not (opt.no_gmlp and opt.no_sca and opt.no_hn) and opt.aggregation == "sum":
                torch.save(model.state_dict(), "ckpt/" + opt.dataset + "-P@10=" + str(round(hit_10, 2))
                           + "-MRR@10=" + str(round(mrr_10, 2)) + "-P@20=" + str(round(hit, 2))
                           + "-MRR@20=" + str(round(mrr, 2)) + ".pkl")
        if mrr > best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            if not (opt.no_gmlp and opt.no_sca and opt.no_hn) and opt.aggregation == "sum":
                torch.save(model.state_dict(), "ckpt/" + opt.dataset + "-P@10=" + str(round(hit_10, 2))
                           + "-MRR@10=" + str(round(mrr_10, 2)) + "-P@20=" + str(round(hit, 2))
                           + "-MRR@20=" + str(round(mrr, 2)) + ".pkl")
        if hit_10 > best_result_10[0]:
            best_result_10[0] = hit_10
            best_epoch_10[0] = epoch
            flag = 1
            if not (opt.no_gmlp and opt.no_sca and opt.no_hn) and opt.aggregation == "sum":
                torch.save(model.state_dict(), "ckpt/" + opt.dataset + "-P@10=" + str(round(hit_10, 2))
                           + "-MRR@10=" + str(round(mrr_10, 2)) + "-P@20=" + str(round(hit, 2))
                           + "-MRR@20=" + str(round(mrr, 2)) + ".pkl")
        if mrr_10 > best_result_10[1]:
            best_result_10[1] = mrr_10
            best_epoch_10[1] = epoch
            flag = 1
            if not (opt.no_gmlp and opt.no_sca and opt.no_hn) and opt.aggregation == "sum":
                torch.save(model.state_dict(), "ckpt/" + opt.dataset + "-P@10=" + str(round(hit_10, 2))
                           + "-MRR@10=" + str(round(mrr_10, 2)) + "-P@20=" + str(round(hit, 2))
                           + "-MRR@20=" + str(round(mrr, 2)) + ".pkl")
        print('Current Result:')
        print('\tHR@20:\t%.2f\tMMR@20:\t%.2f' % (hit, mrr))
        print('Current Result:')
        print('\tHR@10:\t%.2f\tMMR@10:\t%.2f' % (hit_10, mrr_10))
        print('Best Result:')
        print('\tHR@20:\t%.2f\tMMR@20:\t%.2f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('Best Result:')
        print('\tHR@10:\t%.2f\tMMR@10:\t%.2f\tEpoch:\t%d,\t%d' % (
            best_result_10[0], best_result_10[1], best_epoch_10[0], best_epoch_10[1]))
        curr_time = time.time()
        print("current epoch runtime: %f min" % ((curr_time - start) / 60))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f min" % ((end - start) / 60))


if __name__ == '__main__':
    main()
