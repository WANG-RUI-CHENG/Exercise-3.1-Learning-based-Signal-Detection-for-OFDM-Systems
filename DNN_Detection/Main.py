import os
import copy
import numpy as np
from Train import train
from Test import test


class sysconfig(object):
    Pilots = 8        # number of pilots；0 表示 no-pilot，8/16/64 可用於 BER 曲線比較
    with_CP_flag = True
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models/'

    # ===== 題目模式切換 =====
    # 'b_qpsk_8dnns'       : 題目 (b)，QPSK，模擬 8 個小型 DNN 中的一個
    # 'c_64qam_8dnns'      : 題目 (c)，64-QAM，模擬 8 個小型 DNN 中的一個
    # 'd_qpsk_single_dnn'  : 題目 (d)，QPSK，單一大型 DNN 預測全部 128 bits
    experiment = 'b_qpsk_8dnns'

    # 下列預設值會由 apply_experiment() 依照題目模式覆蓋
    mu = 2
    pred_range = np.arange(16, 32)
    n_output = len(pred_range)
    n_hidden_1 = 500
    n_hidden_2 = 250 # 1st layer num features
    n_hidden_3 = 120 # 2nd layer num features

    learning_rate = 0.001
    learning_rate_decrease_step = 2000

    # 保留完整訓練的預設值；Colab 測試時可先調小
    train_epochs = 20000
    total_batch = 50
    train_batch_symbols = 1000
    display_step = 5
    model_saving_step = 5
    test_step = 1000
    small_test_number = 1000
    big_test_number = 10000
    final_test_number = 100000
    model_epoch = 19995

    # 預設只跑一組設定；若要蒐集 BER 曲線可將 RUN_SWEEP 設為 True
    IS_Training = True
    RUN_SWEEP = False
    SNR_LIST = [5, 10, 15, 20, 25]
    PILOT_LIST = [0, 8, 16, 64]


def apply_experiment(config):
    # 集中切換高階設定，避免反覆修改 Train.py 與 Test.py
    if config.experiment == 'b_qpsk_8dnns':
        config.mu = 2
        config.pred_range = np.arange(16, 32)  # 8 個相同 DNN 中的一個；128 / 8 = 16 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'c_64qam_8dnns':
        config.mu = 6
        config.pred_range = np.arange(48, 96)  # 8 個相同 DNN 中的一個；384 / 8 = 48 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'd_qpsk_single_dnn':
        config.mu = 2
        config.pred_range = np.arange(0, 128)  # 單一 DNN 預測全部 QPSK bits
        config.n_output = len(config.pred_range)
        # 輸出層較大，因此提高 hidden layer 容量
        config.n_hidden_1 = 1000
        config.n_hidden_2 = 500
        config.n_hidden_3 = 250
    else:
        raise ValueError('Unknown experiment: {}'.format(config.experiment))
    return config


def run_once(config):
    apply_experiment(config)
    os.makedirs(os.path.join(config.Model_path, config.experiment, 'SNR_' + str(config.SNR)), exist_ok=True)
    print('Experiment:', config.experiment, 'mu:', config.mu, 'SNR:', config.SNR,
          'Pilots:', config.Pilots, 'pred_range:', config.pred_range,
          'n_output:', config.n_output)
    if config.IS_Training:
        train(config)
    else:
        test(config)


def main():
    config = sysconfig()
    if config.RUN_SWEEP:
        # 掃描 SNR 與 pilot 數量，用於整理 BER-vs-SNR 曲線
        for snr in config.SNR_LIST:
            for pilots in config.PILOT_LIST:
                cfg = copy.deepcopy(config)
                cfg.SNR = snr
                cfg.Pilots = pilots
                run_once(cfg)
    else:
        run_once(config)


if __name__ == '__main__':
    main()
