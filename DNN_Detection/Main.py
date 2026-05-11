import os
import copy
import numpy as np
from Train import train
from Test import test


class sysconfig(object):
    Pilots = 8        # number of pilots; use 0 for no-pilot, 8/16/64 for Fig. 3.3-style tests
    with_CP_flag = True
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models/'

    # ===== Exercise switch =====
    # 'b_qpsk_8dnns'       : Task (b), QPSK, one of eight small DNNs
    # 'c_64qam_8dnns'      : Task (c), 64-QAM, one of eight small DNNs
    # 'd_qpsk_single_dnn'  : Task (d), QPSK, one large DNN predicts all 128 bits
    experiment = 'b_qpsk_8dnns'

    # Default values are overwritten by apply_experiment() below.
    mu = 2
    pred_range = np.arange(16, 32)
    n_output = len(pred_range)
    n_hidden_1 = 500
    n_hidden_2 = 250 # 1st layer num features
    n_hidden_3 = 120 # 2nd layer num features

    learning_rate = 0.001
    learning_rate_decrease_step = 2000

    # Keep the original full-training default. For Colab debugging, lower this first.
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

    # Run one configuration by default. Set RUN_SWEEP=True to collect BER curves.
    IS_Training = True
    RUN_SWEEP = False
    SNR_LIST = [5, 10, 15, 20, 25]
    PILOT_LIST = [0, 8, 16, 64]


def apply_experiment(config):
    # Added for Exercise 3.1: change only high-level configuration here instead of editing Train/Test repeatedly.
    if config.experiment == 'b_qpsk_8dnns':
        config.mu = 2
        config.pred_range = np.arange(16, 32)  # one of 8 identical DNNs; 128 / 8 = 16 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'c_64qam_8dnns':
        config.mu = 6
        config.pred_range = np.arange(48, 96)  # one of 8 identical DNNs; 384 / 8 = 48 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'd_qpsk_single_dnn':
        config.mu = 2
        config.pred_range = np.arange(0, 128)  # single DNN predicts all QPSK bits
        config.n_output = len(config.pred_range)
        # Added capacity for the larger output layer in Task (d).
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
        # Added for Task (b)/(c): sweep SNR and pilot number to record BER-vs-SNR curves.
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
