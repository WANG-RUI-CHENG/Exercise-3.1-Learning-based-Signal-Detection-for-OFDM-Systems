import os
import copy
import numpy as np
from Train import train
from Test import test


class sysconfig(object):
    Pilots = 8        # pilot 數量；P=0 代表不放 pilot，P=8/16/64 用來比較不同 pilot 數對 BER 的影響
    with_CP_flag = True
    SNR = 20
    Clipping = False
    Train_set_path = '../H_dataset/'
    Test_set_path = '../H_dataset/'
    Model_path = '../Models/'

    # 將三種實驗集中在 Main.py 切換，這樣 Train.py 與 Test.py 不需要為每一題反覆手動修改
    # 'b_qpsk_8dnns'       : QPSK 情境下，模擬 8 個相同小 DNN 中的一個，單一 DNN 輸出 16 bits
    # 'c_64qam_8dnns'      : 64-QAM 情境下，模擬 8 個相同小 DNN 中的一個，單一 DNN 輸出 48 bits
    # 'd_qpsk_single_dnn'  : QPSK 情境下，改用單一大型 DNN 一次輸出全部 128 bits
    experiment = 'b_qpsk_8dnns'

    # 下列是預設值；真正執行前會由 apply_experiment() 依 experiment 自動覆寫
    mu = 2
    pred_range = np.arange(16, 32)
    n_output = len(pred_range)
    n_hidden_1 = 500
    n_hidden_2 = 250 # 1st layer num features
    n_hidden_3 = 120 # 2nd layer num features

    learning_rate = 0.001
    learning_rate_decrease_step = 2000

    # 原始 reference code 的訓練量較大；這裡預設採用 Colab T4 GPU 可完成的 reduced setting。
    # 這組參數和 README 的 B/C/D 結果一致，方便直接重現表格中的 BER 趨勢。
    train_epochs = 1000
    total_batch = 10
    train_batch_symbols = 500
    display_step = 5
    model_saving_step = 5
    test_step = 1000
    small_test_number = 1000
    big_test_number = 5000
    final_test_number = 10000
    model_epoch = 995

    # 預設只跑單一設定，避免一次掃描所有 SNR/pilot 花太久；需要畫 BER 曲線時再改成 True
    IS_Training = True
    RUN_SWEEP = False
    SNR_LIST = [5, 10, 15, 20, 25]
    PILOT_LIST = [0, 8, 16, 64]


def apply_experiment(config):
    # 依照目前選擇的 experiment 設定 modulation、輸出 bit 範圍與網路大小，避免在多個檔案中分散修改造成錯誤
    if config.experiment == 'b_qpsk_8dnns':
        config.mu = 2
        config.pred_range = np.arange(16, 32)  # QPSK 總共 64×2=128 bits；8 個 DNN 平分後，每個 DNN 預測 16 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'c_64qam_8dnns':
        config.mu = 6
        config.pred_range = np.arange(48, 96)  # 64-QAM 總共 64×6=384 bits；8 個 DNN 平分後，每個 DNN 預測 48 bits
        config.n_output = len(config.pred_range)
        config.n_hidden_1 = 500
        config.n_hidden_2 = 250
        config.n_hidden_3 = 120
    elif config.experiment == 'd_qpsk_single_dnn':
        config.mu = 2
        config.pred_range = np.arange(0, 128)  # 單一大型 DNN 不再分段，直接預測 QPSK 的全部 128 bits
        config.n_output = len(config.pred_range)
        # 輸出層從 16 bits 增加到 128 bits，學習目標變複雜，因此提高 hidden layer 容量以降低欠擬合風險
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
        # 需要重現 BER-vs-SNR 曲線時，才逐一掃描 SNR 與 pilot 數；平常保持 False 可節省訓練時間
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
