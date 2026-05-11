from __future__ import division
import numpy as np
# 原始程式使用 TensorFlow 1.x 的 placeholder、Session 與 Saver；
# Colab 目前多為 TensorFlow 2.x，因此使用 compat.v1 讓原始訓練流程可以維持不變。
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
import os
from utils import *


def _build_pilots(P, K, mu, pilotCarriers):
    # 測試時必須產生與訓練完全相同的 pilot 設定；
    # 因此 pilot bit 長度與檔名規則要和 Train.py 保持一致，避免載入模型後輸入分佈不同。
    pilot_bit_len = len(pilotCarriers) * mu
    if pilot_bit_len == 0:
        return np.array([], dtype=complex)

    Pilot_file_name = 'Pilot_' + str(P) + '_mu' + str(mu) + '.txt'
    if os.path.isfile(Pilot_file_name):
        print ('Load Training Pilots txt')
        bits = np.loadtxt(Pilot_file_name, delimiter=',')
        bits = np.asarray(bits).reshape((-1,))
        if len(bits) != pilot_bit_len:
            raise ValueError('Pilot file length mismatch: {}'.format(Pilot_file_name))
    else:
        bits = np.random.binomial(n=1, p=0.5, size=(pilot_bit_len, ))
        np.savetxt(Pilot_file_name, bits, delimiter=',')
    return Modulation(bits,mu)


def test(config):
        tf.reset_default_graph()
        K = 64
        CP = K//4
        P = config.Pilots # number of pilot carriers per OFDM block
        allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])
        mu = config.mu
        CP_flag = config.with_CP_flag
        if P == 0:
            # P=0 表示 no-pilot 測試；pilotCarriers 設為空陣列，所有 subcarriers 都作為 data carriers。
            # 這樣可避免 K//P 的除以零錯誤，也能和訓練端的 no-pilot 設定一致。
            pilotCarriers = np.asarray([], dtype=int)
            dataCarriers = allCarriers
        elif P<K:
            pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.
            dataCarriers = np.delete(allCarriers, pilotCarriers)
        else:   # K = P
            pilotCarriers = allCarriers
            dataCarriers = []

        payloadBits_per_OFDM = K*mu
        SNRdb = config.SNR  # signal to noise-ratio in dB at the receiver
        Clipping_Flag = config.Clipping
        # 測試端的 pilotValue 必須與訓練端一致，否則接收訊號特徵分佈不同，BER 比較會失去意義。
        pilotValue = _build_pilots(P, K, mu, pilotCarriers)

        training_epochs = 20
        batch_size = 256
        display_step = 5
        model_saving_step = 5
        test_step = 1000
        examples_to_show = 10

        # Network Parameters
        # 測試時必須使用與訓練時完全相同的 hidden layer 與 output size，否則 Saver 無法正確 restore checkpoint。
        n_hidden_1 = config.n_hidden_1
        n_hidden_2 = config.n_hidden_2 # 1st layer num features
        n_hidden_3 = config.n_hidden_3 # 2nd layer num features
        n_input = 256 # MNIST data input (img shape: 28*28)
        # n_output 由 experiment 決定，必須和 Train.py 儲存的模型輸出層維度一致。
        n_output = config.n_output #4
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, n_input])
        #Y = tf.placeholder("float", [None, K*mu])
        Y = tf.placeholder("float", [None, n_output])
        def encoder(x):
            weights = {
                'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output],stddev=0.1)),
            }
            biases = {
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3],stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output],stddev=0.1)),

            }

            # Encoder Hidden layer with sigmoid activation #1
            #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
            return layer_4
        # Building the decoder

        #encoder_op = encoder(X)

        #for network_idx in range(0, int(K*mu/n_output)):
        #    y_pred_cur = encoder(X)
        #    if network_idx == 0:
        #        y_pred = y_pred_cur
        #    else:
        #        y_pred = tf.concat((y_pred, y_pred_cur), axis=1)
        # Prediction
        y_pred = encoder(X)
        # Targets (Labels) are the input data.
        y_true = Y

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 1))
        #cost = tf.reduce_mean(tf.abs(y_true-y_pred))

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Generating Detection
        #code = BinaryLinearBlockCode(parityCheckMatrix='./test/data/BCH_63_36_5_strip.alist')
        #code = PolarCode(6, SNR=4, mu = 16, rate = 0.5)
        #decoders = [IterativeDecoder(code, minSum=True, iterations=50, reencodeOrder=-1, reencodeRange=0.1)]

        # Start Training
        config_GPU = tf.ConfigProto()
        config_GPU.gpu_options.allow_growth = True
        # The H information set
        test_idx_low = 301
        test_idx_high = 401
        channel_response_set_test = []
        H_folder = config.Test_set_path
        for test_idx in range(test_idx_low,test_idx_high):
            H_file = H_folder + str(test_idx) + '.txt'
            with open(H_file) as f:
                for line in f:
                    numbers_str = line.split()
                    numbers_float = [float(x) for x in numbers_str]
                    h_response = np.asarray(numbers_float[0:int(len(numbers_float)/2)])+1j*np.asarray(numbers_float[int(len(numbers_float)/2):len(numbers_float)])
                    channel_response_set_test.append(h_response)




        print ('length of testing channel response', len(channel_response_set_test))



        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        config_GPU = tf.ConfigProto()
        config_GPU.gpu_options.allow_growth = True

        with tf.Session(config=config_GPU) as sess:
            sess.run(init)
            saving_name = getattr(config, 'model_name', None)
            if saving_name is None:
                # checkpoint 路徑包含 experiment 與 SNR，可避免 QPSK、64-QAM、single-DNN 的模型互相混用。
                saving_name = os.path.join(config.Model_path, config.experiment, 'SNR_' + str(SNRdb), 'DetectionModel_SNR_' + str(SNRdb) + '_Pilot_' + str(P) + '_epoch_' + str(config.model_epoch))
            saver.restore(sess, saving_name)
            input_samples_test = []
            input_labels_test = []
            test_number = config.final_test_number
            for i in range(0, test_number):
                bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
                #signal_train, signal_output, para = ofdm_simulate(bits)
                #codeword = code.encode(bits)
                #signal_train, signal_output, para = ofdm_simulate(codeword)
                channel_response= channel_response_set_test[np.random.randint(0,len(channel_response_set_test))]
                #signal_output, para = ofdm_simulate_single(bits,channel_response)
                signal_output, para = ofdm_simulate(bits,channel_response,SNRdb,mu, CP_flag, K, P, CP, pilotValue,pilotCarriers, dataCarriers,Clipping_Flag)
                #input_labels_test.append(codeword)
                # 只比較目前 DNN 應該輸出的 bit 範圍；
                # 若是 single-DNN，pred_range 會是 0:128，因此會計算完整 QPSK data vector 的 BER。
                input_labels_test.append(bits[config.pred_range])
                #input_samples_test.append(np.concatenate((signal_train,signal_output)))
                input_samples_test.append(signal_output)

            batch_x = np.asarray(input_samples_test)
            batch_y = np.asarray(input_labels_test)
            encode_decode = sess.run(y_pred, feed_dict = {X:batch_x})
            mean_error = tf.reduce_mean(abs(y_pred - batch_y))
            BER = 1-tf.reduce_mean(tf.reduce_mean(tf.to_float(tf.equal(tf.sign(y_pred-0.5), tf.cast(tf.sign(batch_y-0.5),tf.float32))),1))

            print("OFDM Detection QAM output number is", n_output, "SNR = ", SNRdb, "Num Pilot", P,"prediction and the mean error on test set are:", mean_error.eval({X:batch_x}), BER.eval({X:batch_x}))
