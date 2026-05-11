# Exercise 3.1 Results Notes

## Common setup

- Platform: Google Colab
- GPU: T4 GPU
- Reduced training setup:
  - `train_epochs = 1000`
  - `total_batch = 10`
  - `train_batch_symbols = 500`
  - `final_test_number = 10000`
- SNR: 20 dB
- Pilots: 8

Note: These are reduced Colab runs, not the full original 20000-epoch reference setup.

## Task (b): QPSK, 8-DNN smaller-output setting

- Experiment: `b_qpsk_8dnns`
- `mu = 2`
- `pred_range = np.arange(16, 32)`
- `n_output = 16`
- Final BER on test set: approximately `0.013812482`
- Best observed BER on test set: approximately `0.009687483` around epoch 926
- Approximate training time: about 20 minutes

## Task (c): 64-QAM, 8-DNN smaller-output setting

- Experiment: `c_64qam_8dnns`
- `mu = 6`
- `pred_range = np.arange(48, 96)`
- `n_output = 48`
- Final BER on test set: approximately `0.21604168`
- Best observed BER on test set: approximately `0.2132917` around epoch 931
- Approximate training time: about 50 minutes

## Task (d): QPSK, single large FC-DNN setting

- Experiment: `d_qpsk_single_dnn`
- `mu = 2`
- `pred_range = np.arange(0, 128)`
- `n_output = 128`
- Final BER on test set: approximately `0.07970315`
- Best observed BER on test set: approximately `0.07970315` around epoch 996
- Approximate training time: about 30 minutes
