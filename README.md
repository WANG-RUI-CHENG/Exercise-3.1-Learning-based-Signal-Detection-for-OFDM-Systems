# Exercise-3.1-Learning-based-Signal-Detection-for-OFDM-Systems


This repository is based on the original OFDM DNN detection code. The goal is to reproduce and extend the learning-based OFDM signal detection experiments in Exercise 3.1.

The implementation keeps the original project structure as much as possible. Only the files required for the exercise are modified, and comments are added where changes are made. Non-essential files and original comments are left unchanged.

---

## 1. Exercise Summary

Section 3.1 introduces learning-based signal detection for OFDM systems using fully-connected deep neural networks (FC-DNNs).

The required tasks are:

- **Task (a)**: Compute the input and output sizes of each FC-DNN for a 64-subcarrier OFDM system with 64-QAM and 8 parallel DNNs.
- **Task (b)**: Reproduce the QPSK simulation setup in Figure 3.3.
- **Task (c)**: Repeat Task (b) with 64-QAM modulation and compare with QPSK.
- **Task (d)**: Train a single large FC-DNN to predict all transmitted bits and compare it with the 8-DNN smaller-output structure.

---

## 2. Answer to Task (a)

For an OFDM system with 64 subcarriers and 64-QAM modulation:

- Each 64-QAM symbol carries:

\[
\log_2(64) = 6 \text{ bits}
\]

- One OFDM data vector contains:

\[
64 \times 6 = 384 \text{ bits}
\]

In the FC-DNN architecture, the input consists of one pilot OFDM symbol and one data OFDM symbol. Each symbol has 64 complex samples, and the real and imaginary parts are separated:

\[
2 \times 64 \times 2 = 256
\]

Therefore, the input size of each FC-DNN is:

```text
n_input = 256
```

If 8 identical DNNs are used to recover the transmitted bits, each DNN predicts:

\[
384 / 8 = 48 \text{ bits}
\]

Therefore, for 64-QAM:

```text
n_output = 48
```

It is acceptable to simulate one DNN instead of all 8 parallel DNNs for BER estimation if the DNNs are identical and each one handles statistically equivalent bit positions. With enough Monte Carlo samples, the BER from one DNN can approximate the average BER of all parallel DNNs.

---

## 3. Modified and Unmodified Files

Only the necessary files are modified.

### Modified files

| File | Purpose |
|---|---|
| `DNN_Detection/Main.py` | Adds experiment selection for Tasks (b), (c), and (d), reduced Colab training parameters, and separate model paths for different experiments. |
| `DNN_Detection/Train.py` | Uses `config.mu`, `config.n_output`, and `config.pred_range` so QPSK, 64-QAM, and single-DNN settings can be selected from `Main.py`. |
| `DNN_Detection/Test.py` | Uses the same config-based settings as training and evaluates BER for the selected experiment. |
| `DNN_Detection/utils.py` | Adds support for the required modulation settings, especially 64-QAM, while keeping the original OFDM flow. |

### Added folder

| Folder | Purpose |
|---|---|
| `results_exercise_3_1/` | Stores summarized BER results, notes, and optional log files. |

### Unmodified files

The following files are intentionally left unchanged:

- `Example.py`
- `OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot.py`
- `OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_with_different_pilots.py`
- `OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_withoutCP.py`
- Original pilot files such as `Pilot_8`, `Pilot_16`, and `Pilot_64`
- Original dataset files under `H_dataset/`
- Generated cache files such as `.pyc`

---

## 4. Environment

The original code was written for an older TensorFlow 1.x environment. The modified version uses TensorFlow compatibility mode so it can run on current Google Colab.

The experiments in this repository were run with:

```text
Platform: Google Colab
GPU: NVIDIA Tesla T4
Python: Colab default Python 3
TensorFlow: TensorFlow 2.x with tensorflow.compat.v1
```

A GPU runtime is strongly recommended. The tests were run using a **T4 GPU**.

---

## 5. Colab Setup

### 5.1 Enable GPU

In Google Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
```

Then verify the GPU:

```bash
!nvidia-smi
```

---

### 5.2 Clone the source repository

```bash
%cd /content
!rm -rf OFDM_DNN
!git clone https://github.com/haoyye/OFDM_DNN.git
```

---

### 5.3 Extract `H_dataset`

The dataset is stored as split zip files. Extract it as follows:

```bash
%cd /content/OFDM_DNN/H_dataset
!cat H_dataset.zip.001 H_dataset.zip.002 H_dataset.zip.003 H_dataset.zip.004 > H_dataset.zip
!unzip -q H_dataset.zip
!if [ -d H_dataset ]; then mv H_dataset/*.txt .; rmdir H_dataset; fi
!find . -maxdepth 1 -name "*.txt" | wc -l
```

The expected number of channel files is:

```text
400
```

You can check important files with:

```bash
!ls 1.txt 300.txt 301.txt 400.txt
```

---

### 5.4 Run the code

Move to the DNN detection folder:

```bash
%cd /content/OFDM_DNN/DNN_Detection
```

Run:

```bash
!python -u Main.py
```

To save a log file while running:

```bash
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_log.log
```

---

## 6. Experiment Selection

Experiments are selected in `Main.py` by changing:

```python
experiment = 'b_qpsk_8dnns'
```

Available options:

| Task | Experiment name | Description |
|---|---|---|
| (b) | `b_qpsk_8dnns` | QPSK, 8 smaller-output DNNs. |
| (c) | `c_64qam_8dnns` | 64-QAM, 8 smaller-output DNNs. |
| (d) | `d_qpsk_single_dnn` | QPSK, one large DNN predicting all bits. |

---

## 7. Training Settings Used in This Repository

The original reference setup uses a much longer training schedule. Due to Colab runtime limitations, reduced training parameters were used:

```python
train_epochs = 1000
total_batch = 10
train_batch_symbols = 500
small_test_number = 1000
big_test_number = 5000
final_test_number = 10000
model_epoch = 995
```

These settings are sufficient to verify the trend and compare relative BER performance among the three tasks, but they are not the full original 20000-epoch reference setup.

---

## 8. Experiment Results

All experiments below use:

```text
SNR = 20 dB
Pilots = 8
GPU = Colab T4
```

### 8.1 Summary Table

| Task | Modulation | DNN structure | `mu` | `n_output` | Final BER | Best observed BER |
|---|---|---|---:|---:|---:|---:|
| (b) | QPSK | 8 smaller-output DNNs | 2 | 16 | 0.013812482 | 0.009687483 |
| (c) | 64-QAM | 8 smaller-output DNNs | 6 | 48 | 0.21604168 | 0.2132917 |
| (d) | QPSK | Single large DNN | 2 | 128 | 0.07970315 | 0.07970315 |

---

### 8.2 Runtime on Colab T4 GPU

Approximate runtime observed in Colab:

| Task | Approximate runtime |
|---|---:|
| (b) QPSK, 8 smaller-output DNNs | about 20 minutes |
| (c) 64-QAM, 8 smaller-output DNNs | about 50 minutes |
| (d) QPSK, single large DNN | about 30 minutes |

These runtimes are reasonable for the reduced 1000-epoch configuration on a T4 GPU.

---

## 9. Discussion

### Task (b): QPSK baseline

Task (b) uses QPSK modulation and 8 smaller-output DNNs. Each DNN predicts 16 bits:

\[
64 \times 2 / 8 = 16
\]

This setting gives the best BER among the three experiments:

```text
Final BER ≈ 0.0138
Best observed BER ≈ 0.00969
```

This is expected because QPSK has a sparse constellation and each DNN only needs to learn a smaller output vector.

---

### Task (c): 64-QAM

Task (c) changes the modulation from QPSK to 64-QAM. Each subcarrier carries 6 bits, so each DNN predicts:

\[
64 \times 6 / 8 = 48
\]

The BER becomes much higher:

```text
Final BER ≈ 0.2160
Best observed BER ≈ 0.2133
```

This is reasonable because 64-QAM has denser constellation points and is more sensitive to noise. Also, each DNN predicts 48 bits instead of 16 bits, making the learning problem harder.

---

### Task (d): Single large DNN

Task (d) returns to QPSK but uses one large DNN to predict all 128 transmitted bits:

\[
64 \times 2 = 128
\]

The BER is:

```text
Final BER ≈ 0.0797
```

This is worse than Task (b), even though both use QPSK. This suggests that splitting the detection problem into 8 smaller DNNs is easier to train than using one large DNN to predict all bits at once.

---

## 10. How to Reproduce the Results

### Task (b)

In `Main.py`:

```python
experiment = 'b_qpsk_8dnns'
```

Run:

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_b_qpsk.log
```

Expected configuration printed at runtime:

```text
Experiment: b_qpsk_8dnns
mu: 2
n_output: 16
```

---

### Task (c)

In `Main.py`:

```python
experiment = 'c_64qam_8dnns'
```

Run:

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_c_64qam.log
```

Expected configuration printed at runtime:

```text
Experiment: c_64qam_8dnns
mu: 6
n_output: 48
```

---

### Task (d)

In `Main.py`:

```python
experiment = 'd_qpsk_single_dnn'
```

Run:

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_d_single_dnn.log
```

Expected configuration printed at runtime:

```text
Experiment: d_qpsk_single_dnn
mu: 2
n_output: 128
```

---

## 11. Notes for GitHub Submission

Recommended files to include:

```text
DNN_Detection/Main.py
DNN_Detection/Train.py
DNN_Detection/Test.py
DNN_Detection/utils.py
results_exercise_3_1/ber_results_summary.csv
results_exercise_3_1/README_notes.md
README.md
```

Recommended files or folders not to upload if they are too large or generated:

```text
Models/
*.pyc
__pycache__/
H_dataset/H_dataset.zip
```

If `H_dataset` is not included in the repository, users should download or extract it from the original `haoyye/OFDM_DNN` repository before running the experiments.

---

## 12. Conclusion

The reduced Colab experiments show the expected trends:

1. QPSK with 8 smaller-output DNNs gives the lowest BER.
2. 64-QAM gives much higher BER due to denser constellation points and larger DNN output size.
3. A single large QPSK DNN performs worse than 8 smaller DNNs, suggesting that dividing the detection task into smaller subproblems is easier to train.

