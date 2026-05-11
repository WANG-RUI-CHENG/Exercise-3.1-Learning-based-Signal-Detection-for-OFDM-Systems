# Exercise 3.1：OFDM 系統的 Learning-based Signal Detection

本 repo 是以原始 `haoyye/OFDM_DNN` 專案為基礎，完成 Exercise 3.1 的 FC-DNN based OFDM signal detection 實驗。

我保留原始專案檔案與目錄結構，只針對完成題目所需的少數檔案進行修改，並新增 `results_exercise_3_1/` 放置本次 Colab 實驗結果。

---

## 修改摘要

### 有修改的檔案

| 檔案 | 修改內容 |
|---|---|
| `DNN_Detection/Main.py` | 新增 `experiment` 設定，用來切換 task (b)、(c)、(d)；新增 reduced Colab training setting；依照實驗自動設定 `mu`、`pred_range`、`n_output` 與 hidden layer size |
| `DNN_Detection/Train.py` | 將原本固定的 `mu = 2`、`n_output = 16` 改成由 `config` 控制；保留原始訓練流程並新增必要註解 |
| `DNN_Detection/Test.py` | 補齊測試流程，使測試時能依照目前 experiment 載入對應模型並計算 BER |
| `DNN_Detection/utils.py` | 補充 64-QAM modulation 支援，並保留 QPSK 相容性 |
| `README.md` | 新增本說明文件 |
| `.gitignore` | 新增 GitHub 建議忽略規則 |

### 新增的資料夾

| 資料夾 | 說明 |
|---|---|
| `results_exercise_3_1/` | 放置本次 B/C/D 實驗結果摘要 |


### 本 zip 未包含的檔案

本次提供的 GitHub zip **沒有包含 `H_dataset/`**，因為檔案較大，且該資料集是原始 `haoyye/OFDM_DNN` 的資料，沒有被本作業修改。

執行前請自行加入下列檔案：

```text
H_dataset/H_dataset.zip.001
H_dataset/H_dataset.zip.002
H_dataset/H_dataset.zip.003
H_dataset/H_dataset.zip.004
```

或依照後面的 Colab 指令，從原始 repo 下載後解壓。

此外，`*.pyc`、`Models/`、`__pycache__/` 都不需要提交到 GitHub。

### 原始保留、未修改的檔案

以下檔案保留原樣，未修改：

| 檔案或資料夾 | 說明 |
|---|---|
| `DNN_Detection/Example.py` | 原始範例程式 |
| `DNN_Detection/OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot.py` | 原始 OFDM channel estimation script |
| `DNN_Detection/OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_with_different_pilots.py` | 原始不同 pilot 數量的 script |
| `DNN_Detection/OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_withoutCP.py` | 原始 without CP script |
| `DNN_Detection/Pilot_8` | 原始 pilot 檔 |
| `DNN_Detection/Pilot_16` | 原始 pilot 檔 |
| `DNN_Detection/Pilot_64` | 原始 pilot 檔 |
| `DNN_Detection/ReadMe.txt` | 原始 DNN_Detection 說明 |
| `ReadMe.rst` | 原始 repo 說明 |

---

## 題目 (a)：Input / Output layer size

OFDM 系統有 64 個 subcarriers。

64-QAM 每個 subcarrier 攜帶：

```text
log2(64) = 6 bits
```

因此一個 OFDM data symbol vector 共有：

```text
64 × 6 = 384 bits
```

Figure 3.2 的 FC-DNN input 由 pilot OFDM symbol 與 data OFDM symbol 的接收訊號組成。兩個 64-point complex vectors 拆成 real part 與 imaginary part：

```text
2 × 64 × 2 = 256
```

所以每個 FC-DNN 的 input layer size 為：

```text
n_input = 256
```

若使用 8 個相同 FC-DNN 平行預測全部 bits，則每個 DNN 的 output size 為：

```text
384 / 8 = 48
```

因此 64-QAM 情況下：

```text
Input layer size  = 256
Output layer size = 48
```

只模擬其中一個 DNN 通常是可接受的，因為 8 個 DNN 架構相同，只是分別預測不同 bit 區段。若測試資料量足夠大，單一 DNN 的 BER 可作為整體 BER 的估計。不過更完整的做法仍可以訓練 8 個 DNN 後平均 BER。

---

## Colab 環境

本次實驗使用：

```text
Google Colab
GPU: T4 GPU
Python: Colab default Python 3
TensorFlow: TensorFlow 2.x with tensorflow.compat.v1
```

原始程式是 TensorFlow 1.x 風格，因此修改後使用 `tensorflow.compat.v1` 維持相容。

在 Colab 開啟 GPU：

```text
執行階段 → 變更執行階段類型 → 硬體加速器選 GPU / T4 GPU
```

確認 GPU：

```bash
!nvidia-smi
```

---

## 資料集準備

若從 GitHub clone 原始專案：

```bash
%cd /content
!rm -rf OFDM_DNN
!git clone https://github.com/haoyye/OFDM_DNN.git
```

解壓 `H_dataset`：

```bash
%cd /content/OFDM_DNN/H_dataset
!cat H_dataset.zip.001 H_dataset.zip.002 H_dataset.zip.003 H_dataset.zip.004 > H_dataset.zip
!unzip -q H_dataset.zip
!if [ -d H_dataset ]; then mv H_dataset/*.txt .; rmdir H_dataset; fi
!find . -maxdepth 1 -name "*.txt" | wc -l
```

正常應得到：

```text
400
```

也可以檢查：

```bash
!ls 1.txt 300.txt 301.txt 400.txt
```

---

## 本次 reduced training setting

原始 reference 設定訓練時間較長，Colab 免費環境不容易完整跑完。因此本次使用 reduced setting：

```python
train_epochs = 1000
total_batch = 10
train_batch_symbols = 500
small_test_number = 1000
big_test_number = 5000
final_test_number = 10000
model_epoch = 995
```

共通條件：

```text
SNR = 20 dB
Pilots = 8
GPU = Colab T4
```

---

## Task (b)：QPSK，8 個小 DNN

在 `DNN_Detection/Main.py` 設定：

```python
experiment = 'b_qpsk_8dnns'
```

此設定會使用：

```text
mu = 2
pred_range = np.arange(16, 32)
n_output = 16
```

執行：

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_b_qpsk.log
```

結果：

| 項目 | 數值 |
|---|---:|
| Modulation | QPSK |
| `mu` | 2 |
| `n_output` | 16 |
| Final BER | 0.013812482 |
| Best observed BER | 0.009687483 |
| Best epoch | 926 |
| 約略訓練時間 | 約 20 分鐘 |

---

## Task (c)：64-QAM，8 個小 DNN

在 `DNN_Detection/Main.py` 設定：

```python
experiment = 'c_64qam_8dnns'
```

此設定會使用：

```text
mu = 6
pred_range = np.arange(48, 96)
n_output = 48
```

執行：

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_c_64qam_real.log
```

結果：

| 項目 | 數值 |
|---|---:|
| Modulation | 64-QAM |
| `mu` | 6 |
| `n_output` | 48 |
| Final BER | 0.21604168 |
| Best observed BER | 0.2132917 |
| Best epoch | 931 |
| 約略訓練時間 | 約 50 分鐘 |

Task (c) 的 BER 明顯高於 task (b)，原因是 64-QAM constellation points 較密，對雜訊較敏感；此外每個 DNN 的 output size 從 16 bits 增加到 48 bits，學習難度也提高。

---

## Task (d)：QPSK，單一大型 DNN

在 `DNN_Detection/Main.py` 設定：

```python
experiment = 'd_qpsk_single_dnn'
```

此設定會使用：

```text
mu = 2
pred_range = np.arange(0, 128)
n_output = 128
```

執行：

```bash
%cd /content/OFDM_DNN/DNN_Detection
!python -u Main.py 2>&1 | tee /content/OFDM_DNN/results_exercise_3_1/task_d_single_dnn.log
```

結果：

| 項目 | 數值 |
|---|---:|
| Modulation | QPSK |
| `mu` | 2 |
| `n_output` | 128 |
| Final BER | 0.07970315 |
| Best observed BER | 0.07970315 |
| Best epoch | 996 |
| 約略訓練時間 | 約 30 分鐘 |

Task (d) 的 BER 高於 task (b)，表示使用單一大型 DNN 一次預測全部 128 bits，比使用 8 個小 DNN 分段預測更難訓練。

---

## B / C / D 結果比較

| Task | Modulation | DNN 設定 | `n_output` | Final BER | Best BER |
|---|---|---|---:|---:|---:|
| (b) | QPSK | 8 個小 DNN | 16 | 0.013812482 | 0.009687483 |
| (c) | 64-QAM | 8 個小 DNN | 48 | 0.21604168 | 0.2132917 |
| (d) | QPSK | 單一大型 DNN | 128 | 0.07970315 | 0.07970315 |

觀察：

- Task (b) BER 最低，表示 QPSK 且分段預測較容易學習。
- Task (c) BER 最高，主要因 64-QAM constellation 較密，且 output bits 較多。
- Task (d) 比 task (b) 差，表示單一大型 DNN 預測全部 bits 的學習難度較高。

---

## 如何重現

1. 開啟 Colab 並選擇 T4 GPU。
2. Clone 原始 `OFDM_DNN` repo，或直接使用本 repo。
3. 解壓 `H_dataset`。
4. 進入 `DNN_Detection/`。
5. 在 `Main.py` 切換 `experiment`：
   - `b_qpsk_8dnns`
   - `c_64qam_8dnns`
   - `d_qpsk_single_dnn`
6. 執行 `python -u Main.py`。
7. 將 console 中的 BER 記錄到 `results_exercise_3_1/ber_results_summary.csv`。

---

## 注意事項

- 本結果為 reduced Colab run，不是原始完整 20000 epochs。
- BER 數值會因 random initialization、training samples 與 Colab GPU 狀態略有差異。
- 若要完整重現 reference 結果，應提高 `train_epochs`、`total_batch` 與測試樣本數。
- `Models/` 是訓練產生的模型資料夾，不建議提交到 GitHub。
