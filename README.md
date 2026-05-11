# Exercise 3.1：OFDM Learning-based Signal Detection

本專案完成 Exercise 3.1 中以 FC-DNN 進行 OFDM signal detection 的實驗。程式以原始 `OFDM_DNN` 專案為基礎，只修改必要檔案，保留原本專案結構與大部分註解，並新增少量註解與設定來完成 task (b)、(c)、(d)。

---

## 1. 題目 (a) 計算

OFDM 系統有 64 個 subcarriers。64-QAM 每個 subcarrier 攜帶 `log2(64)=6` bits，因此一個 OFDM data symbol vector 有：

```text
64 × 6 = 384 bits
```

Figure 3.2 的 FC-DNN input 由 pilot OFDM symbol 與 data OFDM symbol 的接收訊號組成。兩個 64-point complex vectors 拆成 real/imaginary parts：

```text
2 × 64 × 2 = 256
```

所以每個 FC-DNN 的 input layer size 為：

```text
n_input = 256
```

若使用 8 個相同 FC-DNN 平行預測全部 bits，則每個 DNN 的 output layer size 為：

```text
384 / 8 = 48
```

因此 64-QAM 情況下：

```text
Input layer size  = 256
Output layer size = 48
```

只模擬其中一個 DNN 通常是可接受的，因為 8 個 DNN 架構相同，只是分別預測不同 bit 區段。若測試資料量足夠大，單一 DNN 的 BER 可作為整體 BER 的估計。不過完整評估也可以將 8 個 DNN 的 BER 取平均。

---

## 2. 修改原則

本專案刻意維持原始程式的形式，像是從原始 `OFDM_DNN` 複製後，只針對 Exercise 3.1 需求修改少數檔案。

- 不重構整個專案。
- 不刪除原有註解，除非會造成錯誤或衝突。
- 優先新增註解說明修改目的。
- 只修改完成 task (b)、(c)、(d) 所需的程式。
- 額外結果放在 `results_exercise_3_1/`，不混入原始程式資料夾。

---

## 3. 修改與未修改的檔案

### 有修改的檔案

| 檔案 | 修改目的 |
|---|---|
| `DNN_Detection/Main.py` | 新增 `experiment` 設定，用來切換 task (b)、(c)、(d)，並統一設定 SNR、pilot 數、訓練參數與模型儲存路徑 |
| `DNN_Detection/Train.py` | 讓 `mu`、`n_output`、hidden layer size、training epochs 等由 `config` 控制 |
| `DNN_Detection/Test.py` | 補齊測試流程與 BER 計算，使其可依照目前 experiment 載入對應模型 |
| `DNN_Detection/utils.py` | 補充 64-QAM modulation 支援，並保留 QPSK 相容性 |

### 未修改或不需要修改的檔案

| 檔案或資料夾 | 說明 |
|---|---|
| `DNN_Detection/Example.py` | 原始範例程式，不參與本次主要流程 |
| `DNN_Detection/OFDM_ChannelEstimation_*.py` | 原始 paper simulation scripts，不需修改 |
| `DNN_Detection/Pilot_8`, `Pilot_16`, `Pilot_64` | 原始 pilot 檔，不需手動修改 |
| `DNN_Detection/*.pyc` | Python cache，不應提交 |
| `Models/` | 訓練產生的模型檔案，不建議提交到 GitHub |
| `H_dataset/*.txt` | 解壓後的 channel response 很大，不建議提交到 GitHub |

---

## 4. 建議 GitHub 檔案結構

```text
OFDM_DNN/
├── DNN_Detection/
│   ├── Main.py
│   ├── Train.py
│   ├── Test.py
│   ├── utils.py
│   ├── Example.py
│   ├── OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot.py
│   ├── OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_with_different_pilots.py
│   ├── OFDM_ChannelEstimation_DeepLearning_QAM_random_pilot_withoutCP.py
│   ├── Pilot_8
│   ├── Pilot_16
│   ├── Pilot_64
│   └── ReadMe.txt
├── H_dataset/
│   ├── H_dataset.zip.001
│   ├── H_dataset.zip.002
│   ├── H_dataset.zip.003
│   └── H_dataset.zip.004
├── results_exercise_3_1/
│   ├── ber_results_summary.csv
│   ├── README_notes.md
│   ├── task_c_64qam_real.log      # optional
│   └── task_d_single_dnn.log      # optional
├── README.md
└── .gitignore
```

若 GitHub 檔案大小限制不方便上傳 `H_dataset`，可以不放 `H_dataset/`，但需要在 README 說明從原始 repo 下載資料集。

---

## 5. Colab 執行環境

本次實驗使用：

```text
Google Colab
GPU: T4 GPU
Python: Colab default Python 3
TensorFlow: TensorFlow 2.x with tensorflow.compat.v1
```

由於原始 code 是 TensorFlow 1.x 風格，本實作使用 `tensorflow.compat.v1` 維持相容。

在 Colab 中開啟 GPU：

```text
執行階段 → 變更執行階段類型 → 硬體加速器選 GPU / T4 GPU
```

確認 GPU：

```bash
!nvidia-smi
```

---

## 6. 下載與準備資料集

```bash
%cd /content
!rm -rf OFDM_DNN
!git clone https://github.com/haoyye/OFDM_DNN.git
```

確認資料夾：

```bash
%cd /content/OFDM_DNN
!ls
```

應看到：

```text
DNN_Detection  H_dataset  ReadMe.rst
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

也可以確認：

```bash
!ls 1.txt 300.txt 301.txt 400.txt
```

---

## 7. 實驗設定

本次使用 reduced Colab setting，而不是原始完整 20000 epochs。

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

## 8. Task (b)：QPSK，8 個小 DNN

在 `Main.py` 設定：

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

## 9. Task (c)：64-QAM，8 個小 DNN

在 `Main.py` 設定：

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

## 10. Task (d)：QPSK，單一大型 DNN

在 `Main.py` 設定：

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

Task (d) 的 BER 高於 task (b)，表示使用單一大型 DNN 一次預測全部 128 bits，比起使用 8 個小 DNN 分段預測更難訓練。

---

## 11. B / C / D 結果比較

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

## 12. 如何讓別人重現

1. 開啟 Colab 並選擇 T4 GPU。
2. Clone 原始 `OFDM_DNN` repo。
3. 解壓 `H_dataset`。
4. 將本專案修改過的 `Main.py`、`Train.py`、`Test.py`、`utils.py` 放入 `DNN_Detection/`。
5. 在 `Main.py` 中切換 `experiment`：
   - `b_qpsk_8dnns`
   - `c_64qam_8dnns`
   - `d_qpsk_single_dnn`
6. 執行 `python -u Main.py`。
7. 將 console 中的 BER 記錄到 `results_exercise_3_1/ber_results_summary.csv`。

---

## 13. 注意事項

- 本結果為 reduced Colab run，不是原始完整 20000 epochs。
- BER 數值會因 random initialization、training samples 與 Colab GPU 狀態略有差異。
- 若要完整重現 reference 結果，應提高 `train_epochs`、`total_batch` 與測試樣本數。
- `Models/` 可以用於本機保留訓練模型，但不建議提交到 GitHub。
