# H_dataset 放置說明

本資料夾未包含大型 channel response 資料集。
請自行加入原始 `haoyye/OFDM_DNN` 的：

```text
H_dataset.zip.001
H_dataset.zip.002
H_dataset.zip.003
H_dataset.zip.004
```

在 Colab 中可用：

```bash
%cd /content/OFDM_DNN/H_dataset
!cat H_dataset.zip.001 H_dataset.zip.002 H_dataset.zip.003 H_dataset.zip.004 > H_dataset.zip
!unzip -q H_dataset.zip
!if [ -d H_dataset ]; then mv H_dataset/*.txt .; rmdir H_dataset; fi
!find . -maxdepth 1 -name "*.txt" | wc -l
```

正常應得到 400 個 `.txt` channel files。
