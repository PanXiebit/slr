# SLR-SLT

## Preparation
**training data**
```
pip install gdown
gdown https://drive.google.com/uc?id=1y_xaNCjMJdLzE4PQCdTt3Ff4vt2FGy_R&export=download
```

**ctcdecode**
```
git clone --recursive https://github.com/Jevin754/ctcdecode.git
cd ctcdecode
pip install wget
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .
```

****
```
git clone --recursive https://github.com/usnistgov/SCTK.git
cd SCTK
export CXXFLAGS="-std=c++11" && make config && make all && make check && make install && make doc
# add SCTK path to environment variable PATH
export PATH=$PATH:SCTK_PATH/bin/sclite
```

## Experiment1

|model|wer|git branch/commit/description|
|---|---|---|
|model|wer|git branch/commit|
|---|---|---|
|only ctc|epoch8: 32.1|master|
|decoder with tf|epoch12: 38.7/36.0 |tf_decode|


### jointly training with ctcdecode

|parameters/decoder| |
|---|---|
|log file|train-1|
|ctc deocde| dev=31.3/41.2, epoch=14|
|ctc and iterative| dev=31.9/39.2, epoch=22|
|ctc weight|  0.1|
|dec weight| 1.0 |
| early_exit| '3,3,3'|
|no_share_maskpredictor| False |
|no_share_discriminator| False|


|parameters/decoder| |
|---|---|
|log file|train-2|
|ctc deocde| dev=31.2/38.1, epoch=13|
|ctc and iterative| dev=31.2/38.1, epoch=13|
|ctc weight|  0.1|
|dec weight| 1.0 |
|early_exit| '3,3,3'|
|no_share_maskpredictor| **True** |
|no_share_discriminator| **True** |

|parameters/decoder| |
|---|---|
|log file|train-3|
|ctc deocde| dev=31.8/35.5, epoch=13|
|ctc and iterative| dev=31.8/35.5, epoch=13|
|ctc weight|  0.1|
|dec weight| 1.0 |
|early_exit| **'3,6,6'**|
|no_share_maskpredictor| True|
|no_share_discriminator| True|

|parameters/decoder| |
|---|---|
|log file|train-4|
|ctc deocde| dev=31.4/42.0, epoch=11|
|ctc and iterative| dev=32.4/36.7, epoch=9|
|ctc weight|  0.1|
|dec weight| **5.0** |
|early_exit| '3,6,6'|
|no_share_maskpredictor| True|
|no_share_discriminator| True|

|parameters/decoder| |
|---|---|
|log file|train-5|
|ctc deocde| dev=31.2/35.2, epoch=12|
|ctc and iterative| dev=31.2/35.2, epoch=12|
|ctc weight|  0.1|
|dec weight| 5.0 |
|early_exit| '6,6,6'|
|no_share_maskpredictor| True|
|no_share_discriminator| True|



### train decoder after joint trining
|parameters/decoder| |
|---|---|
|log file|train-3-2|
|ctc deocde| dev=31.8, epoch=13|
|ctc and iterative| dev=31.8/35.5, **dev wer increasing**, epoch=13....|
|ctc weight|  **0.0** |
|dec weight| 1.0 |
|early_exit| '3,6,6'|
|no_share_maskpredictor| True|
|no_share_discriminator| True|


### crossing training



