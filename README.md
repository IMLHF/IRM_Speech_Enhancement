# IRM_Speech_Enhancement
## 原理
使用带噪语音的幅度谱作为网络的输入x，纯净语音的幅度谱作为监督训练的标签y，期望训练出的网络具有预测纯净语音幅度谱的能力。
## 网络结构
网络使用两层BLSTM，每层512节点，映射到128点。最后使用一层全连接将维度转换为幅度谱的维度。
## 损失函数
损失函数使用MSE，但不是将网络的输出和标签y直接做MSE。网络的输出为幅度谱上的一个mask，理想情况下x\*mask=y。

所以损失函数为loss=MSE(x\*mask,y)，mask使用psm可以比使用irm取得更好的效果。
## 细节
### 训练参数
#### max_epochs
建议值：max_epochs = 3

训练迭代次数的最大值。3即为最优模型。
#### RAW_DATA
建议值：RAW_DATA = '/xxx/speaker_list'

训练使用的纯净语音的位置。目录中包含多个说话人，每个说话人一个目录。
#### NOISE_DIR
建议值：NOISE_DIR = '/xxx/many_noise'

训练使用的噪音音频数据的位置。目录中包含要使用的若干的噪音音频文件。音频文件为wav格式，单声道，16K采样率，16比特。
#### DATA_DICT_DIR
建议值：DATA_DICT_DIR = '_data/noised_speech'

根据RAW_DATA和NOISE_DIR中纯净语音和噪音生成的混合语音的列表文件保存位置。
#### TFRECORDS_DIR
建议值：TFRECORDS_DIR = '/xxx/feature_tfrecords'

根据DATA_DICT_DIR中各个集合的目录生成的训练数据（TFRecords）的位置。
#### AUDIO_VOLUME_AMP
建议值：AUDIO_VOLUME_AMP = False

是否将降噪结果的声音放到到最大。"False"为否，"True"为是。


## 如何使用
train: run_irm.py
decode/test: decoder.py


### 启动（运行，生成结果）
终端执行"python3 run_irm.py gpu_id"即可，gpu_id为解码使用的gpu编号。

如果不使用gpu，使用"python3 run_irm.py ''"即可。

## 依赖环境
python3             3.5.2<br>
audioread           2.1.6<br>
image               1.5.27<br>
librosa             0.6.2<br>
matplotlib          3.0.0<br>
numpy               1.14.5<br>
Pillow              5.3.0<br>
scipy               1.1.0<br>
tensorflow          1.12.0<br>
