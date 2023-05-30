# BirdClf - 人工智能基础大作业二

首先，建立一个python虚拟环境，如`conda create -n BirdClf python=3.8`, 然后使用 `pip install -r requirements.txt` 来安装必要的python库。

其次，由于有些文件过大，网络学堂没有上传，因此在清华云盘进行下载，云盘链接是[url](https://cloud.tsinghua.edu.cn/d/f5f64c3752f44e059c80/)。云盘中除原始数据集外有7个文件，其中，resnet50-0676ba61.pth是ResNet50预训练文件，请放在checkpoints/下；test_class.zip是为了进行样本类别分析的小数据集，请将其解压为test_class然后放在data/下；test_balance.zip是为了进行样本均衡度分析的小数据集，请将其解压为test_class然后放在data/下。由于进行了非常多实验，有很多ckpt，此处只选取了四个，分别是基本方法部分的最优模型(res/train_05-26_23-35-32/ckpts/epoch_13_acc_0.881143.pth)、进行学习率的调整后不使用预训练的SOTA模型(res/train_lr_05-28_23-13-51/ckpts/epoch_46_acc_0.960381.pth)、
使用预训练后的SOTA模型(res/train_pretrain_05-29_10-34-31/ckpts/epoch_25_acc_0.975238.pth)、使用模型压缩后的最优模型(res/train_compress_05-29_13-50-51/ckpts/epoch_28_acc_0.979048.pth)，若想用它们进行测试，请放入对应的文件夹下。

## 获取小数据集

如报告所述，获取小数据集主要是为了对样本类比和均衡度进行测试。直接在dataset.py文件中运行select_class函数和select_balance函数即可，可以修改其参数。

## 训练

要训练模型，运行`bash train.sh`：

```
PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=$PORT train.py --config configs/train_test_balance.yaml
```
第一行是获取端口号，第二行是运行train.py，可在其中修改GPU，修改配置文件。此外，也可以修改configs/下的各种配置文件中的配置和参数。其中，train.yaml指基础方法的配置文件，加hyperpara代表这是超参数调整的配置文件，加compress代表它是模型压缩的配置文件，以此类推。

训练的所有结果已经在res文件夹下保存，各个文件夹代表的模型在报告中已经指出。

## 测试与评估

测试与评估主要在test.py文件中实现。直接运行`bash test.sh`即可：
```
python test.py --device 0 --config 'res/train_test_balance_05-29_19-40-16/config.yaml' --N_imgs 4
```
其中，device是GPU号，config是配置文件路径，此处注意一定要将resume设为True并写上合理的ckpt_path。N_imgs是可视化图每行的图片数。运行之后就可以得到这个文件在测试集的Accuracy、Precision等指标，得到可视化图，获取模型统计信息。

