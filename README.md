# BirdClf - 人工智能基础大作业二

首先，建立一个python虚拟环境，如`conda create -n BirdClf python=3.8`, 然后使用 `pip install -r requirements.txt` 来安装必要的python库。

其次，由于有些文件过大，网络学堂没有上传，因此在清华云盘进行下载

## Face align

You can align face image by mtcnn or landmark. For mtcnn, run `bash face_detect/run_mtcnn.sh` to get mtcnn face detection results by 8 gpus, then run `face_detect/process_mtcnn.py` to merge mtcnn results and evaluate them. For landmark, run `python face_detect/landmark.py` to get landmark face detection results and evaluate them.

* ckpt `shape_predictor_68_face_landmarks.dat` for landmark face detection can be downloaded  here: [url](download%20https://cloud.tsinghua.edu.cn/d/90f476668cb0498f9882/). put it under `checkpoints/`.

Then, run `python face_align -t mtcnn` or `python face_align -t landmark` to align images of both training_set and test_pair. The default crop size for mtcnn is (112, 96), for landmark is (112, 112). you can edit it in face_align.py.

You can download the face detection and face align results here: [url](download%20https://cloud.tsinghua.edu.cn/d/90f476668cb0498f9882/). Uncompress `data.tar.gz` and put it In the root directory.

**The file structure of data dir is:**

```
data/
├── test/
    ├── landmark/
    ├──align112x112/
    ├──eval_landmark/
    ├──landmarks/
    ├── mtcnn/
        ├──align112x96/
    ├──eval_mtcnn/
    ├──mtcnns/
    ├── test_pair/
├── train/
    ├── landmark/
    ├──align112x112/
    ├──eval_landmark/
    ├──indexes/
    ├──landmarks/
    ├──error_lmk.json
    ├── mtcnn/
        ├──align112x96/
    ├──eval_mtcnn/
    ├──indexes/
    ├──mtcnns/
    ├── training_set/
├── dataset.zip
```

## Train

* `{align_type}` refer to `mtcnn` or `landmark`.

Run `bash train.sh` to train the model:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=23756 train_ddp.py --type {align_type}
```

Also, you can edit `configs/train_{align_type}.yaml` to change training params, like batch_size, max_epochs, val_interval and triplets_num.

The pretrained models can be downloaded here: [url](download%20https://cloud.tsinghua.edu.cn/d/90f476668cb0498f9882/). After download, put `landmark_last_acc_0.8549.pth` in `log\ckpt\landmark_last_acc_0.8549.pth`; put `mtcnn_last_acc_0.8693.pth` in `log\ckpt\mtcnn_last_acc_0.8693.pth`.

What's more, My training logs for the pretrained models are in `log\logs\train_{align_type}.log`.

## Evaluate

Run `python evaluate.py --type {align_type} --val_ratio 0.85 --seed 0` to evaluate model. My evaluate logs for the pretrained models are in `log\logs\evaluate_{align_type}.log`.

## Test

Run `python test.py --type {align_type}` to get test results. My test logs are in `log\logs\test_{align_type}.log` and the test results are in `test_{align_type}.txt`.

## Acknowledgements

[CosFace_pytorch](https://github.com/MuggleWang/CosFace_pytorch), [facenet](https://github.com/tbmoon/facenet), [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).
