# Deep Unsupervised Image Hashing by Maximizing Bit Entropy

This is the PyTorch implementation of accepted AAAI 2021 paper: [Deep Unsupervised Image Hashing by Maximizing Bit Entropy](https://arxiv.org/abs/2012.12334)

<!-- Our paper presentation is on [YouTube](https://www.youtube.com/watch?v=riZDqdTrNrg) -->

<!-- Meantime, a re-implemented version of our work: [Training code](https://github.com/swuxyj/DeepHash-pytorch)
 -->

## Proposed Bi-half layer
Bi-half 层是一种无监督深度哈希技术的关键组件，旨在通过最大化二进制码的比
特熵来提高图像或视频检索的效率和准确性。Bi-half 层的设计理念基于一个核心观察：
在无监督哈希中，当每个哈希位上的 0 和 1 的概率相等时，即达到半半分布，熵达到最
大值。这种状态对于存储和检索来说是最理想的，因为它保证了每个比特位都能携带最
大的信息量。

## Datasets and Architectures on different settings
Experiments on **5 image datasets**:
Flickr25k, Nus-wide, Cifar-10, Mscoco, Mnist, and **2 video
datasets**: Ucf-101 and Hmdb-51. 
According to different settings, we divided them into: i) Train an AutoEncoder on Mnist; ii) Image Hashing on Flickr25k, Nus-wide, Cifar-10, Mscoco using Pre-trained Vgg; iii) Video Hashing on Ucf-101 and Hmdb-51 using Pre-trained 3D models.


### Glance

```
3 settings ── AutoEncoder ── ── ── ── ImageHashing ── ── ── ── VideoHashing      
               ├── Sign.py             ├── Cifar10_I.py          └── main.py
               ├── SignReg.py          ├── Cifar10_II.py
               └── BiHalf.py           ├── Flickr25k.py
    	     			       └── Mscoco.py
```


### Datasets download
使用了 Cifar10、Flickr25k 这两个数据集进行实验
|#|Datasets|Download|
|---|----|-----|
|1|Flick25k|[Link](https://press.liacs.nl/mirflickr/mirdownload.html)
|2|Mscoco|[Link](https://drive.google.com/file/d/0B7IzDz-4yH_HN0Y0SS00eERSUjQ/view?usp=sharing )|
|3|Nuswide|[Link](https://github.com/TreezzZ/DSDH_PyTorch)  |
|4|Cifar10|[Link](https://www.cs.toronto.edu/~kriz/cifar.html)|
|5|Mnist|[Link](http://yann.lecun.com/exdb/mnist/)|
|6|Ucf101|[Link](https://surfdrive.surf.nl/files/index.php/s/dnYpOzKSmZFxvtX)|
|7|Hmdb51|[Link](https://surfdrive.surf.nl/files/index.php/s/q8Oqu4orntKH79p)|


## Citation

If you find the code in this repository useful for your research consider citing it.

```
@article{liAAAI2021,
  title={Deep Unsupervised Image Hashing by Maximizing Bit Entropy},
  author={Li, Yunqiang and van Gemert, Jan},
  journal={AAAI},
  year={2021}
}
```

 
 

 
 
 
 


