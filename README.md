# The code for Blizzard Challenge
### 环境配置
```
pip install -r requirements.txt
```

### 下载预训练模型

预训练模型获取链接如下：

https://pan.baidu.com/s/1XCW9O5YBkn8K3T_aY_GVBA?pwd=1nfy

下载文件夹后将其放置在项目根目录下即可。

### 开启服务
```
(CUDA_VISIBLE_DEVICES=1) python infer_server.py
```
括号内为指定在哪一块显卡上运行，可加可不加，是为了防止在第一块显卡上有程序在运行，本asr服务所需显存不够，则可指定其余空闲的显卡来提供服务
