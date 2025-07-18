# WebDataset

## 简介

- **文件格式**：专门为大规模深度学习数据加载设计的库，数据要求存储在 `.tar` 文件中。遵循以下两个约定：
  1. 在每个 `.tar` 文件中，属于同一训练样本的文件在去掉所有文件扩展名后共享相同的基名（作为 key）。
  2. `.tar` 文件的分片（shards）按编号命名，例如 `something-000000.tar` 到 `something-012345.tar`，通常使用花括号表示法 `something-{000000..012345}.tar`。

- **随机性**：数据读取中需要随机访问，WebDataset 可以将 `.tar` 包打乱顺序，同时在读取一个包时（由于顺序读取速度更快），使用一个 buffer 来打乱样本顺序，通过将新读取的样本与 buffer 中的随机一个交换来打乱数据集。

- **加速原理**：与磁盘存储和网络传输相关。

---

## 使用方法

### 导入

```python
import webdataset as wds  # 用于方便代码编写
```

### 初始化参数

- **数据路径或模式**：

```python
dataset_url = "path/to/dataset-{000..002}.tar"  # 支持大括号扩展模式
# 或者网址
dataset_url = "http://example.com/dataset-{000..002}.tar"
```

- **初始化 WebDataset**：

```python
dataset = wds.WebDataset(dataset_url)
```

### 数据读取及处理

假设数据样本包含一个图像和一个 JSON 文本：

```python
from torchvision.transforms import ToTensor
import json
```

- **定义解码函数**：

```python
def decode(sample):
    # 解码图像
    image = wds.torch_image("rgb")(sample["jpg"])
    # 解码 JSON 标签
    label = json.loads(sample["json"].decode("utf-8"))
    return image, label["class_id"]  # 假设 JSON 文件中包含 {"class_id": ...}
```

- **应用解码**：

```python
dataset = dataset.decode(decode)
```

**注意**：也可以使用 `dataset = dataset.decode("pil")`，这会自动将 `.jpg` 或 `.png` 解码成 PIL 图像。

- **数据预处理**：

```python
from torchvision import transforms
```

- **定义预处理函数**：

```python
def preprocess(image, label):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image, label
```

- **应用预处理**：

```python
dataset = dataset.map(preprocess)
```

**注意**：WebDataset 返回字典格式数据，例如 `{"jpg": <PIL.Image>, "cls": b"0"}`。可以使用 `to_tuple()` 将字典转换成元组格式，例如：

```python
dataset = dataset.to_tuple("jpg", "cls")
```

这样，`dataset` 中的元素就是 `(image, label)`。

### 乱序及分批

- **洗牌**：

```python
dataset = dataset.shuffle(1000)  # 缓冲区大小为 1000
```

- **设置批量大小**：

```python
batch_size = 32
```

- **批处理**：

```python
dataset = dataset.batched(batch_size)
```

### 使用 DataLoader 在训练中加载数据

```python
from torch.utils.data import DataLoader
```

- **创建 DataLoader**：

```python
dataloader = DataLoader(dataset, batch_size=None, num_workers=4)
```

**注意**：
- `num_workers` 是 DataLoader 的子进程数，可根据 CPU 核心数量设置，但受 I/O 设备、内存带宽等影响。
- `batch_size=None` 是因为 WebDataset 已经进行了批处理，无需再合并成新的批次。

如果内存足够，还可以增加 `pin_memory=True`，例如：

```python
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)
```

这适用于 GPU 训练，它将当前批次的数据放到一个“锁定内存”中，减少数据从 CPU 到 GPU 的传输时间。注意批次数据不要过大！

- **训练时迭代数据**：

```python
for batch in dataloader:
    images, labels = batch
    # 这里可以添加训练逻辑
    print(images.shape, labels)
```

---

### 完整示例代码

```python
import webdataset as wds
import json
from torchvision import transforms
from torch.utils.data import DataLoader

# 数据路径或模式
dataset_url = "dataset-{000..002}.tar"  # 支持大括号扩展模式

# 初始化 WebDataset
dataset = wds.WebDataset(dataset_url)

# 定义解码函数
def decode(sample):
    image = wds.torch_image("rgb")(sample["jpg"])
    label = json.loads(sample["json"].decode("utf-8"))
    return image, label["class_id"]  # 假设 JSON 文件中包含 {"class_id": ...}

# 应用解码
dataset = dataset.decode(decode)

# 定义全局预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess(image, label):
    image = transform(image)
    return image, label

# 应用预处理
dataset = dataset.map(preprocess)

# 洗牌
dataset = dataset.shuffle(2000)

# 批处理
batch_size = 32
dataset = dataset.batched(batch_size)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)

# 迭代数据
for batch in dataloader:
    images, labels = batch
    # 这里可以添加训练逻辑
    print(images.shape, labels)
```

---

## 进阶使用

### 分布式训练

在分布式训练中，每个进程加载不同的数据子集。现在官方推荐使用 `ResampledShards`，不能使用 `sampler`。

```python
import os
import torch
import torch.distributed as dist
import webdataset as wds
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
```

- **初始化分布式环境**：

```python
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank))
```

- **Shards 列表**：

```python
shards = [f"/path/to/data-{i:05d}.tar" for i in range(NUM_SHARDS)]
```

- **WebDataset pipeline**：
Fluid接口：
```python
dataset = (
    wds.WebDataset(shards,
                   resampled=True,
                   shardshuffle=True,
                   nodesplitter=wds.split_by_node,
                   splitter=wds.split_by_worker)
    .shuffle(1000)
    .decode("pil")
    .to_tuple("jpg", "cls")
    .map_tuple(transforms.ToTensor(), lambda cls: int(cls))
    .with_epoch(EPOCH_LENGTH)  # 指定每 epoch 抽样总样本数
)
```
显式pipline：这里展示了使用ResampledShards的方案（相当于前面使用resampled=True, shardshuffle=True，实现shards之间顺序随机化），需要将tarfile_to_samples换成cached_tarfile_to_samples，前面一种方式没有tarfile_to_samples是因为被默认执行。
```python
dataset = wds.DataPipeline(
    wds.ResampledShards(urls),
    wds.cached_tarfile_to_samples(),  # Use cached version
    wds.shuffle(shuffle_vals[0]),
    wds.decode(wds.torch_audio),
    wds.map(partial(callback, sample_size=sample_size, sample_rate=sample_rate, verbose=verbose, **kwargs)),
    wds.shuffle(shuffle_vals[1]),
    wds.to_tuple(audio_file_ext),
    wds.batched(batch_size),
).with_epoch(epoch_len)
```
- **WebLoader**：

```python
loader = wds.WebLoader(dataset,
                       batch_size=32,
                       num_workers=4,
                       pin_memory=True)
loader = loader.ddp_equalize(EPOCH_LENGTH // 32)  # 确保每个进程 batch 数一致
```

- **模型构建与 DDP 包装**：

```python
model = ...  # your model
model = model.to(local_rank)
ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

- **训练循环**：

```python
for epoch in range(EPOCHS):
    for batch_imgs, batch_labels in loader:
        batch_imgs = batch_imgs.to(local_rank, non_blocking=True)
        batch_labels = batch_labels.to(local_rank)
        outputs = ddp_model(batch_imgs)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 关键点说明

- **nodesplitter + splitter**：
  - `nodesplitter=wds.split_by_node`：确保不同节点处理不同的 shards。
  - `splitter=wds.split_by_worker`：同一节点内每个 DataLoader worker 独立处理不同的 shards。

- **with_epoch(EPOCH_LENGTH)**：
  - 定义一个 epoch 内总共抽样多少样本，无需预先计算 dataset 长度，可通过经验设置用于训练长度控制。

- **WebLoader + ddp_equalize(...)**：
  - `WebLoader` 是专为 WebDataset 设计的，替代 DataLoader（流式读取大规模数据）。
  - `.ddp_equalize(num_batches_per_epoch)`：确保在 DDP 运行时，各进程收到完全相同数量的 batches，避免训练不一致或 hang 现象。

**注意**：在分布式训练中展示了在 DataLoader 中设置 `batch_size` 的方案。
### 加上Ray
``` python
import os
import sys
import time
import json
import torch
import ray
import webdataset as wds
from collections import deque
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms
from torchvision.models import resnet50

# 在训练函数外部初始化 Ray
if not ray.is_initialized():
    ray.init()

@ray.remote(num_gpus=1)
def train_on_ray(rank, config):
    # 设置分布式环境变量
    os.environ["MASTER_ADDR"] = config["master_addr"]
    os.environ["MASTER_PORT"] = config["master_port"]
    torch.distributed.init_process_group(
        backend=config["backend"],
        rank=rank,
        world_size=config["world_size"]
    )
    config["rank"] = rank
    train(config)

def enumerate_report(seq, delta, growth=1.0):
    last = 0
    for count, item in enumerate(seq):
        now = time.time()
        if now - last > delta:
            last = now
            yield count, item, True
        else:
            yield count, item, False
        delta *= growth

def make_dataloader_train(batch_size, cache_dir):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    def make_sample(sample):
        img = transform(sample["jpg"])
        return img, sample["cls"]

    trainset = (
        wds.WebDataset(
            "https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-train-{000000..001281}.tar",
            resampled=True, shardshuffle=True,
            cache_dir=cache_dir,
            nodesplitter=wds.split_by_node
        )
        .shuffle(1000)
        .decode("pil")
        .map(make_sample)
        .batched(64)
    )
    loader = wds.WebLoader(trainset, batch_size=None, num_workers=4)
    loader = loader.unbatched().shuffle(1000).batched(batch_size)
    loader = loader.with_epoch((1282 * 100) // batch_size)
    return loader

def train(config):
    model = resnet50(pretrained=False).cuda()
    model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    loss_fn = nn.CrossEntropyLoss()

    loader = make_dataloader_train(config["batch_size"], config["cache_dir"])

    losses = deque(maxlen=100)
    accs = deque(maxlen=100)
    steps = 0

    for epoch in range(config["epochs"]):
        for i, (inputs, labels), verbose in enumerate_report(loader, config["report_s"]):
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.argmax(1) == labels).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)
            steps += labels.size(0)

            if verbose:
                print(f"[rank {config['rank']}] epoch {epoch} step {i} loss {sum(losses)/len(losses):.4f} acc {sum(accs)/len(accs):.4f}")

            if steps >= config["max_steps"]:
                print("Finished max_steps")
                return
    print(f"[rank {config['rank']}] Training complete, steps={steps}")

if __name__ == "__main__":
    config = {
        "epochs": 10,
        "max_steps": int(1e6),
        "batch_size": 32,
        "lr": 0.01,
        "momentum": 0.9,
        "world_size": min(2, ray.available_resources().get("GPU", 0)),
        "backend": "nccl",
        "master_addr": "127.0.0.1",
        "master_port": "12355",
        "report_s": 15.0,
        "cache_dir": "./_cache"
    }

    # 异步启动多 GPU 训练进程
    tasks = [train_on_ray.remote(rank, config) for rank in range(config["world_size"])]
    ray.get(tasks)
```
