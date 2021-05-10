Define the forward pass of a Basciblock as follows, where (conv1, conv2, bn1, bn2, shortcut) differs from block to block.
```
def forward(self, x):
    out = relu(bn1(conv1(x)))
    out = bn2(conv2(out))
    out += shortcut(x)
    out = relu(out)
    return out
```
The entire model architecture is listed below.
```
ResNet(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (2): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (linear): Linear(in_features=64, out_features=10, bias=True)
)
```
A model summary with shape
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             432
       BatchNorm2d-2           [-1, 16, 32, 32]              32
            Conv2d-3           [-1, 16, 32, 32]           2,304
       BatchNorm2d-4           [-1, 16, 32, 32]              32
            Conv2d-5           [-1, 16, 32, 32]           2,304
       BatchNorm2d-6           [-1, 16, 32, 32]              32
        BasicBlock-7           [-1, 16, 32, 32]               0
            Conv2d-8           [-1, 16, 32, 32]           2,304
       BatchNorm2d-9           [-1, 16, 32, 32]              32
           Conv2d-10           [-1, 16, 32, 32]           2,304
      BatchNorm2d-11           [-1, 16, 32, 32]              32
       BasicBlock-12           [-1, 16, 32, 32]               0
           Conv2d-13           [-1, 16, 32, 32]           2,304
      BatchNorm2d-14           [-1, 16, 32, 32]              32
           Conv2d-15           [-1, 16, 32, 32]           2,304
      BatchNorm2d-16           [-1, 16, 32, 32]              32
       BasicBlock-17           [-1, 16, 32, 32]               0
           Conv2d-18           [-1, 32, 16, 16]           4,608
      BatchNorm2d-19           [-1, 32, 16, 16]              64
           Conv2d-20           [-1, 32, 16, 16]           9,216
      BatchNorm2d-21           [-1, 32, 16, 16]              64
           Conv2d-22           [-1, 32, 16, 16]             512
      BatchNorm2d-23           [-1, 32, 16, 16]              64
       BasicBlock-24           [-1, 32, 16, 16]               0
           Conv2d-25           [-1, 32, 16, 16]           9,216
      BatchNorm2d-26           [-1, 32, 16, 16]              64
           Conv2d-27           [-1, 32, 16, 16]           9,216
      BatchNorm2d-28           [-1, 32, 16, 16]              64
       BasicBlock-29           [-1, 32, 16, 16]               0
           Conv2d-30           [-1, 32, 16, 16]           9,216
      BatchNorm2d-31           [-1, 32, 16, 16]              64
           Conv2d-32           [-1, 32, 16, 16]           9,216
      BatchNorm2d-33           [-1, 32, 16, 16]              64
       BasicBlock-34           [-1, 32, 16, 16]               0
           Conv2d-35             [-1, 64, 8, 8]          18,432
      BatchNorm2d-36             [-1, 64, 8, 8]             128
           Conv2d-37             [-1, 64, 8, 8]          36,864
      BatchNorm2d-38             [-1, 64, 8, 8]             128
           Conv2d-39             [-1, 64, 8, 8]           2,048
      BatchNorm2d-40             [-1, 64, 8, 8]             128
       BasicBlock-41             [-1, 64, 8, 8]               0
           Conv2d-42             [-1, 64, 8, 8]          36,864
      BatchNorm2d-43             [-1, 64, 8, 8]             128
           Conv2d-44             [-1, 64, 8, 8]          36,864
      BatchNorm2d-45             [-1, 64, 8, 8]             128
       BasicBlock-46             [-1, 64, 8, 8]               0
           Conv2d-47             [-1, 64, 8, 8]          36,864
      BatchNorm2d-48             [-1, 64, 8, 8]             128
           Conv2d-49             [-1, 64, 8, 8]          36,864
      BatchNorm2d-50             [-1, 64, 8, 8]             128
       BasicBlock-51             [-1, 64, 8, 8]               0
           Linear-52                   [-1, 10]             650
================================================================
Total params: 272,474
Trainable params: 272,474
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 3.72
Params size (MB): 1.04
Estimated Total Size (MB): 4.77
```
