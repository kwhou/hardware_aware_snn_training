# Hardware-Aware SNN Training

![NTHU LARC Logo](images/nthu_larc_logo.png?raw=true)

The analog circuit in memristor-based SNN chip is susceptive to the process variation of semiconductor devices such as transistors, resistors, capacitors, memristors, etc. This repository contains the python code based on a popular machine learning framework, PyTorch. Our training method can adaptively tolerate the variation during the training process. Besides, we also take the limited dynamic range of analog circuit into account, adopting quantization for the model parameters.

## Usage

1. Run training and inference of ideal SNN
```
$ python snn_ideal.py --train
$ python snn_ideal.py --load model_ideal.pkl
```

2. Run training and inference of quantized SNN
```
$ python snn_quantization.py --train
$ python snn_quantization.py --load model_quantization.pkl
```

3. Run training and inference of hardware-aware SNN
```
$ python snn_hardware_aware.py --train
$ python snn_hardware_aware.py --load model_hardware_aware.pkl
```

