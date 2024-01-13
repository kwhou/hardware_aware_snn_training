# Hardware-Aware SNN Training

![NTHU LARC Logo](images/nthu_larc_logo.png?raw=true)

The analog circuit in memristor-based SNN chip is susceptive to the process variation of semiconductor devices such as transistors, resistors, capacitors, memristors, etc. This repository contains the python code based on a popular machine learning framework, PyTorch. Our training method can adaptively tolerate the variation during the training process. Besides, we also take the limited dynamic range of analog circuit into account, adopting quantization for the model parameters.

## Usage

1. Run training of ideal SNN
```
$ python snn_ideal.py --train
```

2. Run inference of ideal SNN with pre-trained model
```
$ python snn_ideal.py --load model_ideal.pkl
```

3. Run training of hardware-aware SNN
```
$ python snn_hardware_aware.py --train
```

4. Run inference of hardware-aware SNN with pre-trained model
```
$ python snn_hardware_aware.py --load model_hardware_aware.pkl
```
