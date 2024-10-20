# Hardware-Aware SNN Training

![NTHU LARC Logo](images/nthu_larc_logo.png?raw=true)

The analog circuit in memristor-based SNN chip is susceptive to the process variation of semiconductor devices such as transistors, resistors, capacitors, and memristors. This repository contains the python code based on the machine learning framework, PyTorch. Our training method can tolerate the process variation. We also take the non-linearity, limited precision, and limited dynamic range of analog circuits into account.

## Usage

1. Run training and inference of ideal SNN
```
$ python snn_ideal.py --train --save model_ideal.pkl
$ python snn_ideal.py --load model_ideal.pkl
```

2. Run training and inference of quantized SNN
```
$ python snn_quantization.py --train --save model_quantization.pkl
$ python snn_quantization.py --load model_quantization.pkl
```

3. Run training and inference of hardware aware SNN
```
$ python snn_hardware_aware.py --train --save model_hardware_aware.pkl
$ python snn_hardware_aware.py --load model_hardware_aware.pkl
```

