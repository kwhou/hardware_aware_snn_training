# Hardware-Aware SNN Training

![NTHU LARC Logo](images/nthu_larc_logo.png?raw=true)

The analog circuit in memristor-based SNN chip is susceptive to the process variation of semiconductor devices such as transistors, resistors, capacitors, memristors, etc. This repository contains the python code based on a popular machine learning framework, PyTorch. Our training method can tolerate the variation during the training process. Besides, we also take the limited dynamic range of analog circuit into account, adopting quantization for the model parameters.

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

3. Run training and inference of variation aware SNN
* With Global Variation
```
$ python snn_variation.py --train --vmod global --corner tt --save model_variation_global_tt.pkl
$ python snn_variation.py --train --vmod global --corner fs --save model_variation_global_fs.pkl
$ python snn_variation.py --train --vmod global --corner sf --save model_variation_global_sf.pkl
$ python snn_variation.py --train --vmod global --corner ff --save model_variation_global_ff.pkl
$ python snn_variation.py --train --vmod global --corner ss --save model_variation_global_ss.pkl
$ python snn_variation.py --load model_variation_global_tt.pkl
$ python snn_variation.py --load model_variation_global_fs.pkl
$ python snn_variation.py --load model_variation_global_sf.pkl
$ python snn_variation.py --load model_variation_global_ff.pkl
$ python snn_variation.py --load model_variation_global_ss.pkl
```

* With Global & Local Variation
```
$ python snn_variation.py --train --vmod both --corner tt --range 50,200 --save model_variation_both_tt_50_200.pkl
$ python snn_variation.py --load model_variation_both_tt_50_200.pkl
```

