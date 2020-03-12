### torchsummaryM
---
 - _**Just for myself**_
 - Visualize kernel size, input & output shapes and # parameters
 - Can handle, RNN. Recursice Net, even multi input model
 ---
 
### Ref
 - [torchsummaryX](https://github.com/nmhkahn/torchsummaryX)
 - [torchsummary](https://github.com/sksq96/pytorch-summary)
---

#### Usage
 - input_size must be a tuple without batch size
 - batch size for calculating memory 
 - device type one of 'cuda' or 'cpu'
 - dtypes for RNN ==> torch.long

 ~~~~
 summary(model, tensor, *args)
 ~~~~