import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def summary(model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    net_name = model._get_name()
    result, params_info = summary_string(
        net_name, model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(net_name, model, input_size, batch_size=-1, device=torch.device('cpu'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)
    else:
        dtypes = [dtypes] * len(input_size)

    summary_str = ''
    global max_layer_length
    max_layer_length = 0

    def register_hook(module):
        def hook(module, input, output):
            global max_layer_length

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            module_name = str(module_idx)
            for name, item in module_names.items():
                if item == module:
                    module_name = name
                    break
            
            sep_module_name = module_name.split("-")
            if len(sep_module_name) > 1:
                module_name = "-".join(sep_module_name[:-1])
            length = len(module_name)
            if length > max_layer_length:
                max_layer_length = length


            # m_key = "%i-%s" % (module_idx + 1, module_name)
            m_key = "{:>}> {:<10}".format(module_idx+1, module_name)
            summary[m_key] = OrderedDict()
            summary[m_key]["id"] = id(module)

            summary[m_key]["input_size"] = 0
            if isinstance(input, (list, tuple)):
                summary[m_key]["input_shape"] = []
                for i in input:
                    if isinstance(i, (list, tuple)):
                        for ii in i:
                            try:
                                summary[m_key]["input_shape"].append([batch_size] + list(ii.size()[1:]))
                                summary[m_key]["input_size"] += 1
                            except AttributeError:
                                # pack_padded_seq and pad_packed_seq store feature into data attribute
                                summary[m_key]["input_shape"].append([batch_size] + list(ii.data.size()[1:]))
                                summary[m_key]["input_size"] += 1
                    else:
                        try:
                            summary[m_key]["input_shape"].append([batch_size] + list(i.size()[1:]))
                            summary[m_key]["input_size"] += 1
                        except AttributeError:
                            # pack_padded_seq and pad_packed_seq store feature into data attribute
                            summary[m_key]["input_shape"].append([batch_size] + list(i.data.size()[1:]))
                            summary[m_key]["input_size"] += 1
            else:
                summary[m_key]["input_shape"] = list(input.size())
                summary[m_key]["input_shape"][0] = batch_size
                summary[m_key]["input_size"] += 1


            summary[m_key]["output_size"] = 0
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = []
                for o in output:
                    if isinstance(o, (list, tuple)):
                        for io in o:
                            try:
                                summary[m_key]["output_shape"].append([batch_size] + list(io.size()[1:]))
                                summary[m_key]["output_size"] += 1
                            except AttributeError:
                                # pack_padded_seq and pad_packed_seq store feature into data attribute
                                summary[m_key]["output_shape"].append([batch_size] + list(io.data.size()[1:]))
                                summary[m_key]["output_size"] += 1
                    else:
                        try:
                            summary[m_key]["output_shape"].append([batch_size] + list(o.size()[1:]))
                            summary[m_key]["output_size"] += 1
                        except AttributeError:
                            # pack_padded_seq and pad_packed_seq store feature into data attribute
                            summary[m_key]["output_shape"].append([batch_size] + list(o.data.size()[1:]))
                            summary[m_key]["output_size"] += 1
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
                summary[m_key]["output_size"] += 1

            b_params = 0
            w_params = 0
            summary[m_key]["ksize"] = "-" 
            summary[m_key]["nb_params"] = 0
            for name, param in module.named_parameters():
                if name == "weight":    
                    ksize = list(param.size())
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    summary[m_key]["ksize"] = ksize
            for name, param in module.named_parameters():
                if "weight" in name:
                    w_params += param.nelement()
                    summary[m_key]["trainable"] = param.requires_grad
                elif "bias" in name:
                    b_params += param.nelement()
            summary[m_key]["nb_params"] = w_params + b_params

            if list(module.named_parameters()):
                for k, v in summary.items():
                    if summary[m_key]["id"] == v["id"] and k != m_key:
                        summary[m_key]["nb_params"] = 0

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not module._modules
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    module_names = get_names_dict(model)
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    max_layer_length += 5
    lengths = [max_layer_length, 22, 25, 25, 13]
    total_length = sum(lengths) + 7

    summary_str += "-" * total_length + "\n"
    line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
        "Layer(type)", "Kernel Shape", "Input Shape", "Output Shape", "Param", width=max_layer_length)
    summary_str += line_new + "\n"
    summary_str += "=" * total_length + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        _input_size  = summary[layer]["input_size"]
        _output_size = summary[layer]["output_size"]
        if  _input_size > 1 and _output_size > 1:
            base = 0
            iter_size = _output_size
            if _input_size > _output_size:
                iter_size = _input_size
                base = 1
            
            if _input_size == _output_size:
                base = 2
            
            if base == 1:
                for i in range(iter_size):
                    if i == 0:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                            layer,
                            str(summary[layer]["ksize"]),
                            str(summary[layer]["input_shape"][0]),
                            str(summary[layer]["output_shape"][0]),
                            "{0:,}".format(summary[layer]["nb_params"]),
                            width=max_layer_length
                        )
                        total_params += summary[layer]["nb_params"]
                        total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                        if "trainable" in summary[layer]:
                            if summary[layer]["trainable"] == True:
                                trainable_params += summary[layer]["nb_params"]
                        summary_str += line_new + "\n"  
                    else:
                        if i < _output_size-1:
                            line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", str(summary[layer]["input_shape"][i]), str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                            summary_str += line_new + "\n"  
                        else:
                            line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", str(summary[layer]["input_shape"][i]), "-", "-", width=max_layer_length)
                            summary_str += line_new + "\n"
            elif base == 0:
                for i in range(iter_size):
                    if i == 0:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                            layer,
                            str(summary[layer]["ksize"]),
                            str(summary[layer]["input_shape"][0]),
                            str(summary[layer]["output_shape"][0]),
                            "{0:,}".format(summary[layer]["nb_params"]),
                            width=max_layer_length
                        )
                        total_params += summary[layer]["nb_params"]
                        total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                        if "trainable" in summary[layer]:
                            if summary[layer]["trainable"] == True:
                                trainable_params += summary[layer]["nb_params"]
                        summary_str += line_new + "\n"  
                    else:
                        if i < _input_size-1:
                            line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", str(summary[layer]["input_shape"][i]), str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                            summary_str += line_new + "\n"  
                        else:
                            line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", "-", str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                            summary_str += line_new + "\n"
            else:
                for i in range(iter_size):
                    if i == 0:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                            layer,
                            str(summary[layer]["ksize"]),
                            str(summary[layer]["input_shape"][0]),
                            str(summary[layer]["output_shape"][0]),
                            "{0:,}".format(summary[layer]["nb_params"]),
                            width=max_layer_length
                        )
                        total_params += summary[layer]["nb_params"]
                        total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                        if "trainable" in summary[layer]:
                            if summary[layer]["trainable"] == True:
                                trainable_params += summary[layer]["nb_params"]
                        summary_str += line_new + "\n"  
                    else:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", str(summary[layer]["input_shape"][i]), str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                        summary_str += line_new + "\n"
        elif _input_size > 1:
            for i in range(_input_size):
                    if i == 0:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                            layer,
                            str(summary[layer]["ksize"]),
                            str(summary[layer]["input_shape"][0]),
                            str(summary[layer]["output_shape"]),
                            "{0:,}".format(summary[layer]["nb_params"]),
                            width=max_layer_length
                        )
                        total_params += summary[layer]["nb_params"]
                        total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                        if "trainable" in summary[layer]:
                            if summary[layer]["trainable"] == True:
                                trainable_params += summary[layer]["nb_params"]
                        summary_str += line_new + "\n"  
                    else:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", str(summary[layer]["input_shape"][i]), "-", "-", width=max_layer_length)
                        summary_str += line_new + "\n"
        elif _output_size > 1:
            for i in range(_output_size):
                    if i == 0:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                            layer,
                            str(summary[layer]["ksize"]),
                            str(summary[layer]["input_shape"]),
                            str(summary[layer]["output_shape"][0]),
                            "{0:,}".format(summary[layer]["nb_params"]),
                            width=max_layer_length
                        )
                        total_params += summary[layer]["nb_params"]
                        total_output += np.sum(np.prod(summary[layer]["output_shape"], axis=1))
                        if "trainable" in summary[layer]:
                            if summary[layer]["trainable"] == True:
                                trainable_params += summary[layer]["nb_params"]
                        summary_str += line_new + "\n"  
                    else:
                        line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(" ", "-", "-", str(summary[layer]["output_shape"][i]), "-", width=max_layer_length)
                        summary_str += line_new + "\n"
        else:
            line_new = "{:<{width}}||{:>22} {:>25} {:>25} {:>13}".format(
                layer,
                str(summary[layer]["ksize"]),
                str(summary[layer]["input_shape"]),
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
                width=max_layer_length
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                            * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "=" * total_length + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
                                                        
    summary_str += "-" * total_length + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "-" * total_length + "\n"
    
    return summary_str, (total_params, trainable_params)

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=None):
        for key, m in module.named_children():
            if str.isdigit(key):
                key = int(key) + 1
                key = str(key)

            key = str.capitalize(key)
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))

            if num_named_children > 1:
                name = "{}-{}".format(parent_name, key) if parent_name else key
            else:
                name = "{}-{}-{}".format(parent_name, cls_name, key) if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # class Net(nn.Module):
    #     def __init__(self,
    #              vocab_size=20, embed_dim=300,
    #              hidden_dim=512, num_layers=2):
    #         super().__init__()
    #         self.hidden_dim = hidden_dim
    #         self.embedding = nn.Embedding(vocab_size, embed_dim)
    #         self.encoder = nn.LSTM(embed_dim, hidden_dim,
    #                             num_layers=num_layers, batch_first=True)
    #         self.decoder = nn.Linear(hidden_dim, vocab_size)

    #     def forward(self, x):
    #         embed = self.embedding(x)
    #         out, hidden = self.encoder(embed)
    #         out = self.decoder(out)
    #         out = out.view(-1, out.size(2))
    #         return out, hidden

    # summary(Net().to(device), (100, ), batch_size=1, device=device, dtypes=torch.long)

    # from torchsummaryX import summary as Xsummary
    # inputs = torch.zeros(1, 100, dtype=torch.long).to(device)
    # Xsummary(Net().to(device), inputs)

    from torchsummaryX import summary as Xsummary
    # load WRN-50-2:
    model_50_2 = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=True)
    # or WRN-101-2
    # model_101_2 = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet101_2', pretrained=True)
    summary(model_50_2.to(device), (3, 224, 224), batch_size=16)
    Xsummary(model_50_2.to(device), torch.zeros(16, 3, 224, 224).to(device))
