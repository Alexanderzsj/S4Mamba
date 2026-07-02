import warnings
import torch.nn.functional as F


def get_loss(model_name, model_out, label, args=None):
    if model_name == 'S4Mamba':
        loss_func = torch.nn.CrossEntropyLoss()
        if isinstance(model_out, (tuple, list)):
            logits1, logit2 = model_out
            ls = loss_func(logits1, label) + loss_func(logit2, label) * 0.1
        else:
            ls = loss_func(model_out, label)
            
        return ls
    
    else:
        # 其他基线模型的默认处理
        loss_func = torch.nn.CrossEntropyLoss()
        ls = loss_func(model_out, label)

    return ls


def get_pred(net, model_name, model_type_flag, batch_data, device):
    if model_type_flag == 1:
        X = batch_data.to(device)
        out = net(X)
    else:
        print("model_type_flag error! ERROR in utils/loss.py")
        exit()

    if isinstance(out, dict) and 'logits' in out:
        out = out['logits']
        
    elif isinstance(out, (tuple, list)):
        out = out[-1] 

    pre_y = out.cpu().argmax(axis=1).detach().numpy()
    return pre_y


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def head_loss(loss_func,logits,label,align_corners=True):
    seg_logits = resize(
        input=logits,
        size=label.shape[1:],
        mode='bilinear',
        align_corners=align_corners)

    loss = loss_func(seg_logits,label)
    return loss
