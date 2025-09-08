import torch
import numpy as np
from torchvision import transforms
from datetime import datetime, timezone, timedelta


def ImageStretching(image):
    channels = image.shape[2]
    band_list = []
    for i in range(channels):
        band_data = image[:,:,i]
        band_min = np.percentile(band_data,2)
        band_max = np.percentile(band_data,98)
        band_data = (band_data - band_min) / (band_max - band_min)
        # plt.imshow(band_data)
        # plt.show()
        band_list.append(band_data)
    image_data = np.stack(band_list, axis=-1)
    image_data = np.clip(image_data, 0, 1)
    image_data = (image_data * 255).astype(np.uint8)
    image_data = np.uint8(image_data)
    return image_data


def normlize3D(image,use_group=False,group_num=4):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    size = image.shape
    if size[2]!=3:
        image_norms = []

        for i in range(size[2]):
            image_slice3 = image[:,:,i,:,:]
            image_slice_norm = transform(image_slice3)
            image_norms.append(image_slice_norm.unsqueeze(2))
        image_norms = torch.cat(image_norms,dim=2)
        if use_group:
            image_norms = image_norms.unsqueeze(0)
            grouped_channels = []
            for start_channel in range(0,group_num):
                grouped_channels.append(np.arange(start_channel,(image_norms.shape[2]//group_num)*group_num,group_num))
            grouped_img = torch.cat([image_norms[:, :, channels, :, :] for channels in grouped_channels], dim=0)
            return grouped_img.cuda()
        else:
            return image_norms


def beijing_time() -> str:
    """
    获取北京时间字符串，格式为'%Y-%m-%d,%H:%M:%S'
    """
    # 创建UTC+8时区
    beijing_tz = timezone(timedelta(hours=8))
    
    # 获取当前时间，并将其转换为北京时间
    dt = datetime.now(beijing_tz)
    
    # 返回格式化后的北京时间字符串
    return dt.strftime('%Y-%m-%d,%H:%M:%S')