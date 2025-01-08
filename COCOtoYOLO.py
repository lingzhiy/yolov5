# #COCO 格式的数据集转化为 YOLO 格式的数据集
# #--json_path 输入的json文件路径
# #--save_path 保存的文件夹名字，默认为当前目录下的labels。
#
# import os
# import json
# from tqdm import tqdm
# import argparse
#
# parser = argparse.ArgumentParser()
# #这里根据自己的json文件位置，换成自己的就行
# parser.add_argument('--json_path', default='D:/BaiduNetdiskDownload/COCO2017/annotations_train2017/annotations/instances_train2017.json',type=str, help="input: coco format(json)")
# #这里设置.txt文件保存位置
# parser.add_argument('--save_path', default='D:/BaiduNetdiskDownload/COCO2017/Lable/train2017', type=str, help="specify where to save the output dir of labels")
# arg = parser.parse_args()
#
# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = box[0] + box[2] / 2.0
#     y = box[1] + box[3] / 2.0
#     w = box[2]
#     h = box[3]
# #round函数确定(xmin, ymin, xmax, ymax)的小数位数
#     x = round(x * dw, 6)
#     w = round(w * dw, 6)
#     y = round(y * dh, 6)
#     h = round(h * dh, 6)
#     return (x, y, w, h)
#
# if __name__ == '__main__':
#     json_file =   arg.json_path # COCO Object Instance 类型的标注
#     ana_txt_save_path = arg.save_path  # 保存的路径
#
#     data = json.load(open(json_file, 'r'))
#     if not os.path.exists(ana_txt_save_path):
#         os.makedirs(ana_txt_save_path)
#
#     id_map = {} # coco数据集的id不连续！重新映射一下再输出！
#     with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
#         # 写入classes.txt
#         for i, category in enumerate(data['categories']):
#             f.write(f"{category['name']}\n")
#             id_map[category['id']] = i
#     # print(id_map)
#     #这里需要根据自己的需要，更改写入图像相对路径的文件位置。
#     list_file = open(os.path.join(ana_txt_save_path, 'train2017.txt'), 'w')
#     for img in tqdm(data['images']):
#         filename = img["file_name"]
#         img_width = img["width"]
#         img_height = img["height"]
#         img_id = img["id"]
#         head, tail = os.path.splitext(filename)
#         ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
#         f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
#         for ann in data['annotations']:
#             if ann['image_id'] == img_id:
#                 box = convert((img_width, img_height), ann["bbox"])
#                 f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
#         f_txt.close()
#         #将图片的相对路径写入train2017或val2017的路径
#         list_file.write('./images/train2017/%s.jpg\n' %(head))
#     list_file.close()






































import torch.nn as nn
import torch
# 论文：EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation, CVPR2024
# 论文地址：https://arxiv.org/pdf/2405.06880

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 32, 64, 64)  #B C H W

    block = EUCB(in_channels=32, out_channels=32)

    print(input.size())

    output = block(input)
    print(output.size())