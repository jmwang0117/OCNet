import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class MultiScaleSparseSelfAttention(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_heads=2, dropout=0.0):
        super(MultiScaleSparseSelfAttention, self).__init__()
        self.attention_layers = nn.ModuleList([
            SparseSelfAttention(in_channels, out_channels, num_heads=num_heads, dropout=dropout)
            for in_channels in in_channels_list])

    def forward(self, features_list):
        batch_size, _, height, width = features_list[0].size()
        num_points = height * width
        print("Features list shapes before attention:")
        for feat in features_list:
            print(feat.shape)
        # Project all features to query, key and value.
        query_key_value_list = [
            (att_layer.query_projection(feat.view(batch_size, att_layer.in_channels, -1).permute(0, 2, 1)), 
             att_layer.key_projection(feat.view(batch_size, att_layer.in_channels, -1).permute(0, 2, 1)), 
             att_layer.value_projection(feat.view(batch_size, att_layer.in_channels, -1).permute(0, 2, 1)))
            for att_layer, feat in zip(self.attention_layers, features_list)]

        attended_value_sum = torch.zeros(batch_size, self.attention_layers[0].num_heads, num_points,
                                         self.attention_layers[0].out_channels).to(features_list[0].device)

        local_area_size = (2 * self.attention_layers[0].local_radius + 1) ** 2

        # Compute the local attention scores and attended values using matrix multiplication.
        for idx, (query, key, value) in enumerate(query_key_value_list):
            
            query = query.view(batch_size, self.attention_layers[0].num_heads, num_points, -1)
            key = key.view(batch_size, self.attention_layers[idx].num_heads, num_points, -1)

            value = value.view(batch_size, self.attention_layers[idx].num_heads, num_points, -1)


            attn_scores = torch.matmul(query.permute(0, 2, 1, 3), key.permute(0, 2, 3, 1)) / self.attention_layers[idx].in_channels ** 0.5


            attn_scores = attn_scores.view(batch_size, self.attention_layers[idx].num_heads, num_points,
                                       local_area_size)

            
            attn_scores = F.softmax(attn_scores, dim=-1)
            attn_scores = F.dropout(attn_scores, p=self.attention_layers[0].dropout, training=self.training)

            attended_value = torch.matmul(attn_scores.view(batch_size, -1, local_area_size),
                                          value.view(batch_size, -1).transpose(-2, -1)).view(batch_size, self.attention_layers[idx].num_heads, num_points, -1)
            attended_value_sum += attended_value

        attended_value = attended_value_sum.contiguous().view(batch_size, num_points, -1)
         # 在这里添加 print 语句
        print("Attended features shape after attention:", attended_value.shape)

        return attended_value
        return attended_value







class SparseSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.0, local_radius=2):
        super(SparseSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.local_radius = local_radius

        self.query_projection = nn.Linear(in_channels, out_channels)
        self.key_projection = nn.Linear(in_channels, out_channels)
        self.value_projection = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        num_points = height * width
        x = x.view(batch_size, num_points, -1)

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = query.view(batch_size, num_points, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, num_points, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, num_points, self.num_heads, -1).transpose(1, 2)

        attended_value_sum = torch.zeros_like(value)

        # Compute the local attention scores and attended values in a single loop.
        for h in range(-self.local_radius, self.local_radius + 1):
            for w in range(-self.local_radius, self.local_radius + 1):
                rolled_key = key.roll(shifts=(h, w), dims=(-3, -2))
                attn_scores_hw = torch.matmul(query, rolled_key.transpose(-2, -1)) / (self.out_channels ** 0.5)
                attn_scores_hw = F.softmax(attn_scores_hw, dim=-1)
                attn_scores_hw = F.dropout(attn_scores_hw, p=self.dropout, training=self.training)

                rolled_value = value.roll(shifts=(-h, -w), dims=(-3, -2))
                attended_value_hw = torch.matmul(attn_scores_hw, rolled_value)

                attended_value_sum += attended_value_hw

        attended_value = attended_value_sum.transpose(1, 2).contiguous().view(batch_size, num_points, -1)

        return attended_value



      
      
class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
      
class OCNet(nn.Module):

  def __init__(self, class_num, input_dimensions, class_frequencies):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''

    super().__init__()
    self.nbr_classes = class_num
    self.input_dimensions = input_dimensions  # Grid dimensions should be (W, H, D).. z or height being axis 1
    self.class_frequencies = class_frequencies
    f = self.input_dimensions[1]
    # 在 OCNet 类中添加多尺度稀疏自注意力层
    self.multi_scale_attention_1_1 = MultiScaleSparseSelfAttention([f, int(f/4), int(f/8)], f)
    self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

    self.Encoder_block1 = nn.Sequential(
      DepthwiseSeparableConv(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      DepthwiseSeparableConv(f, f, kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block2 = nn.Sequential(
      nn.MaxPool2d(2),
      DepthwiseSeparableConv(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      DepthwiseSeparableConv(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )

    self.Encoder_block3 = nn.Sequential(
      nn.MaxPool2d(2),
      DepthwiseSeparableConv(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      DepthwiseSeparableConv(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
        
    )

    self.Encoder_block4 = nn.Sequential(
      nn.MaxPool2d(2),
      DepthwiseSeparableConv(int(f * 2), int(f * 2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU(),
      DepthwiseSeparableConv(int(f * 2.5), int(f * 2.5), kernel_size=3, padding=1, stride=1),
      nn.ReLU()
    )
    
    # # 添加稀疏自注意力层
    # self.sparse_self_attention = SparseSelfAttention(in_channels=int(f * 2.5), out_channels=int(f * 2.5),
    #                                                     num_heads=4, dropout=0.0)
    # Treatment output 1:8
    self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
    self.seg_head_1_8       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
    self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
    self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

    # Treatment output 1:4
    self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
    self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
    self.seg_head_1_4       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
    self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

    # Treatment output 1:2
    self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
    self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
    self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)
    self.seg_head_1_2       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

    # Treatment output 1:1
    self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
    self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)
    self.seg_head_1_1       = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

  def forward(self, x):
    
    input = x['3D_OCCUPANCY']  # Input to LMSCNet model is 3D occupancy big scale (1:1) [bs, 1, W, H, D]
    input = torch.squeeze(input, dim=1).permute(0, 2, 1, 3)  # Reshaping to the right way for 2D convs [bs, H, W, D]

    # Encoder block
    _skip_1_1 = self.Encoder_block1(input)
    _skip_1_2 = self.Encoder_block2(_skip_1_1)
    _skip_1_4 = self.Encoder_block3(_skip_1_2)
    _skip_1_8 = self.Encoder_block4(_skip_1_4)

    # Out 1_8
    out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)
    out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D)

    # Out 1_4
    out = self.deconv1_8(out_scale_1_8__2D)
    out = torch.cat((out, _skip_1_4), 1)
    out = F.relu(self.conv1_4(out))
    out_scale_1_4__2D = self.conv_out_scale_1_4(out)
    out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D)

    # Out 1_2
    out = self.deconv1_4(out_scale_1_4__2D)
    out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
    out = F.relu(self.conv1_2(out))
    out_scale_1_2__2D = self.conv_out_scale_1_2(out)
    out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D)
    
    # 修改前向传播部分，以包含多尺度稀疏自注意力层
    attended_features_1_1 = self.multi_scale_attention_1_1([_skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D),
                                                                 self.deconv_1_8__1_1(out_scale_1_8__2D)])

    print("Attended features shape after attention:", attended_value.shape)
    # Out 1_1
    out = self.deconv1_2(out_scale_1_2__2D)
    out = torch.cat((out, attended_features_1_1), 1)  # 使用融合后的特征替换原始特征连接
    #out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
    out_scale_1_1__2D = F.relu(self.conv1_1(out))
    out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)

    # Take back to [W, H, D] axis order
    out_scale_1_8__3D = out_scale_1_8__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    out_scale_1_4__3D = out_scale_1_4__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    out_scale_1_2__3D = out_scale_1_2__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]
    out_scale_1_1__3D = out_scale_1_1__3D.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

    scores = {'pred_semantic_1_1': out_scale_1_1__3D, 'pred_semantic_1_2': out_scale_1_2__3D,
              'pred_semantic_1_4': out_scale_1_4__3D, 'pred_semantic_1_8': out_scale_1_8__3D}

    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    '''

    target = data['3D_LABEL']['1_1']
    device, dtype = target.device, target.dtype
    class_weights = self.get_class_weights().to(device=target.device, dtype=target.dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='mean').to(device=device)

    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL']['1_1'].long())
    loss_1_2 = criterion(scores['pred_semantic_1_2'], data['3D_LABEL']['1_2'].long())
    loss_1_4 = criterion(scores['pred_semantic_1_4'], data['3D_LABEL']['1_4'].long())
    loss_1_8 = criterion(scores['pred_semantic_1_8'], data['3D_LABEL']['1_8'].long())

    loss_total = (loss_1_1 + loss_1_2 + loss_1_4 + loss_1_8) / 4

    loss = {'total': loss_total, 'semantic_1_1': loss_1_1, 'semantic_1_2': loss_1_2, 'semantic_1_4': loss_1_4,
            'semantic_1_8': loss_1_8}

    return loss

  def get_class_weights(self):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(self.class_frequencies + epsilon_w))

    return weights

  def get_target(self, data):
    '''
    Return the target to use for evaluation of the model
    '''
    return {'1_1': data['3D_LABEL']['1_1'], '1_2': data['3D_LABEL']['1_2'],
            '1_4': data['3D_LABEL']['1_4'], '1_8': data['3D_LABEL']['1_8']}
    # return data['3D_LABEL']['1_1'] #.permute(0, 2, 1, 3)

  def get_scales(self):
    '''
    Return scales needed to train the model
    '''
    scales = ['1_1', '1_2', '1_4', '1_8']
    return scales

  def get_validation_loss_keys(self):
    return ['total', 'semantic_1_1','semantic_1_2', 'semantic_1_4', 'semantic_1_8']

  def get_train_loss_keys(self):
    return ['total', 'semantic_1_1','semantic_1_2', 'semantic_1_4', 'semantic_1_8']
