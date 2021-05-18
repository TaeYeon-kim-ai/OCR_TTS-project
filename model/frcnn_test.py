# 패키지 Import, custom utils Load
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import *

# Sample Data, Feature Extraction
image = torch.zeros((1, 3, 800, 800)).float()
image_size = (800, 800)

# bbox -> y1, x1, y2, x2
bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])
labels = torch.LongTensor([6, 8])

sub_sample = 16

vgg16 = torchvision.models.vgg16(pretrained=True)
req_features = vgg16.features[:30]
print(req_features)
output_map = req_features(image)
print(output_map.shape)

'''
샘플 데이터로 800x800 이미지를 사용했고, bounding box는 2개, labels도 2개입니다.

sub_sample은 down sampling이 얼마나 되는지, 800에서 50으로 줄어듦으로 16입니다. (down sample 4번 2**4)
'''

# Anchor 생성

anchor_scale = [8, 16, 32]
ratio = [0.5, 1, 2] # H/W

len_anchor_scale = len(anchor_scale)
len_ratio = len(ratio)
len_anchor_template = len_anchor_scale * len_ratio
anchor_template = np.zeros((9, 4))

for idx, scale in enumerate(anchor_scale):
    h = scale * np.sqrt(ratio) * sub_sample
    w = scale / np.sqrt(ratio) * sub_sample
    y1 = -h/2
    x1 = -w/2
    y2 = h/2
    x2 = w/2
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 0] = y1
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 1] = x1
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 2] = y2
    anchor_template[idx*len_ratio:(idx+1)*len_ratio, 3] = x2

print(anchor_template)

'''
Anchor scale과 ratio로 box를 판단할 모양을 만들어줍니다. (anchor_template)

anchor 하나당 scale 3, ratio 3으로 총 9개입니다.
'''

#template 도식입니다. 크기별로, 모양별로 총 9가지가 나오게 됩니다.
feature_map_size = (50, 50)
# The first center coors is (8, 8)
ctr_y = np.arange(8, 800, 16)
ctr_x = np.arange(8, 800, 16)

ctr = np.zeros((*feature_map_size, 2))
for idx, y in enumerate(ctr_y):
    ctr[idx, :, 0] = y
    ctr[idx, :, 1] = ctr_x
print(ctr.shape)

#anchor_template이 위치할 점들을 가져옵니다.

#위의 점들을 도식하면 아래와 같이 50x50개의 anchor center들이 생성됩니다.

anchors = np.zeros((*feature_map_size, 9, 4))

for idx_y in range(feature_map_size[0]):
    for idx_x in range(feature_map_size[1]):
        anchors[idx_y, idx_x] = (ctr[idx_y, idx_x] + anchor_template.reshape(-1, 2, 2)).reshape(-1, 4)

anchors = anchors.reshape(-1, 4)
print(anchors.shape) # (22500, 4)

#이 점들에 각각 anchor template을 적용해줍니다.
# 파란색 박스가 800x800 이미지고,

# 각각의 anchor별로 template을 적용했을 때, 나오는 22500개 (50x50x9)의 anchor box들입니다.

# anchor box labeling for RPN
valid_index = np.where((anchors[:, 0] >= 0)
                      &(anchors[:, 1] >= 0)
                      &(anchors[:, 2] <= 800)
                      &(anchors[:, 3] <= 800))[0]
print(valid_index.shape) # 8940
#이미지를 넘어가는 박스는 사실상 사용을 못하는 것이기 때문에, 제외해줍니다.

valid_labels = np.empty((valid_index.shape[0],), dtype=np.int32)
valid_labels.fill(-1)

valid_anchors = anchors[valid_index]

print(valid_anchors.shape) # (8940,4)
print(bbox.shape) # torch.Size([2,4])

'''
RPN에서는 Class는 관심없고, Object가 있는 곳을 Region Proposal하는 것이 중요하므로, label은 1, 0이 됩니다.

또한, 8940개의 proposal 중 겹치는 것도 있고, 불필요한 것들도 있으므로 이를 제거하기 위해

-1이라는 label을 할당해줍니다.
'''

ious = bbox_iou(valid_anchors, bbox.numpy()) # anchor 8940 : bbox 2

pos_iou_thres = 0.7
neg_iou_thred = 0.3

# Scenario A
anchor_max_iou = np.amax(ious, axis=1)
pos_iou_anchor_label = np.where(anchor_max_iou >= pos_iou_thres)[0]
neg_iou_anchor_label = np.where(anchor_max_iou < neg_iou_thred)[0]
valid_labels[pos_iou_anchor_label] = 1
valid_labels[neg_iou_anchor_label] = 0

# Scenario B
gt_max_iou = np.amax(ious, axis=0)
gt_max_iou_anchor_label = np.where(ious == gt_max_iou)[0]
print(gt_max_iou_anchor_label)
valid_labels[gt_max_iou_anchor_label] = 1

'''
valid anchor와 bbox 와의 iou를 계산해줍니다.

ious는 행렬이 되는데 행은 각 anchor가 되고 열은 bbox입니다.

8940, 2의 shape로 Anchor별 bbox와의 iou 값들이 들어가게 됩니다.

이 값을 통해 0.7 이상이면 positive, 0.3보다 작으면 negative를 레이블하게 됩니다.

대부분의 경우 0.7이상인 경우가 많지 않기 때문에, Scenario A에서는 논문대로 하고,

Scenario B에서 박스별 iou 최대값인 애들로 positive로 레이블합니다.

'''
n_sample_anchors = 256
pos_ratio = 0.5

total_n_pos = len(np.where(valid_labels == 1)[0])
n_pos_sample = n_sample_anchors*pos_ratio if total_n_pos > n_sample_anchors*pos_ratio else total_n_pos
n_neg_sample = n_sample_anchors - n_pos_sample

pos_index = np.where(valid_labels == 1)[0]
if len(pos_index) > n_sample_anchors*pos_ratio:
    disable_index = np.random.choice(pos_index, size=len(pos_index)-n_pos_sample, replace=False)
    valid_labels[disable_index] = -1

neg_index = np.where(valid_labels == 0)[0]
disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg_sample, replace=False)
valid_labels[disable_index] = -1

# 그리고나서 positive와 negative를 합해 256개만 남기고 제외합니다.

# postive가 128개가 되지 않는 경우, 나머지는 negative로 채우게 됩니다.

# Each anchor corresponds to a box

argmax_iou = np.argmax(ious, axis=1)
max_iou_box = bbox[argmax_iou].numpy()
print(max_iou_box.shape) # 8940, 4
print(valid_anchors.shape) # 8940, 4

anchor_loc_format_target = format_loc(valid_anchors, max_iou_box)
print(anchor_loc_format_target.shape) # 8940, 4

'''
위의 코드를 보면, ious에서 Anchor별로 어떤 박스가 iou가 높은지 확인합니다.

(0.37312, 0.38272) 이면 1, (0.38272, 0.37312) 이면 0

이렇게하면, 1 0 1 0 0 0 0 1 0, ... 이라는 8940개의 배열이 생기게 됩니다.

이 index로 box값들을 하나하나 할당해서 8940, 4의 배열을 만듭니다.

 

그리고나서, utils에 있는(직접 만든) format_loc함수로 anchor box에 location을 할당해줍니다.

(정확히 이해는 못했는데, Regression을 해준다는 의미인 것 같습니다.)
'''

anchor_target_labels = np.empty((len(anchors),), dtype=np.int32)
anchor_target_format_locations = np.zeros((len(anchors), 4), dtype=np.float32)

anchor_target_labels.fill(-1)
anchor_target_labels[valid_index] = valid_labels

anchor_target_format_locations[valid_index] = anchor_loc_format_target

print(anchor_target_labels.shape) # 22500,
print(anchor_target_format_locations.shape) # 22500, 4
#이렇게 하면, 최종적으로 label하고, loc을 계산한 앵커들이 나오게 됩니다.

#=============================================================================
#RPN
#위의 과정이 RPN을 위한 사전 작업이라고 볼 수 있습니다.

mid_channel = 512
in_channel = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channel, mid_channel, 3, 1, 1)
reg_layer = nn.Conv2d(mid_channel, n_anchor*4, 1, 1, 0)
cls_layer = nn.Conv2d(mid_channel, n_anchor*2, 1, 1, 0)
'''
VGG Net 기준 conv layer의 output channel이 512이라서 in channel은 512가 되고,
box regression은 anchor 9 * 4 (location)
box classification은 anchor 9 * 2 (object or not)
이 됩니다.
'''
#============================================================================

x = conv1(output_map)
anchor_pred_format_locations = reg_layer(x)
anchor_pred_scores = cls_layer(x)

print(anchor_pred_format_locations.shape) # torch.Size([1, 36, 50, 50])
print(anchor_pred_scores.shape) # torch.Size([1, 18, 50, 50])

'''
weight 초기화를 거쳐, 추출한 feature map을 conv에 통과시키고,

location과 class를 예측합니다.

이렇게 되면 각 위치별 (50, 50) regression 과 classification 예측 값이 나오게 됩니다.
'''

#============================================================================
anchor_pred_format_locations = anchor_pred_format_locations.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
anchor_pred_scores = anchor_pred_scores.permute(0, 2, 3, 1).contiguous().view(1, -1, 2)
objectness_pred_scores = anchor_pred_scores[:, :, 1]
#위에서 ground truth로 만든 anchor와 비교하기 위해, 형태를 맞춰줍니다.
#============================================================================

print(anchor_target_labels.shape)
print(anchor_target_format_locations.shape)
print(anchor_pred_scores.shape)
print(anchor_pred_format_locations.shape)

gt_rpn_format_locs = torch.from_numpy(anchor_target_format_locations)
gt_rpn_scores = torch.from_numpy(anchor_target_labels)

rpn_format_locs = anchor_pred_format_locations[0]
rpn_scores = anchor_pred_scores[0]

'''
target은 bbox를 통해 만든 ground truth 값들, pred는 RPN으로 예측한 값들입니다.
둘 모두 reg 22500, 4 / cls 22500, 1이 됩니다.
numpy에서 torch로 변환해주고, batch 중 1개만 가져와주는 코드 (여기선 batch가 1)
'''
#============================================================================
####### Object or not loss
rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_scores.long(), ignore_index=-1)
print(rpn_cls_loss)


####### location loss
mask = gt_rpn_scores > 0
mask_target_format_locs = gt_rpn_format_locs[mask]
mask_pred_format_locs = rpn_format_locs[mask]

print(mask_target_format_locs.shape)
print(mask_pred_format_locs.shape)

x = torch.abs(mask_target_format_locs - mask_pred_format_locs)
rpn_loc_loss = ((x<0.5).float()*(x**2)*0.5 + (x>0.5).float()*(x-0.5)).sum()
print(rpn_loc_loss)
#object인지 아닌지는 cross entropy loss를, location은 실제 object인 것만 masking해서 loss 값을 계산합니다.

#============================================================================
rpn_lambda = 10
N_reg = mask.float().sum()

rpn_loss = rpn_cls_loss + rpn_lambda / N_reg * rpn_loc_loss
print(rpn_loss)
#cls loss와 loc loss는 lambda로 적절하게 합쳐주게 됩니다.
#============================================================================

# Generating Proposal to Feed Fast R-CNN
#RPN에서 구한 Proposal을 Fast R-CNN에서 학습할것만 남기는 과정입니다.

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16
'''
non-maximum suppression (NMS)으로 같은 클래스 정보를 가지는 박스들끼리 iou값을 비교해, 중복되는 것들은 제외해줍니다. 이 때 threshold가 nms_thresh 0.7입니다.

nms이전에 12000개만 우선적으로 남기게 되고,

nms를 하면 2000개의 최종 proposal만 남습니다.

이 2000개의 proposal로 Fast RCNN을 학습하게 됩니다.

box의 width 와 height가 16보다 작으면, 해당 proposal도 제외합니다.

Test에선 6000개, 300개만 남기게 됩니다. (현재 코드에선 사용 X)
'''
#============================================================================
print(anchors.shape) # 22500, 4
print(anchor_pred_format_locations.shape) # 22500, 4

rois = deformat_loc(anchors=anchors, formatted_base_anchor=anchor_pred_format_locations[0].data.numpy())
print(rois.shape) # 22500, 4

print(rois)
#[[ -37.56205856  -83.65124834   55.51502551   96.9647187 ]
# [ -59.50866938  -56.68875009   64.91222143   72.23375052]
# [ -81.40298363  -41.99777969   96.39533509   49.35743635]
# ...
# [ 610.35422226  414.3952291   979.0893042  1163.98340092]
# [ 538.20066833  564.81064224 1041.29725647 1063.15491104]
# [ 432.48094419  606.7697889  1166.24708388  973.39356325]]
#이 부분은 좀 헷갈리는데, 예측한 location을 anchors를 통해 rois로 다시 바꿔줍니다. (bounding box)

#============================================================================
rois[:, 0:4:2] = np.clip(rois[:, 0:4:2], a_min=0, a_max=image_size[0])
rois[:, 1:4:2] = np.clip(rois[:, 1:4:2], a_min=0, a_max=image_size[1])
print(rois)

# [[  0.           0.          55.51502551  96.9647187 ]
#  [  0.           0.          64.91222143  72.23375052]
#  [  0.           0.          96.39533509  49.35743635]
#  ...
#  [610.35422226 414.3952291  800.         800.        ]
#  [538.20066833 564.81064224 800.         800.        ]
#  [432.48094419 606.7697889  800.         800.        ]]

#그리고 이미지 사이즈를 벗어나는 값들은 이미지 크기에 맞게 조정해줍니다.

#============================================================================
h = rois[:, 2] - rois[:, 0]
w = rois[:, 3] - rois[:, 1]

valid_index = np.where((h>min_size)&(w>min_size))[0]
valid_rois = rois[valid_index]
valid_scores = objectness_pred_scores[0][valid_index].data.numpy()
#그리고 box크기가 16보다 작은 것들은 제외하고
#object score를 기준으로 정렬해줍니다.

#============================================================================

valid_score_order = valid_scores.ravel().argsort()[::-1]

pre_train_valid_score_order = valid_score_order[:n_train_pre_nms]
pre_train_valid_rois = valid_rois[pre_train_valid_score_order]
pre_train_valid_scores = valid_scores[pre_train_valid_score_order]

print(pre_train_valid_rois.shape) # 12000, 4
print(pre_train_valid_scores.shape) # 12000,
print(pre_train_valid_score_order.shape) # 12000,

#nms를 적용하기 전 12000개만 가져오고

#============================================================================

keep_index = nms(rois=pre_train_valid_rois, scores=pre_train_valid_scores, nms_thresh=nms_thresh)
post_train_valid_rois = pre_train_valid_rois[keep_index][:n_train_post_nms]
post_train_valid_scores = pre_train_valid_scores[keep_index][:n_train_post_nms]
print(post_train_valid_rois.shape) # 2000, 4
print(post_train_valid_scores.shape) # 2000, 

#nms를 적용해 2000개의 roi만 남깁니다.
#2000개도 생각보다 많습니다.

#============================================================================
# anchor box labeling for Fast R-CNN

n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

'''
여기서부턴 RPN에서 ground truth를 만드는 과정과 같습니다.

단지 Fast RCNN을 위한 ground truth를 만드는 것이 차이 (실제 클래스, 실제 bounding box)
'''

#============================================================================
ious = bbox_iou(post_train_valid_rois, bbox)
print(ious.shape) # 2000, 2
#위에서 구한 2000개의 roi와 bbox를 비교해 iou를 계산해줍니다.

#============================================================================
#RPN에선 8940, 2였는데 2000개만 비교해주면 되니, 2000, 2의 배열이 만들어지게 됩니다.
bbox_assignments = ious.argmax(axis=1)
roi_max_ious = ious.max(axis=1)
roi_target_labels = labels[bbox_assignments]
print(roi_target_labels.shape) # 2000
'''
여기선 anchor에서 큰 값인 애들을 실제 label (6, 8)로 각각 할당해주게 됩니다.
0번째가 크면 6 1번째가 크면 8입니다. (헷갈리신다면 코드 구현 맨 위에서 box label값을 확인해보세요)
6 8 6 6 8 6 6 6과 같은 형태의 배열이 만들어지게 되는데
이게 전부 target일수가 없겠죠?
'''
#============================================================================
total_n_pos = len(np.where(roi_max_ious >= pos_iou_thresh)[0])
n_pos_sample = n_sample*pos_ratio if total_n_pos > n_sample*pos_ratio else total_n_pos
n_neg_sample = n_sample - n_pos_sample

print(n_pos_sample) # 10
print(n_neg_sample) # 118
#그래서 positive threshold에 따라 positive인 애들과 negative인 애들을 128개만 sampling합니다. (n_sample)
#============================================================================

pos_index = np.where(roi_max_ious >= pos_iou_thresh)[0]
pos_index = np.random.choice(pos_index, size=n_pos_sample, replace=False)

neg_index = np.where((roi_max_ious < neg_iou_thresh_hi) & (roi_max_ious > neg_iou_thresh_lo))[0]
neg_index = np.random.choice(neg_index, size=n_neg_sample, replace=False)

print(pos_index.shape) # 10
print(neg_index.shape) # 118

#============================================================================

keep_index = np.append(pos_index, neg_index)
post_sample_target_labels = roi_target_labels[keep_index].data.numpy()
post_sample_target_labels[len(pos_index):] = 0
post_sample_rois = post_train_valid_rois[keep_index]

#최종적으로 sampling까지 끝낸 roi들만 남기게 됩니다.
'''
그 중에 positive만 뽑아보면 위의 그래프와 같습니다.
왼쪽 아래 박스가 라벨 6이고, 오른쪽 위 박스가 라벨 8이니
초록색은 6라벨을 위한 roi box가 되고, 빨간색은 라벨 8을 위한 roi box가 됩니다.
'''
#============================================================================
post_sample_bbox = bbox[bbox_assignments[keep_index]]
post_sample_format_rois = format_loc(anchors=post_sample_rois, base_anchors=post_sample_bbox.data.numpy())
print(post_sample_format_rois.shape)
#이를 Fast R-CNN과 비교하기 위한 loc 형태로 변환해주면 target box도 끝

#============================================================================
# Fast R-CNN
rois = torch.from_numpy(post_sample_rois).float()
print(rois.shape) # 128, 4
# roi_indices = torch.zeros((len(rois),1), dtype=torch.float32)
# print(rois.shape, roi_indices.shape)

# indices_and_rois = torch.cat([roi_indices, rois], dim=1)
# print(indices_and_rois.shape)
'''
roi를 torch로 변환해주고,
밑에 주석처리된 코드는 batch별로 계산해주기 위해
batch별 index를 할당해주고 인덱스, roi로 배열을 만들어주는 코드입니다 (여기선 batch 1)
'''
#============================================================================
#RoI Pooling
size = (7, 7)
adaptive_max_pool = nn.AdaptiveMaxPool2d(size)

# correspond to feature map
rois.mul_(1/16.0)
rois = rois.long()

#roi pooling을 통해 고정된 크기로 추출합니다.
#그리고 128개의 rois들은 각각 50,50의 공간에 매핑됩니다.
#============================================================================

output = []
num_rois = len(rois)
for roi in rois:
    roi_feature = output_map[..., roi[0]:roi[2]+1, roi[1]:roi[3]+1]
    output.append(adaptive_max_pool(roi_feature))
output = torch.cat(output, 0)
print(output.shape) # 128, 512, 7, 7
# 각각의 roi를 pooling layer를 거쳐, 고정된 크기로 추출해주면 128, 512, 7, 7의 결과가 나오게 됩니다.
# 이미지 크기에 자유롭기 위해 roi pooling layer를 사용해준 모습이고,

#============================================================================
output_ROI_pooling = output.view(output.size(0), -1)
print(output_ROI_pooling.shape) # 128, 25088

#이를 일자로 펴주게되면 128, 25088의 배열이 나오게 됩니다.

#============================================================================

#RoI Head & Classifier, BBox Regression
roi_head = nn.Sequential(nn.Linear(25088, 4096),
                        nn.Linear(4096, 4096))

cls_loc = nn.Linear(4096, 21*4)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()

cls_score = nn.Linear(4096, 21)
cls_score.weight.data.normal_(0, 0.01)
cls_score.bias.data.zero_()

x = roi_head(output_ROI_pooling)
roi_cls_loc = cls_loc(x)
roi_cls_score = cls_score(x)

print(roi_cls_loc.shape, roi_cls_score.shape) # 128, 84 / 128, 21
# 최종적으로 fully connected layer를 거쳐, 20 (class) + 1 (background)로 분류하게 됩니다.
# location은 *4 (x1,y1,x2,y2)
#============================================================================
#Fast R-CNN Loss
print(roi_cls_loc.shape) # 128, 84
print(roi_cls_score.shape) # 128, 21
#예측값
#============================================================================
print(post_sample_format_rois.shape) # 128, 4
print(post_sample_target_labels.shape) # 128, 
gt_roi_cls_loc = torch.from_numpy(post_sample_format_rois).float()
gt_roi_cls_label = torch.from_numpy(post_sample_target_labels).long()
#실제값 ground truth입니다.
#============================================================================
roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_cls_label)
print(roi_cls_loss)
#cls loss는 cross entropy loss를
#============================================================================
num_roi = roi_cls_loc.size(0)
roi_cls_loc = roi_cls_loc.view(-1, 21, 4)
roi_cls_loc = roi_cls_loc[torch.arange(num_roi), gt_roi_cls_label]
print(roi_cls_loc.shape)

mask = gt_roi_cls_label>0
mask_loc_pred = roi_cls_loc[mask]
mask_loc_target = gt_roi_cls_loc[mask]

print(mask_loc_pred.shape) # 10, 4
print(mask_loc_target.shape) # 10, 4

x = torch.abs(mask_loc_pred-mask_loc_target)
roi_loc_loss = ((x<0.5).float()*x**2*0.5 + (x>0.5).float()*(x-0.5)).sum()
print(roi_loc_loss)
#Fast R-CNN도 마찬가지로 masking처리해서 label이 background가 아닌 것들만 bounding box regression하게 됩니다.
#============================================================================
roi_lambda = 10
N_reg = (gt_roi_cls_label>0).float().sum()
roi_loss = roi_cls_loss + roi_lambda / N_reg * roi_loc_loss
print(roi_loss)
#lambda를 적용해 Fast R-CNN의 Total loss를 구할 수 있습니다.

#============================================================================
# Faster R-CNN Total Loss
total_loss = rpn_loss + roi_loss

'''
Faster R-CNN의 Total loss는 rpn_loss와 roi_loss를 합친 값이 됩니다.
이 loss를 backward해서 network를 update하면 됩니다.
이 과정을 모듈화하는게 헬포인트
'''




