# import the necessary packages
import os

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSAL_INFER = 200

# define the maximum number of positive and negative images to be #최대 양수 및 음수 수
# generated from each image
MAX_POSITIVE = 30 #데이터 세트를 작성 할 때 사용할 양수 및 음수영역
MAX_NEGATIVE = 10

# initialize the input dimensions to the network #wrap up
INPUT_DIMS = (224, 224) # 네트워크 입력차원 초기화 #Line 28 sets the input spatial dimensions to our classification network (MobileNet, pre-trained on ImageNet).

# define the path to the output model and label binarizer #출력모델 경로 지정
MODEL_PATH = "C:/final_project/RCNN/h5" #가중치 저장 루트
ENCODER_PATH = "C:/final_project/RCNN/pickle" #레이블에 대한 출력파일

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99 #정확도 필터 99%정확도 만 출력






