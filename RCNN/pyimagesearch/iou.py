def compute_iou(boxA, boxB):
    		# determine the (x, y)-coordinates of the intersection rectangle #교집합 직사각형 사각형의 X Y좌표
	xA = max(boxA[0], boxB[0]) # 오른쪽 상단 (x, y)
	yA = max(boxA[1], boxB[1]) # 왼쪽 하단 (x, y)
	xB = min(boxA[2], boxB[2]) # (x, y)
	yB = min(boxA[3], boxB[3]) # (x, y)

	# compute the area of intersection rectangle #사각형 면적계산
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the intersection area

	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou










'''
compute_iou 함수는 상자 A와 상자 B의 두 가지 매개 변수를 허용하는데, 
    
이는 우리가 IoU(Intersection over Union)를 계산하려고 하는 지상 실측 및 예측 경계 상자이다. 

매개 변수의 순서는 계산 목적에 상관 없습니다.

내부에서는 경계 상자(라인 3-6)의 오른쪽 상단과 왼쪽 하단(x, y) 좌표를 모두 계산하는 것으로 시작한다.

경계 상자 좌표를 사용하여 경계 상자(라인 9)의 교차점(중첩 영역)을 계산한다. 이 값은 IoT 포럼의 숫자입니다.

분모를 결정하려면 예측 및 지상 실측 경계 상자(라인 13과 14)의 영역을 도출해야 한다.

그런 다음, 교차 구역(숫자)을 두 개의 경계 상자(분모)의 결합 구역으로 나누고 교차 

구역(그렇지 않으면 교차 구역이 두 배로 계산됨)을 빼서 19호선에서 계산할 수 있다.

라인 22는 IOU 결과를 반환합니다.
'''