# Introduction

Faster R-CNN은 두개의 네트워크로 구성이 되어 있습니다. 

 - Deep Convolution Network로서 Region Proposal Network (RPN) 이라고 함
 - Fast R-CNN Detector로서 앞의 proposed regions을 사용하여 object를 감지함
 

 > <span style="color:#777777"> 기존 Fast R-CNN과의 차이점은 Fast R-CNN의 경우 `selective search`를 사용하여 region proposals를 생성합니다. <br>Faster R-CNN의 경우 따로 region proposal method (`selective search`) 를 사용하지 않고, convnet에서 나온 feature maps을 input으로 갖는 region proposal network를 사용하기 때문에 더 빠릅니다. </span>
 
 
Faster R-CNN안에는 2개의 모듈이 존재하지만, 전체적으로는 하나의 object detection network라고 볼 수 있습니다.
이게 중요한 이유는 Fater R-CNN이후부터 fully differentiable model이기 때문입니다.

![Faster R-CNN](images/faster-rcnn.png)

# Architecture

가장 상위단서부터 큰 그림을 그리면서 세부적은 부분을 설명하도록 하겠습니다. 

# Input Images 

Input images는 **Height x Width x Depth** 를 갖고 있는 tensors입니다. 


## Region Proposal Networks


Region Proposal Network (RPN)은 이미지를 (싸이즈 상관 없음) input 으로 받습니다. 
이후 output으로 직사각형의 object 

