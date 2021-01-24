---
title: CenterLoss
tags: Metric-Learning
---

Yandong Wen, Kaipeng Zhang, Zhifeng Li and Yu Qiao

Shenzhen Key Lab of Comp.Vis.&Pat.Rec., Shenzhen Institutes of Advanced Technology, CAS, China

The Chinese University of Hong Kong, Hong Kong



## Introduction

* 기존의 CNN 에서는 test dataset의 클래스가 training dataset의 클래스에 포함되어 있는 형태였다. 
  (close-set identification) 

* 그러나, 얼굴 인식에서는 이런 형태가 불가능하다. 따라서 separable features 뿐만 아니라 discriminative features 도 추출해야만 한다. 

  ![VGG kernelmagic]({{ "/assets/images/centerloss/features.png" | relative_url}}){: width="50%" height="50%"}{: .center}  

* discriminative feature를 추출하기 위하여 global distribution을 조절해야하지만, CNN optimization 과정에서 SGD를 사용하기 때문에 feature들의 global distribution 을 조절하는 것은 어렵다. 

* 이러한 문제의 대안으로 contrastive loss, triplet loss 가 나왔지만, 학습 데이터가 많아짐에 따라서 training pair 가 극도로 늘어나는 문제점이 발생한다. training pairs가 극도로 많아짐에 따라 computational cost 가 늘어나고 training 과정이 오래걸리는 문제가 발생한다. 

* 저자들은 트레이닝 과정을 진행하는 동안 deep feature의 center를 업데이트 하고 center와 deep feature의 거리를 최소화 하기 위하여, center loss 를 제안한다. center loss 와 기존 softmax loss를  joint training 함으로써 discriminative feature를 얻을 수 있다. 



##  The Proposed Approach



