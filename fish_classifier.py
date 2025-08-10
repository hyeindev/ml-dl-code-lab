import sklearn.neighbors
import fish_datas

length = fish_datas.bream_length + fish_datas.smelt_length # 각각 35개, 14개
weight = fish_datas.bream_weight + fish_datas.smelt_weight # 각각 35개, 14개

# 2차원 리스트 생성 : 사이킷런이 요구하는 형태로 변형 
fish_data = [[l,w] for l,w in zip(length,weight)]

fish_target = [1] * 35 + [0] * 14

# install scikit-learn - KNeighborsClassifier : 최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
# print(type(model))

# fish_data, fish_target 로 알고리즘 훈련 
model.fit(fish_data,fish_target) # 다른 알고리즘 훈련에서도 fit() 함수를 사용한다
# kn 훈련 평가
score = model.score(fish_data,fish_target) 

print(score) # 1.0 => 훈련 정확도 100% 0~1.0 까지 출력됨 

# predict : 샘플의 정답을 예측
print(model.predict([[30,600]])) # [1] 이 출력됨. 1= 도미

