import sklearn.neighbors
import fish_datas

length = fish_datas.bream_length + fish_datas.smelt_length # 각각 35개, 14개
weight = fish_datas.bream_weight + fish_datas.smelt_weight # 각각 35개, 14개

# 사이킷런 사용 -> 2차원 리스트 생성
fish_data = [[l,w] for l,w in zip(length,weight)]

fish_target = [1] * 35 + [0] * 14

# install scikit-learn - KNeighborsClassifier : 최근접 이웃 알고리즘
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
print(type(model))

# fish_data, fish_target 로 알고리즘 훈련 
model.fit(fish_data,fish_target)
# kn 훈련 평가
score = model.score(fish_data,fish_target)

print(score)
