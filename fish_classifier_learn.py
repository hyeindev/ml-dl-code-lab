import fish_classifier as fsc

# 훈련한 데이터 외에 다른 데이터로 테스트해야함

train_input = fsc.fish_data[:35]
train_target = fsc.fish_target[:35]  # 훈련 데이터

test_input = fsc.fish_data[35:]
test_target = fsc.fish_target[35:] # 평가 데이터


fsc.model.fit(train_input, train_target); # 도미 데이터를 훈련시키고
testScore = fsc.model.score(test_input,test_target); # 빙어 데이터로 정확도 계산 

print(testScore); # ---> 당연히 0.0 이 출력됨



