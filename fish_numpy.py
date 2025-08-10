# 사이킷런이나 맵플롯립도 numpy 에 대한 의존성이 높음 (입,출력 데이터가 Numpy)
import numpy as np # np 라는 별칭은 프로그래머 관례
import fish_classifier as fs
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

# numpy 객체 준비
input_arr = np.array(fs.fish_data)
target_arr = np.array(fs.fish_target)

#print(input_arr);
#print(tartget_arr);


index = np.arange(49)
np.random.shuffle(index)
#print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]


#import matplotlib.pyplot as plt
#plt.scatter(train_input[:,0],train_input[:, 1]) # : << 전체를 선택하겠다는 의미
#plt.scatter(test_input[:,0],test_input[:, 1])

#plt.show()




model.fit(train_input, train_target);
print(test_input);
print(test_target);
score = model.score(test_input,test_target);

print(score); # 정확도 1.0 출력