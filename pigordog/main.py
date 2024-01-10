from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

pig1 = 0, 1, 0
pig2 = 0, 1, 1
pig3 = 1, 1, 0

dog1 = 0, 1, 1
dog2 = 1, 0, 1
dog3 = 1, 1, 1

data = pig1, pig2, pig3, dog1, dog2, dog3
classifying = 1, 1, 1, 0, 0, 0

model = LinearSVC()
model.fit(data, classifying )

unknown_animal = 1, 1, 1
print(f'{model.predict((unknown_animal,)) = }')

mysterious_animal1 = 1, 1, 1
mysterious_animal2 = 1, 1, 0
mysterious_animal3 = 0, 1, 1

tests = mysterious_animal1, mysterious_animal2, mysterious_animal3
tests_classifying = 0, 1, 1
predictions = model.predict(tests)

correct_predictions = (predictions == tests_classifying).sum()
num_of_tests = len(tests)
accuracy = correct_predictions / num_of_tests

print(f'{accuracy * 100 = }')

sklearn_accuracy = accuracy_score(tests_classifying, predictions)
print(f'{sklearn_accuracy * 100 = }')
