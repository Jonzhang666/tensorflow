from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# method 1: pickle
import pickle   #pickle模块

# save          #保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
# restore       #读取Model
with open('save/clf.pickle', 'rb') as f:
   clf2 = pickle.load(f)
   print(clf2.predict(X[0:1]))


# method 2: joblib
from sklearn.externals import joblib   #jbolib模块

# Save
#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')

# restore
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))
