from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import json
import numpy as np

def read_data(path):
  with open(path) as jf:
    return json.load(jf)
  return None

def get_features(pts):
  ft = [len(pts)]
  x = [p[1] for p in pts]
  y = [p[2] for p in pts]
  z = [p[3] for p in pts]

  dx = np.amax(x) - np.amin(x)
  dy = np.amax(y) - np.amin(y)
  dz = np.amax(z) - np.amin(z)

  ft.append(x[0])
  ft.append(y[0])
  ft.append(z[0])
  ft.append(x[-1])
  ft.append(y[-1])
  ft.append(z[-1])
  ft.append(dx)
  ft.append(dy)
  ft.append(dz)

  return ft

def prepare_data(path = "trdata.json"):
  json_data = read_data(path)
  target = [j["cls"] for j in json_data]
  data = [get_features(j["pts"]) for j in json_data]
  return np.array(data), np.array(target)


data, target = prepare_data()

x_train, x_test, y_train, y_test = train_test_split(data, target,test_size=.33,random_state=123)


clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)