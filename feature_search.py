
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

def normalize(df):
    df.iloc[:,1:] = (df.iloc[:,1:]-df.iloc[:,1:].mean())/df.iloc[:,1:].std()
    return df

def preprocess_realworld():
  data  = arff.loadarff('Rice_Cammeo_Osmancik.arff')
  train = pd.DataFrame(data[0])
  train.head()
  labelencoder = LabelEncoder()
  train['Class'] = labelencoder.fit_transform(train['Class'])
  train = train.rename(columns={'Area':1, 'Perimeter':2, 'Major_Axis_Length':3, 'Minor_Axis_Length':4, 'Eccentricity':5, 'Convex_Area':6, 'Extent':7, 'Class':0})
  train = train[[0,1,2,3,4,5,6,7]]
  train = normalize(train)
  train.to_csv("real_world_rice_data.csv", sep=' ',index=False)



import numpy as np

def nearest_neighbour(X_copy,X,i):
  temp = np.sqrt(np.add.reduce((X_copy - X[i])**2, axis=1)) #Calculate distance
  index = np.argmin(temp,keepdims=True) #find the closest neighbour
  return index


def calc_accuracy(X, Y):
    ''' Uses leave one out Nearest Neighbour classification and returns accuracy of the model on current features'''

    matched_count = 0

    def leave_one_out(i):
        # print(i)
        nonlocal matched_count

        lowest_distance = float('inf')
        curr_class = -1
        X_copy = np.delete(X, i, axis=0) # Create a copy of data by deleting i-th row (leave one out)
        index = nearest_neighbour(X_copy, X, i)

        if i <= index:
            # print(i,index,Y[index + 1],Y[i])
            if Y[index + 1] == Y[i]:
                # print(i,index)
                matched_count += 1
        else:
            if Y[index] == Y[i]:
                # print(i,index)
                matched_count += 1


    np.frompyfunc(leave_one_out, 1, 0)(np.arange(len(X)))
    Accuracy = matched_count / len(Y)
    return Accuracy






def forward_selection(df,es=False):
  '''Performs Forward Selection feature search to find best set of features'''
  Y = df.iloc[:, 0].to_numpy()
  features = [i for i in df.columns[1:]]
#   print(features)
  # df2=df.iloc[:5, 1:]
  selected=[]
  highest=[None,-1]
  print(f"This dataset has {len(features)} features (not including the class feature), with {df.count()[0]} instances.")
  print()
  print(f"Running nearest neighbour on all {len(features)} features using 'leave-one out' evaluation, gives {round(calc_accuracy(df.iloc[:,1:].to_numpy(),Y)*100,4) }% accuracy.")
  print()
  print("Beginning Search")
  avg_count=0
  avg_acc=0
  sum_acc=0
  for k in range(len(features)):
      mx_acc=-1
      temp_f=None
      not_sel=[l for l in features if l not in selected] #not selected features
      for i in not_sel:
          temp_sel=selected+[i] # add one feature to temporary selection
          # print(temp_sel)
          if k==0:
              X= df.iloc[:, temp_sel].to_numpy().reshape(-1,1) # if a single feature is selected
          else:
              X= df.iloc[:, temp_sel].to_numpy()
          # print(X)
          temp=calc_accuracy(X,Y)
          if mx_acc<temp:
              mx_acc=temp
              temp_f= i
          print(f"Using feature(s) {{{temp_sel}}} gives {round(temp*100,4)}% accuracy")
      selected.append(temp_f)
      sum_acc+=mx_acc
      avg_count+=1

      if avg_count==5:
        curr_10_acc=sum_acc/avg_count
        avg_count=0
        sum_acc=0

        if (curr_10_acc<avg_acc) and es:
          print("Early Stopping: Average accuracy for 5 epochs decreased")
          break
        else:
          avg_acc=curr_10_acc
      print(f"Feature set{{{selected}}} was best, accuracy is {round(mx_acc*100,4)}%")
      if mx_acc>highest[1]:
          highest=[selected.copy(),mx_acc]
      else:
        print("(Warning: accuracy has decreased, continuing in case of local Maxima)")
      # print(selected, mx_acc)


  print(f"Fininshed!! Best feature set is {{{highest[0]}}}, which has accuracy of {round(highest[1]*100,4)}%")




forward_selection(df)

def backward_elimination(df,es=False):
  '''BPerforms backward elimination feature search to find best set of features'''
  Y = df.iloc[:, 0].to_numpy()
  temp_sel = [i for i in df.columns[1:]]
  X = df.iloc[:, temp_sel].to_numpy()
  accuracy = calc_accuracy(X, Y)
  final_accuracy = 0
  fina_accuracy_set = []
  highest=[None,-1]
  print(f"This dataset has {len(temp_sel)} features (not including the class feature), with {df.count()[0]} instances.")
  print()
  print(f"Running nearest neighbour on all {len(temp_sel)} features using 'leave-one out' evaluation, gives {round(calc_accuracy(df.iloc[:,1:].to_numpy(),Y)*100,4) }% accuracy.")
  print()
  print("Beginning Search")
  avg_count=0
  avg_acc=0
  sum_acc=0
  for j in range(len(temp_sel)-1):
      index = 0
      accuracy = -1
      for i in temp_sel.copy():
          temp2=temp_sel.copy()
          temp2.remove(i)
          X = df.iloc[:, temp2].to_numpy()
          temp_accuracy = calc_accuracy(X, Y)
          print(f"Using feature(s) {{{temp2}}} gives {round(temp_accuracy*100,4)}% accuracy")
          if temp_accuracy >= accuracy:

              accuracy = temp_accuracy
              final_accuracy_set = temp_sel.copy()
              final_accuracy_set.remove(i)
      temp_sel=final_accuracy_set.copy()
      sum_acc+=accuracy
      avg_count+=1
      if avg_count==5:
        curr_10_acc=sum_acc/avg_count
        avg_count=0
        sum_acc=0
        if (curr_10_acc<avg_acc) and es:
          print("Early Stopping: Average accuracy for 5 epochs decreased")
          break
        else:
          avg_acc=curr_10_acc
      print(f"Feature set{{{final_accuracy_set}}} was best, accuracy is {round(accuracy*100,4)}%")# print(final_accuracy_set,accuracy)
      if(accuracy >= highest[1]):
          highest = [final_accuracy_set.copy(), accuracy]
      else:
        print("(Warning: accuracy has decreased, continuing in case of local Maxima)")
  print(f"Fininshed!! Best feature set is {{{highest[0]}}}, which has accuracy of {round(highest[1]*100,4)}%")


def main():
    print("Welcome to AD's feature search algorithm")
    print("Type in the name of file you want to test")
    filename=input()
    if "real" in filename:
        df=pd.read_csv(filename,delimiter=' ',header=None,index_col=False) # if file is a real_word dataset
    else:
        df=pd.read_csv(filename,delim_whitespace=True ,header=None)

    df=df.sample(frac=1)
    print("Type the number of algorithm you want to test\n1) Forward Selection \n2) Backward Elimination")
    selection=input()
    print("Do you to use Early Stopping? [Y-yes] (Stopping Criteria is average of 5 iterations is decreased (only works for more than 10 features))")
    es= True if input().lower() == 'y' else False
    forward_selection(df,es) if selection=='1' else backward_elimination(df,es)





preprocess_realworld()
main()





