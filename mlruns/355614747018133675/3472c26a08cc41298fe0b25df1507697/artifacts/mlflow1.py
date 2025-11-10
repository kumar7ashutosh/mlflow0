import mlflow,seaborn as sns
import mlflow.sklearn,matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
wine=load_wine()
x=wine.data
y=wine.target 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
max_depth=10
n_estimators=5
mlflow.set_experiment('mlflow_exp')
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')
    plt.savefig('confusion-matrix.png')
    mlflow.log_artifact('confusion-matrix.png')
    mlflow.log_artifact(__file__)