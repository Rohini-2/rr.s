import pandas as pd
df=pd.read_csv("Authentication.csv")
factor=df[["variance","skewness","curtosis","entropy"]]
predict=df["class"]

from sklearn.model_selection import train_test_split
factor_train,factor_test,predict_train,predict_test=train_test_split(factor,predict,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
factor_train=sc_x.fit_transform(factor_train)
factor_test=sc_x.transform(factor_test)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(factor_train,predict_train)
predict_new_test=classifier.predict(factor_test)
print(predict_new_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(predict_test,predict_new_test,[0,1])
import matplotlib.pyplot as plt
import seaborn as sns
ax=plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)
