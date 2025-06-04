import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report , accuracy_score ,precision_score, recall_score, f1_score,  RocCurveDisplay , confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
# spliting the data  

data =pd.read_csv('final_cleaned_data.csv') 
features = ["gender" , 	"age" , 	"hypertension" , 	"heart_disease" , 	"ever_married"  , 	"work_type" , 	"Residence_type" , 	"avg_glucose_level" ,	"bmi" , 	"smoking_status"]
label = ["stroke"]
x = data[features]
y = data[label]
print (f"The cleand data overview :\n{data.head()}")
x_train_orig , x_test , y_train_orig ,y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=52 ) 

# checking the shape of the training data before applying smote
print ( f"x_train shape : {x_train_orig.shape}")

'''
i used smote to make my data balanced therefor some imbalance data which lead to poor performance of the models.
SMOTE (Synthetic Minority Over-sampling Technique) is a popular technique for handling class imbalance in datasets.
It works by creating synthetic samples of the minority class to balance the dataset.
'''
# Applying SMOTE to handle class imbalance

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train_orig, y_train_orig)


# checking the shape of the training data after applying smote
print (f"x_train shape after smote : {x_train_smote.shape}")


# creating fuction to train and evaluate the models
def train_and_evaluate_model(model_name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)

    print(f"********Model: {model_name}********")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.4f}")
    print(f"Precision Score: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall Score: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    # print (f'confusion matrix :\n{pd.crosstab(y_test.values.ravel(),y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)}')
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],  
            yticklabels=['Negative', 'Positive'],
            linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Labels', fontsize=14)
    plt.ylabel('Actual Labels', fontsize=14)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    RocCurveDisplay.from_estimator(model, x_test, y_test)
    plt.title(f"ROC Curve for {model_name}")
    plt.show()
    print("-" * 70)
    
    
# creating the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gaussian Naive Base":GaussianNB()
}

# training and evaluating the models
print ("\n*************** training and evaluating the models ***************")
for model_name, model in models.items():
    train_and_evaluate_model(model_name, model, x_train_smote, x_test, y_train_smote, y_test)

# plotting the feature importance for random forest model
rf = RandomForestClassifier()
rf.fit(x_train_smote, y_train_smote.values.ravel())
importance = rf.feature_importances_
plt.barh(features, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Feature Names")
plt.title("Feature Importance in RandomForest")
# plt.show()

# creating the ensemble model using voting classifier
ensemble_model = VotingClassifier(estimators=[
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=100,max_depth=10)),
    ("DecisionTree ", DecisionTreeClassifier(max_depth=5)),
    ("kNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("Supportvectormachine", SVC(probability=True))
], voting='soft') 

# creating ensemble model using stacking classifier 
ensemble_staking = StackingClassifier(estimators=[
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=100,max_depth=10)),
    ("DecisionTree ", DecisionTreeClassifier(max_depth=5)),
    ("kNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("Supportvectormachine", SVC(probability=True))
]
, final_estimator=GaussianNB(), cv=5, n_jobs=-1) # cv ---> croos validation , n_jobs ----> parallel processing 
    
# train and evaluate the ensemble model (voting classifier)
train_and_evaluate_model("Ensemble_Model_Vot", ensemble_model, x_train_smote, x_test, y_train_smote, y_test)

# train and evaluate the ensemble model (stacking classifier)
train_and_evaluate_model("Ensemble_Model_Stak", ensemble_staking, x_train_smote, x_test, y_train_smote, y_test)

'''
the diffrence between staking and voting is that voting is only choose amoung the predections of all models from the ensemble learning 
but the staking is creating a new model called final estimator or meta model which is the basis of the final prediction this is a more 
accurate method that enhance the preformance by combining the all model preformace on one 
'''


# let's try another ensembel type it's a Gradient boosting classifier
'''
the only problem with this model is the hyperparameters tuning which makes the measurement of the model performance not good untill achieve good tuning
'''
ens_gradient = GradientBoostingClassifier(n_estimators=500,learning_rate=0.0001,max_depth=8,random_state=66)
train_and_evaluate_model("Gradient_Boosting", ens_gradient, x_train_smote, x_test, y_train_smote, y_test)





