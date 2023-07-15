import pandas as pd
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report



# Load the data
df = pd.read_csv('jojo.csv')
df = df.dropna()

# Define the features and the target
X = df[['ndvi', 'nwdi', 'distancefromriver', 'geology','slope', 'soils', 'tri', 'aspect', 'dem']]
y = df['flood']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Prepare and Standardize Predictions
df_pred = pd.read_csv('predictions_no_null.csv')
predict = df_pred[['ndvi', 'nwdi', 'distancefromriver', 'geology','slope', 'soils', 'tri', 'aspect', 'dem']]

predict = scaler.fit_transform(predict)
#print(predict[0])

# Define the models and the hyperparameters to tune
models = [
    {"name": "RandomForestClassifier", "model": RandomForestClassifier(), "params": {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]}},
]



# Train each model, print the classification report, and the probabilities for the positive class
for m in models:
    print("\nTraining", m["name"])
    grid_search = GridSearchCV(m["model"], m["params"], cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    y_pred1 = model.predict(predict)
    print(y_pred1[0])

    prob_list = []
    # Print the probabilities for the positive class

    #y_proba = model.predict_proba(X_test)
    y_proba = model.predict_proba(predict)

    print("\nProbabilities for the positive class (flood):")
    for i, prob in enumerate(y_proba):
        prob_list.append(f"{prob[1]:.2f}")
       # print(f"Sample {i}: P(flood) = {prob[1]:.2f}")

    #print(prob_list)
    predictions = {'predictions': prob_list}
    df = pd.DataFrame(predictions)
    df.to_csv('all_predictions.csv', index=False)


predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
X = X_train

feature_names = ['ndvi', 'nwdi', 'distancefromriver', 'geology', 'slope', 'soils', 'tri', 'aspect', 'dem']
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names = feature_names, class_names=['No flood', 'Flood'], kernel_width=5)

chosen_instance = predict[0]
exp = explainer.explain_instance(chosen_instance, predict_fn_rf, num_features=5)
