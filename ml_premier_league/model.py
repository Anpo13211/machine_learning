from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import pickle

matches_data = pd.read_csv("2024-2021_matches_data.csv")

matches_data_cleaned = matches_data.drop(columns=['date', 'time', 'comp', 'round', 'match report', 'notes', 'season'])

label_encoders = {}
categorical_columns = ['day', 'venue', 'opponent', 'team']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    matches_data_cleaned[column] = label_encoders[column].fit_transform(matches_data_cleaned[column])

matches_data_cleaned = matches_data_cleaned.dropna()

X = matches_data_cleaned.drop(columns=['gf', 'ga'])
y = matches_data_cleaned[['gf', 'ga']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model_gf = GradientBoostingRegressor(random_state=10)
model_ga = GradientBoostingRegressor(random_state=10)

columns_to_drop = ['result', 'captain', 'formation', 'referee']
X_train_dropped = X_train.drop(columns=columns_to_drop)
X_test_dropped = X_test.drop(columns=columns_to_drop)

# Retraining the models with the modified feature set
model_gf.fit(X_train_dropped, y_train['gf'])
model_ga.fit(X_train_dropped, y_train['ga'])

# Predicting on the modified test set
y_pred_gf = model_gf.predict(X_test_dropped)
y_pred_ga = model_ga.predict(X_test_dropped)

# Evaluating the models again
mse_gf = mean_squared_error(y_test['gf'], y_pred_gf)
r2_gf = r2_score(y_test['gf'], y_pred_gf)
mse_ga = mean_squared_error(y_test['ga'], y_pred_ga)
r2_ga = r2_score(y_test['ga'], y_pred_ga)

print(mse_gf, r2_gf, mse_ga, r2_ga)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("model_gf.pkl", "wb") as f:
    pickle.dump(model_gf, f)

with open("model_ga.pkl", "wb") as f:
    pickle.dump(model_ga, f)