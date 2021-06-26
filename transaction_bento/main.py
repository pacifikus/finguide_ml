import pandas as pd
from classification_service import UserClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

if __name__ == "__main__":

    data = pd.read_csv('data/transaction_user_data.csv', index_col='customer_id')
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('label', axis=1),
        data['label'],
        test_size=0.2,
        random_state=2021,
    )

    clf = xgb.XGBClassifier(
        max_depth=5,
        objective='multi:softprob',
        n_estimators=250,
        num_classes=4
    )
    clf.fit(X_train, y_train)

    user_classifier_service = UserClassifier()
    user_classifier_service.pack('model', clf)
    saved_path = user_classifier_service.save()
