# קוד אימון המודלים

# ייבוא ספריות נחוצות
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle
import warnings

# שלב 1: הגדרת תיקיית הפלט
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# שלב 2: טעינת הדאטהסט
file_path = 'DataSet.xlsx'
data = pd.read_excel(file_path)

# שלב 3: הגדרת המטרה והפיצ'רים
# המרת 'DM_Sc' לבעיה בינארית (סיכון גבוה/נמוך)
data['Risk'] = data['DM_Sc'].apply(lambda x: 1 if x > 0.5 else 0)  # 1 לסיכון גבוה, 0 לסיכון נמוך

# הסרת 'DM_Sc' מהפיצ'רים
data = data.drop(columns=['DM_Sc'])

# שלב 4: המרת משתנים קטגוריים ל-One-Hot Encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# הגדרת הפיצ'רים והמטרה
X = data_encoded.drop(columns=['Risk'])  # פיצ'רים
y = data_encoded['Risk']  # מטרה

# שלב 5: חלוקת הנתונים לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 6: נירמול הפיצ'רים למודלים שדורשים זאת
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# התעלמות מאזהרות
warnings.filterwarnings('ignore')

# רשימה לאחסון ביצועי המודלים
model_performance = []

# שלב 7: אימון והערכת מודלים

# 1. Logistic Regression
logreg = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
logreg_acc = accuracy_score(y_test, y_pred_logreg)
logreg_recall = recall_score(y_test, y_pred_logreg)
logreg_f1 = f1_score(y_test, y_pred_logreg)
model_performance.append({
    'Model': 'Logistic Regression',
    'Accuracy': logreg_acc,
    'Recall': logreg_recall,
    'F1 Score': logreg_f1
})

# 2. Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
model_performance.append({
    'Model': 'Random Forest',
    'Accuracy': rf_acc,
    'Recall': rf_recall,
    'F1 Score': rf_f1
})

# 3. Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=3, min_samples_split=20, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)
model_performance.append({
    'Model': 'Decision Tree',
    'Accuracy': dt_acc,
    'Recall': dt_recall,
    'F1 Score': dt_f1
})

# 4. Linear Discriminant Analysis (LDA)
try:
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    y_pred_lda = lda.predict(X_test_scaled)
    lda_acc = accuracy_score(y_test, y_pred_lda)
    lda_recall = recall_score(y_test, y_pred_lda)
    lda_f1 = f1_score(y_test, y_pred_lda)
    model_performance.append({
        'Model': 'LDA',
        'Accuracy': lda_acc,
        'Recall': lda_recall,
        'F1 Score': lda_f1
    })
except Exception as e:
    print(f"LDA did not converge: {e}")
    model_performance.append({
        'Model': 'LDA',
        'Accuracy': np.nan,
        'Recall': np.nan,
        'F1 Score': np.nan
    })

# שלב 8: ביצוע קרוס-ולידציה לכל מודל
for model_dict in model_performance:
    model_name = model_dict['Model']
    if model_name == 'Logistic Regression':
        cv_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5, scoring='accuracy')
    elif model_name == 'Random Forest':
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    elif model_name == 'Decision Tree':
        cv_scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
    elif model_name == 'LDA':
        try:
            cv_scores = cross_val_score(lda, X_train_scaled, y_train, cv=5, scoring='accuracy')
        except Exception as e:
            print(f"LDA Cross-Validation failed: {e}")
            cv_scores = [np.nan]
    else:
        cv_scores = [np.nan]
    model_dict['CV Mean Accuracy'] = np.mean(cv_scores)
    model_dict['CV Std Accuracy'] = np.std(cv_scores)

# שלב 9: שמירת ביצועי המודלים לקובץ CSV
results_df = pd.DataFrame(model_performance)
results_csv_path = os.path.join(output_dir, 'model_performance.csv')
results_df.to_csv(results_csv_path, index=False)

# שלב 10: בחירת המודל הטוב ביותר על פי דיוק
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
if best_model_name == 'Random Forest':
    best_model = rf
elif best_model_name == 'Decision Tree':
    best_model = dt
elif best_model_name == 'Logistic Regression':
    best_model = logreg
elif best_model_name == 'LDA':
    best_model = lda
else:
    best_model = None

print(f"{best_model_name} is the best model.")

# שלב 11: בחירת 10 הפיצ'רים המובילים (ללא פיצ'רים הקשורים לאזור)
# הגדרת רשימת פיצ'רים הקשורים לאזור
area_features = [col for col in X.columns if 'AreaName' in col or 'MSOA_Name' in col]

# קבלת חשיבות הפיצ'רים
if best_model_name in ['Random Forest', 'Decision Tree']:
    # במודלים מבוססי עצים יש feature_importances_
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
elif best_model_name == 'Logistic Regression':
    # בלוגיסטיק רגרשן משתמשים במקדמים
    coefficients = best_model.coef_[0]
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients)
    })
else:
    # במודלים אחרים משתמשים בכל הפיצ'רים
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.nan
    })

# הסרת פיצ'רים הקשורים לאזור
feature_importance_df = feature_importance_df[~feature_importance_df['Feature'].isin(area_features)]

# מיון הפיצ'רים לפי חשיבות
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# בחירת 10 הפיצ'רים המובילים
top_features = feature_importance_df['Feature'].head(10).tolist()

# שמירת הפיצ'רים המובילים לקובץ
with open(os.path.join(output_dir, 'top_features.pkl'), 'wb') as f:
    pickle.dump(top_features, f)

print("Top 10 features (excluding area-related features):")
print(top_features)

# שלב 12: שמירת המודל הטוב ביותר, הסקלר ושמות הפיצ'רים לתיקיית הפלט
with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# שמירת X_test ו-y_test לשימוש בדשבורד
with open(os.path.join(output_dir, 'X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)

with open(os.path.join(output_dir, 'y_test.pkl'), 'wb') as f:
    pickle.dump(y_test, f)

print("All output files have been saved to the 'output' directory.")