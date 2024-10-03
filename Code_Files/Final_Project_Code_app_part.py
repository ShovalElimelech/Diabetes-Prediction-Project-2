# קוד האפליקציה

# ייבוא ספריות נחוצות
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import base64

# הגדרת תיקיית הפלט
output_dir = 'Code_Files/output'

# טעינת המודל והסקלר
with open(os.path.join(output_dir, 'best_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(output_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# טעינת שמות הפיצ'רים, הפיצ'רים המובילים ונתוני הבדיקה לדשבורד
with open(os.path.join(output_dir, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)
with open(os.path.join(output_dir, 'top_features.pkl'), 'rb') as f:
    top_features = pickle.load(f)

with open(os.path.join(output_dir, 'X_test.pkl'), 'rb') as f:
    X_test = pickle.load(f)
with open(os.path.join(output_dir, 'y_test.pkl'), 'rb') as f:
    y_test = pickle.load(f)

# טעינת ביצועי המודל
results_csv_path = os.path.join(output_dir, 'model_performance.csv')
model_results = pd.read_csv(results_csv_path)

# טעינת הדאטהסט המלא
file_path = 'Code_Files/DataSet.xlsx'
data = pd.read_excel(file_path)

# הגדרת תצורת הדף
st.set_page_config(
    page_title="Diabetes Risk Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# פונקציה לטעינת תמונה ל-CSS
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def add_logo(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    logo_css = f'''
    <style>
    [data-testid="stSidebar"]] {{
        position: relative;
    }}
    [data-testid="stSidebar"]::before {{
        content: "";
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        width: 100%;
        height: 150px;
        display: block;
        margin-bottom: 20px;
    }}
    </style>
    '''
    st.markdown(logo_css, unsafe_allow_html=True)

# הוספת לוגו לסרגל הצדדי
add_logo('Code_Files/Diabetes Image.png')

# ניווט בסרגל הצדדי
st.sidebar.title("ניווט")
page = st.sidebar.radio("עבור אל", ["דף הבית", "חיזוי", "לוח מחוונים"])

if page == "דף הבית":
    # עמוד הבית עם הסבר על הפרויקט בעברית
    st.title("פרויקט חיזוי סיכון לסוכרת")
    st.header("שימוש בלמידת מכונה לחיזוי סיכון לסוכרת באנגליה")
    st.write("""
        ברוכים הבאים לאפליקציית חיזוי סיכון לסוכרת. פרויקט זה נועד לנצל טכניקות של למידת מכונה כדי לחזות את הסיכון לסוכרת באזורים שונים ברחבי אנגליה. על ידי ניתוח של דאטהסט מקיף, פיתחנו מודלים המסייעים בזיהוי אזורים עם סיכון גבוה, מה שמאפשר הקצאה טובה יותר של משאבי בריאות וצעדי מניעה.

        **מטרות הפרויקט:**
        - **ניתוח נתונים:** עיבוד והבנת הדאטהסט הניתן הקשור לגורמי סיכון לסוכרת.
        - **אימון מודלים:** אימון מספר מודלים של למידת מכונה לחיזוי סיכון לסוכרת.
        - **הערכת מודלים:** השוואת ביצועי המודלים השונים כדי לבחור את הטוב ביותר.
        - **פריסה:** פיתוח אפליקציה אינטראקטיבית למשתמשים לחזות סיכון לסוכרת ולהציג את ביצועי המודל.

        **טכנולוגיות בשימוש:**
        - **Python וספריות:** pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit.
        - **מודלי למידת מכונה:** רגרסיה לוגיסטית, יער אקראי, עץ החלטה, ניתוח הבחנה לינארית (LDA).

        נווטו באפליקציה באמצעות הסרגל הצדדי כדי לחקור חיזויים ולוח מחוונים.
    """)

elif page == "חיזוי":
    # עמוד החיזוי
    st.title('Diabetes Risk Prediction')

    st.write("""
        **Instructions:** Enter the following health statistics to predict the diabetes risk.
    """)

    # איסוף קלט מהמשתמש עבור כל אחד מהפיצ'רים המובילים
    user_input = {}
    for feature in top_features:
        # התעלמות מפיצ'רים הקשורים לאזור
        if 'AreaName' in feature or 'MSOA_Name' in feature:
            continue

        # בדיקה אם הפיצ'ר קיים בנתונים המקוריים
        if feature in data.columns:
            min_value = int(data[feature].min())
            max_value = int(data[feature].max())
            default_value = int(data[feature].median())
        else:
            # לפיצ'רים בינאריים
            min_value = 0
            max_value = 1
            default_value = 0

        # התאמת ה-step בהתאם לטווח הערכים
        if max_value - min_value > 100:
            step = max((max_value - min_value) // 100, 1)
        else:
            step = 1

        value = st.number_input(f'Enter value for {feature}', min_value=min_value, max_value=max_value, value=default_value, step=step)
        user_input[feature] = value

    # כפתור לחיזוי
    if st.button('Predict Diabetes Risk'):
        # הכנת הנתונים לחיזוי
        input_dict = {}

        for col in feature_names:
            if col in user_input:
                input_dict[col] = user_input[col]
            else:
                # מילוי פיצ'רים חסרים בערך חציוני או 0
                if col in data.columns:
                    input_dict[col] = data[col].median()
                else:
                    input_dict[col] = 0

        # יצירת DataFrame
        input_data = pd.DataFrame([input_dict])

        # הבטחת סוג הנתונים
        input_data = input_data.astype(float)

        # סידור העמודות בהתאם למודל
        input_data = input_data[feature_names]

        # נירמול הנתונים אם נדרש
        if isinstance(model, LogisticRegression) or isinstance(model, LinearDiscriminantAnalysis):
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
        else:
            prediction = model.predict(input_data)

        # הצגת החיזוי
        risk_label = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        st.write(f'The predicted risk is: **{risk_label}**')

elif page == "לוח מחוונים":
    # עמוד הדשבורד
    st.title('Model Performance Dashboard')

    # Display model comparison
    st.subheader('Model Comparison')
    st.write("""
        Below is a comparison of the performance metrics of all trained models.
    """)

    # פונקציות להדגשת הערכים הטובים ביותר
    def highlight_best(s):
        is_best = s == s.max()
        return ['background-color: green' if v else '' for v in is_best]

    def highlight_worst_std(s):
        is_worst_std = s == s.min()
        return ['background-color: green' if v else '' for v in is_worst_std]

    # קביעת המודל הטוב ביותר
    best_model_name = model_results.loc[model_results['Accuracy'].idxmax(), 'Model']

    # יישום הסגנון
    styled_results = model_results.style.applymap(lambda x: 'background-color: green' if x == best_model_name else '', subset=['Model']) \
                                         .apply(highlight_best, subset=['Accuracy', 'Recall', 'F1 Score', 'CV Mean Accuracy']) \
                                         .apply(highlight_worst_std, subset=['CV Std Accuracy'])

    st.dataframe(styled_results)

    # הסבר על המודל הנבחר
    st.write(f"**Selected Model:** {best_model_name}")
    st.write(f"The {best_model_name} was selected as it achieved the highest accuracy on the test data.")

    # הסבר על מודל ה-LDA
    st.subheader('Note on LDA Model')
    st.write("""
        The Linear Discriminant Analysis (LDA) model was also tested but did not converge due to issues such as multicollinearity or singular covariance matrices in the data. Therefore, it was not suitable for our dataset and is not considered in the final model selection.
    """)

    # ביצועי המודל הנבחר
    st.subheader('Selected Model Performance')
    st.write("""
        The following metrics reflect the performance of the selected model on the test dataset.
    """)

    # חישוב ביצועי המודל
    if isinstance(model, LogisticRegression) or isinstance(model, LinearDiscriminantAnalysis):
        y_pred_test = model.predict(scaler.transform(X_test))
    else:
        y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    # מטריצת הבלבול
    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel('Predicted Labels')
    ax_cm.set_ylabel('True Labels')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    # עקומת ROC
    st.subheader('ROC Curve')
    if isinstance(model, LogisticRegression) or isinstance(model, LinearDiscriminantAnalysis):
        y_probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # מספר המקרים לפי אזור
    st.subheader('Diabetes Cases by Region')
    region_diabetes_counts = data.groupby('AreaName')['DM_EstNum'].sum().sort_values(ascending=False).head(10)
    fig_region, ax_region = plt.subplots()
    region_diabetes_counts.plot(kind='bar', color='purple', ax=ax_region)
    ax_region.set_title('Top 10 Regions with Most Estimated Diabetes Cases')
    ax_region.set_xlabel('Region')
    ax_region.set_ylabel('Estimated Number of Diabetes Cases')
    st.pyplot(fig_region)

    # התפלגות החיזויים
    st.subheader('Prediction Distribution')
    unique, counts = np.unique(y_pred_test, return_counts=True)
    fig_dist, ax_dist = plt.subplots()
    ax_dist.bar(['Low Risk', 'High Risk'], counts, color=['lightgreen', 'salmon'])
    ax_dist.set_xlabel('Risk Category')
    ax_dist.set_ylabel('Count')
    ax_dist.set_title('Distribution of Predictions')
    st.pyplot(fig_dist)
