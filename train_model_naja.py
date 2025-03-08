# library ที่ใช้
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# กำหนด seed เพื่อให้ผลลัพธ์คงที่
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ฟังก์ชันโหลด data จาก csv
def load_data(file_path='diabetes.csv'):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.isnull().sum())
    return df

# ฟังก์ชันการจัดเตรียมข้อมูล
def preprocess_data(df):
    """เตรียมข้อมูลสำหรับการทำนาย"""
    column_mapping = {
        'Pregnancies': 'pregnancies',
        'Glucose': 'glucose',
        'BloodPressure': 'blood_pressure',
        'SkinThickness': 'skin_thickness',
        'Insulin': 'insulin',
        'BMI': 'bmi',
        'DiabetesPedigreeFunction': 'diabetes_pedigree',
        'Age': 'age',
        'Outcome': 'diabetes'
    }

    # เปลี่ยนชื่อคอลัมน์ต่างๆ ตามที่ตั้งไว้
    df = df.rename(columns=column_mapping)

    # แทนค่าที่เป็น 0 (ยกเว้นบางคอลัมน์) ด้วยค่าเฉลี่ยของคอลัมน์นั้น
    for col in df.columns:
        if df[col].min() == 0 and col not in ['pregnancies', 'diabetes']:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].mean())
    return df

# ฟังก์ชันสร้างชุดข้อมูลที่แตกต่างกันสำหรับการฝึกโมเดล
def create_feature_sets(df):
    y = df['diabetes'].values # ค่าที่ต้องการพยากรณ์

    feature_sets = {
        'full': df[['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                    'insulin', 'bmi', 'diabetes_pedigree', 'age']].values,

        'no_bp': df[['pregnancies', 'glucose', 'skin_thickness', 'insulin',
                     'bmi', 'diabetes_pedigree', 'age']].values,

        'no_glucose': df[['pregnancies', 'blood_pressure', 'skin_thickness', 'insulin',
                          'bmi', 'diabetes_pedigree', 'age']].values,

        'minimal': df[['pregnancies', 'skin_thickness', 'insulin',
                       'bmi', 'diabetes_pedigree', 'age']].values
    }

    return feature_sets, y

# ฟังก์ชันฝึกโมเดลสำหรับแต่ละชุดข้อมูล
def train_models(feature_sets, y):
    models = {}
    scalers = {}
    results = {}

    for name, X in feature_sets.items():
        # แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ โดย 0.2 คือ แบ่งชุดสอบเป็น 20% จากทั้งหมด
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        # ปรับมาตรฐานข้อมูล
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # สร้างและฝึกโมเดล
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
        model.fit(X_train_scaled, y_train)

        # ทดสอบโมเดล
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        # เก็บผลลัพธ์
        results[name] = {
            'model': model,
            'scaler': scaler,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }

        # บันทึกโมเดลและ scaler
        models[name] = model
        scalers[name] = scaler

    return models, scalers, results

# ฟังก์ชันบันทึกโมเดลลงไฟล์
def save_models(models, scalers, output_dir='models'):
    # ถ้าไม่มีโฟลเดอร์ สำหรับเก็บโมเดลต่างๆ ก็ให้สร้างขึ้นมา
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # สร้างไฟล์โมเดลและ scalers ต่างๆ
    for name, model in models.items():
        with open(f'{output_dir}/model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    for name, scaler in scalers.items():
        with open(f'{output_dir}/scaler_{name}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

# ฟังก์ชันสร้างกราฟเปรียบเทียบความสำคัญของตัวแปร
def plot_feature_importance(results, output_dir='plots'):
    # ถ้าไม่มีโฟลเดอร์ สำหรับเก็บรูปภาพของกราฟแบบต่างๆ ก็ให้สร้างขึ้นมา ในที่นี้คือ plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    feature_names = {
        'full': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
        'no_bp': ['Pregnancies', 'Glucose', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
        'no_glucose': ['Pregnancies', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
        'minimal': ['Pregnancies', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
    }

    # เข้าถึง items ในผลลัพธ์ที่ทำนายได้ แล้วทำเป็นกราฟ features importance จากนั้นบันทึกลงในโฟลเดอร์ plots
    for name, result in results.items():
        model = result['model']
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title(f'Feature Importance - {name.capitalize()} Model')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[name][i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_{name}.png')

# ฟังก์ชันสร้าง confusion matrix สำหรับแต่ละโมเดล
def plot_confusion_matrices(results, output_dir='plots'):
    # ถ้าไม่มีโฟลเดอร์ สำหรับเก็บรูปภาพของกราฟแบบต่างๆ ก็ให้สร้างขึ้นมา ในที่นี้คือ plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # เข้าถึง items ในผลลัพธ์ที่ทำนายได้ แล้วทำเป็นกราฟ confusion matrix จากนั้นบันทึกลงในโฟลเดอร์ plots
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title(f'Confusion Matrix - {name.capitalize()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrix_{name}.png')

# ฟังก์ชันเปรียบเทียบประสิทธิภาพของโมเดลต่างๆ
def compare_models(results):
    # ค่าต่างๆ ที่นำมาเปรียบเทียบ
    comparison = {
        'Model': [],
        'Accuracy': [],
        'CV Mean Accuracy': [],
        'CV Std': []
    }

    # เข้าถึง items ในผลลัพธ์ แล้วเพิ่มเข้าไปใน list ต่างๆ ที่กำหนดไว้
    for name, result in results.items():
        comparison['Model'].append(name.capitalize())
        comparison['Accuracy'].append(result['accuracy'])
        comparison['CV Mean Accuracy'].append(result['cv_scores'].mean())
        comparison['CV Std'].append(result['cv_scores'].std())

    # สร้าง dataFrame ที่มีคอลัมน์ตาม dictionary ในที่นี้คือ comparison
    comparison_df = pd.DataFrame(comparison)

    # พล็อตกราฟออกมา แล้วเซฟไว้ใน folder plots
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(comparison['Model']))

    plt.bar(index, comparison['Accuracy'], bar_width, label='Test Accuracy', color='royalblue')
    plt.bar(index + bar_width, comparison['CV Mean Accuracy'], bar_width, label='CV Accuracy', color='lightcoral')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width / 2, comparison['Model'])
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/model_comparison.png')

    return comparison_df

# ฟังก์ชันหลักสำหรับเรียกใช้งาน
def main():
    # โหลดและเตรียมข้อมูล
    df = load_data()
    df = preprocess_data(df)

    # สร้างชุดข้อมูลต่างๆ
    feature_sets, y = create_feature_sets(df)

    # ฝึกโมเดล
    models, scalers, results = train_models(feature_sets, y)

    # บันทึกโมเดล
    save_models(models, scalers)

    # สร้างกราฟวิเคราะห์ผล
    plot_feature_importance(results)
    plot_confusion_matrices(results)

    # เปรียบเทียบโมเดล
    compare_models(results)

    print("\nเทรนโมเดลเสร็จสมบูรณ์!")

if __name__ == "__main__":
    main()
