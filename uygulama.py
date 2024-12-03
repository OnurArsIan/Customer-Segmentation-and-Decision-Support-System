#!/usr/bin/env python

# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV files
train_path = 'train.csv'
train_df = pd.read_csv(train_path)

# Check for missing values in the training dataset
missing_values_train = train_df.isnull().sum()
print("Missing values in training data:")
print(missing_values_train)

# Fill missing values if any
train_df.fillna(method='ffill', inplace=True)

# Perform One-Hot Encoding for categorical features in the training dataset
categorical_features = ['Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Eğitim Düzeyi', 'İstihdam Durumu', 'Yaşadığı Şehir', 'En Çok İlgilendiği Ürün Grubu', 'Eğitime Devam Etme Durumu']
train_encoded_df = pd.get_dummies(train_df, columns=categorical_features)

# Add new feature: Gelir/Harcama Oranı
train_encoded_df['Gelir_Harcama_Oranı'] = train_encoded_df['Yıllık Ortalama Gelir'] / (train_encoded_df['Yıllık Ortalama Satın Alım Miktarı'] + 1e-5)

# Standardize the numerical features
numerical_features = ['Yıllık Ortalama Gelir', 'Yıllık Ortalama Satın Alım Miktarı', 'Yıllık Ortalama Sipariş Verilen Ürün Adedi', 'Yıllık Ortalama Sepete Atılan Ürün Adedi', 'Gelir_Harcama_Oranı']
scaler = StandardScaler()
train_encoded_df[numerical_features] = scaler.fit_transform(train_encoded_df[numerical_features])

# Split the training data into features and target
X = train_encoded_df.drop(columns=['Öbek İsmi'])
y = train_encoded_df['Öbek İsmi']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_val, y_pred))

# Streamlit user input for real-time prediction
st.title('Müşteri Kredi ve Ürün Öneri Sistemi')

# Collect user inputs
cinsiyet = st.selectbox('Cinsiyet', ['Erkek', 'Kadın'])
yaş_grubu = st.selectbox('Yaş Grubu', ['18-30', '31-40', '41-50', '51-60', '60+'])
medeni_durum = st.selectbox('Medeni Durum', ['Evli', 'Bekar'])
eğitim_düzeyi = st.selectbox('Eğitim Düzeyi', ['İlkokul Mezunu', 'Lise Mezunu', 'Üniversite Mezunu', 'Yüksek Lisans Mezunu', 'Doktora Mezunu'])
istihdam_durumu = st.selectbox('İstihdam Durumu', ['Düzenli ve Ücretli Bir İşi Var', 'Kendi İşinin Sahibi', 'İşsiz veya Düzenli Bir İşi Yok'])
yaşadığı_şehir = st.selectbox('Yaşadığı Şehir', ['Büyük Şehir', 'Küçük Şehir', 'Kırsal'])
ürün_grubu = st.selectbox('En Çok İlgilendiği Ürün Grubu', ['Gıda', 'Teknoloji', 'Giyim', 'Ev', 'Diğer'])
yıllık_gelir = st.number_input('Yıllık Ortalama Gelir', min_value=0)
yıllık_satın_alım = st.number_input('Yıllık Ortalama Satın Alım Miktarı', min_value=0)

# Convert user inputs to a dataframe
user_data = pd.DataFrame({
    'Cinsiyet': [cinsiyet],
    'Yaş Grubu': [yaş_grubu],
    'Medeni Durum': [medeni_durum],
    'Eğitim Düzeyi': [eğitim_düzeyi],
    'İstihdam Durumu': [istihdam_durumu],
    'Yaşadığı Şehir': [yaşadığı_şehir],
    'En Çok İlgilendiği Ürün Grubu': [ürün_grubu],
    'Yıllık Ortalama Gelir': [yıllık_gelir],
    'Yıllık Ortalama Satın Alım Miktarı': [yıllık_satın_alım],
    'Gelir_Harcama_Oranı': [yıllık_gelir / (yıllık_satın_alım + 1e-5)]
})

# One-Hot Encode the user inputs
user_encoded_df = pd.get_dummies(user_data, columns=[col for col in categorical_features if col in user_data.columns])

# Align user data columns with training data columns
user_encoded_df = user_encoded_df.reindex(columns=X_train.columns, fill_value=0)

# Standardize numerical features for user input
user_encoded_df[numerical_features] = scaler.transform(user_encoded_df[numerical_features])

# Make prediction for user
user_cluster = model.predict(user_encoded_df)[0]

# Define credit and product recommendations based on clusters, income, and spending
recommendations = {
    'obek_1': {
        'credit': lambda income, spending, ratio: 'Düşük faizli ihtiyaç kredisi' if ratio < 1.5 else ('Orta düzey ihtiyaç kredisi' if ratio < 3 and income < 200000 else ('Yüksek limitli kredi' if ratio >= 3 and income > 200000 else 'Konut kredisi')),
        'products': lambda interest, ratio: ['Sağlıklı gıda ürünleri', 'Organik ürünler'] if 'Gıda' in interest and ratio < 1.5 else (['Lüks gıda ürünleri', 'Gurme yiyecekler'] if 'Gıda' in interest and ratio >= 3 else ['Teknolojik ürünler', 'Lüks yaşam ürünleri']),
        'services': lambda income, ratio: ['Kişisel finans danışmanlığı'] if ratio < 1.5 else (['Yatırım danışmanlığı', 'Kariyer planlama hizmetleri'] if ratio >= 3 else ['Tasarruf danışmanlığı', 'Bütçe yönetimi hizmetleri'])
    },
    'obek_2': {
        'credit': lambda income, spending: 'Orta düzey ihtiyaç kredisi' if income < 200000 and spending < 100000 else 'Yüksek limitli kredi kartı',
        'products': lambda interest: ['Yüksek teknolojili cihazlar', 'Giyim aksesuarları'] if 'Teknoloji' in interest else ['Kişisel bakım ürünleri', 'Spor ekipmanları'],
        'services': lambda spending: ['Alışveriş indirimleri'] if spending > 50000 else ['Temel müşteri hizmetleri']
    },
    'obek_3': {
        'credit': lambda income, spending: 'Yüksek limitli kredi kartı' if spending > 150000 else 'Konut kredisi',
        'products': lambda interest: ['Modern mobilya', 'Akıllı beyaz eşyalar'] if 'Ev' in interest else ['Lüks moda ürünleri', 'Aksesuarlar'],
        'services': lambda income: ['Premium sağlık sigortası'] if income > 300000 else ['Temel sağlık sigortası']
    },
    'obek_4': {
        'credit': lambda income, spending: 'Konut kredisi' if income > 300000 else 'Orta düzey ihtiyaç kredisi',
        'products': lambda interest: ['Elektrikli ev aletleri', 'Yüksek verimli beyaz eşyalar'] if 'Ev' in interest else ['Araç bakım ürünleri', 'Sigorta hizmetleri'],
        'services': lambda spending: ['Emlak danışmanlığı'] if spending > 100000 else ['Yol yardım hizmeti']
    },
    'obek_5': {
        'credit': lambda income, spending: 'Genel ihtiyaç kredisi',
        'products': lambda interest: ['Genel yaşam ürünleri', 'Finansal güvence paketleri'],
        'services': lambda income: ['Genel müşteri hizmetleri']
    },
    'obek_6': {
        'credit': lambda income, spending: 'Düşük faizli kredi kartı' if income < 150000 else 'Yüksek limitli kredi kartı',
        'products': lambda interest: ['Eğitim malzemeleri', 'Kırtasiye ürünleri'] if 'Eğitim' in interest else ['Kişisel gelişim kitapları', 'Online kurslar'],
        'services': lambda income: ['Eğitim danışmanlığı'] if income < 200000 else ['Kariyer danışmanlığı']
    },
    'obek_7': {
        'credit': lambda income, spending: 'Taşıt kredisi' if income > 250000 else 'Orta düzey ihtiyaç kredisi',
        'products': lambda interest: ['Araç aksesuarları', 'Otomobil bakım ürünleri'] if 'Taşıt' in interest else ['Sigorta hizmetleri', 'Yol yardım hizmetleri'],
        'services': lambda spending: ['Taşıt sigortası'] if spending > 100000 else ['Temel araç bakım hizmetleri']
    },
    'obek_8': {
        'credit': lambda income, spending: 'Yatırım kredisi' if income > 400000 else 'Orta düzey ihtiyaç kredisi',
        'products': lambda interest: ['Yatırım araçları', 'Gayrimenkul danışmanlığı'] if 'Yatırım' in interest else ['Tasarruf planları', 'Sigorta hizmetleri'],
        'services': lambda income: ['Yatırım danışmanlığı'] if income > 300000 else ['Finansal planlama hizmetleri']
    }
}

# Provide recommendations for the user based on predicted cluster
income = yıllık_gelir
spending = yıllık_satın_alım
interest = ürün_grubu
recommendation = recommendations.get(user_cluster, {
    'credit': lambda income, spending, ratio: 'Genel ihtiyaç kredisi',
    'products': lambda interest, ratio: ['Genel ürünler'],
    'services': lambda income, ratio: ['Genel müşteri hizmetleri']
})

# Display recommendations
st.write(f"\nTahmini Öbek: {user_cluster}")
st.write(f"Kredi Teklifi: {recommendation['credit'](income, spending, user_encoded_df['Gelir_Harcama_Oranı'].iloc[0])}")
st.write(f"Ürün Önerileri: {', '.join(recommendation['products'](interest, user_encoded_df['Gelir_Harcama_Oranı'].iloc[0]))}")
st.write(f"Hizmet Önerileri: {', '.join(recommendation['services'](income, user_encoded_df['Gelir_Harcama_Oranı'].iloc[0]))}")
