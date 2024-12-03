import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Veri yükleme ve işleme
@st.cache_data
def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Eksik değerleri doldur
    train_df.fillna(method='ffill', inplace=True)
    test_df.fillna(method='ffill', inplace=True)

    # Yeni özellik: Gelir/Harcama Oranı
    train_df['Gelir/Harcama_Oranı'] = train_df['Yıllık Ortalama Gelir'] / (
        train_df['Yıllık Ortalama Satın Alım Miktarı'] + 1e-5)
    test_df['Gelir/Harcama_Oranı'] = test_df['Yıllık Ortalama Gelir'] / (
        test_df['Yıllık Ortalama Satın Alım Miktarı'] + 1e-5)

    # Kategorik özellikler için One-Hot Encoding
    categorical_features = [
        'Cinsiyet', 'Eğitime Devam Etme Durumu', 'İstihdam Durumu'
    ]
    train_encoded_df = pd.get_dummies(train_df, columns=categorical_features)
    test_encoded_df = pd.get_dummies(test_df, columns=categorical_features)

    # Önemli özellikleri belirle
    important_features = [
        'Yıllık Ortalama Gelir',
        'Yıllık Ortalama Satın Alım Miktarı',
        'Yıllık Ortalama Sipariş Verilen Ürün Adedi',
        'Yıllık Ortalama Sepete Atılan Ürün Adedi',
        'Gelir/Harcama_Oranı'
    ]

    # Sadece önemli özellikleri ölçeklendirme
    scaler = StandardScaler()
    train_encoded_df[important_features] = scaler.fit_transform(train_encoded_df[important_features])
    test_encoded_df[important_features] = scaler.transform(test_encoded_df[important_features])

    return train_encoded_df, test_encoded_df, scaler, important_features


# Veri yükleme ve model eğitimi
train_path = 'train.csv'
test_path = 'test_x.csv'
train_encoded_df, test_encoded_df, scaler, important_features = load_and_preprocess_data(train_path, test_path)

# Eğitim ve test verilerini ayırma
X = train_encoded_df[important_features]
y = train_encoded_df['Öbek İsmi']  # Öbek İsmi sütunu etiket olarak kullanılacak
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modeli tanımla ve eğit
optimized_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='log2',
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
optimized_model.fit(X_train, y_train)

# Öbeklere göre öneriler
def get_recommendations(cluster, user_data):
    income = user_data.get('Yıllık Ortalama Gelir', 0)
    spending = user_data.get('Yıllık Ortalama Satın Alım Miktarı', 0)
    ratio = user_data.get('Gelir/Harcama_Oranı', 0)
    interest = user_data.get('İstihdam Durumu', '')

    recommendations = {
        'obek_1': {
            'credit': lambda income, spending, ratio: 'Düşük faizli ihtiyaç kredisi' if ratio < 1.5 else (
                'Orta düzey ihtiyaç kredisi' if ratio < 3 and income < 200000 else (
                    'Yüksek limitli kredi' if ratio >= 3 and income > 200000 else 'Konut kredisi')),
            'products': lambda interest, ratio: ['Sağlıklı gıda ürünleri', 'Organik ürünler'] if 'Gıda' in interest and ratio < 1.5 else (
                ['Lüks gıda ürünleri', 'Gurme yiyecekler'] if 'Gıda' in interest and ratio >= 3 else ['Teknolojik ürünler', 'Lüks yaşam ürünleri']),
            'services': lambda income, ratio: ['Kişisel finans danışmanlığı'] if ratio < 1.5 else (
                ['Yatırım danışmanlığı', 'Kariyer planlama hizmetleri'] if ratio >= 3 else ['Tasarruf danışmanlığı', 'Bütçe yönetimi hizmetleri'])
        },
        'obek_2': {
            'credit': lambda income, spending, ratio: 'Eğitim kredisi' if income < 100000 else 'Konut kredisi',
            'products': lambda interest, ratio: ['Eğitim materyalleri', 'Kitaplar'] if 'Eğitim' in interest else ['Spor ekipmanları', 'Kişisel gelişim ürünleri'],
            'services': lambda income, ratio: ['Eğitim danışmanlığı', 'Kariyer rehberliği']
        },
        'obek_3': {
            'credit': lambda income, spending, ratio: 'Araç kredisi' if spending > 50000 else 'Tatil kredisi',
            'products': lambda interest, ratio: ['Elektronik cihazlar', 'Tatil paketleri'] if ratio >= 2 else ['Ev eşyaları', 'Giyim ürünleri'],
            'services': lambda income, ratio: ['Yatırım danışmanlığı', 'Sigorta hizmetleri']
        },
        'obek_4': {
            'credit': lambda income, spending, ratio: 'Konut kredisi' if income > 300000 else 'Tüketici kredisi',
            'products': lambda interest, ratio: ['Mobilya', 'Ev dekorasyon ürünleri'],
            'services': lambda income, ratio: ['Ev alımı danışmanlığı', 'İç mekan tasarım hizmetleri']
        },
        'obek_5': {
            'credit': lambda income, spending, ratio: 'İşletme kredisi' if 'Kendi İşinin Sahibi' in interest else 'Ticari kredi',
            'products': lambda interest, ratio: ['Ofis ekipmanları', 'İş geliştirme kaynakları'],
            'services': lambda income, ratio: ['İşletme danışmanlığı', 'Yatırım finansmanı hizmetleri']
        },
        'obek_6': {
            'credit': lambda income, spending, ratio: 'Tatil kredisi' if ratio < 2 else 'Lüks harcamalar için kredi',
            'products': lambda interest, ratio: ['Tatil paketleri', 'Lüks otel konaklamaları'],
            'services': lambda income, ratio: ['Seyahat planlama hizmetleri', 'Lüks yaşam tarzı danışmanlığı']
        },
        'obek_7': {
            'credit': lambda income, spending, ratio: 'Sağlık kredisi' if spending < 20000 else 'Kişisel ihtiyaç kredisi',
            'products': lambda interest, ratio: ['Sağlık ve wellness ürünleri', 'Fitness ekipmanları'],
            'services': lambda income, ratio: ['Sağlık sigortası danışmanlığı', 'Kişisel antrenörlük']
        },
        'obek_8': {
            'credit': lambda income, spending, ratio: 'Evlilik kredisi' if 'Düzenli ve Ücretli Bir İşi Var' in interest else 'Aile kredisi',
            'products': lambda interest, ratio: ['Düğün organizasyon ürünleri', 'Ev eşyaları'],
            'services': lambda income, ratio: ['Evlilik danışmanlığı', 'Aile bütçe yönetimi']
        }
    }

    cluster_recommendations = recommendations.get(cluster, {})
    credit = cluster_recommendations.get('credit', lambda *args: "Kredi önerisi bulunamadı.")(income, spending, ratio)
    products = cluster_recommendations.get('products', lambda *args: ["Ürün önerisi bulunamadı."])(interest, ratio)
    services = cluster_recommendations.get('services', lambda *args: ["Hizmet önerisi bulunamadı."])(income, ratio)

    return {
        "Kredi Önerisi": credit,
        "Ürün Önerileri": products,
        "Hizmet Önerileri": services
    }


# Tahmin sistemi
st.title("Müşteri Segmentasyon ve Karar Destek Sistemi")

# Kullanıcıdan giriş al
st.sidebar.header("Müşteri Bilgilerini Giriniz")

user_data = {
    'Yıllık Ortalama Gelir': st.sidebar.number_input('Yıllık Ortalama Gelir (TL)', min_value=1, help="Müşterinin yıllık ortalama gelirini giriniz."),
    'Yıllık Ortalama Satın Alım Miktarı': st.sidebar.number_input('Yıllık Ortalama Satın Alım Miktarı (TL)', min_value=1, help="Müşterinin yıllık ortalama satın alım miktarını giriniz."),
    'Yıllık Ortalama Sipariş Verilen Ürün Adedi': st.sidebar.number_input('Yıllık Ortalama Sipariş Verilen Ürün Adedi', min_value=0, help="Müşterinin yıllık ortalama sipariş verdiği ürün adedini giriniz."),
    'Yıllık Ortalama Sepete Atılan Ürün Adedi': st.sidebar.number_input('Yıllık Ortalama Sepete Atılan Ürün Adedi', min_value=0, help="Müşterinin yıllık ortalama sepete attığı ürün adedini giriniz."),
    'Cinsiyet': st.sidebar.radio('Cinsiyet', ['Kadın', 'Erkek'], help="Müşterinin cinsiyetini seçiniz."),
    'Yaşadığı Şehir': st.sidebar.selectbox('Yaşadığı Şehir', ['Büyük Şehir', 'Küçük Şehir', 'Kırsal'], help="Müşterinin yaşadığı şehri seçiniz."),
    'İstihdam Durumu': st.sidebar.selectbox('İstihdam Durumu', [
        'İşsiz veya Düzenli Bir İşi Yok', 
        'Düzenli ve Ücretli Bir İşi Var', 
        'Kendi İşinin Sahibi'
    ], help="Müşterinin istihdam durumunu seçiniz.")
}

# Gelir/Harcama Oranı hesapla
user_data['Gelir/Harcama_Oranı'] = user_data['Yıllık Ortalama Gelir'] / user_data['Yıllık Ortalama Satın Alım Miktarı']

# Kullanıcı girişlerini bir veri çerçevesine dönüştür
user_input_df = pd.DataFrame([user_data])
user_input_df = user_input_df.reindex(columns=X_train.columns, fill_value=0)

# Tahmin yap
user_cluster = optimized_model.predict(user_input_df)[0]

# Tahmin sonuçlarını göster
st.success(f"Tahmin Edilen Öbek: {user_cluster}")
st.write("Önerilerimiz:")
recommendations = get_recommendations(user_cluster, user_data)
st.write("Kredi Önerisi:", recommendations["Kredi Önerisi"])
st.write("Ürün Önerileri:")
for product in recommendations["Ürün Önerileri"]:
    st.write(f"- {product}")
st.write("Hizmet Önerileri:")
for service in recommendations["Hizmet Önerileri"]:
    st.write(f"- {service}")
