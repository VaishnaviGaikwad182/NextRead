import streamlit as st
import base64
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= CSS =================
st.markdown(
    """
    <style>
    /* Hide only the deploy/share button */
    button[kind="secondary"][title="Share"] {
        display: none;
    }

    /* Hide footer */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="NextRead",
    page_icon="NextRead.png",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
.metric-card {
    background-color: #f4f8fb;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e3eaf2;
    text-align: center;
}
.spacer { margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================

# Function to convert local image to base64
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your logo
logo_path = "NextRead.png"
logo_base64 = get_base64_image(logo_path)

# Sidebar title with logo + text inline
st.sidebar.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="60" style="border-radius: 12px; margin-right: 12px;">
        <h2 style="margin: 0;">NextRead</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar menu
menu = st.sidebar.radio(
    "",
    ["📖 Book Recommendation", "💸 Discount Prediction", "📊 Clustering", "📈 Summary & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.info("Your Personal Library Assistant\nFind books you’ll love, predict discounts, and explore the library like never before!")
st.sidebar.markdown(
    """
    **✨ About NextRead**  
    - Get personalized book suggestions  
    - Estimate discounts before you buy  
    - Visualize and explore similar books     
    """)


# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("Books.csv")

df = load_data()

def clean_numeric(col):
    col = col.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(col, errors='coerce')

df['Price'] = clean_numeric(df['Price'])
df['Pages'] = clean_numeric(df['Pages'])
df['Discount Percent'] = clean_numeric(df['Discount Percent'])
df = df.dropna(subset=['Price','Pages','Discount Percent']).reset_index(drop=True)

def clean_text(text):
    text = str(text).upper()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# ================= RECOMMENDER =================
@st.cache_resource
def build_recommender(df):
    bt = df.copy().reset_index(drop=True)
    for c in ['Title','Author','Publisher']:
        bt[c] = bt[c].apply(clean_text)

    bt['text'] = bt['Title'] + " " + bt['Author'] + " " + bt['Publisher']
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(bt['text'])
    similarity = cosine_similarity(tfidf_matrix)
    return bt, similarity, tfidf_matrix

bt, similarity, tfidf_matrix = build_recommender(df)
book_titles = bt['Title'].unique().tolist()

def recommend_books(title, top_n=5):
    idx = bt.index[bt['Title'] == title][0]
    scores = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]
    rec_idx = [i[0] for i in scores]
    return bt.iloc[rec_idx][['Title','Author','Publisher','Price']], scores, idx, rec_idx

# ================= DISCOUNT MODEL =================
@st.cache_resource
def train_discount_model(df):
    X = df[['Price','Pages']]
    y = df['Discount Percent']
    scaler = StandardScaler()
    model = LinearRegression()
    model.fit(scaler.fit_transform(X), y)
    return model, scaler

discount_model, scaler = train_discount_model(df)

# ================= TABS =================

# ---------- 1. BOOK RECOMMENDATION ----------
if menu == "📖 Book Recommendation":
    st.title("📖 Smart Library Intelligence and Recommendation System")
    st.caption("Discover personalized book recommendations based on content similarity. Find your next favorite read using TF-IDF and smart similarity analysis.")

    book_name = st.selectbox("🔍 Search Book", book_titles, index=None)

    if st.button("✨ Recommend") and book_name:
        recs, scores, _, rec_idx = recommend_books(book_name)

        st.subheader("📚 Recommended Books")
        st.dataframe(recs, use_container_width=True)

        # ---- BAR CHART ----
        st.subheader("📊 Similarity Scores")
        fig, ax = plt.subplots()
        ax.barh(recs['Title'], [s[1] for s in scores], color="#74b9ff")
        ax.invert_yaxis()
        ax.set_xlabel("Similarity")
        st.pyplot(fig)

# ---------- 2. DISCOUNT ----------
elif menu == "💸 Discount Prediction":
    st.title("💸 Discount Prediction")
    st.caption("Easily Predict Discounts and Discover How Much You Can Save on Any Book Before You Buy")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        price = st.text_input("📘 Book Price (₹)", placeholder="500")
    with col2:
        pages = st.text_input("📄 Pages", placeholder="300")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🎯 Predict")

    if predict and price and pages:
        price, pages = float(price), float(pages)
        X = scaler.transform([[price, pages]])
        discount = discount_model.predict(X)[0]
        final_price = price - (price * discount / 100)

        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><h4>Discount</h4><h2>{discount:.2f}%</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h4>Original Price</h4><h2>₹ {price:.0f}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h4>Final Price</h4><h2>₹ {final_price:.0f}</h2></div>", unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.bar(["Original","Final"], [price, final_price],
               color=["#ffeaa7", "#81ecec"])
        ax.set_ylabel("Price (₹)")
        st.pyplot(fig)

# ---------- 3. CLUSTERING ----------
elif menu == "📊 Clustering":
    st.title("📊 Book Clustering Visualization")
    st.caption("Explore Groups of Similar Books and Visualize Their Relationships to Find Your Next Read")

    cluster_title = st.selectbox("Select Book", book_titles, index=None)

    if st.button("📍 Show Cluster") and cluster_title:
        cluster_books, _, selected_idx, cluster_idx = recommend_books(cluster_title, top_n=8)
        st.dataframe(cluster_books, use_container_width=True)

        pca = PCA(n_components=2)
        points = pca.fit_transform(tfidf_matrix.toarray())

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(points[:,0], points[:,1], color="#dfe6e9", alpha=0.4)
        ax.scatter(points[cluster_idx,0], points[cluster_idx,1],
                   color="#0984e3", s=80, label="Cluster")
        ax.scatter(points[selected_idx,0], points[selected_idx,1],
                   color="red", s=200, marker="*", label="Selected Book")

        ax.set_title("Book Position in Cluster (PCA)")
        ax.legend()
        st.pyplot(fig)

# ---------- 4. SUMMARY ----------
elif menu == "📈 Summary & Insights":
    st.title("📈 Dataset Summary & Insights")
    st.caption("Gain a comprehensive overview of the library dataset, explore key trends, and visualize patterns in book prices, discounts, authors, and publishers")

    st.subheader("📋 Dataset Overview")
    st.dataframe(df.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        df['Price'].plot(kind='hist', bins=20, color='#20b2aa', ax=ax)
        ax.set_title("Price Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        df['Discount Percent'].plot(kind='hist', bins=20, color='#ff6347', ax=ax)
        ax.set_title("Discount Distribution")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        df['Author'].value_counts().head(10).plot(kind='bar', color='#4682b4', ax=ax)
        ax.set_title("Top Authors")
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        df['Publisher'].value_counts().head(10).plot(kind='bar', color='#9acd32', ax=ax)
        ax.set_title("Top Publishers")
        st.pyplot(fig)
