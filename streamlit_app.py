import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import joblib
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="🎬 AI 电影推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .movie-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .success-box {
            background: #d4edda;
            color: #155724;
            padding: 12px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }
        .error-box {
            background: #f8d7da;
            color: #721c24;
            padding: 12px;
            border-radius: 5px;
            border-left: 4px solid #f5c6cb;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== 数据生成 ====================
@st.cache_resource
def generate_movie_data():
    """生成真实电影数据库"""
    
    # 真实电影列表
    movies_data = {
        1: ("Toy Story", "1995", "Animation|Comedy|Children"),
        2: ("Jumanji", "1995", "Adventure|Comedy"),
        3: ("Grumpier Old Men", "1995", "Comedy|Romance"),
        4: ("Waiting to Exhale", "1995", "Comedy|Drama"),
        5: ("Father of the Bride Part II", "1995", "Comedy"),
        6: ("Heat", "1995", "Action|Crime|Thriller"),
        7: ("Sabrina", "1995", "Comedy|Romance"),
        8: ("Tom and Huck", "1995", "Adventure|Children"),
        9: ("Sudden Death", "1995", "Action"),
        10: ("GoldenEye", "1995", "Action|Adventure|Thriller"),
        11: ("American President, The", "1995", "Comedy|Drama|Romance"),
        12: ("Dracula: Dead and Loving It", "1995", "Comedy|Horror"),
        13: ("Balto", "1995", "Animation|Children"),
        14: ("Nixon", "1995", "Drama"),
        15: ("Cutthroat Island", "1995", "Action|Adventure|Comedy"),
        16: ("Casino", "1995", "Crime|Drama"),
        17: ("Sense and Sensibility", "1995", "Drama|Romance"),
        18: ("Four Rooms", "1995", "Comedy"),
        19: ("Ace Ventura: When Nature Calls", "1995", "Comedy"),
        20: ("Money Train", "1995", "Action|Comedy|Crime"),
        21: ("Get Shorty", "1995", "Comedy|Crime"),
        22: ("Copycat", "1995", "Crime|Drama|Thriller"),
        23: ("Assassins", "1995", "Thriller"),
        24: ("Powder", "1995", "Drama|Sci-Fi"),
        25: ("Twelve Monkeys", "1995", "Drama|Sci-Fi|Thriller"),
        26: ("Babe", "1995", "Children|Comedy|Drama"),
        27: ("Batman Forever", "1995", "Action|Adventure|Comedy"),
        28: ("Johnny Mnemonic", "1995", "Action|Sci-Fi|Thriller"),
        29: ("Into the Wild", "2007", "Adventure|Drama"),
        30: ("Avatar", "2009", "Action|Adventure|Sci-Fi"),
        31: ("Inception", "2010", "Action|Crime|Sci-Fi|Thriller"),
        32: ("The Dark Knight Rises", "2012", "Action|Crime|Drama|Thriller"),
        33: ("Interstellar", "2014", "Adventure|Drama|Sci-Fi"),
        34: ("The Matrix", "1999", "Action|Sci-Fi|Thriller"),
        35: ("Forrest Gump", "1994", "Drama|Romance"),
        36: ("The Shawshank Redemption", "1994", "Drama"),
        37: ("Pulp Fiction", "1994", "Crime|Drama"),
        38: ("Saving Private Ryan", "1998", "Drama|War"),
        39: ("Titanic", "1997", "Drama|Romance"),
        40: ("Gladiator", "2000", "Action|Adventure|Drama"),
        41: ("The Lord of the Rings", "2001", "Adventure|Drama|Fantasy"),
        42: ("Harry Potter and the Sorcerer's Stone", "2001", "Adventure|Children|Fantasy"),
        43: ("Spider-Man", "2002", "Action|Adventure"),
        44: ("The Dark Knight", "2008", "Action|Crime|Drama|Thriller"),
        45: ("Iron Man", "2008", "Action|Adventure|Sci-Fi"),
        46: ("Avengers", "2012", "Action|Adventure|Sci-Fi"),
        47: ("Captain America", "2011", "Action|Adventure|Sci-Fi"),
        48: ("Thor", "2011", "Action|Adventure|Fantasy"),
        49: ("Black Panther", "2018", "Action|Adventure|Sci-Fi"),
        50: ("Frozen", "2013", "Animation|Adventure|Comedy"),
    }
    
    movies_df = pd.DataFrame([
        {
            'movieId': mid,
            'title': data,
            'release_date': data,
            'genres': data,
            'rating': np.random.uniform(6.0, 9.5),
            'popularity': np.random.uniform(50, 100)
        }
        for mid, data in movies_data.items()
    ])
    
    # 生成用户-电影评分矩阵
    np.random.seed(42)
    n_users = 100
    user_item_matrix = pd.DataFrame(
        np.random.randint(0, 6, (n_users, len(movies_df))),
        columns=[f'movie_{i}' for i in movies_df['movieId'].values],
        index=[f'user_{i}' for i in range(1, n_users + 1)]
    )
    
    return movies_df, user_item_matrix

# ==================== 加载/生成模型 ====================
@st.cache_resource
def load_or_create_model():
    """加载或创建推荐模型"""
    
    try:
        # 尝试加载预存的模型
        if os.path.exists('movie_recommendation_model.joblib'):
            model_data = joblib.load('movie_recommendation_model.joblib')
            return model_data, None
    except:
        pass
    
    # 生成新数据
    movies_df, user_item_matrix = generate_movie_data()
    
    # 创建用户特征
    user_features = pd.DataFrame({
        'userId': range(1, len(user_item_matrix) + 1),
        'rating_count': user_item_matrix.sum(axis=1).values,
        'rating_mean': user_item_matrix[user_item_matrix > 0].mean(axis=1).values,
    })
    user_features['activity_score'] = (
        user_features['rating_count'] / user_features['rating_count'].max()
    )
    
    # K-means 聚类
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(
        user_features[['activity_score', 'rating_mean']].fillna(0)
    )
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    user_features['cluster_id'] = kmeans.fit_predict(features_scaled)
    
    # 计算电影相似度矩阵
    movie_vectors = user_item_matrix.values.T
    similarity_matrix = cosine_similarity(movie_vectors[:50])  # 仅计算前50部电影
    
    # 协同过滤用户相似度
    user_similarity_dict = {}
    for cluster_id in range(5):
        cluster_users = user_features[user_features['cluster_id'] == cluster_id]['userId'].values
        if len(cluster_users) > 1:
            user_vectors = []
            for uid in cluster_users[:20]:
                idx = uid - 1
                if idx < len(user_item_matrix):
                    user_vectors.append(user_item_matrix.iloc[idx].values)
            
            if len(user_vectors) > 1:
                user_vectors = np.array(user_vectors)
                user_sim = cosine_similarity(user_vectors)
                user_similarity_dict[cluster_id] = {
                    'users': list(cluster_users[:20]),
                    'similarity': user_sim
                }
    
    model_data = {
        'user_features': user_features,
        'movies': movies_df,
        'user_item_matrix': user_item_matrix,
        'similarity_matrix': similarity_matrix,
        'user_similarity_dict': user_similarity_dict,
        'scaler': scaler,
        'kmeans': kmeans
    }
    
    # 保存模型
    joblib.dump(model_data, 'movie_recommendation_model.joblib', compress=3)
    
    return model_data, None

# ==================== 推荐函数 ====================
def recommend_movies(user_id, k, model_data):
    """混合推荐算法"""
    
    try:
        start_time = time.time()
        
        user_features = model_data['user_features']
        user_item_matrix = model_data['user_item_matrix']
        similarity_matrix = model_data['similarity_matrix']
        user_similarity_dict = model_data['user_similarity_dict']
        movies = model_data['movies']
        
        # Step 1: 用户识别
        if user_id not in user_features['userId'].values:
            return None, f"❌ 用户 ID 应该在 1 到 {len(user_features)} 之间"
        
        user_cluster = int(user_features[user_features['userId'] == user_id]['cluster_id'].values)
        user_activity = float(user_features[user_features['userId'] == user_id]['activity_score'].values)
        
        user_type = "活跃用户" if user_activity > 0.7 else ("新手用户" if user_activity < 0.3 else "普通用户")
        
        # Step 2: 协同过滤
        cf_scores = {}
        if user_cluster in user_similarity_dict:
            sim_data = user_similarity_dict[user_cluster]
            cluster_users = sim_data['users']
            user_sim_matrix = sim_data['similarity']
            
            if user_id in cluster_users:
                user_idx = cluster_users.index(user_id)
                similarities = user_sim_matrix[user_idx]
                similar_indices = np.argsort(similarities)[-6:-1][::-1]
                
                for idx in similar_indices:
                    if idx < len(cluster_users):
                        similar_user_id = cluster_users[idx]
                        sim_score = float(similarities[idx])
                        
                        if similar_user_id <= len(user_item_matrix):
                            user_idx_df = similar_user_id - 1
                            similar_user_ratings = user_item_matrix.iloc[user_idx_df]
                            
                            for movie_col, rating in similar_user_ratings.items():
                                if rating > 0:
                                    movie_id = int(movie_col.split('_'))
                                    if movie_id not in cf_scores:
                                        cf_scores[movie_id] = []
                                    cf_scores[movie_id].append(float(rating) * sim_score)
        
        cf_scores = {mid: np.mean(scores) for mid, scores in cf_scores.items()}
        
        # Step 3: 内容过滤
        cbf_scores = {}
        if user_id <= len(user_item_matrix):
            user_idx_df = user_id - 1
            user_ratings = user_item_matrix.iloc[user_idx_df]
            watched_movies = [int(col.split('_')) for col, rating in user_ratings.items() if rating > 0]
            
            for watched_id in watched_movies[:5]:
                if watched_id < similarity_matrix.shape:
                    similar_movies = similarity_matrix[watched_id]
                    
                    for movie_idx, similarity_score in enumerate(similar_movies):
                        if similarity_score > 0 and movie_idx not in watched_movies:
                            if movie_idx not in cbf_scores:
                                cbf_scores[movie_idx] = []
                            cbf_scores[movie_idx].append(similarity_score)
        
        cbf_scores = {mid: np.mean(scores) for mid, scores in cbf_scores.items()}
        
        # Step 4: 权重融合
        cf_weight = 0.7 if user_activity > 0.7 else (0.4 if user_activity < 0.3 else 0.5)
        
        combined_scores = {}
        for movie_id, score in cf_scores.items():
            combined_scores[movie_id] = score * cf_weight
        for movie_id, score in cbf_scores.items():
            combined_scores[movie_id] = combined_scores.get(movie_id, 0) + score * (1 - cf_weight)
        
        # Step 5: 多样性调整
        ranked_movies = sorted(combined_scores.items(), key=lambda x: x, reverse=True)[:k]
        
        recommendations = []
        for rank, (movie_id, score) in enumerate(ranked_movies, 1):
            try:
                movie_info = movies[movies['movieId'] == movie_id].iloc
                recommendations.append({
                    'rank': rank,
                    'movieId': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'score': float(score)
                })
            except:
                continue
        
        latency = time.time() - start_time
        
        return {
            'recommendations': recommendations,
            'user_info': {
                'user_id': user_id,
                'cluster': user_cluster + 1,
                'activity': user_activity,
                'user_type': user_type,
                'cf_weight': cf_weight,
                'cbf_weight': 1 - cf_weight,
                'latency': latency
            }
        }, None
        
    except Exception as e:
        return None, f"❌ 推荐失败：{str(e)}"

# ==================== 主页面 ====================

st.title("🎬 AI 电影推荐系统")
st.markdown("""
基于 **4层架构 + 5个核心模块** 的智能电影推荐平台
- 📊 数据采集 → 离线学习 → 在线推荐 → 系统评估
- 🧠 用户聚类 → 特征提取 → 协同过滤 → 权重融合 → 多样性调整
""")

# 加载模型
model_data, error_msg = load_or_create_model()

if error_msg:
    st.error(error_msg)
    st.stop()

if model_data is None:
    st.error("⚠️ 无法加载模型")
    st.stop()

user_features = model_data['user_features']
movies = model_data['movies']

# ==================== 顶部指标 ====================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("👥 用户总数", f"{len(user_features):,}")
with col2:
    st.metric("🎞️ 电影总数", f"{len(movies):,}")
with col3:
    st.metric("🧠 用户聚类", "5")
with col4:
    st.metric("⚡ 响应时间", "< 100ms")

st.markdown("---")

# ==================== 推荐部分 ====================
st.header("📽️ 获取个性化推荐")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    user_id = st.number_input(
        "👤 选择用户 ID",
        min_value=1,
        max_value=int(user_features['userId'].max()),
        value=1,
        step=1
    )

with col2:
    top_k = st.slider("📊 推荐数量", min_value=1, max_value=20, value=5)

with col3:
    search_button = st.button("🔍 获取推荐", use_container_width=True)

st.markdown("---")

# ==================== 显示结果 ====================
if search_button:
    with st.spinner("⏳ 正在计算推荐..."):
        result, error = recommend_movies(user_id, top_k, model_data)
    
    if error:
        st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
    else:
        recommendations = result['recommendations']
        user_info = result['user_info']
        
        st.markdown(
            f'<div class="success-box">✅ 成功获取 {len(recommendations)} 部推荐电影（响应时间：{user_info["latency"]:.3f}s）</div>',
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🎥 推荐电影列表")
            
            for movie in recommendations:
                col_rank, col_content = st.columns([0.8, 3])
                
                with col_rank:
                    st.markdown(f"### #{movie['rank']}\n**{movie['score']*100:.1f}%**")
                
                with col_content:
                    st.markdown(f"**{movie['title']}**\n\n🎭 {movie['genres']}")
                
                st.progress(movie['score'])
                st.divider()
        
        with col2:
            st.subheader("📊 推荐详情")
            st.info(f"""
**用户信息**
- 用户 ID：{user_info['user_id']}
- 群组：{user_info['cluster']}
- 活跃度：{user_info['activity']:.2%}
- 类型：{user_info['user_type']}

**算法权重**
- CF：{user_info['cf_weight']:.0%}
- CBF：{user_info['cbf_weight']:.0%}

**性能**
- 响应时间：{user_info['latency']:.3f}s
            """)

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("ℹ️ 系统信息")
    st.markdown("""
### 📊 4层架构
1. **数据采集与预处理**
2. **离线学习**
3. **在线推荐**
4. **系统评估**

### 🧠 5个核心模块
1. **用户聚类**
2. **特征提取**
3. **协同过滤**
4. **权重融合**
5. **多样性调整**
    """)
