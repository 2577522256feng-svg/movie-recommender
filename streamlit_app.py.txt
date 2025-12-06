# ==================== 第1步：安装依赖 ====================
!pip install streamlit pyngrok -q

# ==================== 第2步：创建 Streamlit 应用 ====================
# 使用双引号作为外层，避免冲突
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="🎬 电影推荐系统", page_icon="🎬", layout="wide")

st.markdown('''
    <style>
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
''', unsafe_allow_html=True)

# 从全局变量获取数据
@st.cache_resource
def get_model_data():
    try:
        return {
            'user_features': user_features,
            'user_item_matrix': user_item_matrix,
            'similarity_matrix': similarity_matrix,
            'user_similarity_dict': user_similarity_dict,
            'movies': movies
        }, None
    except Exception as e:
        return None, f"❌ 加载失败：{str(e)}"

def recommend_movies(user_id, k, model_data):
    try:
        start_time = time.time()
        
        user_features = model_data['user_features']
        user_item_matrix = model_data['user_item_matrix']
        similarity_matrix = model_data['similarity_matrix']
        user_similarity_dict = model_data['user_similarity_dict']
        movies = model_data['movies']
        
        if user_id not in user_features['userId'].values:
            return None, "❌ 用户不存在"
        
        user_cluster = int(user_features[user_features['userId'] == user_id]['cluster_id'].values)
        user_activity = float(user_features[user_features['userId'] == user_id]['activity_score'].values)
        
        # 协同过滤
        cf_candidates = {}
        if user_cluster in user_similarity_dict:
            sim_data = user_similarity_dict[user_cluster]
            cluster_users = list(sim_data['users']) if not isinstance(sim_data['users'], list) else sim_data['users']
            user_sim_matrix = sim_data['similarity']
            
            if user_id in cluster_users:
                user_idx = cluster_users.index(user_id)
                similarities = user_sim_matrix[user_idx]
                similar_indices = np.argsort(similarities)[-6:-1][::-1]
                
                for idx in similar_indices:
                    if idx < len(cluster_users):
                        similar_user_id = cluster_users[idx]
                        sim_score = float(similarities[idx])
                        
                        if similar_user_id in user_item_matrix.index:
                            similar_user_ratings = user_item_matrix.loc[similar_user_id]
                            
                            for movie_id, rating in similar_user_ratings.items():
                                if rating > 0:
                                    if movie_id not in cf_candidates:
                                        cf_candidates[movie_id] = []
                                    cf_candidates[movie_id].append(float(rating) * sim_score)
        
        cf_scores = {mid: np.mean(scores) for mid, scores in cf_candidates.items()}
        
        # 内容过滤
        cbf_candidates = {}
        if user_id in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user_id]
            watched_movies = [mid for mid in user_ratings.index if user_ratings[mid] > 0]
            
            for watched_id in watched_movies[:5]:
                if watched_id < similarity_matrix.shape:
                    similar_movies = similarity_matrix[watched_id]
                    
                    for movie_idx, similarity_score in enumerate(similar_movies):
                        if similarity_score > 0 and movie_idx not in watched_movies:
                            if movie_idx not in cbf_candidates:
                                cbf_candidates[movie_idx] = []
                            cbf_candidates[movie_idx].append(similarity_score)
        
        cbf_scores = {mid: np.mean(scores) for mid, scores in cbf_candidates.items()}
        
        # 权重融合
        cf_weight = 0.7 if user_activity > 0.7 else (0.4 if user_activity < 0.3 else 0.5)
        
        combined_scores = {}
        for movie_id, score in cf_scores.items():
            combined_scores[movie_id] = score * cf_weight
        
        for movie_id, score in cbf_scores.items():
            if movie_id in combined_scores:
                combined_scores[movie_id] += score * (1 - cf_weight)
            else:
                combined_scores[movie_id] = score * (1 - cf_weight)
        
        # 多样性调整
        if user_id in user_item_matrix.index:
            watched_ids = user_item_matrix.index.tolist()
            for movie_id in list(combined_scores.keys()):
                for watched_id in watched_ids[:3]:
                    if watched_id < similarity_matrix.shape and movie_id < similarity_matrix.shape:
                        if similarity_matrix[watched_id, movie_id] > 0.9:
                            combined_scores[movie_id] *= 0.7
        
        ranked_movies = sorted(combined_scores.items(), key=lambda x: x, reverse=True)[:k]
        
        recommendations = []
        for rank, (movie_id, score) in enumerate(ranked_movies, 1):
            try:
                movie_info = movies[movies['movieId'] == movie_id].iloc
                recommendations.append({
                    'rank': rank,
                    'title': str(movie_info['title']),
                    'genres': str(movie_info['genres']),
                    'score': float(score)
                })
            except:
                continue
        
        latency = time.time() - start_time
        return {'recommendations': recommendations, 'latency': latency}, None
        
    except Exception as e:
        return None, f"❌ 推荐失败：{str(e)}"

# UI
st.title("🎬 AI 电影推荐系统")

model_data, error_msg = get_model_data()

if error_msg:
    st.error(error_msg)
    st.stop()

user_features = model_data['user_features']
movies = model_data['movies']
user_item_matrix = model_data['user_item_matrix']

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("👥 用户", f"{len(user_features):,}")
with col2:
    st.metric("🎞️ 电影", f"{len(movies):,}")
with col3:
    st.metric("⭐ 评分", f"{len(user_item_matrix) * len(user_item_matrix.columns):,}")
with col4:
    st.metric("🧠 聚类", "5")
st.markdown("---")

st.header("📽️ 获取推荐")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    user_id = st.number_input("👤 用户ID", min_value=1, max_value=int(user_features['userId'].max()), value=100)
with col2:
    top_k = st.slider("📊 数量", 1, 20, 5)
with col3:
    search_button = st.button("🔍 推荐", use_container_width=True)

st.markdown("---")

if search_button:
    with st.spinner("⏳ 计算中..."):
        result, error = recommend_movies(user_id, top_k, model_data)
    
    if error:
        st.markdown(f'<div class="error-box">{error}</div>', unsafe_allow_html=True)
    else:
        recommendations = result['recommendations']
        st.markdown(f'<div class="success-box">✅ 成功获取 {len(recommendations)} 部电影（{result["latency"]:.3f}s）</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🎥 推荐结果")
            for movie in recommendations:
                col_rank, col_content = st.columns([0.8, 3])
                with col_rank:
                    st.markdown(f"### #{movie['rank']}\\n**{movie['score']*100:.1f}%**")
                with col_content:
                    st.markdown(f"**{movie['title']}**\\n🎭 {movie['genres']}")
                st.progress(movie['score'])
                st.divider()
        
        with col2:
            st.subheader("📊 详情")
            st.info(f"用户ID: {user_id}\\n推荐数: {len(recommendations)}\\n平均分: {np.mean([m['score'] for m in recommendations]):.2%}")

with st.sidebar:
    st.header("ℹ️ 系统")
    st.markdown("### 📊 4层架构\\n1. 数据采集\\n2. 离线学习\\n3. 在线推荐\\n4. 系统评估\\n\\n### 🧠 5个模块\\n1. 用户聚类\\n2. 特征提取\\n3. 协同过滤\\n4. 权重融合\\n5. 多样性调整")
"""

# 保存到文件
with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(streamlit_code)

print("✅ Streamlit 应用已创建")

# ==================== 第3步：启动应用 ====================
import subprocess
import time
from pyngrok import ngrok

# 启动 Streamlit
print("🚀 启动 Streamlit 应用...")
process = subprocess.Popen(
    ['streamlit', 'run', 'streamlit_app.py', '--server.port', '8501', '--server.headless', 'true'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# 等待启动
time.sleep(5)

# 创建 ngrok 链接
print("🌐 创建公网链接...")
try:
    public_url = ngrok.connect(8501)
    print(f"\n{'='*60}")
    print(f"✅ 应用已启动！")
    print(f"{'='*60}")
    print(f"🌐 公网链接：{public_url}")
    print(f"⏱️  应用在后台运行")
    print(f"📱 你现在可以在浏览器中访问应用了！")
    print(f"{'='*60}\n")
except Exception as e:
    print(f"⚠️  ngrok 连接失败：{str(e)}")
    print(f"📱 本地链接：http://localhost:8501")
