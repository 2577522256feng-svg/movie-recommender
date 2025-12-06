import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==================== 第1步：加载数据 ====================
@st.cache_resource
def load_data():
    # 生成示例电影数据
    movies = pd.DataFrame({
        'movie_id': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'genre': ['Action', 'Comedy', 'Drama', 'Horror', 'Romance'] * 20,
        'rating': np.random.uniform(5, 9, 100)
    })
    
    # 生成用户-电影评分矩阵
    ratings = pd.DataFrame(
        np.random.randint(1, 6, (943, 100)),
        columns=[f'movie_{i}' for i in range(1, 101)]
    )
    
    return movies, ratings

# ==================== 第2步：推荐算法 ====================
def collaborative_filtering(user_id, movies_df, ratings_df, n_recommendations=5):
    # 用户向量
    user_vector = ratings_df.iloc[user_id-1].values.reshape(1, -1)
    
    # 计算相似度
    similarities = cosine_similarity(user_vector, ratings_df.values)[0]
    
    # 找相似用户
    similar_users = np.argsort(similarities)[-6:-1][::-1]
    
    # 推荐电影
    recommended_movies = []
    for sim_user in similar_users:
        user_ratings = ratings_df.iloc[sim_user]
        high_rated = user_ratings[user_ratings > 3].index.tolist()
        recommended_movies.extend(high_rated)
    
    # 获取唯一电影并返回
    unique_movies = list(set(recommended_movies))[:n_recommendations]
    return unique_movies

# ==================== 第3步：Streamlit 应用 ====================
st.set_page_config(page_title="🎬 电影推荐系统", layout="wide")

st.title("🎬 电影推荐系统")
st.markdown("---")

# 加载数据
movies_df, ratings_df = load_data()

# 侧边栏输入
st.sidebar.header("⚙️ 设置")
user_id = st.sidebar.slider("选择用户 ID", 1, 943, 1)
n_recommendations = st.sidebar.slider("推荐数量", 1, 10, 5)

# 获取推荐
recommended_movie_ids = collaborative_filtering(user_id, movies_df, ratings_df, n_recommendations)

# 显示结果
st.subheader(f"👤 用户 {user_id} 的推荐电影")
st.markdown("---")

if recommended_movie_ids:
    for idx, movie_id in enumerate(recommended_movie_ids, 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.metric("编号", idx)
            
            with col2:
                movie = movies_df[movies_df['movie_id'] == int(movie_id.split('_')[1])].iloc[0]
                st.write(f"**{movie['title']}**")
                st.write(f"类型: {movie['genre']} | ⭐ {movie['rating']:.1f}")
            
            st.markdown("---")
else:
    st.warning("暂无推荐")

# ==================== 第4步：页脚 ====================
st.markdown("---")
st.markdown("✨ **电影推荐系统** - 基于协同过滤算法")
