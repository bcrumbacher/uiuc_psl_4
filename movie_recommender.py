import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    sim_mat = pd.read_csv('sim_mat.csv', index_col=0)
    
    movies_df = pd.read_csv('movies.csv', header=None, names=['id', 'title', 'genres'])
    
    pop_df = pd.read_csv('pop_rank.csv')
    popularity_list = pop_df['id']
    
    movies_dict = movies_df.set_index('id')[['title', 'genres']].to_dict('index')
    
    return sim_mat, movies_dict, popularity_list

def get_poster_url(movie_id):
    return f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg?raw=true"

def myIBCF(newuser, ratings_matrix, similarity_matrix, popularity_list):
    popularity_ranks = {movie: rank for rank, movie in enumerate(popularity_list)}
    
    assert len(newuser) == similarity_matrix.shape[0], "Newuser vector must match similarity matrix dimensions"
    
    newuser_series = pd.Series(newuser, index=similarity_matrix.columns)
    
    rated_movies = newuser_series.index[~newuser_series.isna()]
    
    predictions = {}
    for target_movie in similarity_matrix.index:
        if target_movie in rated_movies:
            continue
        
        movie_similarities = similarity_matrix.loc[target_movie]
        
        valid_movies = movie_similarities.index[
            (~movie_similarities.isna()) & 
            (~newuser_series.isna())
        ]
        
        if len(valid_movies) == 0:
            continue
        
        similarities = movie_similarities.loc[valid_movies]
        user_ratings = newuser_series.loc[valid_movies]
        
        numerator = np.sum(similarities * user_ratings)
        denominator = np.sum(similarities)
        
        prediction = numerator / denominator if denominator != 0 else np.nan
        predictions[target_movie] = prediction
    
    predictions_series = pd.Series(predictions)
    
    if len(predictions_series) == 0:
        return popularity_list[:10]
    
    pred_df = pd.DataFrame({
        'prediction': predictions_series,
        'popularity_rank': pd.Series({
            movie: popularity_ranks.get(movie, len(popularity_list)) 
            for movie in predictions_series.index
        })
    })
    
    sorted_predictions = pred_df.sort_values(
        ['prediction', 'popularity_rank'],
        ascending=[False, True]
    )
    
    if len(sorted_predictions) >= 10:
        top_recommendations = sorted_predictions.index[:10].tolist()
    else:
        top_predictions = sorted_predictions.index.tolist()
        
        remaining_popular = [
            movie for movie in popularity_list 
            if movie not in rated_movies and movie not in top_predictions
        ]
        
        additional_needed = 10 - len(top_predictions)
        
        top_recommendations = top_predictions + remaining_popular[:additional_needed]
    
    return top_recommendations

def get_recommendations(user_ratings, sim_mat, movies_dict, popularity_list, n_recommendations=10):
    newuser = pd.Series(np.nan, index=sim_mat.columns)
    for movie_id, rating in user_ratings.items():
        movie_col = f'm{movie_id}'
        if movie_col in newuser.index:
            newuser[movie_col] = rating
    
    recommended_movies = myIBCF(newuser, None, sim_mat, popularity_list)
    
    recommendations = []
    for movie_col in recommended_movies:
        if movie_col.startswith('m'): 
            movie_id = int(movie_col[1:])  # Remove 'm' prefix
            if movie_id in movies_dict:
                movie_details = movies_dict[movie_id]
                score = 0
                for rated_movie, rating in user_ratings.items():
                    rated_movie_col = f'm{rated_movie}'
                    if rated_movie_col in sim_mat.index:
                        similarity = sim_mat.loc[movie_col, rated_movie_col]
                        if not pd.isna(similarity):
                            score += similarity * rating
                
                recommendations.append({
                    'id': movie_id,
                    'title': movie_details['title'],
                    'genres': movie_details['genres'],
                    'score': score
                })
    
    return recommendations


if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = pd.Series(dtype=float)
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
if 'rated_count' not in st.session_state:
    st.session_state.rated_count = 0
if 'movies_per_page' not in st.session_state:
    st.session_state.movies_per_page = 8

sim_mat, movies_dict, popularity_list = load_data()
valid_movie_ids = [int(col[1:]) for col in sim_mat.columns if col.startswith('m')]

st.title("Movie Recommendations")

total_pages = len(valid_movie_ids) // st.session_state.movies_per_page

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("Previous Page", disabled=st.session_state.current_page <= 0):
        st.session_state.current_page -= 1
        st.rerun()
with col2:
    st.write(f"Page {st.session_state.current_page + 1} of {total_pages + 1}")
with col3:
    if st.button("Next Page", disabled=st.session_state.current_page >= total_pages):
        st.session_state.current_page += 1
        st.rerun()

progress_message = f"You have rated {st.session_state.rated_count} movies"
st.write(progress_message)

start_idx = st.session_state.current_page * st.session_state.movies_per_page
end_idx = min(start_idx + st.session_state.movies_per_page, len(valid_movie_ids))

grid_placeholder = st.empty()

with grid_placeholder.container():
    for row in range(2):
        cols = st.columns(4)
        for col in range(4):
            idx = row * 4 + col
            if start_idx + idx < end_idx:
                movie_id = valid_movie_ids[start_idx + idx]
                
                with cols[col]:
                    with st.container():
                        if movie_id in movies_dict:
                            movie_details = movies_dict[movie_id]
                            st.image(get_poster_url(movie_id), use_container_width=True)
                            
                            st.markdown(f"**{movie_details['title']}**")
                            
                            st.markdown("<div style='height: 100px;'>", unsafe_allow_html=True)
                            
                            current_rating = st.session_state.user_ratings.get(movie_id, None)
                            if current_rating:
                                st.write(f"Your Rating: {current_rating}/5")
                            else:
                                rating = st.slider(
                                    "Rate",
                                    min_value=1,
                                    max_value=5,
                                    value=3,
                                    key=f"slider_{movie_id}"
                                )
                                if st.button("Submit", key=f"submit_{movie_id}"):
                                    st.session_state.user_ratings[movie_id] = rating
                                    st.session_state.rated_count += 1
                                    st.rerun()
                            
                            st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Your Recommended Movies:")

recommendations = get_recommendations(st.session_state.user_ratings, sim_mat, movies_dict, popularity_list)

if recommendations:
    for i, movie in enumerate(recommendations, 1):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(get_poster_url(movie['id']), use_container_width=True)
        
        with col2:
            st.subheader(f"{i}. {movie['title']}")
            st.write(f"Genres: {movie['genres']}")
        st.markdown("---")

st.subheader("Your Ratings:")
rating_cols = st.columns(4) 
for idx, (movie_id, rating) in enumerate(st.session_state.user_ratings.items()):
    if movie_id in movies_dict:
        with rating_cols[idx % 4]:
            st.image(get_poster_url(movie_id), use_container_width=True)
            st.write(f"**{movies_dict[movie_id]['title']}**")
            st.write(f"Rating: {rating}/5")
