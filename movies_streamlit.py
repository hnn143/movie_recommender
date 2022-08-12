import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))

movies = pd.read_csv('C:/Users/chicmachina/Desktop/WBS/recommender/ml-latest-small/movies.csv')
ratings = pd.read_csv('C:/Users/chicmachina/Desktop/WBS/recommender/ml-latest-small/ratings.csv')
links = pd.read_csv('C:/Users/chicmachina/Desktop/WBS/recommender/ml-latest-small/links.csv')
tags = pd.read_csv('C:/Users/chicmachina/Desktop/WBS/recommender/ml-latest-small/tags.csv')

## Data preparation

genres = movies.pop('genres').str.split('|', expand=True)

genres = genres.stack().reset_index().drop(['level_1'], axis=1).rename(columns={'level_0':'old_index', 0:'genre'})

genres['movieId'] = genres.old_index+1

genres_wide = pd.get_dummies(genres['genre'])

genres_wide = genres_wide.join(genres['movieId'])

movies_full = movies.merge(genres_wide.groupby('movieId').sum().reset_index(), on='movieId')

# to only get years, not sequels
movies_full['year']= movies_full.title.str.extract('([\(\d\)]+)', expand=False).str.replace('\(','').str.replace('\)','')
# to only remove years, not sequels
movies_full['title_only']=movies_full.title.str.replace('([\(\d\)]+)', '').str.strip()

## recommend

## by ratings

m_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
m_rating['rating_count'] = ratings.groupby('movieId')['rating'].count()

def top_movies(n):
    top = m_rating[m_rating['rating_count']>1].sort_values(['rating', 'rating_count'], ascending=False).head(n)
    top = top.merge(movies, on='movieId')
    return top

## by popularity weighted ratings

m_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
m_rating['rating_count'] = ratings.groupby('movieId')['rating'].count()
m_rating['weight'] = m_rating['rating_count']/m_rating['rating_count'].sum()
m_rating['weighted_rating'] = m_rating['rating']*m_rating['weight']

def top_movies_weight(n):
    top = m_rating.sort_values(['weighted_rating'], ascending=False).head(n)
    top = top.merge(movies, on='movieId')
    return top

### by popularity rated ranking for specific genre

def top_movies_weight_option(option='all', n=5):
    if option == 'all':
        # list of all movies
        option_movies = movies_full['movieId'].to_list()
    else:
        # list of movies from the selected genre by movieId
        option_movies = movies_full.query(f'{option}==1')['movieId'].to_list()
        
    # average ratings and rating count for movies of the same genre
    m_rating_option = pd.DataFrame(ratings.loc[ratings.movieId.isin(option_movies)].groupby('movieId')['rating'].mean())
    m_rating_option['rating_count'] = ratings.loc[ratings.movieId.isin(option_movies)].groupby('movieId')['rating'].count()
    # popularity weights
    m_rating_option['weight'] = m_rating_option['rating_count']/m_rating_option['rating_count'].sum()
    m_rating_option['weighted_rating'] = m_rating_option['rating']*m_rating_option['weight']
    top = m_rating_option.sort_values(['weighted_rating'], ascending=False).head(n)
    top = top.merge(movies, on='movieId')
    return top

### by movie similarity

user_movie_pivot = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')

# at least 10 recommendations

def recomm_item_based(movie_name, n):
    if len(movies.loc[movies.title.str.contains(movie_name),'movieId'])>1:
        return 'ambigious name'
    else:
        movie = movies.loc[movies.title.str.contains(movie_name),'movieId'].item()
        Id_ratings = user_movie_pivot[movie]
        corr_to_Id = pd.DataFrame(user_movie_pivot.corrwith(Id_ratings), columns=['PearsonR'])
        corr_to_Id.dropna(inplace=True)
        corr_to_Id = corr_to_Id.join(m_rating[['rating', 'rating_count']])
        corr_to_Id.drop(movie, inplace=True) # drop movie itself
        top = corr_to_Id[corr_to_Id['rating_count']>=10].sort_values(['PearsonR', 'rating', 'rating_count'], ascending=False).head(n)
        top = top.merge(movies, left_index=True, right_on="movieId")
        name_movie = movies.loc[movies.movieId==movie]['title'].item()
        print(f'The {n} most similar movies to {name_movie} are:')
        return top

# at least 10 common recommenders

def recomm_item_based10(movie_name, n):
    if len(movies.loc[movies.title.str.contains(movie_name),'movieId'])>1:
        return 'ambigious name'
    else:
        movie = movies.loc[movies.title.str.contains(movie_name),'movieId'].item()
        common_viewers = user_movie_pivot.loc[user_movie_pivot[movie].notna(), :] # only those viewers that saw the movie
        common_viewers = common_viewers.loc[:,common_viewers.notna().count()>=10] # and only those movies with 10 common viewers
        rest_ratings = common_viewers[movie] # ratings of these viewers
        corr_to_Id = pd.DataFrame(common_viewers.corrwith(rest_ratings), columns=['PearsonR']) # corr with these movies
        corr_to_Id.dropna(inplace=True)
        corr_to_Id = corr_to_Id.join(m_rating[['rating','rating_count']], how='left')
        corr_to_Id.drop(movie, inplace=True) # drop movie itself
        top = corr_to_Id.sort_values(['PearsonR','rating', 'rating_count'], ascending=False).head(n)
        top = top.merge(movies, left_index=True, right_on="movieId")
        name_movie = movies.loc[movies.movieId==movie]['title'].item()
        print(f'The {n} most similar movies to {name_movie} are:')
        return top

### by viewer similarities

user_movie_null = user_movie_pivot.fillna(0)

viewer_sim = pd.DataFrame(cosine_similarity(user_movie_null),
                                 columns=user_movie_null.index, 
                                 index=user_movie_null.index)

def recomm_user_based(viewer,n=5):
    weights = viewer_sim.query('userId!=@viewer')[viewer]/sum(viewer_sim.query('userId!=@viewer')[viewer])
    unseen_movies = user_movie_null.loc[user_movie_null.index!=viewer, user_movie_null.loc[viewer,:]==0]
    weighted_average = pd.DataFrame(unseen_movies.T.dot(weights), columns=['weighted_rating'])
    top = weighted_average.join(m_rating[['rating','rating_count']], how='left')
    top = top.sort_values(['weighted_rating','rating', 'rating_count'], ascending=False).head(n)
    top = top.merge(movies, left_index=True, right_on="movieId")
    print(f'for viewer {viewer} we recommend the following {n} movies based on ratings of viewers with similar taste:')
    return top    

