import pandas as pd
import numpy as np
import ast
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of books_recommender.py
glove_file = os.path.join(BASE_DIR, "assets", "glove.twitter.27B.50d.txt")
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
books_file = os.path.join(BASE_DIR, "assets", "books_1.Best_Books_Ever.csv")
books_data = pd.read_csv(books_file,
                         usecols=['title', 'author', 'rating', 'numRatings', 'description', 'language', 'genres'])
books_data = books_data.reset_index(drop=True)
keybert_file = os.path.join(BASE_DIR, "assets", "books_modified.csv")
keybert_keywords = pd.read_csv(keybert_file, usecols = ['keywords1'])

def get_top_keywords(row_idx, tfidf_mat, features, top_n=5):
    row = tfidf_mat.getrow(row_idx)
    nonzero_idx = row.nonzero()[1]
    scores = row.data
    top_indices = scores.argsort()[::-1][:top_n]
    return features[nonzero_idx][top_indices].tolist()


def extract_tfidf_keywords():
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[a-zA-Z]+\b', stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(books_data['description'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    books_data['keywords'] = [get_top_keywords(i, tfidf_matrix, feature_names) for i in range(len(books_data))]
    keybert_keywords['keywords1'] = keybert_keywords['keywords1'].apply(lambda keywords: keywords.split())
    books_data['keywords_comb'] = [
        list(set(k1 + k2))
        for k1, k2 in zip(books_data['keywords'], keybert_keywords['keywords1'])
    ]
    books_data['keywords'] = books_data['keywords_comb']
    return books_data


def words_to_vector(words, model=model):
    vectors = [model.get_vector(word) if word in model else np.zeros(model.vector_size) for word in words]
    return np.mean(vectors, axis=0)


def add_book_vectors():
    books_data['vector'] = books_data['keywords'].apply(words_to_vector)
    return books_data

books_data.drop_duplicates(subset=['title'], inplace=True)
books_data = books_data[books_data['language'] == 'English']
books_data["length"] = books_data['description'].apply(lambda d: len(d.split()) if isinstance(d, str) else 0)
books_data = books_data[books_data["length"] >= 4].copy()
books_data.dropna(subset=["description"], inplace=True)

books_data['genres'] = books_data['genres'].apply(ast.literal_eval)
books_data['genres'] = books_data['genres'].apply(lambda x: [g.lower() for g in x])
books_data['author'] = books_data['author'].apply(lambda x: x.lower())

# Rating score
books_data['rating_total'] = books_data['rating'] * books_data['numRatings']
extract_tfidf_keywords()
add_book_vectors()

X = np.stack(books_data['vector'].values, axis=0)
nn = NearestNeighbors(n_neighbors=6, metric="cosine")
nn.fit(X)


def get_preference():
  category = input("Do you want to look for recommendation according to author or genre? ")
  return category

def get_genres():
  genre = input("What kind of book genre do you like? If you want to put in 2 or more genres please separate them with a comma. ")
  return genre

def genre_match(gs, genre_input):
    return all(g.lower() in (x.lower() for x in gs) for g in genre_input)

def get_author():
  author = input("Who is the author you want to look for? ")
  return author

def get_keywords():
  keywords = input("What are the keywords you want to look for? ")
  return keywords.split()

def get_recommendation(user_input):
  if 'selected_idx' in user_input:
    selected_idx = user_input['selected_idx']
    book_vector = books_data.loc[selected_idx, 'vector']
    distances, indices = nn.kneighbors([book_vector]) # For two dimensional array as input
    recommended_books = books_data.iloc[indices.flatten()[1:]][['title', 'author', 'genres']]
  else :
    keywords = user_input['keywords']
    user_vector = words_to_vector(keywords)
    distances, indices = nn.kneighbors([user_vector])
    recommended_books = books_data.iloc[indices.flatten()[1:]][['title', 'author', 'genres']]

  return recommended_books

def show_books(book_list):
        for idx, row in book_list.iterrows():
            print(f"{row['title']} by {row['author']} (Genres: {', '.join(row['genres'])})")

def ablauf():
  global filtered_books
  user_input = {}
  preference = get_preference().lower()
  if preference == 'genre':
        genre_input = get_genres().lower().split(',')
        genre_input = [g.strip() for g in genre_input]
        print("User input:", genre_input)
        print("First book genres:", books_data['genres'].iloc[0])

        filtered_books = books_data[books_data['genres'].apply(lambda gs: genre_match(gs, genre_input))]

  elif preference == 'author':
        author_input = get_author().lower().strip()
        filtered_books = books_data[books_data['author'].str.contains(author_input)]

  else:
        print("Invalid preference. Please choose 'author' or 'genre'.")

  top_books = filtered_books.sort_values('rating_total', ascending=False).head(5)
  for idx, row in top_books.iterrows():
            print(f"{idx}. {row['title']} by {row['author']}")
  selected_idx = input("Pick the number of the book you like the most (or type 'none'):")

  if selected_idx.lower() == 'none':
        next_books = filtered_books.sort_values('rating', ascending=False).iloc[5:10]
        print("Here are the next 5 books:")
        for idx, row in next_books.iterrows():
            print(f"{idx}. {row['title']} by {row['author']}")
        selected_idx = input("Pick the number of the book you like the most (or type 'none'):")

  if selected_idx.lower() == 'none':
        keywords_input = get_keywords()
        user_input = {'keywords': keywords_input}
        recommendeds = get_recommendation(user_input)
        print("Here are the recommended books based on your keywords:")
        show_books(recommendeds)
        return recommendeds

  selected_idx = int(selected_idx)
  user_input['selected_idx'] = selected_idx
  selected_book = books_data.iloc[selected_idx]
  print(f"You selected: {selected_book['title']} by {selected_book['author']}")

  recommendeds = get_recommendation(user_input)
  print("Here are the recommended books:")
  show_books(recommendeds)

  return recommendeds

if __name__ == '__main__':
    ablauf()