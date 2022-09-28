import os
import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


books = pd.read_csv('books.csv')
tags = pd.read_csv('tags.csv')
book_tags = pd.read_csv('book_tags.csv')


tags = pd.merge( book_tags,tags, left_on = 'tag_id', right_on = 'tag_id', how = 'inner')

books_merged = pd.merge(books, tags, left_on ='book_id', right_on = 'goodreads_book_id',how = 'inner')

for i in books_merged.columns:
    books_merged[i] = books_merged[i].fillna(' ')

for i in books.columns:
    books[i] = books[i].fillna(' ')

joined_merged_books = books_merged.groupby('book_id')['tag_name'].apply(' '.join).reset_index()

books= pd.merge(books,joined_merged_books,left_on = 'book_id', right_on = 'book_id', how= 'left')


columns = ['authors','title','language_code','tag_name']
for i in columns:
    books[i] = books[i].fillna(' ')

def combine_features(row):
    return row['authors']+' '+row['tag_name']+' '+row['title']

books['combined_features'] = books.apply(combine_features,axis = 1)


vectorizer = CountVectorizer(ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = vectorizer.fit_transform(books['combined_features'])

cosine_similarity_score = cosine_similarity(count_matrix)


liked_book = input("Enter the book name that you like: ")
n = int(input("No of similar books you want to display: "))

def from_title_get_index(title):
    return books[books.title == title].index.values[0]

index_of_books = from_title_get_index(liked_book)

books_similar = list(enumerate(cosine_similarity_score[index_of_books]))

sorted_books_similar =  sorted(books_similar, key = lambda x:x[1], reverse = True)

def from_index_get_title(index):
    return books[books.index == index].title.values[0]


i = 0
for book_i in sorted_books_similar:
    print(i,from_index_get_title(book_i[0]))
    i=i+1
    if i>n:
        break
