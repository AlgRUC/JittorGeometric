INTERACTIONs DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file ml-1m.inter comprising the ratings of users over the movies.
Each record/line in the file has the following fields: user_id, item_id, rating, timestamp

user_id: the id of the users and its type is token. 
item_id: the id of the movies and its type is token.
rating: the rating of the users over the movies, and its type is float.
timestamp: the UNIX timestamp of the rating, and its type is float.

MOVIES INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file ml-1m.item comprising the attributes of the movies.
Each record/line in the file has the following fields: item_id, movie_title, release_year, genre
 
item_id: the id of the movies and its type is token.
movie_title: the title of the movies, and its type is token_seq.
release_year: the year when movies were released, and its type is float.
genre: the genres of the movies, and its type is token_seq.


USERS INFORMATION DATASET FILE DESCRIPTION
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
The file ml-1m.user comprising the attributes of the users.
Each record/line in the file has the following fields: user_id, age, gender, occupation and zip_code
 
user_id: the id of the users and its type is token.
age: the age of the users, and its type is float.
gender: the gender of the users, and its type is token.
occupation: the occupation of the users, and its type is token.
zip_code: the zip_code of the users, and its type is token.