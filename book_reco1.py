#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# In[2]:


st.header("Book Recommender Engine") 


# In[3]:


books=pd.read_csv("C:\\Users\\adite\\Downloads\\Dataset1\\Books.csv",low_memory=False)
ratings=pd.read_csv("C:\\Users\\adite\\Downloads\\Dataset1\\Ratings.csv",low_memory=False)
users=pd.read_csv("C:\\Users\\adite\\Downloads\\Dataset1\\Users.csv",low_memory=False)


# In[4]:


#add_selectbox = st.sidebar.selectbox(
#    "What do you want to see?",
#    ("Datasets and information", "Visualization and EDA", "Recommendations"))


# In[5]:


#st.sidebar.radio("What do you want to see?",["Datasets and information", "Visualization and EDA", "Recommendations"])


# In[ ]:





# In[6]:


books['ISBN'] = books['ISBN'].str.upper()


# In[7]:


books.drop_duplicates(keep='last', inplace=True)  #We have droped duplicates and reset index
books.reset_index(drop = True, inplace = True)


# In[8]:


#books["Year-Of-Publication"].unique()


# In[9]:


pd.set_option('display.max_colwidth', None)
#books[books["Year-Of-Publication"]=="DK Publishing Inc"]


# In[10]:


#books[books["Year-Of-Publication"]=="Gallimard"]


# In[11]:


def replace_correct_value(df,idx,column,val):
    df.loc[idx,column] = val
    #return df


# In[12]:


replace_correct_value(books,209232,"Book-Title","DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers")
replace_correct_value(books,209232,"Book-Author","Michael Teitelbaum")
replace_correct_value(books,209232,"Year-Of-Publication",2000)
replace_correct_value(books,209232,"Publisher","DK Publishing Inc")


replace_correct_value(books,221369,"Book-Title","DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers")
replace_correct_value(books,221369,"Book-Author","James Buckley")
replace_correct_value(books,221369,"Year-Of-Publication",2000)
replace_correct_value(books,221369,"Publisher","DK Publishing Inc")


replace_correct_value(books,220422,"Book-Title","Peuple du ciel, suivi de 'Les Bergers")
replace_correct_value(books,220422,"Book-Author","Jean-Marie Gustave Le ClÃ?Â©zio")
replace_correct_value(books,220422,"Year-Of-Publication",2003)
replace_correct_value(books,220422,"Publisher","Gallimard")


# In[13]:


mode=books["Year-Of-Publication"].mode()


# In[14]:


books["Year-Of-Publication"]=books["Year-Of-Publication"].replace([0],2002)


# In[15]:


books["Year-Of-Publication"]=books["Year-Of-Publication"].astype(int)


# In[16]:


books.loc[books['Year-Of-Publication'] > 2022, 'Year-Of-Publication'] = 2002


# In[17]:


books["Year-Of-Publication"]=books["Year-Of-Publication"].replace([0],2002)


# In[18]:


books["Book-Author"]=books[["Book-Author"]].fillna("Others")


# In[19]:


books["Publisher"]=books[["Publisher"]].fillna("Others")


# In[20]:


books=books.drop(["Image-URL-S","Image-URL-L"],axis=1)


# In[21]:


ratings['ISBN'] = ratings['ISBN'].str.upper()


# In[22]:


age_less_than90 = users[users['Age'] <= 90]
age_df = age_less_than90[age_less_than90['Age'] >= 15]


# In[23]:


mean=np.round(age_df.Age.mean(),0)


# In[24]:


users.loc[users['Age'] > 90, 'Age'] = mean    
users.loc[users['Age'] < 15, 'Age'] = mean    
users['Age'] = users['Age'].fillna(mean)      


# In[25]:


users['Age'] = users['Age'].astype(int)


# In[26]:


loca=users["Location"].str.split(",", n = 2, expand = True)


# In[27]:


loca.rename(columns = {0:"City",1:"State",2:"Country"},inplace=True)


# In[28]:


users=users.join(loca)


# In[29]:


users=users.drop("Location",axis=1)


# In[30]:


dataset = pd.merge(books, ratings, on='ISBN', how='inner')


# In[31]:


dataset = pd.merge(dataset, users, on='User-ID', how='inner')


# In[32]:


ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]


# In[33]:


ratings_explicit = ratings_new[ratings_new['Book-Rating'] != 0]
ratings_implicit = ratings_new[ratings_new['Book-Rating'] == 0]
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['Book-Rating'].count())


# In[34]:


# Create column Rating average 
ratings_explicit['Avg_Rating']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('mean')
# Create column Rating sum
ratings_explicit['Total_No_Of_Users_Rated']=ratings_explicit.groupby('ISBN')['Book-Rating'].transform('count')


# In[35]:


implicite=dataset


# In[36]:


explicite=dataset[dataset["Book-Rating"]>0]


# In[37]:


step1=dataset.groupby("User-ID").count()["Book-Rating"]


# In[38]:


step2=step1>5
people_rate_greater_than5=step2[step2].index


# In[39]:


filtered_ratings=dataset[dataset["User-ID"].isin(people_rate_greater_than5)]


# In[40]:


step3=filtered_ratings.groupby("Book-Title").count()["Book-Rating"]>3


# In[41]:


Top_rated_books=step3[step3].index


# In[42]:


final_ratings=filtered_ratings[filtered_ratings["Book-Title"].isin(Top_rated_books)]


# In[43]:


Pivot_table=final_ratings.pivot_table(index="Book-Title",columns="User-ID",values="Book-Rating")
Pivot_table.fillna(0,inplace=True)


# In[44]:


from sklearn.metrics.pairwise import cosine_similarity


# In[45]:


similarity_score=cosine_similarity(Pivot_table)


# In[46]:


def recommender(book_name):
  index=np.where(Pivot_table.index==book_name)[0][0]
  similar_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:10]

  for i in similar_items:
    print(Pivot_table.index[i[0]])


# In[47]:


Final_Dataset=users.copy()
Final_Dataset=pd.merge(Final_Dataset,ratings_explicit,on='User-ID')
Final_Dataset=pd.merge(Final_Dataset,books,on='ISBN')


# In[48]:


ratings_explicit.head()
ratings_explicit.rename(columns={'user_id':'User-ID','isbn':'ISBN','book_rating':'Book-Rating'},inplace=True)


# In[49]:


users_interactions_count_df = ratings_explicit.groupby(['ISBN', 'User-ID']).size().groupby('User-ID').size()
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 100].reset_index()[['User-ID']]


# In[50]:


interactions_from_selected_users_df = ratings_explicit.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'User-ID',
               right_on = 'User-ID')


# In[51]:


import math
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df.groupby(['ISBN', 'User-ID'])['Book-Rating'].sum().apply(smooth_user_preference).reset_index()


# In[52]:


from sklearn.model_selection import train_test_split

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['User-ID'], 
                                   test_size=0.20,
                                   random_state=42)


# In[53]:


users_items_pivot_matrix_df = interactions_train_df.pivot(index='User-ID', 
                                                          columns='ISBN', 
                                                          values='Book-Rating').fillna(0)


# In[54]:


users_items_pivot_matrix = users_items_pivot_matrix_df.values


# In[55]:


users_ids = list(users_items_pivot_matrix_df.index)


# In[56]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)


# In[57]:


sigma = np.diag(sigma)


# In[58]:


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 


# In[59]:


cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()


# In[60]:


#global books


# In[61]:


class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df):
        self.cf_predictions_df = cf_predictions_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False).reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating content that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['ISBN'].isin(items_to_ignore)].sort_values('recStrength', ascending = False).head(topn)
        recommendations_df=recommendations_df.merge(books,on='ISBN',how='inner')
        recommendations_df=recommendations_df[['ISBN','Book-Title','recStrength']]

        return recommendations_df



cf_recommender_model = CFRecommender(cf_preds_df)


# In[62]:


interactions_full_indexed_df = interactions_full_df.set_index('User-ID')
interactions_train_indexed_df = interactions_train_df.set_index('User-ID')
interactions_test_indexed_df = interactions_test_df.set_index('User-ID')


# In[63]:


def get_items_interacted(UserID, interactions_df):
    interacted_items = interactions_df.loc[UserID]['ISBN']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# In[75]:


class ModelRecommender:

    # Function for getting the set of items which a user has not interacted with
    def get_not_interacted_items_sample(self, UserID, sample_size, seed=42):
        interacted_items = get_items_interacted(UserID, interactions_full_indexed_df)
        all_items = set(ratings_explicit['ISBN'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    # Function to verify whether a particular item_id was present in the set of top N recommended items
    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index
    
    # Function to evaluate the performance of model for each user
    def evaluate_model_for_user(self, model, person_id):
        
        # Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        
        if type(interacted_values_testset['ISBN']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['ISBN'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['ISBN'])])
            
        interacted_items_count_testset = len(person_interacted_items_testset) 

        # Getting a ranked recommendation list from the model for a given user
        person_recs_df = model.recommend_items(person_id, items_to_ignore=get_items_interacted(person_id, interactions_train_indexed_df),topn=10000000000)
        print('Recommendation for User-ID = ',person_id)
        print(person_recs_df.head(5))

        # Function to evaluate the performance of model at overall level
    def recommend_book(self, model ,userid):
        
        person_metrics = self.evaluate_model_for_user(model, userid)  
        return

model_recommender = ModelRecommender()


# In[76]:


user_list=list(interactions_full_indexed_df.index.values)


# In[77]:


user_list_df=pd.DataFrame(user_list,columns=["User_IDs"])


# In[81]:


#user=int(input("Enter User ID from above list for book recommendation  "))
#model_recommender.recommend_book(cf_recommender_model,user)


# In[82]:


menu=["Recommendations","Datasets and information"]
choice=st.sidebar.selectbox("What do you want to see?",menu)

if choice=="Recommendations":
    st.subheader("Recommendations")
    st.subheader("Select user ID here and get books recommendations :")
    userid=st.selectbox("select user-id from here",user_list_df["User_IDs"].unique())
    st.write("You have just selected",userid)
    button=st.button("Show Book Recommendations")
    if button:
        st.write('Recommended books are :')
        
        #model_recommender.recommend_book(cf_recommender_model,userid)
        #st.write(recomend)
       
    

    


# In[84]:


st.success(model_recommender.recommend_book(cf_recommender_model,userid))


# In[ ]:


#model_recommender


# In[ ]:


#elif choice=="Datasets and information":
  #  st.subheader("Datasets and information")
  #  st.text("Books dataset")
  #  st.write(books)
 #   st.table(books.describe())
  #  st.text("Book Ratings dataset")
  #  st.write(ratings)
  #  st.table(ratings.describe())
  #  st.text("Users dataset")
  #  st.write(users)
  #  st.table(users.describe())
  #  st.text("merged dataset")
  #  st.write(dataset)
  #  st.table(dataset.describe())


# In[ ]:




