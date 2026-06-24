import streamlit as st,pandas as pd,pickle,numpy as np,os
from sklearn.metrics.pairwise import cosine_similarity
from warnings import filterwarnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
load_dotenv()
filterwarnings('ignore')

os.environ['HF_TOKEN']=st.secrets.get('HF_API_KEY')
st.set_page_config(page_title="Job Recommender", page_icon=":guardsman:", layout="wide")

st.title("Job Recommender System")
@st.cache_resource
def load_assets():
    with open('search_assets.pkl', 'rb') as f:
        assets=pickle.load(f)
    return assets

@st.cache_resource
def sent_model():
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

def get_semantic_recommendations(job_title, req_skills, min_exp, df, sent_model, n=5):
    filtered_df = df[df['YearsOfExperience'] >= min_exp].copy()
    if filtered_df.empty:
        return 'No results found for the given request.'
    request_text = f'{job_title} {req_skills}'
    query_embedding = sent_model.encode([request_text]).reshape(1, -1)
    dataset_embeddings = np.vstack(filtered_df['Transformer_Embeddings'].values)
    cos_scores = cosine_similarity(query_embedding,dataset_embeddings).flatten()
    filtered_df['Transformer_Score'] = cos_scores
    sorted_df = filtered_df.sort_values(by='Transformer_Score', ascending=False)
    return sorted_df[['Title', 'Skills', 'Responsibilities', 'Keywords', 'YearsOfExperience', 'Transformer_Score']].head(n)

def tfidf_get_recommendations(job_title,req_skills,min_exp,df,n=5):
    filtered_df = df[df['YearsOfExperience'] >= min_exp].copy()
    if filtered_df.empty:
        return 'No results found for the given request.'
    request_text=f'{job_title} {req_skills}'
    filtered_tfidf_matrix=tfidf_matrix[filtered_df.index]
    request_tfidf=tfidf_vectorizer.transform([request_text])
    cos_scores=cosine_similarity(request_tfidf,filtered_tfidf_matrix).flatten()
    filtered_df['Cosine_Score']=cos_scores    
    sorted_df=filtered_df.sort_values(by='Cosine_Score',ascending=False)
    return sorted_df[['Title','Skills','Responsibilities','Keywords','YearsOfExperience','Cosine_Score']].head(n)


def hybrid_recommendations(job_title, req_skills, min_exp, df, sent_model, tfidf_weight, transformer_weight, n=5):
    filtered_df = df[df['YearsOfExperience'] >= min_exp].copy()
    if filtered_df.empty:
        return 'No results found for the given request.'
    request_text=f'{job_title} {req_skills}'
    filtered_tfidf_matrix=tfidf_matrix[filtered_df.index]
    request_tfidf=tfidf_vectorizer.transform([request_text])
    cos_scores=cosine_similarity(request_tfidf,filtered_tfidf_matrix).flatten()
    filtered_df['Cosine_Score']=cos_scores
    query_embedding = sent_model.encode([request_text]).reshape(1, -1)
    dataset_embeddings = np.vstack(filtered_df.loc[filtered_df.index,'Transformer_Embeddings'].values)
    cos_scores = cosine_similarity(query_embedding, dataset_embeddings).flatten()
    filtered_df['Transformer_Score'] = cos_scores
    filtered_df['Hybrid_Score'] = (tfidf_weight * filtered_df['Cosine_Score']) + (transformer_weight * filtered_df['Transformer_Score'])
    sorted_df = filtered_df.sort_values(by='Hybrid_Score', ascending=False)
    return sorted_df[['Title', 'Skills', 'Responsibilities', 'Keywords', 'YearsOfExperience', 'Cosine_Score', 'Transformer_Score', 'Hybrid_Score']].head(n)



def reciprocal_rank_fusion_recommendations(job_title, req_skills, min_exp, df, sent_model, n=5,k=60):
    filtered_df = df[df['YearsOfExperience'] >= min_exp].copy()
    if filtered_df.empty:
        return 'No results found for the given request.'
    request_text=f'{job_title} {req_skills}'
    filtered_tfidf_matrix=tfidf_matrix[filtered_df.index]
    request_tfidf=tfidf_vectorizer.transform([request_text])
    filtered_df['Cosine_Score']=cosine_similarity(request_tfidf,filtered_tfidf_matrix).flatten()
    query_embedding = sent_model.encode([request_text]).reshape(1, -1)
    dataset_embeddings = np.vstack(filtered_df.loc[filtered_df.index,'Transformer_Embeddings'].values)
    filtered_df['Transformer_Score'] = cosine_similarity(query_embedding, dataset_embeddings).flatten()
    tfidf_rank = filtered_df['Cosine_Score'].rank(ascending=False, method='min')
    transformer_rank = filtered_df['Transformer_Score'].rank(ascending=False, method='min')
    filtered_df['RRF_Score'] = (1 / (k + tfidf_rank)) + (1 / (k + transformer_rank))
    sorted_df = filtered_df.sort_values(by='RRF_Score', ascending=False)
    return sorted_df[['Title', 'Skills', 'Responsibilities', 'Keywords', 'YearsOfExperience', 'Cosine_Score', 'Transformer_Score', 'RRF_Score']].head(n)


st.sidebar.title("Job Recommender")
title=st.sidebar.text_input(label='Please enter the job title',placeholder='Data Scientist, Data Analyst, Data Engineer',key='job_title')
skills=st.sidebar.text_input(label='Please enter the skills',placeholder='Python, SQL, JAVA, Machine Learning',key='skills')
years_of_experience=st.sidebar.slider(label='Please select the minimum years of experience',min_value=0,max_value=20,value=0,key='min_exp')
n=st.sidebar.slider(label='Please select the number of recommendations',min_value=1,max_value=10,value=5,key='n_recommendations')
recommender_type=st.sidebar.radio(label='Please select the recommender type',options=['TF-IDF','Semantic','Hybrid','Reciprocal Rank Fusion'],key='recommender_type')
if recommender_type=='Hybrid':
    tfidf_weight=st.sidebar.slider(label='Please select the weight for TF-IDF',min_value=0.0,max_value=1.0,value=0.5,key='tfidf_weight')
    transformer_weight=st.sidebar.slider(label='Please select the weight for Transformer',min_value=0.0,max_value=1.0,value=0.5,key='transformer_weight')
elif recommender_type=='Reciprocal Rank Fusion':
    k=st.sidebar.slider(label='Please select the value of k for Reciprocal Rank Fusion',min_value=1,max_value=100,value=60,key='k')
assets=load_assets()
loaded_df=assets['dataframe'].drop(['Cosine_Score','Cosine_Score_Sentence_Transformer'],axis=1)
tfidf_vectorizer=assets['tfidf_vectorizer']
tfidf_matrix=assets['tfidf_matrix']

with st.expander('Snippet of the dataset'):
    st.dataframe(loaded_df.head(10))

if 'recommendation' not in st.session_state:
    st.session_state['recommendation'] = None


if st.sidebar.button('Get Recommendations') and title and skills:
    if recommender_type=='TF-IDF':
        recommendations=tfidf_get_recommendations(title, skills, years_of_experience, loaded_df, n)
    elif recommender_type=='Semantic':
        recommendations=get_semantic_recommendations(title, skills, years_of_experience, loaded_df, sent_model(), n)
    elif recommender_type=='Hybrid':
        recommendations=hybrid_recommendations(title, skills, years_of_experience, loaded_df, sent_model(), tfidf_weight, transformer_weight, n)
    else:
        recommendations=reciprocal_rank_fusion_recommendations(title, skills, years_of_experience, loaded_df, sent_model(), n, k)
    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.session_state.recommendation = recommendations
if st.session_state.recommendation is not None: 
        st.subheader("Top Job Recommendations")
        st.dataframe(st.session_state.recommendation)