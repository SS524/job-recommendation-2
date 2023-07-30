import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object,text_preprocessing
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendPipeline:
    def __init__(self):
        pass

    def recommend(self,job_title,skills,experience):
        try:
            refdf_path=os.path.join('artifacts','reference_df.csv')
            processeddf_path=os.path.join('artifacts','processed_df.csv')

            #ref_df=load_object(refdf_path)
            ref_df = pd.read_csv(refdf_path)
            #processed_df=load_object(processeddf_path)
            processed_df = pd.read_csv(processeddf_path)

            job_title="".join(job_title.split())
            df=pd.DataFrame({'Location':['Not Specified'],'Tags':[job_title+" "+skills+" "+experience]})
            df['Tags']=df['Tags'].apply(lambda x: text_preprocessing(x))
            output_df=pd.concat([df,processed_df],axis=0)

            cv=CountVectorizer()
            vectors=cv.fit_transform(output_df['Tags'].values.astype('U')).toarray()
            similarity=cosine_similarity(vectors)
            top_5_recommended=sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]
            list_of_index_values=[]
            for item in top_5_recommended:
                list_of_index_values.append(item[0])
            output_list=[]
            for ind in list_of_index_values:
                dic={}
                dic['Job_title'] = ref_df.iloc[ind]['Job_title']
                dic['Company'] =  ref_df.iloc[ind]['Company_name']
                dic['Skills'] =  ref_df.iloc[ind]['Skills']
                dic['Experience'] =  ref_df.iloc[ind]['Experience']
                dic['Location'] =  ref_df.iloc[ind]['Location']
                
                output_list.append(dic)

            logging.info("Top 5 recommended jobs are fetched")
            return output_list


        except Exception as e:
            logging.info("Exception occured in recommendation")
            raise CustomException(e,sys)