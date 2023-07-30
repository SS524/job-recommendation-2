from flask import Flask,request,render_template,jsonify
from src.pipeline.recommendation_pipeline import RecommendPipeline
import sys
import logging


app=Flask(__name__)

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

@app.route('/home')
def home_page():
    return render_template('index.html')


@app.route('/recommend',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        
        skills=request.form.getlist("skills")
        experience = request.form.get('experience')
        job_title = request.form.get('job_title')

        skill_string = " ".join(skills)

        recommend_obj=RecommendPipeline()
        top_5_recommended_movies = recommend_obj.recommend(job_title,skill_string,experience)
         

        return render_template('result.html',output=top_5_recommended_movies)


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)