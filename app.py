from flask import Flask, render_template, request, redirect, url_for
import  predict
import pandas as pd
import re
import json


from jinja2 import Template
#from bokeh.embed import json_item
#from bokeh.plotting import figure
#from bokeh.resources import CDN
#from bokeh.sampledata.iris import flowers


## initialize app
#create an instance
app = Flask(__name__)  


@app.route('/')
def index():
        
    return render_template('index.html')


@app.route('/', methods=['POST'])
def classify():
    
    if request.form["submit_button"] == "Submit File":
    
        uploaded_file = request.files['file']
        column_name = request.form.get("column")
        
        if uploaded_file.filename == "" or not uploaded_file.filename.endswith(".csv"):
            return render_template('index.html', no_file_err="No file chosen or the format is not csv.")
            
        if column_name=="":
            return render_template('index.html', no_column_err="No column name entered, please enter the column name.")
            
         
        
        if uploaded_file.filename != '':  
            uploaded_file.save('feedback.csv')
            df = pd.read_csv("feedback.csv")
            
            if column_name not in df.columns:
                 return    render_template('index.html', column_not_exists="The entered column name does not exist in the dataframe.")    
            
            
            pred = predict.predict(df[[column_name]])
            
            # print(pred)
            df_in_html = pred.to_html(classes='predictions')
            df_in_html = re.sub(r'right', r'center', df_in_html)
            df_in_html = re.sub(r'left', r'center', df_in_html)
            
        
        return render_template('index.html', tables=[df_in_html], titles = ['na', 'Predictions'] ) 

    elif request.form["submit_button"] == "Submit Comment":
            
        comment = request.form['comment_name']   
        #print(comment)
        pred = predict.predict_comment(comment)
        
            
        # print(pred)
        df_in_html = pred.to_html(classes='predictions')
        df_in_html = re.sub(r'right', r'center', df_in_html)
        df_in_html = re.sub(r'left', r'center', df_in_html)
        
        return render_template('index.html', tables=[df_in_html], titles = ['na', 'Predictions'] ) 



if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
    