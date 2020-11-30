from flask import Flask, render_template, request, redirect, url_for
import  predict_bert
import pandas as pd
import re
#import json


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
def upload_file():
    
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        uploaded_file.save('feedback.csv')
        df = pd.read_csv("feedback.csv")
        print(df)
        pred = predict_bert.predict(df)
        print(pred)
        #pred = predict.predict(df)
        
        # print(pred)
        df_in_html = pred.to_html(classes='predictions')
        df_in_html = re.sub(r'right', r'center', df_in_html)
        df_in_html = re.sub(r'left', r'center', df_in_html)
    # return redirect(url_for('index'), predictions = pred)  ()
    return render_template('index.html',tables=[df_in_html], 
    titles = ['na', 'Predictions'] )            





if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
    