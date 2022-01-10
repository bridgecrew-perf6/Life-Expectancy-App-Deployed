from flask import Flask, request, jsonify, render_template
#import pickle 
import life_expectancy
import numpy as np


app=Flask(__name__)

@app.route('/')
def form():
    return render_template('index.html')

@app.route("/",methods=["POST","Get"])
def le_ip():
    if request.method=="POST":
        v1=request.form["Country"]
        v2=request.form["Year"]
        v3=request.form["AdultMortality"]
        v4=request.form["Alcohol"]
        v5=request.form["percentageexpenditure"]
        v6=request.form["HepatitisB"]
        v7=request.form["Measles"]
        v8=request.form["BMI"]
        v9=request.form["Totalexpenditure"]
        v10=request.form["HIV/AIDS"]
        v11=request.form["GDP"]
        v12=request.form["thinness1-19years"]
        v13=request.form["thinness5-9years"]

        final_features=[v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13]
        
        w=life_expectancy.Loaded_model(final_features)
        
    return render_template("index.html",prediction=w)
if __name__=="__main__":
    app.run(debug=True)
