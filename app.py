from flask import Flask, render_template, url_for, request, Response,jsonify,flash
import sqlite3
# import cv2
import os
# from flask_cors import cross_origin
import pickle

app = Flask(__name__)
import os
cv = pickle.load(open("vector.pkl", "rb"))

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':

#         connection = sqlite3.connect('user_data.db')
#         cursor = connection.cursor()

#         name = request.form['name']
#         password = request.form['password']

#         query = "SELECT name, password FROM admin WHERE name = '"+name+"' AND password= '"+password+"'"
#         cursor.execute(query)

#         result = cursor.fetchall()

#         if len(result) == 0:
#             return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
#         else:
#             return render_template('message.html')

#     return render_template('index.html')



# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     return render_template('signup.html')



# @app.route('/registration', methods=['GET', 'POST'])
# def registration():
#     if request.method == 'POST':

#         connection = sqlite3.connect('user_data.db')
#         cursor = connection.cursor()

#         name = request.form['name']
#         password = request.form['password']
#         mobile = request.form['phone']
#         email = request.form['email']
        
#         print(name, mobile, email, password)

#         command = """CREATE TABLE IF NOT EXISTS admin(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
#         cursor.execute(command)

#         cursor.execute("INSERT INTO admin VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
#         connection.commit()

#         return render_template('index.html', msg='Successfully Registered')
    
#     return render_template('signup.html')
    


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     return render_template('index.html')


    

model = pickle.load(open("hindi.pkl", "rb"))
@app.route('/')
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        review= request.form['review']
        vect = cv.transform([review]).toarray()
        my_prediction = model.predict(vect)
        print(my_prediction)
        a=''
        if my_prediction == 0:
            a ="Offensive"
        elif my_prediction == 1:
            a ="Hate"
        elif my_prediction == 2:
            a ="Neither"    
    
        return render_template('message.html', pred=a)
    return render_template('message.html')    

@app.route("/logout")
def logout():
   return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
