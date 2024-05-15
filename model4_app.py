from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, abort
from sqlalchemy import Column, String
import numpy as np
import os
from flask_cors import CORS
from model4_main import main, test
app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS for all routes

DB_URL = "postgresql://postgres:postgreSQL@localhost:5432/cv"
app.config['SQLALCHEMY_DATABASE_URI'] = DB_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.app_context().push()
db = SQLAlchemy(app)

app.config['UPLOAD_FOLDER'] = 'photos'

class persons(db.Model):
    user_id = Column(String, primary_key=True)
    user_name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    img_path = Column(String)
    face_coordinates = Column(String(255))
# db.create_all()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def store(file, user_id):
    try:
        if not allowed_file(file.filename):
            print("message: photo format is not allowed")
        
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
            print(user_folder)
        
        # Save file to user's folder
        file.save(os.path.join(user_folder, file.filename))

        print('File uploaded successfully')
    except Exception as e:
        print(e)
    
class UserRegistration(Resource):
    def post(self):
        user_id = request.form.get('user_id')
        user_name = request.form.get('user_name')
        password = request.form.get('password')
        a = request.files.get('a')
        b = request.files.get('b')
        c = request.files.get('c')

        if not all([user_id, user_name, password, a, b, c]):
            return {'message': 'All fields are required'}, 401
    
        store(a,user_id)
        store(b,user_id)
        store(c,user_id)

        user = persons.query.filter_by(user_id=user_id).first()
        if user:
            abort(401, message="This User ID is already registered")
        else:
            add_user = persons(user_id = user_id,user_name = user_name, password = password)    # , img_path = filepath
        try:
            db.session.add(add_user)
            db.session.commit()
            print("Data is stored !!")
            status_code = main()
            if status_code == 200:
                return "Registration successful !!", 201
            else:
                return "error in model", 500
        except Exception as e:
                db.session.rollback()
                abort(500, message="Error occurred while registering user")

class IdentifyStudent(Resource):
    def post(self):
        # Check if image file is present in the request
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        image_bytes = file.read()     # file object to bytes
        user_ids, status_code = test(image_bytes)
        if status_code == 200:
            print("The students who are present are: ", user_ids)

api.add_resource(IdentifyStudent, '/IdentifyStudent')
api.add_resource(UserRegistration, '/UserRegistration')

if __name__ == '__main__':
    app.run(debug=True)
