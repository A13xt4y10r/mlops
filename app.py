from flask import Flask, request # type: ignore
from flask_restx import Api, Resource, fields # type: ignore
from werkzeug.datastructures import FileStorage # type: ignore
import os
import pandas as pd # type: ignore
from MLModel import MLModel

# Flask alkalmazás inicializálása
app = Flask(__name__)
api = Api(app, version='1.0', title='Invoice Clear Date Predictor')

# MLModel objektum létrehozása
obj_mlmodel = MLModel()

# Predikciós modell definíciója
predict_model = api.model('PredictModel', {
    'business_code': fields.String(required=True, description='Business code', example="U001"),
    'cust_number': fields.String(required=True, description='Customer number', example="200769623"),
    'name_customer': fields.String(required=True, description='Customer name', example="WAL-MAR corp"),
    'clear_date': fields.String(required=True, description='Clear date', example="2/11/2020 0:00"),
    'buisness_year': fields.Integer(required=True, description='Business year', example=2020),
    'doc_id': fields.Integer(required=True, description='Document ID', example=1930438491),
    'posting_date': fields.String(required=True, description='Posting date', example="1/26/2020"),
    'document_create_date': fields.Integer(required=True, description='Document create date', example=20200125),
    'document_create_date.1': fields.Integer(required=True, description='Document create date.1', example=20200126),
    'due_in_date': fields.Integer(required=True, description='Due in date', example=20200210),
    'invoice_currency': fields.String(required=True, description='Invoice currency', example="USD"),
    'document_type': fields.String(required=True, description='Document type', example="RV"),
    'posting_id': fields.Integer(required=True, description='Posting ID', example=1),
    'area_business': fields.String(required=False, description='Area business', example=""),
    'total_open_amount': fields.Float(required=True, description='Total open amount', example=54273.28),
    'baseline_create_date': fields.Integer(required=True, description='Baseline create date', example=20200126),
    'cust_payment_terms': fields.String(required=True, description='Customer payment terms', example="NAH4"),
    'invoice_id': fields.Integer(required=True, description='Invoice ID', example=1930438491),
    'isOpen': fields.Integer(required=True, description='Is open', example=0)
})

#Feltöltés (upload() objeltum
file_upload = api.parser()
file_upload.add_argument('file', location='files',
                         type=FileStorage, required=True,
                         help='CSV file for training')

# Namespace létrehozása a modell műveletekhez
ns = api.namespace('model', description='Model operations')

#train végpont és logika
@ns.route('/train')
class Train(Resource):
    @ns.expect(file_upload)
    def post(self): #post kérés definiálása
        args = file_upload.parse_args()
        uploaded_file = args['file'] #itt lesz a feltöltött adat
        if os.path.splitext(uploaded_file.filename)[1] != '.csv':
            return {'error': 'Invalid file type'}, 400
        
        data_path = 'data/temp_payment_dataset.csv'
        uploaded_file.save(data_path) #mentjük a TEMP fájlt
        
        try:
            df = pd.read_csv(data_path)         
            train_accuracy, test_accuracy = obj_mlmodel.train_and_save_model(df) #betanítás
            os.remove(data_path) #töröljük a temp fájlt

            return {'message': 'Model Trained Successfully', 
                    'train_accuracy (MAPE)': train_accuracy, 'test_accuracy (MAPE)': test_accuracy}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 501

#predict végpont és logika
@ns.route('/predict')
class Predict(Resource):
    @api.expect(predict_model)
    def post(self): #post-ot hívunk
        try:
            infer_data = request.get_json() #itt kapjuk meg az adatot
            predicted_clear_date, due_date, days = obj_mlmodel.predict(infer_data,True) #preprocesszálunk majd prediktálunk
            return {'message': 'Inference Successful', 'due_in_date': due_date, 'predicted_clear_date': predicted_clear_date, 'day_diff':days}, 200
        except Exception as e:
            return {'message': 'Internal Server Error', 'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1080, debug=False)
