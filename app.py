from flask import Flask, request, jsonify
import joblib
import numpy as np
import time

app = Flask(__name__)
model = joblib.load('fraud_model.pkl')

@app.route('/detect', methods=['POST'])
def detect_fraud():
    try:
        data = request.get_json()
        # Ensure data is converted to a 2D array for the model
        features = np.array(data['features']).reshape(1, -1)
        
        # Get the prediction (0 or 1)
        prediction = model.predict(features)
        
        # Get the probability score
        # Using [0][1] only if the model was trained on both classes
        probs = model.predict_proba(features)[0] 
        fraud_chance = probs[1] if len(probs) > 1 else probs[0]

        return jsonify({
            'is_fraud': int(prediction[0]),
            'fraud_probability': round(float(fraud_chance), 4),
            'decision': 'BLOCK' if prediction[0] == 1 else 'APPROVE',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
