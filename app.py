from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Add your fraud detection model logic here
    amount = request.form.get('amount')
    v1 = request.form.get('v1')
    v2 = request.form.get('v2')
    
    # Placeholder prediction
    return jsonify({'result': 'Transaction scanned'})

if __name__ == '__main__':
    app.run(debug=True)
