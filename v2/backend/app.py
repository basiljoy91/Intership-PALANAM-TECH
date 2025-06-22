from flask import Flask, request, jsonify, make_response
from transformers import pipeline
import logging

app = Flask(__name__)

# Load the pre-trained model
try:
    classifier = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    classifier = None

# Label to category mapping
LABEL_MAPPING = {
    'toxic': 'Malicious',
    'severe_toxic': 'Malicious',
    'obscene': 'Sensitive',
    'threat': 'Malicious',
    'insult': 'Sensitive',
    'identity_hate': 'Sensitive'
}
DEFAULT_CATEGORY = 'Safe'

# Enhanced CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

def classify_text(text):
    """Classify text using the loaded model."""
    if not text.strip():
        return {'category': DEFAULT_CATEGORY, 'confidence': 1.0}
    
    try:
        results = classifier(text)[0]
        category_scores = {}
        
        for result in results:
            label = result['label']
            score = result['score']
            category = LABEL_MAPPING.get(label, DEFAULT_CATEGORY)
            
            if category not in category_scores or score > category_scores[category]:
                category_scores[category] = score
        
        if not category_scores:
            return {'category': DEFAULT_CATEGORY, 'confidence': 1.0}
        
        max_category = max(category_scores.items(), key=lambda x: x[1])
        
        return {
            'category': max_category[0],
            'confidence': float(max_category[1])
        }
        
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return {'category': 'Error', 'confidence': 0.0}

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not isinstance(text, str):
        return jsonify({'error': 'Text must be a string'}), 400
    
    result = classify_text(text)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)