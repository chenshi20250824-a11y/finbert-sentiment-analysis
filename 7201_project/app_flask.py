"""
FinBERT 7-class News Sentiment Analysis - Flask Website
Complete Fixed Version
"""

from flask import Flask, render_template_string, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

# ==================== Initialize Flask ====================
app = Flask(__name__)

# ==================== Force Offline Mode ====================
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# ==================== Load Model ====================
MODEL_PATH = 'finbert_7class_model'

print("\n" + "=" * 80)
print("🔄 Loading Model (Full Local Mode)")
print("=" * 80)
print(f"Model Path: {os.path.abspath(MODEL_PATH)}")

try:
    # 1. Check model folder
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")
    
    # 2. List files
    files = os.listdir(MODEL_PATH)
    print(f"Folder contents: {len(files)} files")
    
    # 3. Load tokenizer
    print("\n📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False
    )
    print("✅ Tokenizer loaded successfully")
    
    # 4. Load model
    print("🤖 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=False
    )
    model.eval()
    print("✅ Model loaded successfully")
    
    # 5. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"✅ Device: {device}")
    
    # 6. Load label mapping (using os.path.join)
    print("🏷️ Loading label mapping...")
    label_mapping_path = os.path.join(MODEL_PATH, 'label_mapping.json')
    
    with open(label_mapping_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    
    LABELS_ORDER = label_mapping['labels_order']
    id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
    LABEL_INFO = label_mapping['label_info']
    print("✅ Label mapping loaded successfully")
    
    print("\n" + "=" * 80)
    print("✅ All components loaded successfully!")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Classes: {LABELS_ORDER}")
    print("=" * 80 + "\n")
    
except Exception as e:
    print(f"\n❌ Loading failed: {e}")
    print("\n💡 Please ensure:")
    print("   1. Running this script from 7201_project directory")
    print("   2. finbert_7class_model folder exists with all files")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== HTML Template ====================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📰 FinBERT Financial News Sentiment Analysis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-section {
            margin-bottom: 30px;
        }
        
        label {
            display: block;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .button-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: bold;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f5f5f5;
            color: #666;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
        }
        
        .examples {
            margin-bottom: 30px;
        }
        
        .examples h3 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .example-btns {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .example-btn {
            padding: 10px 15px;
            background: #e3f2fd;
            border: 1px solid #90caf9;
            border-radius: 8px;
            cursor: pointer;
            text-align: left;
            font-size: 0.9em;
            transition: all 0.3s;
        }
        
        .example-btn:hover {
            background: #bbdefb;
            transform: translateX(5px);
        }
        
        .result {
            display: none;
            background: #f9f9f9;
            border-radius: 15px;
            padding: 30px;
            margin-top: 30px;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .result-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .result-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .result-item:last-child {
            border-bottom: none;
        }
        
        .result-label {
            font-weight: bold;
            color: #666;
        }
        
        .result-value {
            color: #333;
            font-weight: bold;
        }
        
        .advice {
            padding: 20px;
            border-left: 4px solid #667eea;
            background: white;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .prob-bar {
            margin: 10px 0;
        }
        
        .prob-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        
        .prob-progress {
            height: 25px;
            background: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
        }
        
        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.8s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .footer {
            background: #f5f5f5;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📰 FinBERT Financial News Sentiment Analysis</h1>
            <p>7-class Financial News Sentiment Analysis System based on FinBERT</p>
        </div>
        
        <div class="content">
            <div class="input-section">
                <label for="newsInput">📝 Enter News Headline or Content</label>
                <textarea id="newsInput" placeholder="Example: Apple announces record quarterly earnings beating all estimates, stock soars 10%"></textarea>
            </div>
            
            <div class="button-group">
                <button class="btn-primary" onclick="analyze()">🔍 Analyze</button>
                <button class="btn-secondary" onclick="clearInput()">🔄 Clear</button>
            </div>
            
            <div class="examples">
                <h3>💡 Example News (Click to Use)</h3>
                <div class="example-btns">
                    <div class="example-btn" onclick="useExample(0)">
                        📈 Apple Exceeds Earnings Expectations
                    </div>
                    <div class="example-btn" onclick="useExample(1)">
                        📉 Tesla Massive Recall
                    </div>
                    <div class="example-btn" onclick="useExample(2)">
                        📊 Microsoft Steady Growth
                    </div>
                    <div class="example-btn" onclick="useExample(3)">
                        🚀 NVIDIA Releases Revolutionary Chip
                    </div>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing...</p>
            </div>
            
            <div class="result" id="result">
                <h2>🎯 Analysis Results</h2>
                
                <div class="result-card">
                    <div class="result-item">
                        <span class="result-label">Predicted Class</span>
                        <span class="result-value" id="predLabel">-</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Chinese Name</span>
                        <span class="result-value" id="predName">-</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Expected Change</span>
                        <span class="result-value" id="predRange">-</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence</span>
                        <span class="result-value" id="predConf">-</span>
                    </div>
                </div>
                
                <div class="advice" id="advice">
                    <strong>💡 Investment Advice:</strong>
                    <p id="adviceText">-</p>
                </div>
                
                <h3 style="margin: 20px 0 10px 0;">📊 Probability Distribution by Class</h3>
                <div id="probBars"></div>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Model Info: </strong> FinBERT (ProsusAI) + 7-class Fine-tuning</p>
            <p><strong>Training Data:</strong> 8,377 Financial News + Actual Stock Price Changes</p>
            <p style="margin-top: 10px;">⚠️ Prediction results are for reference only. Investment involves risk, decisions should be made cautiously.</p>
        </div>
    </div>

    <script>
        const examples = [
            "Apple announces record quarterly earnings beating all estimates, stock soars 10%",
            "Tesla faces massive safety recall affecting 2 million vehicles",
            "Microsoft reports steady cloud revenue growth in line with expectations",
            "NVIDIA unveils revolutionary AI chip exceeding analyst expectations"
        ];
        
        function useExample(index) {
            document.getElementById('newsInput').value = examples[index];
        }
        
        function clearInput() {
            document.getElementById('newsInput').value = '';
            document.getElementById('result').style.display = 'none';
        }
        
        async function analyze() {
            const text = document.getElementById('newsInput').value.trim();
            
            if (!text) {
                alert('Please enter news content');
                return;
            }
            
            // Show loading animation
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                // Hide loading animation
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display results
                displayResult(data);
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Request failed: ' + error);
            }
        }
        
        function displayResult(data) {
            document.getElementById('predLabel').textContent = data.predicted_label.toUpperCase();
            document.getElementById('predName').textContent = data.predicted_name;
            document.getElementById('predRange').textContent = data.predicted_range;
            document.getElementById('predConf').textContent = (data.confidence * 100).toFixed(2) + '%';
            document.getElementById('adviceText').textContent = data.advice;
            
            // Probability bars
            const probBars = document.getElementById('probBars');
            probBars.innerHTML = '';
            
            for (const [label, prob] of Object.entries(data.all_probabilities)) {
                const barHtml = `
                    <div class="prob-bar">
                        <div class="prob-label">
                            <span><strong>${label}</strong> (${data.label_info[label].name} ${data.label_info[label].range})</span>
                            <span>${(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div class="prob-progress">
                            <div class="prob-fill" style="width: ${prob * 100}%">
                                ${prob > 0.15 ? (prob * 100).toFixed(1) + '%' : ''}
                            </div>
                        </div>
                    </div>
                `;
                probBars.innerHTML += barHtml;
            }
            
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Enter key triggers analysis
        document.getElementById('newsInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyze();
            }
        });
    </script>
</body>
</html>
'''

# ==================== Prediction Function ====================
def predict_news(text):
    """Predict news sentiment"""
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    
    predicted_idx = probabilities.argmax()
    predicted_label = id2label[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    
    info = LABEL_INFO[predicted_label]
    
    # Investment advice
    if confidence < 0.4:
        advice = "⚠️ Low confidence, recommended to wait and observe"
    elif predicted_label in ['surge', 'rise']: 
        advice = "📈 Bullish signal, consider buying" if confidence < 0.7 else "📈 Strong bullish signal, consider buying"
    elif predicted_label in ['crash', 'drop']:
        advice = "📉 Bearish signal, proceed with caution" if confidence < 0.7 else "📉 Strong bearish signal, consider selling or waiting"
    elif predicted_label in ['small_rise']: 
        advice = "📊 Slightly bullish, consider small position buying"
    elif predicted_label in ['small_drop']:
        advice = "📊 Slightly bearish, consider reducing position"
    else: 
        advice = "➡️ Volatility signal, maintain current position"
    
    # Sort probabilities
    prob_dict = {LABELS_ORDER[i]: float(probabilities[i]) for i in range(len(LABELS_ORDER))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'predicted_label': predicted_label,
        'predicted_name': info['name'],
        'predicted_range': info['range'],
        'confidence': confidence,
        'advice': advice,
        'all_probabilities': sorted_probs,
        'label_info': LABEL_INFO
    }

# ==================== Routes ====================
@app.route('/')
def home():
    """Home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Please enter news content'}), 400
        
        result = predict_news(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== Start Server ====================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🚀 FinBERT Website Started Successfully!")
    print("=" * 80)
    
    # Generate public URL
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000)
        print(f"\nLocal Access:   http://localhost:5000")
        print(f"🌐 Public Access:   {public_url}")
        print("\n✨ Public link can be shared with anyone!")
    except Exception as e:
        print(f"\nLocal Access:   http://localhost:5000")
        print(f"⚠️  Public link generation failed:  {e}")
        print("💡 For public access, please run:  pip install pyngrok")
    
    print("\nPress Ctrl+C to stop the service")
    print("=" * 80)
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=False)