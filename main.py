from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your RAG system
from app.rag_model import EnhancedRAGSystemWithGroq

app = Flask(__name__)
CORS(app)

# Initialize RAG system
print("🚀 Initializing Bengali RAG System...")
rag_system = EnhancedRAGSystemWithGroq(vector_store_path='app/vector_store.pkl')
print("✅ RAG System initialized successfully")

# HTML Template for the UI
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bengali RAG System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 1200px;
            height: 90vh;
            display: grid;
            grid-template-rows: auto 1fr auto;
            gap: 20px;
        }

        .header {
            text-align: center;
        }

        .header h1 {
            color: #2c3e50;
            font-size: 2rem;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            overflow: hidden;
        }

        .left-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .question-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            height: 200px;
            display: flex;
            flex-direction: column;
        }

        .question-input {
            flex: 1;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 15px;
            font-size: 16px;
            font-family: inherit;
            resize: none;
            outline: none;
            transition: border-color 0.3s;
        }

        .question-input:focus {
            border-color: #667eea;
        }

        .ask-button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.2s;
        }

        .ask-button:hover {
            transform: translateY(-2px);
        }

        .ask-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .status-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            flex: 1;
        }

        .status-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            height: 100%;
        }

        .status-item {
            background: white;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .status-label {
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 5px;
            font-weight: 600;
        }

        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }

        .model-status {
            grid-column: 1 / -1;
            background: #d4edda;
        }

        .confidence .status-value {
            color: #28a745;
        }

        .matches .status-value {
            color: #dc3545;
        }
        .question-display-section {
            background: #e3f2fd;
            border: 2px solid #bbdefb;
            border-radius: 15px;
            padding: 0;
            height: 100px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            margin-bottom: 15px;
        }

        .question-display-header {
            background: #2196f3;
            color: white;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 13px 13px 0 0;
            text-align: center;
        }

        .question-display-content {
            flex: 1;
            padding: 15px 20px;
            overflow-y: auto;
        }

        .question-display-text {
            font-size: 16px;
            font-weight: 500;
            color: #2c3e50;
            line-height: 1.5;
        }
        .keywords .status-value {
            color: #6f42c1;
            font-size: 14px;
        }

        .similarity .status-value {
            color: #fd7e14;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .answer-section {
            background: #e8f5e8;
            border: 2px solid #c3e6c3;
            border-radius: 15px;
            padding: 0;
            height: 150px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .answer-header {
            background: #28a745;
            color: white;
            padding: 12px 20px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 13px 13px 0 0;
            text-align: center;
            box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
        }

        .answer-content {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }

        .answer-text {
            font-size: 16px;
            font-weight: 500;
            color: #2c3e50;
            line-height: 1.6;
        }

        .chunk-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            flex: 1;
            overflow-y: auto;
        }

        .chunk-title {
            font-size: 16px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }

        .chunk-text {
            font-size: 14px;
            line-height: 1.6;
            color: #6c757d;
        }

        .loading {
            opacity: 0.6;
            pointer-events: none;
        }

        .empty-state {
            color: #6c757d;
            font-style: italic;
            text-align: center;
        }

        .error-state {
            color: #dc3545;
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Bengali RAG System</h1>
        </div>

        <div class="main-content">
            <div class="left-panel">
                <div class="question-section">
                    <textarea 
                        id="questionInput" 
                        class="question-input" 
                        placeholder="আপনার প্রশ্ন এখানে লিখুন..."
                        rows="3"
                    ></textarea>
                    <button id="askButton" class="ask-button">প্রশ্ন করুন</button>
                </div>

                <div class="status-section">
                    <div class="status-grid">
                        <div class="status-item model-status">
                            <div class="status-label">✅ MODEL STATUS</div>
                            <div class="status-value" id="modelStatus">Ready</div>
                        </div>
                        <div class="status-item confidence">
                            <div class="status-label">📊 CONFIDENCE</div>
                            <div class="status-value" id="confidence">-</div>
                        </div>
                        <div class="status-item matches">
                            <div class="status-label">🔢 KEYWORD MATCHES</div>
                            <div class="status-value" id="matches">-</div>
                        </div>
                        <div class="status-item similarity">
                            <div class="status-label">🎯 COSINE SIMILARITY</div>
                            <div class="status-value" id="similarity">-</div>
                        </div>
                        <div class="status-item keywords" style="grid-column: 1 / -1;">
                            <div class="status-label">🔍 KEYWORDS FOUND</div>
                            <div class="status-value" id="keywords">-</div>
                        </div>
                    </div>
                </div>
            </div>
            

            <div class="right-panel">
                <div class="question-display-section">
                    <div class="question-display-header">❓ প্রশ্ন</div>
                    <div class="question-display-content">
                        <div class="question-display-text" id="questionDisplay">
                            <div class="empty-state">এখানে প্রশ্ন দেখা যাবে...</div>
                        </div>
                    </div>
                </div>
            
                <div class="answer-section">
                    <div class="answer-header">📝 উত্তর</div>
                    <div class="answer-content">
                        <div class="answer-text" id="answerText">
                            <div class="empty-state">এখানে উত্তর দেখা যাবে...</div>
                        </div>
                    </div>
                </div>

                <div class="chunk-section">
                    <div class="chunk-title">📄 Full Chunk:</div>
                    <div class="chunk-text" id="chunkText">
                        <div class="empty-state">এখানে সম্পূর্ণ চাংক দেখা যাবে...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const questionInput = document.getElementById('questionInput');
        const askButton = document.getElementById('askButton');
        const modelStatus = document.getElementById('modelStatus');
        const confidence = document.getElementById('confidence');
        const matches = document.getElementById('matches');
        const similarity = document.getElementById('similarity');
        const keywords = document.getElementById('keywords');
        const answerText = document.getElementById('answerText');
        const chunkText = document.getElementById('chunkText');
        const questionDisplay = document.getElementById('questionDisplay');

        async function handleQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            // Set loading state
            askButton.disabled = true;
            askButton.textContent = 'প্রক্রিয়াকরণ...';
            // Set loading state এর পরে যোগ করুন
            questionDisplay.textContent = question;
            
            // Update model status
            modelStatus.textContent = 'Processing...';
            modelStatus.parentElement.style.background = '#fff3cd';
            
            // Clear previous results
            answerText.innerHTML = '<div class="empty-state">উত্তর তৈরি হচ্ছে...</div>';
            chunkText.innerHTML = '<div class="empty-state">চাংক লোড হচ্ছে...</div>';
            confidence.textContent = '-';
            matches.textContent = '-';
            similarity.textContent = '-';
            keywords.textContent = '-';

            try {
                // Call your RAG API
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: question })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();
                
                // Update model status
                modelStatus.textContent = data.model || 'llama-3.3-70b-versatile';
                modelStatus.parentElement.style.background = '#d4edda';
                
                // Update answer
                answerText.textContent = data.answer || 'উত্তর পাওয়া যায়নি';
                
                // Update stats
                confidence.textContent = (data.confidence || 0).toFixed(3);
                matches.textContent = (data.keyword_matches || 0).toFixed(1);
                similarity.textContent = (data.semantic_score || 0).toFixed(3);
                keywords.textContent = (data.matching_keywords || []).join(', ') || '-';
                
                // Update chunk
                chunkText.textContent = data.full_chunk || 'চাংক পাওয়া যায়নি';
                
            } catch (error) {
                console.error('Error:', error);
                modelStatus.textContent = 'Error';
                modelStatus.parentElement.style.background = '#f8d7da';
                answerText.innerHTML = '<div class="error-state">ত্রুটি ঘটেছে। আবার চেষ্টা করুন।</div>';
            }
            
            // Reset button
            askButton.disabled = false;
            askButton.textContent = 'প্রশ্ন করুন';
        }

        // Event listeners
        askButton.addEventListener('click', handleQuestion);
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                handleQuestion();
            }
        });
    </script>
</body>
</html>'''

@app.route('/')
def home():
    """Serve the main UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests from the UI"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'error': 'No query provided',
                'answer': 'প্রশ্ন লিখুন',
                'confidence': 0.0,
                'keyword_matches': 0.0,
                'semantic_score': 0.0,
                'matching_keywords': [],
                'full_chunk': '',
                'model': 'Error'
            }), 400
        
        # Get answer from RAG system
        result = rag_system.search_and_answer(query, top_k=3, use_groq=True)
        
        # Extract semantic score from the best chunk if available
        semantic_score = 0.0
        if result.get('source_chunks') and len(result['source_chunks']) > 0:
            semantic_score = result['source_chunks'][0].get('semantic_score', 0.0)
        
        # Format response
        response = {
            'answer': result.get('answer', 'উত্তর পাওয়া যায়নি'),
            'confidence': result.get('confidence', 0.0),
            'keyword_matches': result.get('keyword_matches', 0.0),
            'semantic_score': semantic_score,
            'matching_keywords': result.get('matching_keywords', []),
            'full_chunk': result.get('full_chunk', ''),
            'model': 'llama-3.3-70b-versatile',
            'success': True
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in search endpoint: {e}")
        return jsonify({
            'error': str(e),
            'answer': 'সিস্টেম ত্রুটি',
            'confidence': 0.0,
            'keyword_matches': 0.0,
            'semantic_score': 0.0,
            'matching_keywords': [],
            'full_chunk': '',
            'model': 'Error',
            'success': False
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Health check endpoint"""
    try:
        # Check if RAG system is working
        chunks_count = len(rag_system.vector_store['chunks']) if rag_system.vector_store else 0
        
        return jsonify({
            'status': 'healthy',
            'chunks_loaded': chunks_count,
            'groq_available': rag_system.groq_client is not None,
            'model': 'llama-3.3-70b-versatile'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Bengali RAG System REST API Server")
    print("="*80)
    print("📡 Server URL: http://localhost:8000")
    print("🌐 Web Interface: http://localhost:8000")
    print("📊 API Endpoint: http://localhost:8000/api/search")
    print("❤️  Status Check: http://localhost:8000/api/status")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )