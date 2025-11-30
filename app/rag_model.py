import os
os.environ["USE_TF"] = "0"

import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv

load_dotenv()  # Loads .env file

groq_api_key = os.getenv("GROQ_API_KEY")
print(groq_api_key)

# Groq API imports
from groq import Groq
import json

class EnhancedRAGSystemWithGroq:
    def __init__(self, vector_store_path='app/vector_store.pkl', groq_api_key=None):
        """Enhanced RAG System for Bengali Text with Groq API"""
        
        # Load vector store
        self.vector_store = self.load_vector_store(vector_store_path)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize Groq client
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('GROQ_API_KEY')
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                print("⚠️ Groq API key not provided. Add your API key to use LLM features.")
                self.groq_client = None
        
        # Initialize TF-IDF vectorizer for keyword matching
        self.tfidf_vectorizer = None
        self.chunk_tfidf_matrix = None
        
        # Bengali stop words - expanded list
        self.bengali_stop_words = {
            'আর', 'এর', 'তার', 'সে', 'তিনি', 'আমি', 'আমার', 'তুমি', 'তোমার', 
            'আপনি', 'আপনার', 'এই', 'ওই', 'সেই', 'ঐ', 'যে', 'যার', 'যাকে',
            'কে', 'কাকে', 'কার', 'কি', 'কী', 'কোন', 'কোনো', 'কত', 'কখন', 
            'কোথায়', 'কিভাবে', 'কেন', 'কেমন', 'কিসের', 'কাছে', 'কাছ থেকে',
            'হয়', 'হলো', 'হয়েছে', 'হইল', 'হইয়াছে', 'হয়েছিল', 'হয়ে', 'হয়ত',
            'বলা', 'বলে', 'বলেছে', 'বলেছিল', 'বলেন', 'বলল', 'বললেন',
            'উল্লেখ', 'উল্লিখিত', 'উল্লেখ করা', 'উল্লেখিত হয়েছে',
            'করা', 'করে', 'করেছে', 'করেছিল', 'করল', 'করলেন', 'করতে',
            'এবং', 'ও', 'আও', 'বা', 'অথবা', 'তবে', 'যদি', 'তাহলে', 
            'এর', 'এক', 'একটি', 'একজন', 'দুই', 'দুইজন', 'তিন', 'চার',
            'আছে', 'আছিল', 'ছিল', 'ছিলেন', 'থাকে', 'থেকে', 'থাকতে',
            'গেছে', 'গেল', 'গেলেন', 'এল', 'এলেন', 'আসল', 'আসলেন',
            'যায়', 'যেতে', 'আসতে', 'পারে', 'পারেন', 'পারি', 'পারো',
            'দিয়ে', 'নিয়ে', 'সাথে', 'সময়', 'সময়ে','ভাষায়','আসিতে', 'মধ্যে', 'ভিতরে',
            'উপর', 'নিচে', 'পাশে', 'কাছে', 'দূরে', 'সামনে', 'পিছনে',
            'জন্য', 'কারণে', 'ফলে', 'তাই', 'সেজন্য', 'এজন্য', 'যেহেতু',
            'একটা', 'একটু', 'অনেক', 'কিছু', 'সব', 'সকল', 'প্রতি', 'বেশি',
            'ভাষায়','অনুপম','অনুপমের','কল্যাণী','কল্যাণীর','কম', 'বড়', 'ছোট', 'ভাল', 'ভালো', 'মন্দ', 'ভাগ্য', 'সৌভাগ্য'
        }
        
        # Bengali suffixes that should be considered for word matching
        self.bengali_suffixes = [
            'কে', 'কেই', 'টি', 'টিকে', 'টা', 'টাকে', 'খানি', 'খানা', 'গুলি', 'গুলো',
            'দের', 'রা', 'েরা', 'গণ', 'বৃন্দ', 'মালা', 'পুঞ্জ', 'নিচয়', 'সমূহ',
            'এ', 'তে', 'য়', 'র', 'এর', 'ের', 'ইতে', 'েতে', 'হতে', 'থেকে',
            'দিয়ে', 'নিয়ে', 'ছাড়া', 'বিনা', 'ব্যতীত', 'সহ', 'সহিত', 'মত',
            'ও', 'ই', 'ও', 'ই', 'টো', 'টাই', 'টাও', 'খানিই', 'খানাও',
            'গুলিই', 'গুলোও', 'দেরই', 'রাই', 'দেরও', 'রাও', 'গণই', 'গণও'
        ]
        
        # Prepare chunks for processing
        self.prepare_chunks()
    
    def load_vector_store(self, path):
        """Load the vector store from pickle file"""
        try:
            with open(path, 'rb') as f:
                store = pickle.load(f)
            print(f"✅ Vector store loaded successfully with {len(store['chunks'])} chunks")
            return store
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return None
    
    def clean_text(self, text):
        """Advanced text cleaning for Bengali"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove punctuation but keep Bengali punctuation
        bengali_punctuation = '।,;!?()[]{}"\'-–—'
        text = re.sub(f'[{re.escape(string.punctuation + bengali_punctuation)}]', ' ', text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_word_stem(self, word):
        """Get the stem of a Bengali word by removing common suffixes"""
        original_word = word
        
        # Sort suffixes by length (longest first) to match longer suffixes first
        sorted_suffixes = sorted(self.bengali_suffixes, key=len, reverse=True)
        
        for suffix in sorted_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:  # Ensure stem is not too short
                stem = word[:-len(suffix)]
                # Only return stem if it's meaningful (at least 2 characters)
                if len(stem) >= 2:
                    return stem
        
        return original_word
    
    def extract_keywords(self, text, min_length=2):
        """Extract meaningful keywords from Bengali text with stemming"""
        if not text:
            return []
        
        # Clean text
        clean_text = self.clean_text(text.lower())
        words = clean_text.split()
        
        # Filter and stem keywords
        keywords = []
        for word in words:
            if (len(word) >= min_length and 
                word not in self.bengali_stop_words and
                not word.isdigit()):
                
                # Get stem of the word
                stem = self.get_word_stem(word)
                keywords.append({
                    'original': word,
                    'stem': stem
                })
        
        return keywords
    
    def prepare_chunks(self):
        """Prepare chunks with TF-IDF processing"""
        if not self.vector_store or not self.vector_store['chunks']:
            print("❌ No chunks found in vector store")
            return
        
        chunks = self.vector_store['chunks']
        
        # Clean and prepare chunks for TF-IDF
        cleaned_chunks = []
        for chunk in chunks:
            cleaned = self.clean_text(chunk)
            if cleaned:
                cleaned_chunks.append(cleaned)
            else:
                cleaned_chunks.append(chunk)  # fallback to original
        
        # Create TF-IDF matrix
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,  # We'll handle stop words manually
                ngram_range=(1, 2),  # Include bigrams
                max_features=5000,
                min_df=1,
                max_df=0.8
            )
            
            self.chunk_tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_chunks)
            print(f"✅ TF-IDF matrix created: {self.chunk_tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"❌ Error creating TF-IDF matrix: {e}")
            self.tfidf_vectorizer = None
            self.chunk_tfidf_matrix = None
    
    def improved_keyword_matching(self, query_keywords, chunk_text):
        """Improved keyword matching with stem-based comparison"""
        if not query_keywords:
            return 0, []
        
        chunk_keywords = self.extract_keywords(chunk_text)
        
        exact_matches = []
        stem_matches = []
        
        for q_keyword in query_keywords:
            q_original = q_keyword['original']
            q_stem = q_keyword['stem']
            
            # Check for exact matches first
            found_exact = False
            for c_keyword in chunk_keywords:
                c_original = c_keyword['original']
                c_stem = c_keyword['stem']
                
                # Exact original word match
                if q_original == c_original:
                    exact_matches.append(q_original)
                    found_exact = True
                    break
                
                # Check if query word appears as part of chunk word (but with suffixes)
                # This handles cases like "ভাগ্যদেবতা" matching "ভাগ্যদেবতাকে"
                elif c_original.startswith(q_original) and len(c_original) > len(q_original):
                    # Check if the extra part is a valid suffix
                    extra_part = c_original[len(q_original):]
                    if extra_part in self.bengali_suffixes:
                        exact_matches.append(q_original)
                        found_exact = True
                        break
                
                # Check reverse: chunk word appears in query word with suffix
                elif q_original.startswith(c_original) and len(q_original) > len(c_original):
                    extra_part = q_original[len(c_original):]
                    if extra_part in self.bengali_suffixes:
                        exact_matches.append(q_original)
                        found_exact = True
                        break
            
            # If no exact match found, check stem matches
            if not found_exact:
                for c_keyword in chunk_keywords:
                    c_stem = c_keyword['stem']
                    
                    # Stem match (less weight)
                    if q_stem == c_stem and len(q_stem) >= 3:  # Only consider meaningful stems
                        stem_matches.append(q_original)
                        break
        
        # Calculate match count with different weights
        exact_count = len(exact_matches)
        stem_count = len(stem_matches) * 0.7  # Stem matches get less weight
        
        total_matches = exact_count + stem_count
        all_matches = exact_matches + stem_matches
        
        return total_matches, all_matches
    
    def keyword_similarity_score(self, query_keywords, chunk_text):
        """Calculate keyword-based similarity score with improved matching"""
        if not query_keywords:
            return 0.0
        
        match_count, matching_keywords = self.improved_keyword_matching(query_keywords, chunk_text)
        
        if len(query_keywords) == 0:
            return 0.0
        
        # Calculate similarity based on match ratio
        similarity = match_count / len(query_keywords)
        
        return min(1.0, similarity)
    
    def tfidf_similarity_score(self, query):
        """Calculate TF-IDF based similarity"""
        if not self.tfidf_vectorizer or self.chunk_tfidf_matrix is None:
            return []
        
        try:
            # Transform query using fitted vectorizer
            query_cleaned = self.clean_text(query)
            query_tfidf = self.tfidf_vectorizer.transform([query_cleaned])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_tfidf, self.chunk_tfidf_matrix).flatten()
            
            return similarities
        except Exception as e:
            print(f"❌ Error in TF-IDF similarity: {e}")
            return []
    
    def semantic_similarity_score(self, query, chunk):
        """Calculate semantic similarity using sentence transformers"""
        try:
            query_emb = self.embedding_model.encode([query])
            chunk_emb = self.embedding_model.encode([chunk])
            
            similarity = cosine_similarity(query_emb, chunk_emb)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"❌ Error in semantic similarity: {e}")
            return 0.0
    
    def find_best_chunks(self, query, top_k=10, debug=True):
        """Find the most relevant chunks with improved keyword matching"""
        
        if not self.vector_store or not self.vector_store['chunks']:
            print("❌ No chunks available")
            return []
        
        chunks = self.vector_store['chunks']
        query_keywords = self.extract_keywords(query)
        
        if debug:
            print(f"\n🔍 Query: {query}")
            print(f"📝 Extracted Keywords: {[kw['original'] for kw in query_keywords]}")
            print(f"📊 Total Chunks to Search: {len(chunks)}")
            print("-" * 80)
        
        # Get TF-IDF similarities for all chunks
        tfidf_similarities = self.tfidf_similarity_score(query)
        
        scored_chunks = []
        
        for i, chunk in enumerate(chunks):
            # 1. Improved keyword matching (PRIMARY CRITERIA)
            match_count, matching_kw = self.improved_keyword_matching(query_keywords, chunk)
            
            # 2. Keyword similarity score
            keyword_score = self.keyword_similarity_score(query_keywords, chunk)
            
            # 3. TF-IDF similarity
            tfidf_score = tfidf_similarities[i] if i < len(tfidf_similarities) else 0.0
            
            # 4. Semantic similarity (only for candidates with keyword matches)
            semantic_score = 0.0
            if match_count > 0:
                semantic_score = self.semantic_similarity_score(query, chunk)
            
            # 5. Length normalization
            chunk_words = len(chunk.split())
            length_score = min(1.0, chunk_words / 50) if chunk_words > 0 else 0.0
            
            # Combined weighted score with heavy emphasis on keyword matching
            combined_score = (
                0.60 * match_count / len(query_keywords) if query_keywords else 0 +  # Match count ratio (highest weight)
                0.20 * keyword_score +          # Keyword similarity
                0.10 * tfidf_score +           # TF-IDF similarity  
                0.05 * semantic_score +        # Semantic similarity
                0.05 * length_score            # Length normalization
            )
            
            # Only include chunks with keyword matches
            if match_count > 0:
                scored_chunks.append({
                    'text': chunk,
                    'index': i,
                    'combined_score': combined_score,
                    'match_count': match_count,
                    'keyword_score': keyword_score,
                    'tfidf_score': tfidf_score,
                    'semantic_score': semantic_score,
                    'length_score': length_score,
                    'matching_keywords': matching_kw
                })
        
        # Sort by match count first, then by combined score
        scored_chunks.sort(key=lambda x: (x['match_count'], x['combined_score']), reverse=True)
        
        if debug and scored_chunks:
            print(f"\n🎯 Top {min(5, len(scored_chunks))} Results (Sorted by Keyword Matches):")
            print("=" * 120)
            
            for rank, chunk_info in enumerate(scored_chunks[:5], 1):
                print(f"\n[Rank {rank}] 🔥 Keyword Matches: {chunk_info['match_count']:.1f} | Combined Score: {chunk_info['combined_score']:.3f}")
                print(f"   📊 Keyword: {chunk_info['keyword_score']:.3f} | "
                      f"TF-IDF: {chunk_info['tfidf_score']:.3f} | "
                      f"Semantic: {chunk_info['semantic_score']:.3f}")
                print(f"   🎯 Matching Keywords: {chunk_info['matching_keywords']}")
                print(f"   📄 Text Preview: {chunk_info['text'][:200]}...")
                print("-" * 80)
        
        return scored_chunks[:top_k]
    
    def generate_answer_with_groq(self, query, chunk_text):
        """Generate concise answer using Groq API"""
        if not self.groq_client:
            print("⚠️ Groq API not available. Using fallback method.")
            return self.extract_answer_fallback(query, chunk_text)
        
        try:
            # Create prompt for Groq
            prompt = f"""তুমি একজন বাংলা ভাষা বিশেষজ্ঞ। নিচের প্রসঙ্গ থেকে প্রশ্নের উত্তর খুঁজে বের করো।

প্রশ্ন: {query}

প্রসঙ্গ: {chunk_text}

নির্দেশনা:
- শুধুমাত্র প্রশ্নের সরাসরি উত্তর দাও
- এক কথায় উত্তর দাও
- যদি উত্তর একটি নাম হয়, শুধু নামটি দাও
- যদি উত্তর একটি সংখ্যা হয়, শুধু সংখ্যাটি দাও
- যদি উত্তর একটি স্থানের নাম হয়, শুধু স্থানের নামটি দাও
- অতিরিক্ত ব্যাখ্যা দিও না
- যদি উত্তর না পাও, "তথ্য পাওয়া যায়নি" বলো

উত্তর:"""

            # Try different models in order of preference
            models_to_try = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant", 
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
            
            response = None
            for model in models_to_try:
                try:
                    response = self.groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model=model,
                        temperature=0.1,  # Low temperature for more focused answers
                        max_tokens=50,  # Short answers
                        top_p=0.9
                    )
                    print(f"✅ Using model: {model}")
                    break
                except Exception as model_error:
                    print(f"❌ Model {model} failed: {model_error}")
                    continue
            
            if response is None:
                raise Exception("All models failed")
            
            answer = response.choices[0].message.content.strip()
            
            # Clean up the answer
            answer = re.sub(r'^(উত্তর:|Answer:|A:)', '', answer).strip()
            answer = answer.strip('।')  # Remove ending punctuation if any
            
            return answer
            
        except Exception as e:
            print(f"❌ Error with Groq API: {e}")
            return self.extract_answer_fallback(query, chunk_text)
   
    
    def extract_answer_fallback(self, query, chunk_text):
        """Fallback method for answer extraction without API"""
        # Split into sentences
        sentences = re.split(r'[।!?]', chunk_text)
        query_keywords = self.extract_keywords(query)
        
        if not query_keywords or not sentences:
            return chunk_text.strip()[:100]  # Return first 100 chars
        
        # Score each sentence
        sentence_scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Calculate keyword overlap for this sentence
                match_count, _ = self.improved_keyword_matching(query_keywords, sentence)
                
                # Calculate score based on keyword matches
                if match_count > 0:
                    sentence_scores.append((sentence, match_count))
        
        if sentence_scores:
            # Sort by match count
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return best sentence if it has good overlap
            if sentence_scores[0][1] >= 1:  # At least 1 keyword match
                return sentence_scores[0][0]
        
        # Fallback to first 100 characters
        return chunk_text.strip()[:100]
    
    def search_and_answer(self, query, top_k=5, use_groq=True):
        """Main function to search and provide answer"""
        
        best_chunks = self.find_best_chunks(query, top_k=top_k, debug=True)
        
        if not best_chunks:
            return {
                'answer': "দুঃখিত, এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।",
                'confidence': 0.0,
                'keyword_matches': 0,  # Added missing key
                'source_chunks': [],
                'full_chunk': "",
                'matching_keywords': []  # Added missing key
            }
        
        # Get the best chunk (highest keyword match count)
        best_chunk = best_chunks[0]
        
        # Get full chunk text (this was the issue - it was being truncated)
        full_chunk_text = best_chunk['text']
        
        # Generate short answer using Groq API
        if use_groq and self.groq_client:
            short_answer = self.generate_answer_with_groq(query, full_chunk_text)
        else:
            short_answer = self.extract_answer_fallback(query, full_chunk_text)
        
        return {
            'answer': short_answer,  # Short answer from Groq
            'full_chunk': full_chunk_text,  # Complete chunk text
            'confidence': best_chunk['combined_score'],
            'keyword_matches': best_chunk['match_count'],
            'source_chunks': best_chunks[:3],  # Top 3 chunks for reference
            'matching_keywords': best_chunk['matching_keywords']
        } 
    
    def interactive_mode(self):
        """Interactive question-answering mode"""
        print("\n" + "="*60)
        print("🤖 Enhanced Bengali RAG System with Groq Integration")
        print("="*60)
        print("📝 Type your questions in Bengali")
        print("⏹️  Type 'exit', 'quit', or 'bye' to stop")
        print("="*60)
        
        while True:
            try:
                user_query = input("\n❓ আপনার প্রশ্ন লিখুন: ").strip()
                
                if user_query.lower() in ['exit', 'quit', 'bye', 'বাই', 'বন্ধ']:
                    print("👋 ধন্যবাদ! RAG System বন্ধ করা হচ্ছে...")
                    break
                
                if not user_query:
                    print("⚠️ প্রশ্ন লিখুন!")
                    continue
                
                # Get answer
                result = self.search_and_answer(user_query, use_groq=True)
                
                print(f"\n🎯 সংক্ষিপ্ত উত্তর: {result['answer']}")
                print(f"📊 কনফিডেন্স স্কোর: {result['confidence']:.3f}")
                print(f"🔢 কীওয়ার্ড ম্যাচ: {result['keyword_matches']:.1f}")
                
                if result['matching_keywords']:
                    print(f"🔍 মিলেছে এই শব্দগুলো: {', '.join(result['matching_keywords'])}")
                
                # Show full chunk if requested
                show_full = input("\n📄 সম্পূর্ণ প্রসঙ্গ দেখতে চান? (y/n): ").lower()
                if show_full in ['y', 'yes', 'হ্যাঁ', 'হা']:
                    print(f"\n📄 সম্পূর্ণ প্রসঙ্গ:\n{result['full_chunk']}")
                
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 RAG System বন্ধ করা হচ্ছে...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def test_rag_system_with_groq():
    """Test the RAG system with Groq API"""
    
    # Initialize RAG system with Groq
    rag = EnhancedRAGSystemWithGroq(groq_api_key=None)  # Will use GROQ_API_KEY env var
    
    # Test queries
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ে উপলক্ষে কন্যাপক্ষকে কোথায় আসিতে হইল?"
    ]
    
    print("\n" + "="*80)
    print("📋 RAG System Testing with Groq Integration")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}] 🔍 Query: {query}")
        print("-" * 80)
        
        result = rag.search_and_answer(query, top_k=3, use_groq=True)
        
        print(f"✅ Short Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"🔢 Keyword Matches: {result.get('keyword_matches', 0):.1f}")
        
        if result.get('matching_keywords'):
            print(f"🎯 Keywords Found: {', '.join(result['matching_keywords'])}")
        
        if result.get('full_chunk'):
            print(f"\n📄 Full Chunk Preview: {result['full_chunk'][:300]}...")
        print("=" * 80)


if __name__ == "__main__":
    # Test the system first
    print("🚀 Testing Enhanced RAG System with Groq Integration...")
    test_rag_system_with_groq()
    
    # Start interactive mode
    print("\n🔄 Starting Interactive Mode...")
    rag = EnhancedRAGSystemWithGroq()
    rag.interactive_mode()