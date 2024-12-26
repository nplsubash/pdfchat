from fastapi.responses import HTMLResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import jwt
import sqlite3
import logging
import uvicorn
from pydantic import BaseModel
import shutil
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, validator
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database setup
def setup_database():
    conn = sqlite3.connect('chatpdf.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    # Create documents table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP,
            uploader_id INTEGER,
            description TEXT,
            tags TEXT,
            FOREIGN KEY (uploader_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

setup_database()

class User(BaseModel):
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def username_must_be_valid(cls, v):
        if not re.match("^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError('Username must be 3-20 characters and contain only letters, numbers, and underscore')
        return v

    @validator('password')
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search("[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search("[0-9]", v):
            raise ValueError('Password must contain at least one number')
        return v

class Document(BaseModel):
    filename: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class DocumentProcessor:
    def __init__(self, storage_dir: str = "pdf_storage"):
        self.storage_dir = storage_dir
        self.embeddings_dir = "embeddings"
        self.ensure_directories()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def ensure_directories(self):
        """Ensure required directories exist."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            os.makedirs(self.embeddings_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract text from PDF and split into chunks with error handling."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            chunks = []
            current_chunk = ""
            for paragraph in text.split('\n\n'):
                if len(current_chunk) + len(paragraph) < 500:
                    current_chunk += paragraph + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph + " "
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            raise

    def process_pdf(self, pdf_path: str, pdf_name: str, description: str = "", tags: List[str] = None):
        """Process PDF with enhanced error handling and metadata."""
        try:
            chunks = self.extract_text_from_pdf(pdf_path)
            embeddings = self.compute_embeddings(chunks)
            metadata = {
                'filename': pdf_name,
                'description': description,
                'tags': tags or [],
                'processed_date': datetime.now().isoformat(),
                'chunk_count': len(chunks)
            }
            self.save_document_data(pdf_name, chunks, embeddings, metadata)
            return {"message": f"Successfully processed {pdf_name}", "metadata": metadata}
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

class MultiDocumentSearchEngine:
    def __init__(self, doc_processor: DocumentProcessor):
        self.doc_processor = doc_processor

    def search_across_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search across all available documents."""
        try:
            results = []
            query_embedding = self.doc_processor.model.encode([query])[0]
            
            # Search through all available documents
            for filename in os.listdir(self.doc_processor.embeddings_dir):
                if filename.endswith('.pkl'):
                    doc_data = self.doc_processor.load_document_data(filename[:-4])
                    doc_embeddings = doc_data['embeddings']
                    chunks = doc_data['chunks']
                    metadata = doc_data.get('metadata', {})
                    
                    # Calculate similarities
                    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                    top_indices = np.argsort(similarities)[-top_k:]
                    
                    for idx in top_indices:
                        results.append({
                            'document': metadata.get('filename', filename[:-4]),
                            'chunk': chunks[idx],
                            'similarity': float(similarities[idx]),
                            'metadata': metadata
                        })
            
            # Sort results by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in multi-document search: {e}")
            raise HTTPException(status_code=500, detail="Search failed")

class DocumentManager:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        
    def get_document_list(self, user_id: int = None) -> List[Dict]:
        """Get list of documents with metadata."""
        conn = sqlite3.connect('chatpdf.db')
        try:
            c = conn.cursor()
            if user_id:
                c.execute('''
                    SELECT id, filename, upload_date, description, tags 
                    FROM documents WHERE uploader_id = ?
                ''', (user_id,))
            else:
                c.execute('''
                    SELECT id, filename, upload_date, description, tags 
                    FROM documents
                ''')
            documents = []
            for row in c.fetchall():
                documents.append({
                    'id': row[0],
                    'filename': row[1],
                    'upload_date': row[2],
                    'description': row[3],
                    'tags': row[4].split(',') if row[4] else []
                })
            return documents
        finally:
            conn.close()

    def delete_document(self, doc_id: int, user_id: int):
        """Delete document and its associated data."""
        conn = sqlite3.connect('chatpdf.db')
        try:
            c = conn.cursor()
            # Check ownership
            c.execute('SELECT filename FROM documents WHERE id = ? AND uploader_id = ?', 
                     (doc_id, user_id))
            result = c.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Document not found or unauthorized")
                
            filename = result[0]
            # Delete file and embeddings
            pdf_path = os.path.join(self.storage_dir, filename)
            embeddings_path = os.path.join('embeddings', f"{filename}.pkl")
            
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
                
            # Delete from database
            c.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            
            return {"message": f"Document {filename} deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete document")
        finally:
            conn.close()

# User authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    conn = sqlite3.connect('chatpdf.db')
    try:
        c = conn.cursor()
        c.execute('SELECT id, username, hashed_password FROM users WHERE username = ?', 
                 (username,))
        user = c.fetchone()
        if not user or not verify_password(password, user[2]):
            return False
        return {'id': user[0], 'username': user[1]}
    finally:
        conn.close()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
        
    conn = sqlite3.connect('chatpdf.db')
    try:
        c = conn.cursor()
        c.execute('SELECT id, username FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        if user is None:
            raise credentials_exception
        return {'id': user[0], 'username': user[1]}
    finally:
        conn.close()

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
search_engine = MultiDocumentSearchEngine(doc_processor)
doc_manager = DocumentManager("pdf_storage")

# Auth endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(user: User):
    conn = sqlite3.connect('chatpdf.db')
    try:
        c = conn.cursor()
        # Check if username exists
        c.execute('SELECT username FROM users WHERE username = ?', (user.username,))
        if c.fetchone():
            return JSONResponse(
                status_code=400,
                content={"detail": "Username already exists"}
            )
        
        # Check if email exists
        c.execute('SELECT email FROM users WHERE email = ?', (user.email,))
        if c.fetchone():
            return JSONResponse(
                status_code=400,
                content={"detail": "Email already registered"}
            )

        hashed_password = get_password_hash(user.password)
        c.execute('''
            INSERT INTO users (username, email, hashed_password) 
            VALUES (?, ?, ?)
        ''', (user.username, user.email, hashed_password))
        conn.commit()
        return {"message": "User created successfully"}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Registration failed. Please try again."}
        )
    finally:
        conn.close()

# Document management endpoints
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    description: str = "",
    tags: str = "",
    current_user: dict = Depends(get_current_user)
):
    """Upload and process a PDF file with metadata."""
    try:
        # Save the uploaded file
        pdf_path = os.path.join("pdf_storage", file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        tags_list = [tag.strip() for tag in tags.split(',')] if tags else []
        result = doc_processor.process_pdf(pdf_path, file.filename, description, tags_list)
        
        # Save to database
        conn = sqlite3.connect('chatpdf.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO documents (filename, upload_date, uploader_id, description, tags)
            VALUES (?, ?, ?, ?, ?)
        ''', (file.filename, datetime.now(), current_user['id'], description, tags))
        conn.commit()
        conn.close()
        
        return result
    except Exception as e:
        logger.error(f"Error in upload_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all documents for the current user."""
    return doc_manager.get_document_list(current_user['id'])

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    """Delete a document."""
    return doc_manager.delete_document(doc_id, current_user['id'])

# Chat endpoints
@app.post("/chat")
async def chat(
    query: str,
    current_user: dict = Depends(get_current_user)
):
    """Search across all documents and return relevant results."""
    try:
        results = search_engine.search_across_documents(query)
        return {
            "results": results,
            "query": query
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Chat query failed")

# Enhanced frontend with authentication and document management
# [Previous code remains the same until HTML_CONTENT]

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced ChatPDF</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .loading {
            position: relative;
        }
        .loading:after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            top: 50%;
            left: 50%;
            margin: -10px 0 0 -10px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .fade-enter {
            opacity: 0;
            transform: translateY(-10px);
        }
        .fade-enter-active {
            opacity: 1;
            transform: translateY(0);
            transition: opacity 300ms, transform 300ms;
        }
        .chat-bubble {
            position: relative;
            background: #E3F2FD;
            border-radius: .4em;
            padding: 15px;
            margin: 10px;
        }
        .chat-bubble:after {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            width: 0;
            height: 0;
            border: 20px solid transparent;
            border-right-color: #E3F2FD;
            border-left: 0;
            margin-top: -20px;
            margin-left: -20px;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Auth Section -->
    <div id="authSection" class="container mx-auto px-4 py-8">
        <div class="max-w-md mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
            <div class="p-6">
                <div class="flex justify-between mb-6">
                    <button onclick="showLoginForm()" 
                            id="loginTab" 
                            class="text-lg font-semibold text-blue-600 pb-2 border-b-2 border-blue-600">
                        Login
                    </button>
                    <button onclick="showRegisterForm()" 
                            id="registerTab" 
                            class="text-lg font-semibold text-gray-500 pb-2">
                        Register
                    </button>
                </div>

                <!-- Login Form -->
                <form id="loginForm" class="space-y-4">
                    <div>
                        <label class="block text-gray-700">Username</label>
                        <input type="text" id="loginUsername" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                    </div>
                    <div>
                        <label class="block text-gray-700">Password</label>
                        <input type="password" id="loginPassword" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                    </div>
                    <button type="submit" 
                            class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-200">
                        Login
                    </button>
                    <div id="loginError" class="text-red-500 text-sm hidden"></div>
                </form>

                <!-- Register Form -->
                <form id="registerForm" class="space-y-4 hidden">
                    <div>
                        <label class="block text-gray-700">Username</label>
                        <input type="text" id="registerUsername" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                        <p class="text-sm text-gray-500">3-20 characters, letters, numbers, and underscore only</p>
                    </div>
                    <div>
                        <label class="block text-gray-700">Email</label>
                        <input type="email" id="registerEmail" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                    </div>
                    <div>
                        <label class="block text-gray-700">Password</label>
                        <input type="password" id="registerPassword" 
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500">
                        <p class="text-sm text-gray-500">Minimum 8 characters, at least one uppercase letter and one number</p>
                    </div>
                    <button type="submit" 
                            class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-200">
                        Register
                    </button>
                    <div id="registerError" class="text-red-500 text-sm hidden"></div>
                    <div id="registerSuccess" class="text-green-500 text-sm hidden"></div>
                </form>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div id="mainContent" class="hidden container mx-auto px-4 py-8">
        <!-- Rest of your existing main content HTML with Tailwind classes -->
    </div>

    <script>
        // Show/Hide form functions
        function showLoginForm() {
            document.getElementById('loginForm').classList.remove('hidden');
            document.getElementById('registerForm').classList.add('hidden');
            document.getElementById('loginTab').classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
            document.getElementById('loginTab').classList.remove('text-gray-500');
            document.getElementById('registerTab').classList.add('text-gray-500');
            document.getElementById('registerTab').classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
        }

        function showRegisterForm() {
            document.getElementById('loginForm').classList.add('hidden');
            document.getElementById('registerForm').classList.remove('hidden');
            document.getElementById('registerTab').classList.add('text-blue-600', 'border-b-2', 'border-blue-600');
            document.getElementById('registerTab').classList.remove('text-gray-500');
            document.getElementById('loginTab').classList.add('text-gray-500');
            document.getElementById('loginTab').classList.remove('text-blue-600', 'border-b-2', 'border-blue-600');
        }

        // Form submission handlers
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const loginError = document.getElementById('loginError');
            loginError.classList.add('hidden');

            try {
                const response = await fetch('/token', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        'username': document.getElementById('loginUsername').value,
                        'password': document.getElementById('loginPassword').value
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    localStorage.setItem('token', data.access_token);
                    document.getElementById('authSection').classList.add('hidden');
                    document.getElementById('mainContent').classList.remove('hidden');
                    loadDocuments();
                } else {
                    loginError.textContent = 'Invalid credentials';
                    loginError.classList.remove('hidden');
                }
            } catch (error) {
                loginError.textContent = 'Login failed. Please try again.';
                loginError.classList.remove('hidden');
            }
        });

        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const registerError = document.getElementById('registerError');
            const registerSuccess = document.getElementById('registerSuccess');
            registerError.classList.add('hidden');
            registerSuccess.classList.add('hidden');

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        username: document.getElementById('registerUsername').value,
                        email: document.getElementById('registerEmail').value,
                        password: document.getElementById('registerPassword').value
                    })
                });

                const data = await response.json();

                if (response.ok) {
                    registerSuccess.textContent = 'Registration successful! Please login.';
                    registerSuccess.classList.remove('hidden');
                    setTimeout(() => showLoginForm(), 2000);
                } else {
                    registerError.textContent = data.detail || 'Registration failed';
                    registerError.classList.remove('hidden');
                }
            } catch (error) {
                registerError.textContent = 'Registration failed. Please try again.';
                registerError.classList.remove('hidden');
            }
        });

        // Add loading indicators and other interactive features...
        // [Rest of your existing JavaScript code]
    </script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)