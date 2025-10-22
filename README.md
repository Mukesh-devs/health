# Health LLM Application

AI-powered health information system using Knowledge Graphs, Semantic Search, and Large Language Models.

## 🎯 Overview

This application provides intelligent health information retrieval by combining:
- **Knowledge Graph**: 162,212 medical entities with 1,017,284 relationships
- **Semantic Search**: Using Sentence Transformers for entity matching
- **LLM Integration**: Google Gemini for natural language understanding
- **Information Retrieval**: Real-time performance metrics and evaluation

## 📁 Project Structure

```
health/
├── dataset/                    # Knowledge graph and test data
│   ├── neo4j_node.csv         # 162,212 medical entities (UMLS CUIs)
│   ├── neo4j_rel.csv          # 1,017,284 relationships
│   ├── test_queries.csv       # Test queries for evaluation
│   └── test_queries_original.csv
│
├── cache/                      # Pre-computed embeddings
│   └── entity_embeddings_cache.npz  # 162K entity embeddings (236MB)
│
├── static/                     # Frontend files
│   ├── index.html             # Main UI
│   └── login.html             # Login page
│
├── healthapp/                  # Backend application
│   ├── __init__.py            # App factory
│   ├── config.py              # Configuration
│   ├── models.py              # Database models
│   ├── auth.py                # Authentication
│   ├── query.py               # Query processing
│   ├── embedding.py           # Semantic search & retrieval
│   ├── kg_loader.py           # Knowledge graph loader
│   ├── chat_sessions.py       # Session management
│   └── metrics_logger.py      # Performance metrics
│
├── logs/                       # Application logs
│   ├── retrieval_metrics.log  # IR metrics (startup)
│   └── query_execution_metrics.jsonl  # Query logs
│
├── instance/                   # Instance-specific files (auto-generated)
├── archived/                   # Archived code
├── .env                        # Environment variables (DO NOT COMMIT)
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── app_new.py                 # Application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/health.git
cd health
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use any text editor
```

**Required environment variables in `.env`:**
```env
GEMINI_API_KEY=your-actual-api-key-here
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```

### 6. Run the Application

```bash
python3 app_new.py
```

The application will start on: **http://localhost:5001**

**On startup, you will see:**
- Knowledge graph loading progress
- Entity embeddings cache loading
- Information Retrieval metrics evaluation
- F1 Score, Precision, Recall, MRR, MAP

### 7. Access the Application

- **Frontend**: http://localhost:5001/static/login.html
- **API Base**: http://localhost:5001

## 📊 Performance Metrics

Current retrieval performance (Information Retrieval metrics):

| Metric | Score | Description |
|--------|-------|-------------|
| **F1 Score** | 63.53% | Harmonic mean of precision and recall |
| **Precision** | 75.00% | Accuracy of retrieved entities |
| **Recall** | 55.10% | Coverage of expected entities |
| **MRR** | 0.7576 | Mean Reciprocal Rank |
| **MAP** | 0.6364 | Mean Average Precision |

**Evaluation Dataset:**
- 33 Alzheimer's disease queries
- Evaluated on startup automatically
- Results logged to `logs/retrieval_metrics.log`

## 🔧 Configuration

All configuration is managed through the `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | - | ✅ Yes |
| `SECRET_KEY` | Flask secret key | - | ✅ Yes |
| `JWT_SECRET_KEY` | JWT token secret | - | ✅ Yes |
| `DATABASE_URL` | Database connection | `sqlite:///health_app.db` | No |
| `KG_NODE_PATH` | Path to node CSV | `dataset/neo4j_node.csv` | No |
| `KG_REL_PATH` | Path to relationships CSV | `dataset/neo4j_rel.csv` | No |
| `EMBEDDINGS_CACHE_PATH` | Path to embeddings cache | `cache/entity_embeddings_cache.npz` | No |
| `SENTENCE_MODEL_NAME` | Sentence transformer model | `all-MiniLM-L6-v2` | No |
| `FLASK_ENV` | Flask environment | `development` | No |
| `FLASK_DEBUG` | Debug mode | `1` | No |

## 📝 Logs

### Query Execution Logs

View formatted JSON logs of each query:

```bash
# Pretty-print JSON logs
jq . logs/query_execution_metrics.jsonl

# View last 5 queries
tail -5 logs/query_execution_metrics.jsonl | jq .

# Filter by session_id
jq 'select(.session_id == 21)' logs/query_execution_metrics.jsonl
```

**Log format:**
```json
{
  "timestamp": "2025-10-22T18:38:15.001357",
  "query": "does vitamin e cures ad",
  "session_id": 21,
  "response_time_seconds": 12.244,
  "entities_found": 8,
  "triples_extracted": 5,
  "coverage": {
    "entities_found": 8,
    "triples_extracted": 5
  }
}
```

### Retrieval Metrics Log

View Information Retrieval performance:

```bash
cat logs/retrieval_metrics.log
```

## 🔐 Security

- ✅ API keys stored in `.env` file (excluded from Git)
- ✅ JWT-based authentication
- ✅ Bcrypt password hashing
- ✅ Session-based security
- ✅ CORS protection
- ⚠️ **Never commit `.env` file to Git**
- ⚠️ **Always use `.env.example` as template**

## 📈 Dataset Information

### Knowledge Graph
- **Entities**: 162,212 medical concepts (UMLS CUIs)
- **Relationships**: 1,017,284 medical relationships
- **Format**: Neo4j CSV export
- **Source**: UMLS (Unified Medical Language System)

### Test Queries
- **Count**: 33 queries
- **Domain**: Alzheimer's disease
- **Format**: CSV with expected CUIs
- **Purpose**: Automated retrieval evaluation

### Embeddings
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Size**: 236 MB (compressed NPZ format)
- **Count**: 162,212 entity embeddings
- **Load Time**: ~0.8 seconds

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask 3.0 |
| **Database** | SQLite with SQLAlchemy |
| **Authentication** | JWT + Bcrypt |
| **AI/ML** | Sentence Transformers, PyTorch |
| **LLM** | Google Gemini 2.5 Flash |
| **Semantic Search** | Cosine Similarity |
| **Frontend** | HTML, JavaScript |
| **Knowledge Graph** | Neo4j CSV format |
| **Embeddings** | all-MiniLM-L6-v2 (384d) |

## 📚 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login user
- `POST /auth/logout` - Logout user

### Queries
- `POST /query` - Process health query
- `GET /history?session_id={id}` - Get chat history

### Sessions
- `GET /sessions` - List all sessions
- `POST /sessions` - Create new session
- `DELETE /sessions/{id}` - Delete session

## 🧪 Development

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application with debug mode
python3 app_new.py
```

### Viewing Logs

```bash
# Watch query logs in real-time
tail -f logs/query_execution_metrics.jsonl

# View retrieval metrics
cat logs/retrieval_metrics.log
```

### Updating Dependencies

```bash
pip install --upgrade -r requirements.txt
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Issue: Module not found

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: API key error

```bash
# Check .env file exists
ls -la .env

# Verify API key is set
grep GEMINI_API_KEY .env
```

### Issue: Database not found

```bash
# The database is auto-created on first run
# Delete and recreate if corrupted
rm -f health_app.db
python3 app_new.py
```

### Issue: Embeddings cache missing

```bash
# Verify cache file exists
ls -lh cache/entity_embeddings_cache.npz

# If missing, you need to regenerate it
# (Contact maintainer for cache generation script)
```

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using Flask, Sentence Transformers, and Google Gemini**
