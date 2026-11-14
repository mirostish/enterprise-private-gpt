from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"
MODEL_DIR = BASE_DIR / "models" / "multilingual-e5-base"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "corporate_knowledge"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 32
TOP_K = 4

# Конфигурация LLM
LLM_MODEL = "qwen3:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"

LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 4096
LLM_MAX_TOKENS = 400

# None = порог отключён
# При включении: меньшее расстояние лучше
MAX_COSINE_DISTANCE = None

NO_ANSWER = (
    "В предоставленных документах "
    "нет информации для ответа на этот вопрос."
)