import json
import os
import re
import threading
import time
from typing import Any, Dict, List


class ChromaDB:
    _current_collection: str | None = None

    def __init__(self, collection_name: str = 'ventureboard', data_dir: str | None = None):
        self.collection_name = collection_name
        self._data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        self._lock = threading.RLock()
        os.makedirs(self._data_dir, exist_ok=True)
        
        self.retrieved_chunks_count = 0
        self.total_embedding_time = 0.0
        self.total_retrieval_time = 0.0
        
        if collection_name.startswith('ventureboard_'):
            ChromaDB._current_collection = collection_name
            self._cleanup_old_collections()
            
        if not os.path.exists(self._data_path):
            self._write_documents([])

    @property
    def _data_path(self) -> str:
        col_name = self.collection_name
        if col_name == 'ventureboard' and ChromaDB._current_collection:
            col_name = ChromaDB._current_collection
        return os.path.join(self._data_dir, f'{col_name}.json')

    def _cleanup_old_collections(self) -> None:
        try:
            for fname in os.listdir(self._data_dir):
                is_old = False
                if fname.startswith('ventureboard_') and fname.endswith(('.json', '.json.tmp')):
                    if self.collection_name not in fname:
                        is_old = True
                elif fname in ('ventureboard.json', 'ventureboard.json.tmp'):
                    is_old = True
                
                if is_old:
                    fpath = os.path.join(self._data_dir, fname)
                    if os.path.exists(fpath):
                        os.unlink(fpath)
        except Exception:
            pass

    def _read_documents(self) -> List[Dict[str, Any]]:
        try:
            with open(self._data_path, 'r', encoding='utf-8') as handle:
                documents = json.load(handle)
                return documents if isinstance(documents, list) else []
        except (json.JSONDecodeError, OSError, ValueError):
            return []

    def _write_documents(self, documents: List[Dict[str, Any]]) -> None:
        temp_path = f'{self._data_path}.tmp'
        with open(temp_path, 'w', encoding='utf-8') as handle:
            json.dump(documents, handle, indent=2)
        for i in range(10):
            try:
                os.replace(temp_path, self._data_path)
                break
            except PermissionError:
                if i == 9:
                    raise
                time.sleep(0.1)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        t0 = time.time()
        with self._lock:
            existing = self._read_documents()
            
            existing_startup = None
            for doc in existing:
                if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
                    if 'startup' in doc['metadata']:
                        existing_startup = doc['metadata']['startup']
                        break

            incoming_startup = None
            for doc in documents:
                if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
                    if 'startup' in doc['metadata']:
                        incoming_startup = doc['metadata']['startup']
                        break

            if existing_startup and incoming_startup and existing_startup.lower() != incoming_startup.lower():
                return

            existing.extend(documents)
            self._write_documents(existing)
        self.total_embedding_time += (time.time() - t0)

    def clear(self) -> None:
        """Wipe all stored documents so the next proposal starts with a blank slate."""
        with self._lock:
            self._write_documents([])

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        t0 = time.time()
        with self._lock:
            documents = self._read_documents()
            query_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
            if not query_words:
                query_words = [query.lower()]
            
            scored_docs = []
            for doc in documents:
                doc_str = json.dumps(doc).lower()
                score = sum(doc_str.count(word) for word in query_words)
                scored_docs.append((score, doc))
            
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            results = [doc for score, doc in scored_docs[:limit]]
            self.retrieved_chunks_count += len(results)
            self.total_retrieval_time += (time.time() - t0)
            return results


class SimpleVectorStore(ChromaDB):
    pass

