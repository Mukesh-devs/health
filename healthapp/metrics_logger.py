"""
Metrics Logger for Retrieval Accuracy and Query Performance
Logs retrieval metrics at startup and query metrics per execution
"""
import json
import time
from datetime import datetime
from pathlib import Path

class MetricsLogger:
    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Log files
        self.retrieval_log = self.logs_dir / "retrieval_accuracy_metrics.jsonl"
        self.query_log = self.logs_dir / "query_execution_metrics.jsonl"
    
    def log_retrieval_accuracy(self, precision, recall, f1_score, top_k_accuracy, 
                               total_queries, perfect_matches, evaluation_time):
        """
        Log retrieval accuracy metrics (run at startup or after evaluation)
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "top_k_accuracy": round(top_k_accuracy, 4),
            "total_queries_evaluated": total_queries,
            "perfect_matches": perfect_matches,
            "evaluation_time_seconds": round(evaluation_time, 2),
            "strategy": "ADAPTIVE_PRECISION"
        }
        
        self._write_log(self.retrieval_log, metrics)
        print(f"✅ Retrieval accuracy logged: F1={f1_score:.3f}, P={precision:.3f}, R={recall:.3f}")
    
    def log_query_execution(self, query, entities_found, triples_extracted, 
                           response_time, llm_time=0, session_id=None):
        """
        Log metrics for each query execution
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "session_id": session_id,
            "response_time_seconds": round(response_time, 3),
            "entities_found": entities_found,
            "triples_extracted": triples_extracted,
            "coverage": {
                "entities_found": entities_found,
                "triples_extracted": triples_extracted
            }
        }
        
        self._write_log(self.query_log, metrics)
    
    def log_query_metrics(self, query, latency, entities_found, triples_extracted, session_id=None):
        """
        Alias for log_query_execution with simplified parameters
        """
        self.log_query_execution(
            query=query,
            entities_found=entities_found,
            triples_extracted=triples_extracted,
            response_time=latency,
            session_id=session_id
        )
    
    def _write_log(self, log_file, data):
        """Write a single JSON line to log file"""
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
    
    def get_recent_retrieval_metrics(self, limit=10):
        """Get recent retrieval accuracy metrics"""
        return self._read_recent_logs(self.retrieval_log, limit)
    
    def get_recent_query_metrics(self, limit=100):
        """Get recent query execution metrics"""
        return self._read_recent_logs(self.query_log, limit)
    
    def _read_recent_logs(self, log_file, limit):
        """Read recent logs from file"""
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return logs[-limit:] if logs else []

# Global metrics logger instance
metrics_logger = MetricsLogger()

# Aliases for backward compatibility
retrieval_logger = metrics_logger
query_logger = metrics_logger
