"""
Performance Monitoring
"""
import time
from datetime import datetime, timezone
from functools import wraps
from flask import Blueprint, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
import psutil
from . import db
from .models import User, ChatHistory

performance_bp = Blueprint('performance', __name__)

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0,
            'avg_response_time': 0,
            'embedding_load_time': 0,
            'kg_load_time': 0,
            'semantic_search_times': [],
            'llm_call_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'entities_loaded': 0,
            'start_time': datetime.now(timezone.utc),
            'query_history': [],
            # F1 Score tracking
            'f1_metrics': {
                'total_entity_lookups': 0,
                'exact_matches': 0,
                'semantic_matches': 0,
                'fuzzy_matches': 0,
                'not_found': 0,
                'semantic_scores': [],
                'fuzzy_scores': []
            }
        }
    
    def record_query_start(self):
        """Record the start of a query"""
        return time.time()
    
    def record_query_end(self, start_time, success=True, query_text="", response_time=None):
        """Record the end of a query"""
        if response_time is None:
            response_time = time.time() - start_time
        
        self.metrics['total_queries'] += 1
        if success:
            self.metrics['successful_queries'] += 1
        else:
            self.metrics['failed_queries'] += 1
        
        self.metrics['total_response_time'] += response_time
        self.metrics['avg_response_time'] = (
            self.metrics['total_response_time'] / self.metrics['total_queries']
        )
        
        # Keep last 100 queries
        self.metrics['query_history'].append({
            'query': query_text,
            'response_time': response_time,
            'response_time_seconds': round(response_time, 3),
            'entities': 0,  # Will be updated by query handler
            'triples': 0,  # Will be updated by query handler
            'success': success,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        if len(self.metrics['query_history']) > 100:
            self.metrics['query_history'].pop(0)
        
        return response_time
    
    def record_embedding_load_time(self, load_time):
        """Record embedding loading time"""
        self.metrics['embedding_load_time'] = load_time
    
    def record_kg_load_time(self, load_time):
        """Record knowledge graph loading time"""
        self.metrics['kg_load_time'] = load_time
    
    def record_semantic_search_time(self, search_time):
        """Record semantic search time"""
        self.metrics['semantic_search_times'].append(search_time)
        if len(self.metrics['semantic_search_times']) > 100:
            self.metrics['semantic_search_times'].pop(0)
    
    def record_llm_call_time(self, call_time):
        """Record LLM API call time"""
        self.metrics['llm_call_times'].append(call_time)
        if len(self.metrics['llm_call_times']) > 100:
            self.metrics['llm_call_times'].pop(0)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.metrics['cache_misses'] += 1
    
    def set_entities_loaded(self, count):
        """Set the number of entities loaded"""
        self.metrics['entities_loaded'] = count
    
    def update_last_query_details(self, entities_count=0, triples_count=0):
        """Update the last query in history with entity and triple counts"""
        if self.metrics['query_history']:
            self.metrics['query_history'][-1]['entities'] = entities_count
            self.metrics['query_history'][-1]['triples'] = triples_count
    
    def record_entity_lookup(self, match_type='not_found', score=None):
        """Record entity lookup results for F1 tracking"""
        self.metrics['f1_metrics']['total_entity_lookups'] += 1
        
        if match_type == 'exact':
            self.metrics['f1_metrics']['exact_matches'] += 1
        elif match_type == 'semantic':
            self.metrics['f1_metrics']['semantic_matches'] += 1
            if score is not None:
                self.metrics['f1_metrics']['semantic_scores'].append(score)
        elif match_type == 'fuzzy':
            self.metrics['f1_metrics']['fuzzy_matches'] += 1
            if score is not None:
                self.metrics['f1_metrics']['fuzzy_scores'].append(score)
        else:
            self.metrics['f1_metrics']['not_found'] += 1
    
    def get_f1_accuracy_metrics(self):
        """Calculate accuracy metrics for F1 dashboard display"""
        f1_data = self.metrics['f1_metrics']
        total = f1_data['total_entity_lookups']
        
        if total == 0:
            return {
                'total_entity_lookups': 0,
                'overall_success_rate': 0,
                'exact_match_rate': 0,
                'semantic_match_rate': 0,
                'semantic_avg_score': 0,
                'fuzzy_match_rate': 0,
                'fuzzy_avg_score': 0,
                'not_found_rate': 0
            }
        
        # Calculate rates
        success_count = f1_data['exact_matches'] + f1_data['semantic_matches'] + f1_data['fuzzy_matches']
        overall_success_rate = round((success_count / total) * 100, 2)
        exact_match_rate = round((f1_data['exact_matches'] / total) * 100, 2)
        semantic_match_rate = round((f1_data['semantic_matches'] / total) * 100, 2)
        fuzzy_match_rate = round((f1_data['fuzzy_matches'] / total) * 100, 2)
        not_found_rate = round((f1_data['not_found'] / total) * 100, 2)
        
        # Calculate average scores
        semantic_avg_score = round(sum(f1_data['semantic_scores']) / len(f1_data['semantic_scores']), 3) if f1_data['semantic_scores'] else 0
        fuzzy_avg_score = round(sum(f1_data['fuzzy_scores']) / len(f1_data['fuzzy_scores']), 3) if f1_data['fuzzy_scores'] else 0
        
        return {
            'total_entity_lookups': total,
            'overall_success_rate': overall_success_rate,
            'exact_match_rate': exact_match_rate,
            'semantic_match_rate': semantic_match_rate,
            'semantic_avg_score': semantic_avg_score,
            'fuzzy_match_rate': fuzzy_match_rate,
            'fuzzy_avg_score': fuzzy_avg_score,
            'not_found_rate': not_found_rate
        }
    
    def get_metrics(self):
        """Get all metrics"""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if self.metrics['semantic_search_times']:
            metrics['avg_semantic_search_time'] = (
                sum(self.metrics['semantic_search_times']) / 
                len(self.metrics['semantic_search_times'])
            )
        else:
            metrics['avg_semantic_search_time'] = 0
        
        if self.metrics['llm_call_times']:
            metrics['avg_llm_call_time'] = (
                sum(self.metrics['llm_call_times']) / 
                len(self.metrics['llm_call_times'])
            )
        else:
            metrics['avg_llm_call_time'] = 0
        
        # Calculate cache hit rate
        total_cache_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total_cache_requests > 0:
            metrics['cache_hit_rate'] = (
                self.metrics['cache_hits'] / total_cache_requests * 100
            )
        else:
            metrics['cache_hit_rate'] = 0
        
        # Calculate uptime
        uptime = datetime.now(timezone.utc) - self.metrics['start_time']
        metrics['uptime_seconds'] = uptime.total_seconds()
        metrics['uptime_formatted'] = str(uptime).split('.')[0]
        uptime_hours = round(uptime.total_seconds() / 3600, 2)
        
        # Add system metrics
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics['system'] = {
                'cpu_percent': round(psutil.cpu_percent(interval=0.1), 2),
                'memory_mb': round(memory_info.rss / (1024 * 1024), 2),
                'memory_percent': round(process.memory_percent(), 2),
                'threads': process.num_threads(),
                'uptime_hours': uptime_hours
            }
        except Exception:
            # If psutil fails, provide default values
            metrics['system'] = {
                'cpu_percent': 0,
                'memory_mb': 0,
                'memory_percent': 0,
                'threads': 0,
                'uptime_hours': 0
            }
        
        # Add database statistics
        try:
            metrics['database'] = {
                'total_users': db.session.query(db.func.count(User.id)).scalar() or 0,
                'total_chats': db.session.query(db.func.count(ChatHistory.id)).scalar() or 0
            }
        except Exception:
            metrics['database'] = {
                'total_users': 0,
                'total_chats': 0
            }
        
        # Add query_details for dashboard compatibility
        if 'query_history' in metrics and metrics['query_history']:
            total_queries = len(metrics['query_history'])
            avg_response = sum(q['response_time'] for q in metrics['query_history']) / total_queries if total_queries > 0 else 0
            
            metrics['query_details'] = {
                'total_queries': metrics.get('total_queries', 0),
                'avg_response_time_seconds': round(avg_response, 3),
                'avg_llm_time_seconds': round(metrics.get('avg_llm_call_time', 0), 3),
                'avg_entities_per_query': 0,  # Can be calculated from query history if needed
                'recent_queries': metrics['query_history'][-10:]  # Last 10 queries
            }
        else:
            metrics['query_details'] = {
                'total_queries': 0,
                'avg_response_time_seconds': 0,
                'avg_llm_time_seconds': 0,
                'avg_entities_per_query': 0,
                'recent_queries': []
            }
        
        # Add accuracy metrics with real F1 data
        metrics['accuracy'] = self.get_f1_accuracy_metrics()
        
        # Add API endpoints placeholder for dashboard compatibility
        metrics['api_endpoints'] = {}
        
        # Add processing operations placeholder for dashboard compatibility
        metrics['processing'] = {}
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.__init__()

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def track_performance(f):
    """Decorator to track endpoint performance"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        g.request_start_time = time.time()
        try:
            result = f(*args, **kwargs)
            performance_monitor.record_query_end(
                g.request_start_time,
                success=True
            )
            return result
        except Exception as e:
            performance_monitor.record_query_end(
                g.request_start_time,
                success=False
            )
            raise e
    return decorated_function

@performance_bp.route('/metrics', methods=['GET'])
@jwt_required()
def get_metrics():
    """Get performance metrics"""
    metrics = performance_monitor.get_metrics()
    return jsonify(metrics), 200

@performance_bp.route('/metrics/reset', methods=['POST'])
@jwt_required()
def reset_metrics():
    """Reset performance metrics"""
    performance_monitor.reset_metrics()
    return jsonify({"message": "Metrics reset successfully"}), 200

@performance_bp.route('/f1-evaluation', methods=['POST'])
@jwt_required()
def run_f1_evaluation():
    """Run F1 evaluation using the test dataset"""
    try:
        import subprocess
        import os
        
        # Get the current working directory
        script_path = os.path.join(os.getcwd(), 'evaluate_f1.py')
        
        if not os.path.exists(script_path):
            return jsonify({"error": "F1 evaluation script not found"}), 404
        
        # Run the evaluation script
        result = subprocess.run(
            ['python3', script_path],
            input='1\n',  # Choose single evaluation
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode == 0:
            # Parse the output to extract key metrics
            output_lines = result.stdout.split('\n')
            metrics = {}
            
            for line in output_lines:
                if 'F1 Score:' in line and '(' in line:
                    try:
                        # Extract F1 score from lines like "  F1 Score:  0.1156 (11.56%)"
                        parts = line.split('F1 Score:')[1].strip()
                        f1_value = float(parts.split('(')[0].strip())
                        metrics['f1_score'] = f1_value
                    except:
                        pass
                elif 'Precision:' in line and '(' in line:
                    try:
                        parts = line.split('Precision:')[1].strip()
                        precision_value = float(parts.split('(')[0].strip())
                        metrics['precision'] = precision_value
                    except:
                        pass
                elif 'Recall:' in line and '(' in line:
                    try:
                        parts = line.split('Recall:')[1].strip()
                        recall_value = float(parts.split('(')[0].strip())
                        metrics['recall'] = recall_value
                    except:
                        pass
            
            return jsonify({
                "message": "F1 evaluation completed successfully",
                "metrics": metrics,
                "full_output": result.stdout
            }), 200
        else:
            return jsonify({
                "error": "F1 evaluation failed",
                "stderr": result.stderr,
                "stdout": result.stdout
            }), 500
            
    except subprocess.TimeoutExpired:
        return jsonify({"error": "F1 evaluation timed out"}), 408
    except Exception as e:
        return jsonify({"error": f"Failed to run F1 evaluation: {str(e)}"}), 500
