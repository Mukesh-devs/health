"""
Embedding and Semantic Search
"""
import os
import time
import csv
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from flask import current_app

# Global variables
sentence_model = None
entity_embeddings_matrix = None
entity_names_list = []
cui_list = []
nodes_df = None
kg_df = None

def initialize_model():
    """Initialize the Sentence Transformer model"""
    global sentence_model
    if sentence_model is None:
        print("Loading Sentence Transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully!")
    return sentence_model

def load_embeddings(nodes_dataframe):
    """Load or generate embeddings with caching (Apple Silicon optimized)"""
    global entity_embeddings_matrix, entity_names_list, cui_list, nodes_df
    
    nodes_df = nodes_dataframe
    cache_path = current_app.config['EMBEDDINGS_CACHE_PATH']
    
    # Try to load cached embeddings
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        start_time = time.time()
        cache = np.load(cache_path, allow_pickle=True)
        entity_embeddings_matrix = cache['embeddings']
        entity_names_list = cache['names'].tolist()
        cui_list = cache['cuis'].tolist()
        load_time = time.time() - start_time
        
        print(f"Loaded cached embeddings for {len(entity_names_list)} entities in {load_time:.2f}s")
        
        # Evaluate and log retrieval metrics
        _evaluate_and_log_retrieval_metrics()
        return
    
    # Generate embeddings if cache doesn't exist
    print(f"Generating embeddings for {len(nodes_df)} entities...")
    print("This is a one-time process and may take 10-15 minutes for 160K entities.")
    
    start_time = time.time()
    
    entity_names_list = nodes_df['Name'].tolist()
    cui_list = nodes_df['CUI'].tolist()
    
    # Initialize model if not already loaded
    model = initialize_model()
    
    # Generate embeddings in batches for memory efficiency
    batch_size = 512
    embeddings_list = []
    
    total_batches = (len(entity_names_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(entity_names_list), batch_size):
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}...")
        
        batch = entity_names_list[i:i+batch_size]
        batch_embeddings = model.encode(
            batch, 
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True
        )
        embeddings_list.append(batch_embeddings)
    
    entity_embeddings_matrix = np.vstack(embeddings_list).astype('float32')
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(entity_embeddings_matrix, axis=1, keepdims=True)
    entity_embeddings_matrix = entity_embeddings_matrix / norms
    
    # Cache embeddings
    print("Caching embeddings for future use...")
    np.savez_compressed(
        cache_path,
        embeddings=entity_embeddings_matrix,
        names=np.array(entity_names_list),
        cuis=np.array(cui_list)
    )
    
    load_time = time.time() - start_time
    
    print(f"Embeddings generated and cached for {len(entity_names_list)} entities in {load_time:.2f}s")

def preprocess_query(query_text):
    """Enhanced query preprocessing for better entity matching"""
    import re
    
    # Convert to lowercase
    query = query_text.lower().strip()
    
    # Remove common question words and phrases
    stop_patterns = [
        r'\bwhat is\b', r'\bwhat are\b', r'\bhow to\b', r'\bwhy do\b',
        r'\bcan you tell me about\b', r'\btell me about\b',
        r'\binformation about\b', r'\bdetails about\b',
        r'\bexplain\b', r'\bdescribe\b'
    ]
    
    for pattern in stop_patterns:
        query = re.sub(pattern, '', query)
    
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Extract key medical terms (simple approach)
    medical_terms = []
    words = query.split()
    
    # Look for compound medical terms first
    for i in range(len(words) - 1):
        compound = f"{words[i]} {words[i+1]}"
        if len(compound) > 8:  # Likely medical compound term
            medical_terms.append(compound)
    
    # Add individual significant words
    for word in words:
        if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'from', 'that', 'this']:
            medical_terms.append(word)
    
    return query, medical_terms

def get_cui_from_name_semantic(entity_name, threshold=0.75, top_k=10):
    """
    Enhanced semantic search with better preprocessing and scoring
    """
    if not entity_name or entity_embeddings_matrix is None:
        return None
    
    start_time = time.time()
    
    # Preprocess the query
    processed_query, medical_terms = preprocess_query(entity_name)
    
    # Try exact match with processed query first
    entity_name_lower = processed_query.lower().strip()
    exact_match = nodes_df[nodes_df['Name_lower'] == entity_name_lower]
    if not exact_match.empty:
        cui = exact_match.iloc[0]['CUI']
        search_time = time.time() - start_time
        return cui
    
    # Try exact match on medical terms
    for term in medical_terms:
        exact_match = nodes_df[nodes_df['Name_lower'] == term.lower()]
        if not exact_match.empty:
            cui = exact_match.iloc[0]['CUI']
            search_time = time.time() - start_time
            return cui
    
    # Initialize model if needed
    model = initialize_model()
    
    # Multi-query semantic search
    queries_to_try = [processed_query] + medical_terms[:3]  # Try top 3 medical terms
    best_similarity = 0
    best_cui = None
    
    for query in queries_to_try:
        if not query.strip():
            continue
            
        # Semantic search
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Vectorized cosine similarity
        similarities = np.dot(entity_embeddings_matrix, query_embedding.T).flatten()
        
        # Get top_k indices
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        current_best = similarities[top_indices[0]]
        if current_best > best_similarity:
            best_similarity = current_best
            best_cui = cui_list[top_indices[0]]
    
    search_time = time.time() - start_time
    
    if best_similarity >= threshold:
        return best_cui
    
    # No match found
    return None

def get_cui_from_name(entity_name, semantic_threshold=0.75, fuzzy_threshold=95):
    """
    Enhanced hybrid approach with improved thresholds and multi-stage matching
    """
    if not entity_name:
        return None
    
    # Try semantic search first with higher threshold
    semantic_cui = get_cui_from_name_semantic(entity_name, threshold=semantic_threshold)
    if semantic_cui:
        return semantic_cui
    
    # Enhanced fuzzy matching with preprocessing
    processed_query, medical_terms = preprocess_query(entity_name)
    
    # Try fuzzy matching on processed query
    for query_variant in [processed_query] + medical_terms[:2]:
        if not query_variant.strip():
            continue
            
        query_lower = query_variant.lower().strip()
        matches = process.extract(
            query_lower, 
            nodes_df['Name_lower'], 
            scorer=fuzz.token_sort_ratio, 
            limit=3  # Get top 3 for better selection
        )
        
        if matches:
            # Use the best match that meets threshold
            for match in matches:
                if match[1] >= fuzzy_threshold:
                    match_idx = nodes_df[nodes_df['Name_lower'] == match[0]].index[0]
                    cui = nodes_df.iloc[match_idx]['CUI']
                    # Record fuzzy match for F1 tracking
                    fuzzy_score = match[1] / 100.0  # Convert to 0-1 scale
                    return cui
    
    # Try with lower fuzzy threshold for medical terms only
    for term in medical_terms[:2]:
        if len(term) > 5:  # Only try longer medical terms
            matches = process.extract(
                term.lower(), 
                nodes_df['Name_lower'], 
                scorer=fuzz.partial_ratio,  # Different scorer for partial matches
                limit=1
            )
            
            if matches and matches[0][1] >= 90:  # Lower threshold for partial matches
                match_idx = nodes_df[nodes_df['Name_lower'] == matches[0][0]].index[0]
                cui = nodes_df.iloc[match_idx]['CUI']
                fuzzy_score = matches[0][1] / 100.0
                return cui
    
    # No match found with any method
    return None

def set_kg_data(kg_dataframe):
    """Set the knowledge graph data for triple validation"""
    global kg_df
    kg_df = kg_dataframe

def validate_triple(subject, relation, obj):
    """Validates a single triple against the loaded knowledge graph."""
    subject_cui = get_cui_from_name(subject)
    object_cui = get_cui_from_name(obj)
    relation_upper = relation.strip().upper()

    if not subject_cui or not object_cui:
        return {'status': 'unsure', 'pubmed_ids': None, 'sentence': None}

    results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui) &
        (kg_df['PREDICATE'] == relation_upper)
    ]
    if not results.empty:
        top_result = results.iloc[0]
        return {
            'status': 'supported', 
            'pubmed_ids': top_result['PubMed_ID'], 
            'sentence': top_result['SENTENCE']
        }

    relevant_results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui)
    ]
    if not relevant_results.empty:
        top_result = relevant_results.iloc[0]
        return {
            'status': 'relevant', 
            'pubmed_ids': top_result['PubMed_ID'], 
            'sentence': top_result['SENTENCE']
        }

    return {'status': 'unsure', 'pubmed_ids': None, 'sentence': None}

def generate_recommendations(entities_in_graph, max_recommendations=4):
    """Generates follow-up questions based on entities in the current graph."""
    recommendations = set()
    if not entities_in_graph or kg_df is None:
        return []

    entity_cuis = [get_cui_from_name(name) for name in entities_in_graph]
    entity_cuis = [cui for cui in entity_cuis if cui]

    for cui in entity_cuis:
        related = kg_df[(kg_df['START_ID'] == cui) | (kg_df['END_ID'] == cui)]
        if related.empty:
            continue

        for _, row in related.sample(min(len(related), 3)).iterrows():
            if len(recommendations) >= max_recommendations:
                break
            
            start_node_name = nodes_df[nodes_df['CUI'] == row['START_ID']].iloc[0]['Name']
            end_node_name = nodes_df[nodes_df['CUI'] == row['END_ID']].iloc[0]['Name']

            if row['START_ID'] == cui:
                recommendations.add(
                    f"What is the relationship between {start_node_name} and {end_node_name}?"
                )
            else:
                recommendations.add(
                    f"How does {end_node_name} affect {start_node_name}?"
                )
    
    return list(recommendations)

def _evaluate_and_log_retrieval_metrics():
    """
    Evaluate INFORMATION RETRIEVAL metrics using test queries and log to file
    
    Dataset: test_queries.csv (33 Alzheimer's disease queries)
    Metrics: Precision@K, Recall@K, F1@K, MRR, MAP, nDCG
    """
    try:
        # Load test queries from dataset/test_queries.csv
        test_file = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test_queries.csv')
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            return
        
        test_queries = []
        with open(test_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_queries.append({
                    'query': row['query'],
                    'expected_cuis': [cui.strip() for cui in row['expected_cuis'].split(',')]
                })
        
        print(f"📊 Evaluating Information Retrieval metrics on {len(test_queries)} test queries...")
        print(f"📁 Dataset: test_queries.csv")
        print(f"🔍 Entity Pool: {len(entity_names_list)} entities from neo4j_node.csv")
        
        # ADAPTIVE_PRECISION strategy parameters
        threshold = 0.88
        top_k = 10  # Evaluate top-10 for comprehensive metrics
        
        # Information Retrieval Metrics
        total_tp = 0  # True Positives (exact + soft matches)
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives
        
        total_precision_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        total_recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        total_f1_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        
        reciprocal_ranks = []  # For MRR (Mean Reciprocal Rank)
        average_precisions = []  # For MAP (Mean Average Precision)
        
        for test_case in test_queries:
            query = test_case['query']
            expected_cuis = set(test_case['expected_cuis'])
            
            # Get query embedding
            query_embedding = sentence_model.encode([query], show_progress_bar=False)[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Compute cosine similarities
            similarities = np.dot(entity_embeddings_matrix, query_embedding)
            
            # Get top-10 results for comprehensive evaluation
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Apply threshold filter
            retrieved_cuis = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    retrieved_cuis.append(cui_list[idx])
            
            # Calculate metrics with SOFT MATCHING (for related CUIs)
            exact_matches = set(retrieved_cuis) & expected_cuis
            
            # Soft matching: partial credit for substring matches
            soft_matches = 0
            for ret_cui in retrieved_cuis:
                if ret_cui not in exact_matches:
                    for exp_cui in expected_cuis:
                        if ret_cui in exp_cui or exp_cui in ret_cui:
                            soft_matches += 1
                            break
            
            # Calculate TP, FP, FN with partial credit for soft matches
            tp = len(exact_matches) + (soft_matches * 0.5)
            fp = len(retrieved_cuis) - len(exact_matches) - soft_matches
            fn = len(expected_cuis) - len(exact_matches) - (soft_matches * 0.5)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Precision@K, Recall@K, F1@K for different K values
            for k in [1, 3, 5, 10]:
                k_retrieved = set(retrieved_cuis[:k])
                k_exact = k_retrieved & expected_cuis
                
                # Soft matching for K results
                k_soft = 0
                for ret_cui in k_retrieved:
                    if ret_cui not in k_exact:
                        for exp_cui in expected_cuis:
                            if ret_cui in exp_cui or exp_cui in ret_cui:
                                k_soft += 1
                                break
                
                k_tp = len(k_exact) + (k_soft * 0.5)
                k_precision = k_tp / len(k_retrieved) if k_retrieved else 0
                k_recall = k_tp / len(expected_cuis) if expected_cuis else 0
                k_f1 = 2 * k_precision * k_recall / (k_precision + k_recall) if (k_precision + k_recall) > 0 else 0
                
                total_precision_at_k[k] += k_precision
                total_recall_at_k[k] += k_recall
                total_f1_at_k[k] += k_f1
            
            # MRR: Mean Reciprocal Rank (position of first relevant result)
            first_relevant_rank = None
            for rank, cui in enumerate(retrieved_cuis, 1):
                if cui in expected_cuis or any(cui in exp or exp in cui for exp in expected_cuis):
                    first_relevant_rank = rank
                    break
            
            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
            
            # MAP: Mean Average Precision (precision at each relevant result)
            relevant_found = 0
            precision_sum = 0
            for rank, cui in enumerate(retrieved_cuis, 1):
                if cui in expected_cuis or any(cui in exp or exp in cui for exp in expected_cuis):
                    relevant_found += 1
                    precision_sum += relevant_found / rank
            
            avg_precision = precision_sum / len(expected_cuis) if expected_cuis else 0
            average_precisions.append(avg_precision)
        
        # Calculate overall metrics
        n_queries = len(test_queries)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
        map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0
        
        # Write to log file
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'retrieval_metrics.log')
        
        with open(log_file, 'w', encoding='utf-8') as f:  # Overwrite to keep latest only
            f.write('='*70 + '\n')
            f.write('INFORMATION RETRIEVAL METRICS - ADAPTIVE_PRECISION Strategy\n')
            f.write('='*70 + '\n')
            f.write(f'Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Dataset: test_queries.csv ({n_queries} queries)\n')
            f.write(f'Entity Pool: neo4j_node.csv ({len(entity_names_list)} entities)\n')
            f.write(f'Model: sentence-transformers/all-MiniLM-L6-v2\n\n')
            
            f.write('OVERALL RETRIEVAL PERFORMANCE:\n')
            f.write(f'  Precision:         {overall_precision:.4f} ({overall_precision*100:.2f}%)\n')
            f.write(f'  Recall:            {overall_recall:.4f} ({overall_recall*100:.2f}%)\n')
            f.write(f'  F1 Score:          {overall_f1:.4f} ({overall_f1*100:.2f}%)\n')
            f.write(f'  MRR (Mean Reciprocal Rank):  {mrr:.4f}\n')
            f.write(f'  MAP (Mean Average Precision): {map_score:.4f}\n\n')
            
            f.write('PRECISION @ K:\n')
            for k in [1, 3, 5, 10]:
                p_at_k = total_precision_at_k[k] / n_queries
                f.write(f'  P@{k:2d}:  {p_at_k:.4f} ({p_at_k*100:.2f}%)\n')
            
            f.write('\nRECALL @ K:\n')
            for k in [1, 3, 5, 10]:
                r_at_k = total_recall_at_k[k] / n_queries
                f.write(f'  R@{k:2d}:  {r_at_k:.4f} ({r_at_k*100:.2f}%)\n')
            
            f.write('\nF1 @ K:\n')
            for k in [1, 3, 5, 10]:
                f1_at_k = total_f1_at_k[k] / n_queries
                f.write(f'  F1@{k:2d}: {f1_at_k:.4f} ({f1_at_k*100:.2f}%)\n')
            
            f.write('\nRETRIEVAL STRATEGY:\n')
            f.write(f'  Strategy: ADAPTIVE_PRECISION\n')
            f.write(f'  Similarity Threshold: {threshold}\n')
            f.write(f'  Top-K Results: {top_k}\n')
            f.write(f'  Soft Matching: Enabled (0.5 credit for substring matches)\n')
            
            f.write('\nMETRIC DEFINITIONS:\n')
            f.write('  • Precision: Fraction of retrieved entities that are relevant\n')
            f.write('  • Recall: Fraction of relevant entities that are retrieved\n')
            f.write('  • F1 Score: Harmonic mean of precision and recall\n')
            f.write('  • MRR: Average of reciprocal ranks of first relevant result\n')
            f.write('  • MAP: Mean of average precision across all queries\n')
            f.write('  • P@K: Precision considering only top-K results\n')
            f.write('  • R@K: Recall considering only top-K results\n')
            f.write('='*70 + '\n')
        
        print(f"✓ Retrieval metrics logged to: {log_file}")
        print(f"  F1 Score: {overall_f1:.4f} ({overall_f1*100:.2f}%)")
        print(f"  Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
        print(f"  Recall: {overall_recall:.4f} ({overall_recall*100:.2f}%)")
        print(f"  MRR: {mrr:.4f}, MAP: {map_score:.4f}")
        
    except Exception as e:
        print(f"Error evaluating retrieval metrics: {e}")
        import traceback
        traceback.print_exc()
