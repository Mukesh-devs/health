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
import pickle
import faiss

faiss_index = None
metadata = None

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
    # global entity_embeddings_matrix, entity_names_list, cui_list, nodes_df
    global faiss_index, metadata
    global entity_names_list, cui_list, nodes_df

    nodes_df = nodes_dataframe
    if (os.path.exists(current_app.config['FAISS_INDEX_PATH']) and os.path.exists(current_app.config['FAISS_METADATA_PATH'])):
        print("Loading FAISS index...")
        start_time = time.time()

        faiss_index = faiss.read_index(
            current_app.config['FAISS_INDEX_PATH']
        )

        with open(
            current_app.config['FAISS_METADATA_PATH'],
            'rb'
        ) as f:
            metadata = pickle.load(f)

        entity_names_list = metadata["names"]
        cui_list = metadata["cuis"]

        load_time = time.time() - start_time

        print(
            f"Loaded FAISS index for "
            f"{len(entity_names_list)} entities "
            f"in {load_time:.2f}s"
        )

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
        

    dimension = entity_embeddings_matrix.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(entity_embeddings_matrix)

    faiss.write_index(
        faiss_index,
        current_app.config['FAISS_INDEX_PATH']
    )

    metadata = {
        "names": entity_names_list,
        "cuis": cui_list
    }

    with open(
        current_app.config['FAISS_METADATA_PATH'],
        "wb"
    ) as f:
        pickle.dump(metadata, f)

    # Cache embeddings
    # print("Caching embeddings for future use...")
    # np.savez_compressed(
    #     cache_path,
    #     embeddings=entity_embeddings_matrix,
    #     names=np.array(entity_names_list),
    #     cuis=np.array(cui_list)
    # )
    
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
    # ANSI colors
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    if not entity_name or faiss_index is None:
        return None
    
    start_time = time.time()
    
    print(f"\n{CYAN}[ENTITY SEARCH] Searching for: '{entity_name}'{RESET}")
    
    # Preprocess the query
    processed_query, medical_terms = preprocess_query(entity_name)
    print(f"{BLUE}  Preprocessed query: '{processed_query}'{RESET}")
    if medical_terms:
        print(f"{BLUE}  Medical terms extracted: {medical_terms[:3]}{RESET}")
    
    # Try exact match with processed query first
    entity_name_lower = processed_query.lower().strip()
    exact_match = nodes_df[nodes_df['Name_lower'] == entity_name_lower]
    if not exact_match.empty:
        cui = exact_match.iloc[0]['CUI']
        search_time = time.time() - start_time
        print(f"{GREEN}  ✓ EXACT MATCH found!{RESET}")
        print(f"{GREEN}    CUI: {cui}{RESET}")
        print(f"{GREEN}    Confidence: 1.0000 (100.00%){RESET}")
        print(f"{GREEN}    Search time: {search_time:.4f}s{RESET}")
        return cui
    
    # Try exact match on medical terms
    for term in medical_terms:
        exact_match = nodes_df[nodes_df['Name_lower'] == term.lower()]
        if not exact_match.empty:
            cui = exact_match.iloc[0]['CUI']
            search_time = time.time() - start_time
            print(f"{GREEN}  ✓ EXACT MATCH on medical term: '{term}'{RESET}")
            print(f"{GREEN}    CUI: {cui}{RESET}")
            print(f"{GREEN}    Confidence: 1.0000 (100.00%){RESET}")
            print(f"{GREEN}    Search time: {search_time:.4f}s{RESET}")
            return cui
    
    # Initialize model if needed
    model = initialize_model()
    
    print(f"{YELLOW}  Performing semantic search via embeddings...{RESET}")
    print(f"{BLUE}    Embedding cache: {len(entity_names_list):,} entities{RESET}")
    print(f"{BLUE}    Embedding dimension: {faiss_index.d}{RESET}")
    print(f"{BLUE}    Search threshold: {threshold:.2f}{RESET}")
    
    # Multi-query semantic search
    queries_to_try = [processed_query] + medical_terms[:3]  # Try top 3 medical terms
    best_similarity = 0
    best_cui = None
    best_entity_name = None
    best_query_variant = None
    all_top_results = []
    
    for query_idx, query in enumerate(queries_to_try):
        if not query.strip():
            continue
        
        print(f"{BLUE}    Trying query variant {query_idx + 1}: '{query}'{RESET}")
            
        # Semantic search
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        scores, ids = faiss_index.search(
            query_embedding,
            top_k
        )
        # top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        current_best = scores[0][0]

        current_best_cui = cui_list[ids[0][0]]

        current_best_name = entity_names_list[ids[0][0]]        
        # Show top results for this query variant
        if query_idx == 0:  # Only show detailed results for main query
            print(f"{YELLOW}      Top {min(5, top_k)} candidates:{RESET}")
            for i, idx in enumerate(ids[0][:5], 1):
                candidate_name = entity_names_list[idx]
                candidate_cui = cui_list[idx]
                candidate_sim = scores[0][i - 1]
                status = "✓" if candidate_sim >= threshold else "✗"
                print(f"{BLUE}        {i}. [{status}] {candidate_name} (CUI: {candidate_cui}){RESET}")
                print(f"{BLUE}           Similarity: {candidate_sim:.4f} ({candidate_sim*100:.2f}%){RESET}")
        
        if current_best > best_similarity:
            best_similarity = current_best
            best_cui = current_best_cui
            best_entity_name = current_best_name
            best_query_variant = query
    
    search_time = time.time() - start_time
    
    print(f"\n{BLUE}  Semantic search completed in {search_time:.4f}s{RESET}")
    
    if best_similarity >= threshold:
        print(f"{GREEN}  ✓ SEMANTIC MATCH found!{RESET}")
        print(f"{GREEN}    Best match: {best_entity_name}{RESET}")
        print(f"{GREEN}    CUI: {best_cui}{RESET}")
        print(f"{GREEN}    Confidence: {best_similarity:.4f} ({best_similarity*100:.2f}%){RESET}")
        print(f"{GREEN}    Query variant used: '{best_query_variant}'{RESET}")
        print(f"{GREEN}    Total search time: {search_time:.4f}s{RESET}")
        return best_cui
    
    # No match found
    print(f"{YELLOW}  ⚠ No match above threshold{RESET}")
    print(f"{YELLOW}    Best similarity: {best_similarity:.4f} ({best_similarity*100:.2f}%){RESET}")
    print(f"{YELLOW}    Threshold required: {threshold:.4f} ({threshold*100:.2f}%){RESET}")
    if best_entity_name:
        print(f"{YELLOW}    Closest entity: {best_entity_name} (CUI: {best_cui}){RESET}")
    return None

def get_cui_from_name(entity_name, semantic_threshold=0.75, fuzzy_threshold=95):
    """
    Enhanced hybrid approach with improved thresholds and multi-stage matching
    """
    # ANSI colors
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    if not entity_name:
        return None
    
    print(f"\n{CYAN}[HYBRID ENTITY RETRIEVAL] Starting for: '{entity_name}'{RESET}")
    print(f"{BLUE}  Strategy: Semantic (threshold={semantic_threshold}) + Fuzzy (threshold={fuzzy_threshold}){RESET}")
    
    # Try semantic search first with higher threshold
    print(f"{YELLOW}  Stage 1/2: Semantic search via embedding cache...{RESET}")
    semantic_cui = get_cui_from_name_semantic(entity_name, threshold=semantic_threshold)
    if semantic_cui:
        return semantic_cui
    
    print(f"{YELLOW}  Stage 2/2: Fallback to fuzzy string matching...{RESET}")
    
    # Enhanced fuzzy matching with preprocessing
    processed_query, medical_terms = preprocess_query(entity_name)
    
    # Try fuzzy matching on processed query
    print(f"{BLUE}    Search space: {len(nodes_df):,} entities{RESET}")
    
    for query_idx, query_variant in enumerate([processed_query] + medical_terms[:2], 1):
        if not query_variant.strip():
            continue
        
        print(f"{BLUE}    Fuzzy match attempt {query_idx}: '{query_variant}'{RESET}")
            
        query_lower = query_variant.lower().strip()
        matches = process.extract(
            query_lower, 
            nodes_df['Name_lower'], 
            scorer=fuzz.token_sort_ratio, 
            limit=3  # Get top 3 for better selection
        )
        
        if matches:
            print(f"{YELLOW}      Top 3 fuzzy matches:{RESET}")
            for i, match in enumerate(matches, 1):
                match_idx = nodes_df[nodes_df['Name_lower'] == match[0]].index[0]
                match_cui = nodes_df.iloc[match_idx]['CUI']
                match_name = nodes_df.iloc[match_idx]['Name']
                fuzzy_score = match[1]
                status = "✓" if fuzzy_score >= fuzzy_threshold else "✗"
                print(f"{BLUE}        {i}. [{status}] {match_name} (CUI: {match_cui}){RESET}")
                print(f"{BLUE}           Fuzzy score: {fuzzy_score:.1f}/100 ({fuzzy_score:.1f}%){RESET}")
            
            # Use the best match that meets threshold
            for match in matches:
                if match[1] >= fuzzy_threshold:
                    match_idx = nodes_df[nodes_df['Name_lower'] == match[0]].index[0]
                    cui = nodes_df.iloc[match_idx]['CUI']
                    match_name = nodes_df.iloc[match_idx]['Name']
                    # Record fuzzy match for F1 tracking
                    fuzzy_score = match[1] / 100.0  # Convert to 0-1 scale
                    print(f"{GREEN}  ✓ FUZZY MATCH found!{RESET}")
                    print(f"{GREEN}    Match: {match_name}{RESET}")
                    print(f"{GREEN}    CUI: {cui}{RESET}")
                    print(f"{GREEN}    Confidence: {fuzzy_score:.4f} ({fuzzy_score*100:.2f}%){RESET}")
                    print(f"{GREEN}    Method: Fuzzy string matching (token_sort_ratio){RESET}")
                    return cui
    
    # Try with lower fuzzy threshold for medical terms only
    print(f"{YELLOW}    Trying partial ratio matching on medical terms...{RESET}")
    for term_idx, term in enumerate(medical_terms[:2], 1):
        if len(term) > 5:  # Only try longer medical terms
            print(f"{BLUE}      Term {term_idx}: '{term}'{RESET}")
            matches = process.extract(
                term.lower(), 
                nodes_df['Name_lower'], 
                scorer=fuzz.partial_ratio,  # Different scorer for partial matches
                limit=1
            )
            
            if matches and matches[0][1] >= 90:  # Lower threshold for partial matches
                match_idx = nodes_df[nodes_df['Name_lower'] == matches[0][0]].index[0]
                cui = nodes_df.iloc[match_idx]['CUI']
                match_name = nodes_df.iloc[match_idx]['Name']
                fuzzy_score = matches[0][1] / 100.0
                print(f"{GREEN}  ✓ PARTIAL FUZZY MATCH found!{RESET}")
                print(f"{GREEN}    Match: {match_name}{RESET}")
                print(f"{GREEN}    CUI: {cui}{RESET}")
                print(f"{GREEN}    Confidence: {fuzzy_score:.4f} ({fuzzy_score*100:.2f}%){RESET}")
                print(f"{GREEN}    Method: Fuzzy partial matching{RESET}")
                return cui
    
    # No match found with any method
    print(f"{YELLOW}  ✗ NO MATCH FOUND with any method{RESET}")
    print(f"{YELLOW}    Entity '{entity_name}' not found in knowledge graph{RESET}")
    return None

def set_kg_data(kg_dataframe):
    """Set the knowledge graph data for triple validation"""
    global kg_df
    kg_df = kg_dataframe

def validate_triple(subject, relation, obj):
    """
    Enhanced validation with contradiction detection and live PubMed integration.
    ALWAYS fetches BOTH static KG evidence AND live PubMed evidence to show combined results.
    
    Returns:
        - 'supported': Triple found in KG (may also have live evidence)
        - 'relevant': Related triple found (may also have live evidence)
        - 'contradictory_evidence': Conflicting relationships detected
        - 'emerging_evidence': New evidence from PubMed not in static KG
        - 'unsure': No evidence found anywhere
    """
    from healthapp.contradiction_detector import contradiction_detector
    from healthapp.pubmed_live import pubmed_live
    import logging
    
    # ANSI color codes
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    
    logger = logging.getLogger(__name__)
    
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}[TRIPLE VALIDATION MODULE] Starting validation...{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{MAGENTA}Triple to validate:{RESET}")
    print(f"{MAGENTA}  Subject:  {subject}{RESET}")
    print(f"{MAGENTA}  Relation: {relation}{RESET}")
    print(f"{MAGENTA}  Object:   {obj}{RESET}")
    
    print(f"\n{YELLOW}{'─'*80}{RESET}")
    print(f"{YELLOW}[Stage 1/5] Entity Matching in Knowledge Graph{RESET}")
    print(f"{YELLOW}{'─'*80}{RESET}")
    
    print(f"\n{CYAN}Retrieving Subject CUI for: '{subject}'{RESET}")
    subject_start = time.time()
    subject_cui = get_cui_from_name(subject)
    subject_time = time.time() - subject_start
    relation_upper = relation.strip().upper()
    
    if subject_cui:
        # Get entity details
        subject_entity = nodes_df[nodes_df['CUI'] == subject_cui].iloc[0]
        print(f"\n{GREEN}✓ Subject entity resolved successfully!{RESET}")
        print(f"{GREEN}  CUI: {subject_cui}{RESET}")
        print(f"{GREEN}  Name: {subject_entity['Name']}{RESET}")
        print(f"{GREEN}  Type: {subject_entity['Semantic_Type']}{RESET}")
        print(f"{GREEN}  Category: {subject_entity['Category']}{RESET}")
        print(f"{GREEN}  Retrieval time: {subject_time:.4f}s{RESET}")
    else:
        print(f"\n{YELLOW}⚠ Subject CUI not found in KG{RESET}")
        print(f"{YELLOW}  Search time: {subject_time:.4f}s{RESET}")
    
    print(f"\n{CYAN}Retrieving Object CUI for: '{obj}'{RESET}")
    object_start = time.time()
    object_cui = get_cui_from_name(obj)
    object_time = time.time() - object_start
    
    if object_cui:
        # Get entity details
        object_entity = nodes_df[nodes_df['CUI'] == object_cui].iloc[0]
        print(f"\n{GREEN}✓ Object entity resolved successfully!{RESET}")
        print(f"{GREEN}  CUI: {object_cui}{RESET}")
        print(f"{GREEN}  Name: {object_entity['Name']}{RESET}")
        print(f"{GREEN}  Type: {object_entity['Semantic_Type']}{RESET}")
        print(f"{GREEN}  Category: {object_entity['Category']}{RESET}")
        print(f"{GREEN}  Retrieval time: {object_time:.4f}s{RESET}")
    else:
        print(f"\n{YELLOW}⚠ Object CUI not found in KG{RESET}")
        print(f"{YELLOW}  Search time: {object_time:.4f}s{RESET}")

    # ALWAYS fetch live PubMed evidence regardless of KG findings
    print(f"\n{YELLOW}{'─'*80}{RESET}")
    print(f"{YELLOW}[Stage 2/5] Fetching Live PubMed Evidence{RESET}")
    print(f"{YELLOW}{'─'*80}{RESET}")
    
    live_evidence = {'found_new_evidence': False}
    try:
        print(f"{CYAN}Querying PubMed API for: '{subject}' AND '{obj}'{RESET}")
        pubmed_start = time.time()
        live_evidence = pubmed_live.fetch_latest_evidence(subject, obj, max_results=5)
        pubmed_time = time.time() - pubmed_start
        
        if live_evidence.get('found_new_evidence', False):
            print(f"{GREEN}✓ Found {live_evidence.get('total_papers', 0)} recent PubMed papers{RESET}")
            print(f"{GREEN}  Search time: {pubmed_time:.4f}s{RESET}")
            if 'papers' in live_evidence and live_evidence['papers']:
                print(f"{BLUE}  Sample papers:{RESET}")
                for i, paper in enumerate(live_evidence['papers'][:3], 1):
                    print(f"{BLUE}    {i}. PMID: {paper.get('pmid', 'N/A')}{RESET}")
                    print(f"{BLUE}       Title: {paper.get('title', 'N/A')[:80]}...{RESET}")
        else:
            print(f"{YELLOW}⚠ No recent PubMed evidence found{RESET}")
            print(f"{YELLOW}  Search time: {pubmed_time:.4f}s{RESET}")
        logger.info(f"Live PubMed evidence fetched for {subject} + {obj}: {live_evidence.get('total_papers', 0)} papers")
    except Exception as e:
        print(f"{YELLOW}⚠ PubMed fetch failed: {str(e)}{RESET}")
        logger.warning(f"Live PubMed fetch failed: {e}")

    if not subject_cui or not object_cui:
        print(f"{YELLOW}⚠ Cannot validate - CUI mapping incomplete{RESET}")
        # Even without CUI mapping, return live evidence if found
        if live_evidence.get('found_new_evidence', False):
            print(f"{GREEN}✓ Returning emerging evidence from PubMed{RESET}")
            return {
                'status': 'emerging_evidence',
                'pubmed_ids': None,
                'sentence': None,
                'static_kg': {'found_in_kg': False, 'pubmed_ids': None, 'sentence': None},
                'live_pubmed': live_evidence,
                'evidence_quality': 'new',
                'interpretation': f"⚡ New evidence from recent research"
            }
        print(f"{YELLOW}✓ No evidence found (unsure){RESET}")
        return {'status': 'unsure', 'pubmed_ids': None, 'sentence': None}

    # Check for contradictions FIRST
    print(f"\n{YELLOW}{'─'*80}{RESET}")
    print(f"{YELLOW}[Stage 3/5] Checking for Contradictions{RESET}")
    print(f"{YELLOW}{'─'*80}{RESET}")
    
    if contradiction_detector is not None:
        try:
            print(f"{CYAN}Analyzing relationship for conflicting evidence...{RESET}")
            contradiction_start = time.time()
            contradiction_analysis = contradiction_detector.detect_contradictions(
                subject_cui, object_cui
            )
            contradiction_time = time.time() - contradiction_start
            
            if contradiction_analysis.get('has_contradictions', False):
                print(f"{MAGENTA}⚠ Contradictions detected!{RESET}")
                print(f"{MAGENTA}  Analysis time: {contradiction_time:.4f}s{RESET}")
                print(f"{MAGENTA}  Conflicting relationships found in KG{RESET}")
                # Include live evidence even for contradictions
                contradiction_analysis['live_pubmed'] = live_evidence
                return {
                    'status': 'contradictory_evidence',
                    'subject': subject,
                    'object': obj,
                    'relation': relation,
                    **contradiction_analysis
                }
            else:
                print(f"{GREEN}✓ No contradictions found{RESET}")
                print(f"{GREEN}  Analysis time: {contradiction_time:.4f}s{RESET}")
        except Exception as e:
            print(f"{YELLOW}⚠ Contradiction check failed: {str(e)}{RESET}")
            logger.warning(f"Contradiction detection failed: {e}")
    else:
        print(f"{YELLOW}⚠ Contradiction detector not available{RESET}")

    # Check static knowledge graph - exact match
    print(f"\n{YELLOW}{'─'*80}{RESET}")
    print(f"{YELLOW}[Stage 4/5] Searching Static Knowledge Graph{RESET}")
    print(f"{YELLOW}{'─'*80}{RESET}")
    
    print(f"{CYAN}Searching for exact triple match...{RESET}")
    print(f"{BLUE}  START_ID: {subject_cui} ({subject}){RESET}")
    print(f"{BLUE}  PREDICATE: {relation_upper}{RESET}")
    print(f"{BLUE}  END_ID: {object_cui} ({obj}){RESET}")
    
    kg_start = time.time()
    results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui) &
        (kg_df['PREDICATE'] == relation_upper)
    ]
    kg_time = time.time() - kg_start
    
    static_evidence = {
        'found_in_kg': not results.empty,
        'pubmed_ids': results.iloc[0]['PubMed_ID'] if not results.empty else None,
        'sentence': results.iloc[0]['SENTENCE'] if not results.empty else None
    }
    
    if static_evidence['found_in_kg']:
        print(f"{GREEN}✓ EXACT MATCH found in static KG!{RESET}")
        print(f"{GREEN}  Matches found: {len(results)}{RESET}")
        print(f"{GREEN}  Search time: {kg_time:.4f}s{RESET}")
        if static_evidence['pubmed_ids']:
            print(f"{GREEN}  PubMed IDs: {static_evidence['pubmed_ids']}{RESET}")
        if static_evidence['sentence']:
            print(f"{GREEN}  Evidence: {static_evidence['sentence'][:150]}...{RESET}")
    else:
        print(f"{YELLOW}⚠ No exact match in static KG{RESET}")
        print(f"{YELLOW}  Search time: {kg_time:.4f}s{RESET}")
    
    # Check for relevant relationships (any relation between entities)
    print(f"\n{YELLOW}{'─'*80}{RESET}")
    print(f"{YELLOW}[Stage 5/5] Searching for Related Triples{RESET}")
    print(f"{YELLOW}{'─'*80}{RESET}")
    
    print(f"{CYAN}Searching for any relationship between entities...{RESET}")
    related_start = time.time()
    relevant_results = kg_df[
        (kg_df['START_ID'] == subject_cui) &
        (kg_df['END_ID'] == object_cui)
    ]
    related_time = time.time() - related_start
    
    relevant_evidence = {
        'found_relevant': not relevant_results.empty,
        'pubmed_ids': relevant_results.iloc[0]['PubMed_ID'] if not relevant_results.empty else None,
        'sentence': relevant_results.iloc[0]['SENTENCE'] if not relevant_results.empty else None,
        'actual_relation': relevant_results.iloc[0]['PREDICATE'] if not relevant_results.empty else None
    }
    
    if relevant_evidence['found_relevant']:
        print(f"{GREEN}✓ Related triples found!{RESET}")
        print(f"{GREEN}  Total relations: {len(relevant_results)}{RESET}")
        print(f"{GREEN}  Search time: {related_time:.4f}s{RESET}")
        
        # Show all unique relations
        unique_relations = relevant_results['PREDICATE'].unique()
        print(f"{BLUE}  Unique relations found:{RESET}")
        for rel in unique_relations:
            count = len(relevant_results[relevant_results['PREDICATE'] == rel])
            print(f"{BLUE}    - {rel}: {count} triple(s){RESET}")
    else:
        print(f"{YELLOW}⚠ No related triples found{RESET}")
        print(f"{YELLOW}  Search time: {related_time:.4f}s{RESET}")
    
    # COMBINED EVIDENCE LOGIC - Always include both static + live
    
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}[VALIDATION RESULT SUMMARY]{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    
    # Case 1: Found in static KG (exact match)
    if static_evidence['found_in_kg']:
        print(f"{GREEN}✓ Status: SUPPORTED{RESET}")
        print(f"{GREEN}  Evidence source: Static Knowledge Graph (exact match){RESET}")
        print(f"{GREEN}  Confidence: HIGH{RESET}")
        if live_evidence.get('found_new_evidence'):
            print(f"{GREEN}  Additional evidence: {live_evidence.get('total_papers', 0)} recent PubMed papers{RESET}")
            print(f"{GREEN}  Evidence quality: HIGH (KG + Recent literature){RESET}")
        else:
            print(f"{GREEN}  Evidence quality: MEDIUM (KG only){RESET}")
        
        return {
            'status': 'supported',
            'pubmed_ids': static_evidence['pubmed_ids'],
            'sentence': static_evidence['sentence'],
            'static_kg': static_evidence,
            'live_pubmed': live_evidence,  # Always include live evidence
            'evidence_quality': 'high' if live_evidence.get('found_new_evidence') else 'medium',
            'interpretation': f"✓ Found in KG dataset" + 
                            (f" + {live_evidence.get('total_papers', 0)} recent papers" 
                             if live_evidence.get('found_new_evidence') else "")
        }
    
    # Case 2: Related evidence in static KG (different relation)
    elif relevant_evidence['found_relevant']:
        print(f"{YELLOW}✓ Status: RELEVANT{RESET}")
        print(f"{YELLOW}  Evidence source: Static Knowledge Graph (different relation){RESET}")
        print(f"{YELLOW}  Found relation: {relevant_evidence['actual_relation']}{RESET}")
        print(f"{YELLOW}  Requested relation: {relation_upper}{RESET}")
        print(f"{YELLOW}  Confidence: MEDIUM{RESET}")
        if live_evidence.get('found_new_evidence'):
            print(f"{GREEN}  Additional evidence: {live_evidence.get('total_papers', 0)} recent PubMed papers{RESET}")
        print(f"{YELLOW}  Evidence quality: MEDIUM{RESET}")
        
        return {
            'status': 'relevant',
            'pubmed_ids': relevant_evidence['pubmed_ids'],
            'sentence': relevant_evidence['sentence'],
            'actual_relation': relevant_evidence['actual_relation'],
            'static_kg': static_evidence,
            'relevant_kg': relevant_evidence,
            'live_pubmed': live_evidence,  # Always include live evidence
            'evidence_quality': 'medium',
            'interpretation': f"Different relation in KG: {relevant_evidence['actual_relation']}" +
                            (f" | {live_evidence.get('total_papers', 0)} recent papers found"
                             if live_evidence.get('found_new_evidence') else "")
        }
    
    # Case 3: Only live PubMed evidence (not in static KG)
    elif live_evidence.get('found_new_evidence', False):
        sentiment = live_evidence.get('sentiment_summary', {})
        print(f"{MAGENTA}✓ Status: EMERGING EVIDENCE{RESET}")
        print(f"{MAGENTA}  Evidence source: Live PubMed search only{RESET}")
        print(f"{MAGENTA}  Papers found: {live_evidence.get('total_papers', 0)}{RESET}")
        print(f"{MAGENTA}  Overall sentiment: {sentiment.get('overall_sentiment', 'mixed').upper()}{RESET}")
        print(f"{MAGENTA}  Confidence: LOW-MEDIUM (new/emerging research){RESET}")
        print(f"{MAGENTA}  Evidence quality: NEW{RESET}")
        
        return {
            'status': 'emerging_evidence',
            'pubmed_ids': None,
            'sentence': None,
            'static_kg': static_evidence,
            'live_pubmed': live_evidence,
            'evidence_quality': 'new',
            'interpretation': f"⚡ New evidence: {sentiment.get('overall_sentiment', 'mixed')} ({live_evidence.get('total_papers', 0)} papers)"
        }
    
    # Case 4: No evidence found anywhere
    print(f"{YELLOW}✓ Status: UNSURE{RESET}")
    print(f"{YELLOW}  Evidence source: None found{RESET}")
    print(f"{YELLOW}  Confidence: NONE{RESET}")
    print(f"{YELLOW}  Evidence quality: NONE{RESET}")
    print(f"{YELLOW}  Recommendation: Consult medical literature or expert{RESET}")
    
    return {
        'status': 'unsure',
        'pubmed_ids': None,
        'sentence': None,
        'static_kg': static_evidence,
        'live_pubmed': live_evidence,
        'evidence_quality': 'none',
        'interpretation': '❓ No evidence found in KG or recent literature'
    }

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

# def _evaluate_and_log_retrieval_metrics():
#     """
#     Evaluate INFORMATION RETRIEVAL metrics using test queries and log to file
    
#     Dataset: test_queries.csv (33 Alzheimer's disease queries)
#     Metrics: Precision@K, Recall@K, F1@K, MRR, MAP, nDCG
#     """
#     try:
#         # Load test queries from dataset/test_queries.csv
#         test_file = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'test_queries.csv')
#         if not os.path.exists(test_file):
#             print(f"Test file not found: {test_file}")
#             return
        
#         test_queries = []
#         with open(test_file, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 test_queries.append({
#                     'query': row['query'],
#                     'expected_cuis': [cui.strip() for cui in row['expected_cuis'].split(',')]
#                 })
        
#         print(f"📊 Evaluating Information Retrieval metrics on {len(test_queries)} test queries...")
#         print(f"📁 Dataset: test_queries.csv")
#         print(f"🔍 Entity Pool: {len(entity_names_list)} entities from neo4j_node.csv")
        
#         # ADAPTIVE_PRECISION strategy parameters
#         threshold = 0.88
#         top_k = 10  # Evaluate top-10 for comprehensive metrics
        
#         # Information Retrieval Metrics
#         total_tp = 0  # True Positives (exact + soft matches)
#         total_fp = 0  # False Positives
#         total_fn = 0  # False Negatives
        
#         total_precision_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
#         total_recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
#         total_f1_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        
#         reciprocal_ranks = []  # For MRR (Mean Reciprocal Rank)
#         average_precisions = []  # For MAP (Mean Average Precision)
        
#         # Initialize detailed calculation log
#         log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
#         os.makedirs(log_dir, exist_ok=True)
#         detailed_log_file = os.path.join(log_dir, 'retrieval_metrics_detailed_calculations.log')
#         detailed_log = open(detailed_log_file, 'w', encoding='utf-8')
        
#         # Write header to detailed log
#         detailed_log.write('='*100 + '\n')
#         detailed_log.write('DETAILED INFORMATION RETRIEVAL METRICS CALCULATION LOG\n')
#         detailed_log.write('='*100 + '\n')
#         detailed_log.write(f'Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
#         detailed_log.write(f'Dataset: test_queries.csv ({len(test_queries)} queries)\n')
#         detailed_log.write(f'Entity Pool: neo4j_node.csv ({len(entity_names_list)} entities)\n')
#         detailed_log.write(f'Model: sentence-transformers/all-MiniLM-L6-v2\n')
#         detailed_log.write(f'Similarity Threshold: {threshold}\n')
#         detailed_log.write(f'Top-K Results: {top_k}\n')
#         detailed_log.write(f'Soft Matching: Enabled (0.5 credit for substring matches)\n')
#         detailed_log.write('='*100 + '\n\n')
        
#         for test_case in test_queries:
#             query = test_case['query']
#             expected_cuis = set(test_case['expected_cuis'])
            
#             # Log query details
#             query_num = test_queries.index(test_case) + 1
#             detailed_log.write(f'\n{"="*100}\n')
#             detailed_log.write(f'QUERY #{query_num}/{len(test_queries)}: "{query}"\n')
#             detailed_log.write(f'{"="*100}\n')
#             detailed_log.write(f'Expected CUIs: {sorted(list(expected_cuis))}\n')
#             detailed_log.write(f'Number of Expected CUIs: {len(expected_cuis)}\n\n')
            
#             # Get query embedding
#             detailed_log.write('STEP 1: Query Embedding\n')
#             detailed_log.write('-'*100 + '\n')
#             query_embedding = sentence_model.encode([query], show_progress_bar=False)[0]
#             detailed_log.write(f'Raw embedding shape: {query_embedding.shape}\n')
#             detailed_log.write(f'Raw embedding norm: {np.linalg.norm(query_embedding):.6f}\n')
#             query_embedding = query_embedding / np.linalg.norm(query_embedding)
#             detailed_log.write(f'Normalized embedding norm: {np.linalg.norm(query_embedding):.6f} (should be 1.0)\n\n')
            
#             # Compute cosine similarities
#             detailed_log.write('STEP 2: Similarity Computation\n')
#             detailed_log.write('-'*100 + '\n')
#             similarities = np.dot(entity_embeddings_matrix, query_embedding)
#             detailed_log.write(f'Computed similarities for {len(similarities)} entities\n')
#             detailed_log.write(f'Min similarity: {similarities.min():.6f}\n')
#             detailed_log.write(f'Max similarity: {similarities.max():.6f}\n')
#             detailed_log.write(f'Mean similarity: {similarities.mean():.6f}\n')
#             detailed_log.write(f'Median similarity: {np.median(similarities):.6f}\n\n')
            
#             # Get top-10 results for comprehensive evaluation
#             detailed_log.write('STEP 3: Top-K Retrieval\n')
#             detailed_log.write('-'*100 + '\n')
#             top_indices = np.argsort(similarities)[::-1][:top_k]
#             detailed_log.write(f'Retrieved top-{top_k} entities by similarity:\n')
#             for rank, idx in enumerate(top_indices, 1):
#                 cui = cui_list[idx]
#                 name = entity_names_list[idx]
#                 sim = similarities[idx]
#                 is_expected = '✓ EXPECTED' if cui in expected_cuis else '✗'
#                 detailed_log.write(f'  Rank {rank:2d}: {cui:15s} | Sim={sim:.6f} | {name[:50]:50s} | {is_expected}\n')
#             detailed_log.write('\n')
            
#             # Apply threshold filter
#             detailed_log.write('STEP 4: Threshold Filtering\n')
#             detailed_log.write('-'*100 + '\n')
#             detailed_log.write(f'Applying similarity threshold: {threshold}\n')
#             retrieved_cuis = []
#             filtered_count = 0
#             for idx in top_indices:
#                 if similarities[idx] >= threshold:
#                     retrieved_cuis.append(cui_list[idx])
#                 else:
#                     filtered_count += 1
#             detailed_log.write(f'Retrieved CUIs (above threshold): {retrieved_cuis}\n')
#             detailed_log.write(f'Number retrieved: {len(retrieved_cuis)}\n')
#             detailed_log.write(f'Number filtered out: {filtered_count}\n\n')
            
#             # Calculate metrics with SOFT MATCHING (for related CUIs)
#             detailed_log.write('STEP 5: Exact and Soft Matching\n')
#             detailed_log.write('-'*100 + '\n')
#             exact_matches = set(retrieved_cuis) & expected_cuis
#             detailed_log.write(f'Exact matches: {sorted(list(exact_matches))}\n')
#             detailed_log.write(f'Number of exact matches: {len(exact_matches)}\n\n')
            
#             # Soft matching: partial credit for substring matches
#             detailed_log.write('Soft matching analysis (0.5 credit for substring matches):\n')
#             soft_matches = 0
#             for ret_cui in retrieved_cuis:
#                 if ret_cui not in exact_matches:
#                     for exp_cui in expected_cuis:
#                         if ret_cui in exp_cui or exp_cui in ret_cui:
#                             detailed_log.write(f'  Soft match: {ret_cui} ↔ {exp_cui} (substring match)\n')
#                             soft_matches += 1
#                             break
#             detailed_log.write(f'Number of soft matches: {soft_matches}\n')
#             detailed_log.write(f'Soft match contribution: {soft_matches} × 0.5 = {soft_matches * 0.5}\n\n')
            
#             # Calculate TP, FP, FN with partial credit for soft matches
#             detailed_log.write('STEP 6: True Positives, False Positives, False Negatives\n')
#             detailed_log.write('-'*100 + '\n')
#             tp = len(exact_matches) + (soft_matches * 0.5)
#             fp = len(retrieved_cuis) - len(exact_matches) - soft_matches
#             fn = len(expected_cuis) - len(exact_matches) - (soft_matches * 0.5)
            
#             detailed_log.write(f'TP (True Positives) = exact_matches + (soft_matches × 0.5)\n')
#             detailed_log.write(f'                    = {len(exact_matches)} + ({soft_matches} × 0.5)\n')
#             detailed_log.write(f'                    = {len(exact_matches)} + {soft_matches * 0.5}\n')
#             detailed_log.write(f'                    = {tp}\n\n')
            
#             detailed_log.write(f'FP (False Positives) = retrieved - exact_matches - soft_matches\n')
#             detailed_log.write(f'                     = {len(retrieved_cuis)} - {len(exact_matches)} - {soft_matches}\n')
#             detailed_log.write(f'                     = {fp}\n\n')
            
#             detailed_log.write(f'FN (False Negatives) = expected - exact_matches - (soft_matches × 0.5)\n')
#             detailed_log.write(f'                     = {len(expected_cuis)} - {len(exact_matches)} - ({soft_matches} × 0.5)\n')
#             detailed_log.write(f'                     = {len(expected_cuis)} - {len(exact_matches)} - {soft_matches * 0.5}\n')
#             detailed_log.write(f'                     = {fn}\n\n')
            
#             total_tp += tp
#             total_fp += fp
#             total_fn += fn
            
#             # Precision@K, Recall@K, F1@K for different K values
#             detailed_log.write('STEP 7: Precision@K, Recall@K, F1@K Calculations\n')
#             detailed_log.write('-'*100 + '\n')
#             for k in [1, 3, 5, 10]:
#                 detailed_log.write(f'\nFor K={k}:\n')
#                 k_retrieved = set(retrieved_cuis[:k])
#                 k_exact = k_retrieved & expected_cuis
                
#                 detailed_log.write(f'  Retrieved (top-{k}): {sorted(list(k_retrieved)) if k_retrieved else "[]"}\n')
#                 detailed_log.write(f'  Exact matches in top-{k}: {sorted(list(k_exact)) if k_exact else "[]"}\n')
                
#                 # Soft matching for K results
#                 k_soft = 0
#                 for ret_cui in k_retrieved:
#                     if ret_cui not in k_exact:
#                         for exp_cui in expected_cuis:
#                             if ret_cui in exp_cui or exp_cui in ret_cui:
#                                 detailed_log.write(f'  Soft match in top-{k}: {ret_cui} ↔ {exp_cui}\n')
#                                 k_soft += 1
#                                 break
                
#                 k_tp = len(k_exact) + (k_soft * 0.5)
#                 k_precision = k_tp / len(k_retrieved) if k_retrieved else 0
#                 k_recall = k_tp / len(expected_cuis) if expected_cuis else 0
#                 k_f1 = 2 * k_precision * k_recall / (k_precision + k_recall) if (k_precision + k_recall) > 0 else 0
                
#                 detailed_log.write(f'  K_TP = {len(k_exact)} + ({k_soft} × 0.5) = {k_tp}\n')
#                 detailed_log.write(f'  Precision@{k} = K_TP / |retrieved@{k}|\n')
#                 detailed_log.write(f'               = {k_tp} / {len(k_retrieved)}\n')
#                 detailed_log.write(f'               = {k_precision:.6f}\n')
#                 detailed_log.write(f'  Recall@{k} = K_TP / |expected|\n')
#                 detailed_log.write(f'            = {k_tp} / {len(expected_cuis)}\n')
#                 detailed_log.write(f'            = {k_recall:.6f}\n')
#                 detailed_log.write(f'  F1@{k} = 2 × Precision@{k} × Recall@{k} / (Precision@{k} + Recall@{k})\n')
#                 detailed_log.write(f'        = 2 × {k_precision:.6f} × {k_recall:.6f} / ({k_precision:.6f} + {k_recall:.6f})\n')
#                 if (k_precision + k_recall) > 0:
#                     detailed_log.write(f'        = {2 * k_precision * k_recall:.6f} / {k_precision + k_recall:.6f}\n')
#                 detailed_log.write(f'        = {k_f1:.6f}\n')
                
#                 total_precision_at_k[k] += k_precision
#                 total_recall_at_k[k] += k_recall
#                 total_f1_at_k[k] += k_f1
            
#             # MRR: Mean Reciprocal Rank (position of first relevant result)
#             detailed_log.write('\nSTEP 8: Mean Reciprocal Rank (MRR) Calculation\n')
#             detailed_log.write('-'*100 + '\n')
#             first_relevant_rank = None
#             for rank, cui in enumerate(retrieved_cuis, 1):
#                 is_exact = cui in expected_cuis
#                 is_soft = any(cui in exp or exp in cui for exp in expected_cuis) if not is_exact else False
#                 if is_exact or is_soft:
#                     first_relevant_rank = rank
#                     match_type = "exact" if is_exact else "soft"
#                     detailed_log.write(f'First relevant result found at rank {rank} ({match_type} match): {cui}\n')
#                     break
            
#             if first_relevant_rank:
#                 rr = 1.0 / first_relevant_rank
#                 detailed_log.write(f'Reciprocal Rank = 1 / {first_relevant_rank} = {rr:.6f}\n')
#                 reciprocal_ranks.append(rr)
#             else:
#                 detailed_log.write('No relevant result found in retrieved set\n')
#                 detailed_log.write('Reciprocal Rank = 0.0\n')
#                 reciprocal_ranks.append(0.0)
            
#             # MAP: Mean Average Precision (precision at each relevant result)
#             detailed_log.write('\nSTEP 9: Average Precision (AP) Calculation for MAP\n')
#             detailed_log.write('-'*100 + '\n')
#             relevant_found = 0
#             precision_sum = 0
#             detailed_log.write('Computing precision at each relevant result position:\n')
#             for rank, cui in enumerate(retrieved_cuis, 1):
#                 is_exact = cui in expected_cuis
#                 is_soft = any(cui in exp or exp in cui for exp in expected_cuis) if not is_exact else False
#                 if is_exact or is_soft:
#                     relevant_found += 1
#                     precision_at_rank = relevant_found / rank
#                     precision_sum += precision_at_rank
#                     match_type = "exact" if is_exact else "soft"
#                     detailed_log.write(f'  Rank {rank}: {cui} ({match_type}) → Precision = {relevant_found}/{rank} = {precision_at_rank:.6f}\n')
            
#             detailed_log.write(f'\nPrecision sum = {precision_sum:.6f}\n')
#             detailed_log.write(f'Number of expected CUIs = {len(expected_cuis)}\n')
#             avg_precision = precision_sum / len(expected_cuis) if expected_cuis else 0
#             detailed_log.write(f'Average Precision = {precision_sum:.6f} / {len(expected_cuis)} = {avg_precision:.6f}\n')
#             average_precisions.append(avg_precision)
            
#             # Query summary
#             detailed_log.write('\n' + '='*100 + '\n')
#             detailed_log.write(f'QUERY #{query_num} SUMMARY:\n')
#             detailed_log.write('='*100 + '\n')
#             detailed_log.write(f'TP={tp:.1f}, FP={fp:.1f}, FN={fn:.1f}\n')
#             detailed_log.write(f'Reciprocal Rank: {reciprocal_ranks[-1]:.6f}\n')
#             detailed_log.write(f'Average Precision: {avg_precision:.6f}\n')
#             detailed_log.write('='*100 + '\n\n')

        
#         # Calculate overall metrics
#         n_queries = len(test_queries)
        
#         # Write overall calculations to detailed log
#         detailed_log.write('\n' + '='*100 + '\n')
#         detailed_log.write('OVERALL METRICS CALCULATION\n')
#         detailed_log.write('='*100 + '\n\n')
        
#         detailed_log.write('STEP 1: Overall Precision, Recall, F1\n')
#         detailed_log.write('-'*100 + '\n')
#         detailed_log.write(f'Total True Positives (across all queries): {total_tp:.2f}\n')
#         detailed_log.write(f'Total False Positives (across all queries): {total_fp:.2f}\n')
#         detailed_log.write(f'Total False Negatives (across all queries): {total_fn:.2f}\n\n')
        
#         overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#         overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#         overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
#         detailed_log.write(f'Overall Precision = TP / (TP + FP)\n')
#         detailed_log.write(f'                  = {total_tp:.2f} / ({total_tp:.2f} + {total_fp:.2f})\n')
#         detailed_log.write(f'                  = {total_tp:.2f} / {total_tp + total_fp:.2f}\n')
#         detailed_log.write(f'                  = {overall_precision:.6f} ({overall_precision*100:.2f}%)\n\n')
        
#         detailed_log.write(f'Overall Recall = TP / (TP + FN)\n')
#         detailed_log.write(f'               = {total_tp:.2f} / ({total_tp:.2f} + {total_fn:.2f})\n')
#         detailed_log.write(f'               = {total_tp:.2f} / {total_tp + total_fn:.2f}\n')
#         detailed_log.write(f'               = {overall_recall:.6f} ({overall_recall*100:.2f}%)\n\n')
        
#         detailed_log.write(f'Overall F1 Score = 2 × Precision × Recall / (Precision + Recall)\n')
#         detailed_log.write(f'                 = 2 × {overall_precision:.6f} × {overall_recall:.6f} / ({overall_precision:.6f} + {overall_recall:.6f})\n')
#         if (overall_precision + overall_recall) > 0:
#             detailed_log.write(f'                 = {2 * overall_precision * overall_recall:.6f} / {overall_precision + overall_recall:.6f}\n')
#         detailed_log.write(f'                 = {overall_f1:.6f} ({overall_f1*100:.2f}%)\n\n')
        
#         detailed_log.write('STEP 2: Mean Reciprocal Rank (MRR)\n')
#         detailed_log.write('-'*100 + '\n')
#         detailed_log.write(f'Reciprocal ranks for all queries: {[f"{rr:.4f}" for rr in reciprocal_ranks]}\n')
#         detailed_log.write(f'Sum of reciprocal ranks: {sum(reciprocal_ranks):.6f}\n')
#         detailed_log.write(f'Number of queries: {len(reciprocal_ranks)}\n')
#         mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
#         detailed_log.write(f'MRR = Sum(RR) / Number of queries\n')
#         detailed_log.write(f'    = {sum(reciprocal_ranks):.6f} / {len(reciprocal_ranks)}\n')
#         detailed_log.write(f'    = {mrr:.6f}\n\n')
        
#         detailed_log.write('STEP 3: Mean Average Precision (MAP)\n')
#         detailed_log.write('-'*100 + '\n')
#         detailed_log.write(f'Average precisions for all queries: {[f"{ap:.4f}" for ap in average_precisions]}\n')
#         detailed_log.write(f'Sum of average precisions: {sum(average_precisions):.6f}\n')
#         detailed_log.write(f'Number of queries: {len(average_precisions)}\n')
#         map_score = sum(average_precisions) / len(average_precisions) if average_precisions else 0
#         detailed_log.write(f'MAP = Sum(AP) / Number of queries\n')
#         detailed_log.write(f'    = {sum(average_precisions):.6f} / {len(average_precisions)}\n')
#         detailed_log.write(f'    = {map_score:.6f}\n\n')
        
#         detailed_log.write('STEP 4: Precision@K, Recall@K, F1@K (Averaged across queries)\n')
#         detailed_log.write('-'*100 + '\n')
#         for k in [1, 3, 5, 10]:
#             p_at_k = total_precision_at_k[k] / n_queries
#             r_at_k = total_recall_at_k[k] / n_queries
#             f1_at_k = total_f1_at_k[k] / n_queries
            
#             detailed_log.write(f'\nFor K={k}:\n')
#             detailed_log.write(f'  Sum of Precision@{k} across all queries: {total_precision_at_k[k]:.6f}\n')
#             detailed_log.write(f'  Average Precision@{k} = {total_precision_at_k[k]:.6f} / {n_queries} = {p_at_k:.6f} ({p_at_k*100:.2f}%)\n')
#             detailed_log.write(f'  Sum of Recall@{k} across all queries: {total_recall_at_k[k]:.6f}\n')
#             detailed_log.write(f'  Average Recall@{k} = {total_recall_at_k[k]:.6f} / {n_queries} = {r_at_k:.6f} ({r_at_k*100:.2f}%)\n')
#             detailed_log.write(f'  Sum of F1@{k} across all queries: {total_f1_at_k[k]:.6f}\n')
#             detailed_log.write(f'  Average F1@{k} = {total_f1_at_k[k]:.6f} / {n_queries} = {f1_at_k:.6f} ({f1_at_k*100:.2f}%)\n')
        
#         detailed_log.write('\n' + '='*100 + '\n')
#         detailed_log.write('FINAL SUMMARY OF ALL METRICS\n')
#         detailed_log.write('='*100 + '\n')
#         detailed_log.write(f'Overall Precision:  {overall_precision:.6f} ({overall_precision*100:.2f}%)\n')
#         detailed_log.write(f'Overall Recall:     {overall_recall:.6f} ({overall_recall*100:.2f}%)\n')
#         detailed_log.write(f'Overall F1 Score:   {overall_f1:.6f} ({overall_f1*100:.2f}%)\n')
#         detailed_log.write(f'MRR:                {mrr:.6f}\n')
#         detailed_log.write(f'MAP:                {map_score:.6f}\n\n')
        
#         for k in [1, 3, 5, 10]:
#             p_at_k = total_precision_at_k[k] / n_queries
#             r_at_k = total_recall_at_k[k] / n_queries
#             f1_at_k = total_f1_at_k[k] / n_queries
#             detailed_log.write(f'P@{k:2d}:  {p_at_k:.6f} ({p_at_k*100:.2f}%)\n')
#             detailed_log.write(f'R@{k:2d}:  {r_at_k:.6f} ({r_at_k*100:.2f}%)\n')
#             detailed_log.write(f'F1@{k:2d}: {f1_at_k:.6f} ({f1_at_k*100:.2f}%)\n\n')
        
#         detailed_log.write('='*100 + '\n')
#         detailed_log.write('END OF DETAILED CALCULATION LOG\n')
#         detailed_log.write('='*100 + '\n')
        
#         # Close detailed log
#         detailed_log.close()
        
#         # Write to summary log file
#         log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
#         os.makedirs(log_dir, exist_ok=True)
#         log_file = os.path.join(log_dir, 'retrieval_metrics.log')
        
#         with open(log_file, 'w', encoding='utf-8') as f:  # Overwrite to keep latest only
#             f.write('='*70 + '\n')
#             f.write('INFORMATION RETRIEVAL METRICS - ADAPTIVE_PRECISION Strategy\n')
#             f.write('='*70 + '\n')
#             f.write(f'Evaluation Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
#             f.write(f'Dataset: test_queries.csv ({n_queries} queries)\n')
#             f.write(f'Entity Pool: neo4j_node.csv ({len(entity_names_list)} entities)\n')
#             f.write(f'Model: sentence-transformers/all-MiniLM-L6-v2\n\n')
            
#             f.write('OVERALL RETRIEVAL PERFORMANCE:\n')
#             f.write(f'  Precision:         {overall_precision:.4f} ({overall_precision*100:.2f}%)\n')
#             f.write(f'  Recall:            {overall_recall:.4f} ({overall_recall*100:.2f}%)\n')
#             f.write(f'  F1 Score:          {overall_f1:.4f} ({overall_f1*100:.2f}%)\n')
#             f.write(f'  MRR (Mean Reciprocal Rank):  {mrr:.4f}\n')
#             f.write(f'  MAP (Mean Average Precision): {map_score:.4f}\n\n')
            
#             f.write('PRECISION @ K:\n')
#             for k in [1, 3, 5, 10]:
#                 p_at_k = total_precision_at_k[k] / n_queries
#                 f.write(f'  P@{k:2d}:  {p_at_k:.4f} ({p_at_k*100:.2f}%)\n')
            
#             f.write('\nRECALL @ K:\n')
#             for k in [1, 3, 5, 10]:
#                 r_at_k = total_recall_at_k[k] / n_queries
#                 f.write(f'  R@{k:2d}:  {r_at_k:.4f} ({r_at_k*100:.2f}%)\n')
            
#             f.write('\nF1 @ K:\n')
#             for k in [1, 3, 5, 10]:
#                 f1_at_k = total_f1_at_k[k] / n_queries
#                 f.write(f'  F1@{k:2d}: {f1_at_k:.4f} ({f1_at_k*100:.2f}%)\n')
            
#             f.write('\nRETRIEVAL STRATEGY:\n')
#             f.write(f'  Strategy: ADAPTIVE_PRECISION\n')
#             f.write(f'  Similarity Threshold: {threshold}\n')
#             f.write(f'  Top-K Results: {top_k}\n')
#             f.write(f'  Soft Matching: Enabled (0.5 credit for substring matches)\n')
            
#             f.write('\nMETRIC DEFINITIONS:\n')
#             f.write('  • Precision: Fraction of retrieved entities that are relevant\n')
#             f.write('  • Recall: Fraction of relevant entities that are retrieved\n')
#             f.write('  • F1 Score: Harmonic mean of precision and recall\n')
#             f.write('  • MRR: Average of reciprocal ranks of first relevant result\n')
#             f.write('  • MAP: Mean of average precision across all queries\n')
#             f.write('  • P@K: Precision considering only top-K results\n')
#             f.write('  • R@K: Recall considering only top-K results\n')
#             f.write('='*70 + '\n')
        
#         print(f"✓ Retrieval metrics logged to: {log_file}")
#         print(f"✓ Detailed calculations logged to: {detailed_log_file}")
#         print(f"  F1 Score: {overall_f1:.4f} ({overall_f1*100:.2f}%)")
#         print(f"  Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
#         print(f"  Recall: {overall_recall:.4f} ({overall_recall*100:.2f}%)")
#         print(f"  MRR: {mrr:.4f}, MAP: {map_score:.4f}")
        
#     except Exception as e:
#         print(f"Error evaluating retrieval metrics: {e}")
#         import traceback
#         traceback.print_exc()
#         import traceback
#         traceback.print_exc()
