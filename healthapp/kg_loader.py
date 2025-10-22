"""
Knowledge Graph Loader
"""
import os
import pandas as pd
from flask import current_app

def load_knowledge_graph():
    """Load knowledge graph nodes and relationships from CSV files"""
    try:
        rel_path = current_app.config['KG_REL_PATH']
        node_path = current_app.config['KG_NODE_PATH']
        
        if not os.path.exists(rel_path) or not os.path.exists(node_path):
            raise FileNotFoundError(
                f"Knowledge graph files not found. "
                f"Expected: {rel_path} and {node_path}"
            )
        
        # Load relationships
        kg_df = pd.read_csv(rel_path, dtype=str)
        kg_df.fillna('', inplace=True)
        
        # Load nodes
        nodes_df = pd.read_csv(node_path, dtype=str)
        nodes_df.fillna('', inplace=True)
        nodes_df['Name_lower'] = nodes_df['Name'].str.lower().str.strip()
        
        print(f"Knowledge graph loaded: {len(nodes_df)} nodes, {len(kg_df)} relationships")
        
        return kg_df, nodes_df
        
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        raise
