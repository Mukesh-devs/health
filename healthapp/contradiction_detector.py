"""
Detects and analyzes contradictions in medical research.
Identifies opposing relationships and provides temporal analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContradictionDetector:
    """Detects and analyzes contradictions in medical research"""
    
    # Define opposing relationship pairs
    CONTRADICTORY_PREDICATES = {
        'TREATS': ['NO_EFFECT', 'INEFFECTIVE_FOR', 'DOES_NOT_TREAT'],
        'PREVENTS': ['DOES_NOT_PREVENT', 'NO_PREVENTIVE_EFFECT'],
        'CAUSES': ['DOES_NOT_CAUSE', 'NOT_ASSOCIATED_WITH'],
        'INCREASES': ['DECREASES', 'REDUCES'],
        'ASSOCIATED_WITH': ['NOT_ASSOCIATED_WITH'],
        'BENEFICIAL_FOR': ['HARMFUL_FOR', 'ADVERSE_FOR'],
        'STIMULATES': ['INHIBITS', 'SUPPRESSES'],
        'INTERACTS_WITH': ['NO_INTERACTION'],
    }
    
    def __init__(self, kg_df: pd.DataFrame):
        self.kg_df = kg_df
        self._preprocess_dates()
    
    
    def _preprocess_dates(self):
        """Extract publication years from PubMed data"""
        # Estimate year from PubMed IDs (older IDs = older papers)
        self.kg_df['pub_year_estimate'] = self.kg_df['PubMed_ID'].apply(
            self._estimate_year_from_pmid
        )
    
    
    def _estimate_year_from_pmid(self, pmid: str) -> int:
        """Rough estimation: lower PMID = older paper"""
        try:
            pmid_str = str(pmid).split(';')[0]  # Take first PMID if multiple
            pmid_int = int(pmid_str)
            # PubMed started at ~1 in 1996, reaches ~35M by 2024
            # Rough formula: year ≈ 1996 + (pmid / 1,250,000)
            estimated_year = 1996 + (pmid_int // 1_250_000)
            return min(estimated_year, 2024)  # Cap at current year
        except:
            return 2020  # Default to recent if can't parse
    
    
    def detect_contradictions(self, entity1_cui: str, entity2_cui: str) -> Dict:
        """
        Find contradictory relationships between two entities
        
        Args:
            entity1_cui: CUI of first entity
            entity2_cui: CUI of second entity
            
        Returns:
            Dictionary with contradiction analysis
        """
        
        # Find ALL relationships between entities (both directions)
        relationships = self._get_all_relationships(entity1_cui, entity2_cui)
        
        if not relationships:
            return {
                'has_contradictions': False,
                'message': 'No relationships found'
            }
        
        # Group by predicate type
        grouped_predicates = self._group_predicates(relationships)
        
        # Detect contradictory pairs
        contradictions = self._find_contradictory_pairs(grouped_predicates)
        
        if not contradictions:
            return {
                'has_contradictions': False,
                'message': 'No contradictions detected',
                'total_relationships': len(relationships)
            }
        
        # Analyze temporal patterns
        temporal_analysis = self._analyze_temporal_trends(contradictions)
        
        # Analyze study quality
        quality_analysis = self._analyze_study_quality(contradictions)
        
        # Generate clinical interpretation
        interpretation = self._generate_interpretation(
            contradictions, temporal_analysis, quality_analysis
        )
        
        return {
            'has_contradictions': True,
            'total_relationships': len(relationships),
            'contradictions': contradictions,
            'temporal_analysis': temporal_analysis,
            'quality_analysis': quality_analysis,
            'interpretation': interpretation,
            'recommendation': self._get_clinical_recommendation(interpretation)
        }
    
    
    def _get_all_relationships(self, cui1: str, cui2: str) -> List[Dict]:
        """Get all relationships between two entities"""
        
        # Forward direction
        forward = self.kg_df[
            (self.kg_df['START_ID'] == cui1) &
            (self.kg_df['END_ID'] == cui2)
        ]
        
        # Backward direction (some predicates might be reversed)
        backward = self.kg_df[
            (self.kg_df['START_ID'] == cui2) &
            (self.kg_df['END_ID'] == cui1)
        ]
        
        all_rels = []
        
        for _, row in forward.iterrows():
            all_rels.append({
                'predicate': row['PREDICATE'],
                'pubmed_ids': row['PubMed_ID'],
                'sentence': row['SENTENCE'],
                'direction': 'forward',
                'year': row['pub_year_estimate']
            })
        
        for _, row in backward.iterrows():
            all_rels.append({
                'predicate': row['PREDICATE'],
                'pubmed_ids': row['PubMed_ID'],
                'sentence': row['SENTENCE'],
                'direction': 'backward',
                'year': row['pub_year_estimate']
            })
        
        return all_rels
    
    
    def _group_predicates(self, relationships: List[Dict]) -> Dict[str, List[Dict]]:
        """Group relationships by predicate type"""
        
        grouped = defaultdict(list)
        for rel in relationships:
            grouped[rel['predicate']].append(rel)
        
        return dict(grouped)
    
    
    def _find_contradictory_pairs(self, grouped: Dict) -> List[Dict]:
        """Identify contradictory predicate pairs"""
        
        contradictions = []
        
        for positive_pred, negative_preds in self.CONTRADICTORY_PREDICATES.items():
            if positive_pred in grouped:
                for negative_pred in negative_preds:
                    if negative_pred in grouped:
                        # Found contradiction!
                        contradictions.append({
                            'positive': {
                                'predicate': positive_pred,
                                'evidence': grouped[positive_pred],
                                'count': len(grouped[positive_pred])
                            },
                            'negative': {
                                'predicate': negative_pred,
                                'evidence': grouped[negative_pred],
                                'count': len(grouped[negative_pred])
                            }
                        })
        
        return contradictions
    
    
    def _analyze_temporal_trends(self, contradictions: List[Dict]) -> Dict:
        """Analyze how evidence has evolved over time"""
        
        trends = []
        
        for contra in contradictions:
            positive_years = [e['year'] for e in contra['positive']['evidence']]
            negative_years = [e['year'] for e in contra['negative']['evidence']]
            
            avg_positive_year = np.mean(positive_years) if positive_years else 0
            avg_negative_year = np.mean(negative_years) if negative_years else 0
            
            if avg_positive_year < avg_negative_year:
                trend = 'initial_optimism_later_skepticism'
                message = f"Early studies ({int(avg_positive_year)}) showed benefits, but recent research ({int(avg_negative_year)}) questions this"
            elif avg_negative_year < avg_positive_year:
                trend = 'evolving_understanding'
                message = f"Research is evolving - newer studies ({int(avg_positive_year)}) show different results"
            else:
                trend = 'ongoing_debate'
                message = "Contradictory evidence published simultaneously - active research area"
            
            trends.append({
                'positive_predicate': contra['positive']['predicate'],
                'negative_predicate': contra['negative']['predicate'],
                'trend_type': trend,
                'message': message,
                'positive_avg_year': int(avg_positive_year),
                'negative_avg_year': int(avg_negative_year)
            })
        
        return {
            'trends': trends,
            'overall_pattern': self._determine_overall_pattern(trends)
        }
    
    
    def _determine_overall_pattern(self, trends: List[Dict]) -> str:
        """Determine overall temporal pattern"""
        
        patterns = [t['trend_type'] for t in trends]
        
        if 'initial_optimism_later_skepticism' in patterns:
            return 'Evidence weakening over time - be cautious'
        elif 'evolving_understanding' in patterns:
            return 'Research is actively evolving - monitor for updates'
        else:
            return 'Ongoing scientific debate - consult latest guidelines'
    
    
    def _analyze_study_quality(self, contradictions: List[Dict]) -> Dict:
        """Analyze quality indicators of conflicting studies"""
        
        quality_metrics = []
        
        for contra in contradictions:
            positive_pmids = []
            negative_pmids = []
            
            for ev in contra['positive']['evidence']:
                positive_pmids.extend(str(ev['pubmed_ids']).split(';'))
            
            for ev in contra['negative']['evidence']:
                negative_pmids.extend(str(ev['pubmed_ids']).split(';'))
            
            quality_metrics.append({
                'positive_study_count': len(positive_pmids),
                'negative_study_count': len(negative_pmids),
                'evidence_balance': len(negative_pmids) / (len(positive_pmids) + len(negative_pmids)) if (len(positive_pmids) + len(negative_pmids)) > 0 else 0.5
            })
        
        avg_balance = np.mean([m['evidence_balance'] for m in quality_metrics]) if quality_metrics else 0.5
        
        if avg_balance > 0.6:
            quality_assessment = 'Preponderance of evidence leans negative'
        elif avg_balance < 0.4:
            quality_assessment = 'Preponderance of evidence leans positive'
        else:
            quality_assessment = 'Evidence is evenly balanced - true uncertainty'
        
        return {
            'metrics': quality_metrics,
            'assessment': quality_assessment
        }
    
    
    def _generate_interpretation(self, contradictions: List[Dict], 
                                 temporal: Dict, quality: Dict) -> Dict:
        """Generate human-readable interpretation"""
        
        total_positive = sum(c['positive']['count'] for c in contradictions)
        total_negative = sum(c['negative']['count'] for c in contradictions)
        total_evidence = total_positive + total_negative
        
        positive_pct = (total_positive / total_evidence) * 100 if total_evidence > 0 else 0
        negative_pct = (total_negative / total_evidence) * 100 if total_evidence > 0 else 0
        
        # Determine confidence level
        if abs(positive_pct - negative_pct) < 10:
            confidence = 'very_low'
            confidence_msg = 'Highly uncertain - evidence is evenly split'
        elif abs(positive_pct - negative_pct) < 25:
            confidence = 'low'
            confidence_msg = 'Low confidence - significant disagreement in literature'
        elif abs(positive_pct - negative_pct) < 40:
            confidence = 'moderate'
            confidence_msg = 'Moderate confidence - some consensus emerging'
        else:
            confidence = 'high'
            confidence_msg = 'Higher confidence - clear preponderance of evidence'
        
        return {
            'evidence_summary': {
                'total_papers': total_evidence,
                'positive_papers': total_positive,
                'negative_papers': total_negative,
                'positive_percentage': round(positive_pct, 1),
                'negative_percentage': round(negative_pct, 1)
            },
            'confidence_level': confidence,
            'confidence_message': confidence_msg,
            'temporal_pattern': temporal['overall_pattern'],
            'quality_assessment': quality['assessment']
        }
    
    
    def _get_clinical_recommendation(self, interpretation: Dict) -> str:
        """Generate clinical recommendation based on analysis"""
        
        confidence = interpretation['confidence_level']
        summary = interpretation['evidence_summary']
        
        if confidence in ['very_low', 'low']:
            return (
                f"⚠️ <strong>CONFLICTING EVIDENCE</strong><br>"
                f"Evidence is divided: {summary['positive_percentage']}% positive vs "
                f"{summary['negative_percentage']}% negative from {summary['total_papers']} studies.<br>"
                f"<strong>Recommendation:</strong> Consult current clinical guidelines and healthcare provider. "
                f"Individual patient factors should guide decision-making."
            )
        
        elif summary['positive_percentage'] > summary['negative_percentage']:
            return (
                f"⚡ <strong>MIXED EVIDENCE - LEANS POSITIVE</strong><br>"
                f"{summary['positive_percentage']}% of {summary['total_papers']} studies support this relationship.<br>"
                f"<strong>Recommendation:</strong> May be beneficial but not universally accepted. "
                f"Review latest clinical guidelines."
            )
        
        else:
            return (
                f"⚡ <strong>MIXED EVIDENCE - LEANS NEGATIVE</strong><br>"
                f"{summary['negative_percentage']}% of {summary['total_papers']} studies question this relationship.<br>"
                f"<strong>Recommendation:</strong> Current evidence does not strongly support this. "
                f"Consult healthcare provider for alternatives."
            )


# Global detector instance (will be initialized with kg_df)
contradiction_detector = None

def init_contradiction_detector(kg_df: pd.DataFrame):
    """Initialize detector with knowledge graph"""
    global contradiction_detector
    contradiction_detector = ContradictionDetector(kg_df)
