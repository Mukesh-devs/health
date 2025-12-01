"""
Real-time PubMed integration for latest medical evidence.
Fetches and analyzes recent research papers to complement static knowledge graph.
"""

from Bio import Entrez, Medline
import requests
from datetime import datetime, timedelta
import re
from typing import List, Dict
import logging

# Configure PubMed API
Entrez.email = "health_app@example.com"  # REQUIRED by NCBI - change to your email
Entrez.tool = "HealthLLMApp"

logger = logging.getLogger(__name__)


class PubMedLiveSearch:
    """Real-time PubMed integration for latest medical evidence"""
    
    def __init__(self, months_back=12):
        self.months_back = months_back
        self.date_filter = self._get_date_filter()
    
    def _get_date_filter(self):
        """Generate date filter for recent papers"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * self.months_back)
        return f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[PDAT]"
    
    
    def fetch_latest_evidence(self, entity1: str, entity2: str, 
                             max_results: int = 10) -> Dict:
        """
        Fetch latest research papers for entity relationship
        
        Args:
            entity1: First medical entity (e.g., "Aspirin")
            entity2: Second medical entity (e.g., "Heart Disease")
            max_results: Maximum papers to retrieve
            
        Returns:
            Dictionary with papers, sentiment, and metadata
        """
        
        # ANSI color codes
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'
        
        print(f"\n{CYAN}  [PUBMED MODULE] Fetching live evidence...{RESET}")
        print(f"{YELLOW}  Entities: {entity1} + {entity2}{RESET}")
        
        logger.info(f"Fetching live evidence for: {entity1} + {entity2}")
        
        # Build search query with filters
        search_query = self._build_search_query(entity1, entity2)
        print(f"{YELLOW}  Query: {search_query}{RESET}")
        
        try:
            # Step 1: Search PubMed
            print(f"{YELLOW}  [Step 1] Searching PubMed database...{RESET}")
            search_handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=max_results,
                sort="relevance",
                usehistory="y"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmid_list = search_results.get("IdList", [])
            print(f"{GREEN}  ✓ Found {len(pmid_list)} papers{RESET}")
            
            if not pmid_list:
                print(f"{YELLOW}  ⚠ No recent papers found{RESET}")
                return {
                    'found_new_evidence': False,
                    'message': 'No recent papers found',
                    'papers': []
                }
            
            # Step 2: Fetch paper details
            print(f"{YELLOW}  [Step 2] Fetching paper details...{RESET}")
            papers = self._fetch_paper_details(pmid_list)
            print(f"{GREEN}  ✓ Retrieved {len(papers)} full papers{RESET}")
            
            # Step 3: Analyze evidence sentiment
            print(f"{YELLOW}  [Step 3] Analyzing sentiment...{RESET}")
            analyzed_papers = self._analyze_papers(papers, entity1, entity2)
            print(f"{GREEN}  ✓ Analyzed {len(analyzed_papers)} papers{RESET}")
            
            # Step 4: Calculate aggregate sentiment
            print(f"{YELLOW}  [Step 4] Calculating aggregate sentiment...{RESET}")
            sentiment_summary = self._calculate_sentiment(analyzed_papers)
            print(f"{GREEN}  ✓ Overall sentiment: {sentiment_summary.get('overall_sentiment', 'unknown')}{RESET}")
            
            return {
                'found_new_evidence': True,
                'total_papers': len(analyzed_papers),
                'papers': analyzed_papers,
                'sentiment_summary': sentiment_summary,
                'search_query': search_query,
                'date_range': self.date_filter
            }
            
        except Exception as e:
            print(f"{YELLOW}  ⚠ PubMed API error: {str(e)}{RESET}")
            logger.error(f"PubMed API error: {str(e)}")
            return {
                'found_new_evidence': False,
                'error': str(e),
                'papers': []
            }
    
    
    def _build_search_query(self, entity1: str, entity2: str) -> str:
        """Build optimized PubMed search query"""
        
        # Add MeSH terms for better precision
        query_parts = [
            f'("{entity1}"[Title/Abstract] OR "{entity1}"[MeSH Terms])',
            'AND',
            f'("{entity2}"[Title/Abstract] OR "{entity2}"[MeSH Terms])',
            'AND',
            self.date_filter,
            'AND',
            # Filter for high-quality studies
            '(Clinical Trial[ptyp] OR Meta-Analysis[ptyp] OR Review[ptyp] OR Randomized Controlled Trial[ptyp])',
            'AND',
            'humans[MeSH Terms]',  # Human studies only
            'AND',
            'English[lang]'  # English papers only
        ]
        
        return ' '.join(query_parts)
    
    
    def _fetch_paper_details(self, pmid_list: List[str]) -> List[Dict]:
        """Fetch full paper metadata and abstracts"""
        
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmid_list,
                rettype="medline",
                retmode="text"
            )
            records = Medline.parse(fetch_handle)
            
            papers = []
            for record in records:
                papers.append({
                    'pmid': record.get('PMID', ''),
                    'title': record.get('TI', ''),
                    'abstract': record.get('AB', ''),
                    'authors': record.get('AU', []),
                    'journal': record.get('JT', ''),
                    'pub_date': record.get('DP', ''),
                    'publication_type': record.get('PT', []),
                    'mesh_terms': record.get('MH', [])
                })
            
            fetch_handle.close()
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching paper details: {str(e)}")
            return []
    
    
    def _analyze_papers(self, papers: List[Dict], entity1: str, 
                       entity2: str) -> List[Dict]:
        """Analyze relationship sentiment in papers using NLP"""
        
        analyzed = []
        
        # Keywords for relationship detection
        positive_keywords = ['effective', 'beneficial', 'improves', 'treats', 
                            'reduces', 'prevents', 'significant improvement',
                            'positive effect', 'therapeutic', 'efficacious',
                            'associated with reduced', 'protective effect']
        
        negative_keywords = ['ineffective', 'no benefit', 'not recommended',
                            'no significant', 'failed to', 'did not improve',
                            'contraindicated', 'adverse', 'harmful',
                            'no association', 'not effective', 'no evidence']
        
        neutral_keywords = ['investigated', 'examined', 'studied', 'analyzed',
                           'requires further', 'needs more research', 'unclear']
        
        for paper in papers:
            abstract = paper.get('abstract', '').lower()
            title = paper.get('title', '').lower()
            
            # Combine title + abstract for analysis
            full_text = f"{title} {abstract}"
            
            # Count sentiment indicators
            positive_count = sum(1 for kw in positive_keywords if kw in full_text)
            negative_count = sum(1 for kw in negative_keywords if kw in full_text)
            neutral_count = sum(1 for kw in neutral_keywords if kw in full_text)
            
            # Determine overall sentiment
            if positive_count > negative_count and positive_count > neutral_count:
                sentiment = 'supports'
                confidence = min(positive_count / (positive_count + negative_count + 1), 1.0)
            elif negative_count > positive_count:
                sentiment = 'contradicts'
                confidence = min(negative_count / (positive_count + negative_count + 1), 1.0)
            else:
                sentiment = 'neutral'
                confidence = 0.5
            
            # Extract key sentence mentioning both entities
            key_sentence = self._extract_key_sentence(full_text, entity1, entity2)
            
            paper['sentiment'] = sentiment
            paper['confidence'] = round(confidence, 2)
            paper['key_sentence'] = key_sentence
            paper['positive_score'] = positive_count
            paper['negative_score'] = negative_count
            
            analyzed.append(paper)
        
        return analyzed
    
    
    def _extract_key_sentence(self, text: str, entity1: str, entity2: str) -> str:
        """Extract sentence mentioning both entities"""
        
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if entity1.lower() in sentence and entity2.lower() in sentence:
                return sentence.strip()[:200]  # Limit to 200 chars
        
        # If no sentence with both entities, try finding one with either
        for sentence in sentences:
            if entity1.lower() in sentence or entity2.lower() in sentence:
                return sentence.strip()[:200]
        
        return ""
    
    
    def _calculate_sentiment(self, papers: List[Dict]) -> Dict:
        """Calculate aggregate sentiment from all papers"""
        
        if not papers:
            return {}
        
        total = len(papers)
        supports = sum(1 for p in papers if p['sentiment'] == 'supports')
        contradicts = sum(1 for p in papers if p['sentiment'] == 'contradicts')
        neutral = sum(1 for p in papers if p['sentiment'] == 'neutral')
        
        avg_confidence = sum(p['confidence'] for p in papers) / total
        
        # Determine overall trend
        if supports / total >= 0.6:
            overall = 'strongly_supports'
        elif supports / total >= 0.4:
            overall = 'moderately_supports'
        elif contradicts / total >= 0.6:
            overall = 'strongly_contradicts'
        else:
            overall = 'mixed_evidence'
        
        return {
            'overall_sentiment': overall,
            'supports_percentage': round((supports / total) * 100, 1),
            'contradicts_percentage': round((contradicts / total) * 100, 1),
            'neutral_percentage': round((neutral / total) * 100, 1),
            'average_confidence': round(avg_confidence, 2),
            'total_papers_analyzed': total,
            'interpretation': self._get_interpretation(overall, supports, total)
        }
    
    
    def _get_interpretation(self, overall: str, supports: int, total: int) -> str:
        """Generate human-readable interpretation"""
        
        interpretations = {
            'strongly_supports': f'Recent research strongly supports this relationship ({supports}/{total} papers)',
            'moderately_supports': f'Recent research moderately supports this relationship ({supports}/{total} papers)',
            'strongly_contradicts': f'Recent research contradicts this relationship',
            'mixed_evidence': f'Recent research shows mixed results - requires clinical judgment'
        }
        
        return interpretations.get(overall, 'Insufficient evidence')


# Initialize global instance
pubmed_live = PubMedLiveSearch(months_back=12)
