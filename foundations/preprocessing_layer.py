"""
Enhanced Preprocessing Layer for AI Assistant
Provides fast query classification and routing using lightweight regex and fuzzy matching.
Target: <100ms preprocessing time for simple queries
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from rapidfuzz import fuzz, process
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classifications for optimal routing."""
    GREETING = "greeting"
    CONTACT = "contact" 
    RESUME = "resume"
    SKILLS = "skills"
    EXPERIENCE = "experience"
    PROJECTS = "projects"
    EDUCATION = "education"
    CONVERSATIONAL = "conversational"
    COMPLEX_ANALYSIS = "complex_analysis"
    UNKNOWN = "unknown"

class RoutingDecision(Enum):
    """Routing decision for query processing."""
    DIRECT_GEMINI = "direct_gemini"  # Bypass gRPC, go straight to Gemini
    ENHANCED_PIPELINE = "enhanced_pipeline"  # Full pipeline with knowledge retrieval
    STATIC_RESPONSE = "static_response"  # Pre-built static response
    CACHE_LOOKUP = "cache_lookup"  # Check cache first

@dataclass
class QueryAnalysis:
    """Results of query preprocessing analysis."""
    intent: QueryIntent
    routing: RoutingDecision
    confidence: float
    keywords: List[str]
    complexity: str  # simple, medium, complex
    estimated_response_time: float  # seconds
    cache_key: Optional[str] = None
    suggested_template: Optional[str] = None

class EnhancedPreprocessor:
    """
    Lightning-fast preprocessing layer with intelligent routing.
    Target: <100ms for simple queries, <200ms for complex queries.
    """
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
        self.keyword_database = self._build_keyword_database()
        self.response_templates = self._build_response_templates()
        self._performance_stats = {"total_queries": 0, "avg_time": 0.0}
        
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build regex patterns for intent classification."""
        return {
            QueryIntent.GREETING: [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\b(how are you|what\'s up|nice to meet)\b',
                r'^\s*(hi|hello|hey)\s*[!.]*\s*$'
            ],
            QueryIntent.CONTACT: [
                r'\b(contact|email|phone|linkedin|github|reach|calendly)\b',
                r'\b(how to contact|get in touch|schedule|meeting)\b',
                r'\b(url|link|social|professional)\b'
            ],
            QueryIntent.RESUME: [
                r'\b(resume|cv|curriculum vitae|download|pdf)\b',
                r'\b(professional summary|profile|overview)\b'
            ],
            QueryIntent.SKILLS: [
                r'\b(skills|technical|programming|languages|frameworks)\b',
                r'\b(technologies|tools|expertise|proficient)\b',
                r'\b(python|java|javascript|aws|docker|ai|ml)\b'
            ],
            QueryIntent.EXPERIENCE: [
                r'\b(experience|work|job|employment|career|years)\b',
                r'\b(company|role|position|responsibilities)\b',
                r'\b(veritis|tcs|virtusa|professional)\b'
            ],
            QueryIntent.PROJECTS: [
                r'\b(projects|built|developed|created|implemented)\b',
                r'\b(portfolio|achievements|accomplishments)\b',
                r'\b(testing agent|clinical|loan|chat platform)\b'
            ],
            QueryIntent.EDUCATION: [
                r'\b(education|degree|university|college|graduation)\b',
                r'\b(masters|bachelor|ms|computer science|gmu|george mason)\b'
            ]
        }
    
    def _build_keyword_database(self) -> Dict[str, List[str]]:
        """Build fuzzy match keyword database for enhanced matching."""
        return {
            "technical_skills": [
                "python", "java", "javascript", "typescript", "react", "angular",
                "fastapi", "django", "flask", "aws", "docker", "kubernetes",
                "ai", "ml", "machine learning", "tensorflow", "pytorch", "langchain"
            ],
            "companies": [
                "veritis", "veritis group", "tcs", "tata consultancy", "virtusa",
                "george mason", "gmu", "gitam"
            ],
            "projects": [
                "testing agent", "clinical api", "loan origination", "chat platform",
                "rag", "retrieval augmented generation", "multi-modal"
            ],
            "contact_terms": [
                "email", "phone", "linkedin", "github", "contact", "reach",
                "schedule", "calendly", "meeting", "touch"
            ]
        }
    
    def _build_response_templates(self) -> Dict[QueryIntent, str]:
        """Build response templates for different intents."""
        return {
            QueryIntent.GREETING: "conversational",
            QueryIntent.CONTACT: "action_oriented", 
            QueryIntent.RESUME: "descriptive",
            QueryIntent.SKILLS: "descriptive",
            QueryIntent.EXPERIENCE: "descriptive",
            QueryIntent.PROJECTS: "descriptive",
            QueryIntent.EDUCATION: "descriptive",
            QueryIntent.CONVERSATIONAL: "conversational",
            QueryIntent.COMPLEX_ANALYSIS: "action_oriented"
        }
    
    async def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Lightning-fast query analysis and routing decision.
        Target: <100ms for simple queries.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Normalize query
        query_lower = query.lower().strip()
        
        # Fast regex-based intent detection
        intent = await self._detect_intent_fast(query_lower)
        
        # Extract keywords using fuzzy matching
        keywords = await self._extract_keywords_fuzzy(query_lower)
        
        # Determine complexity and routing
        complexity = self._assess_complexity(query_lower, keywords)
        routing = self._determine_routing(intent, complexity, keywords)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(intent, keywords, complexity)
        
        # Estimate response time
        estimated_time = self._estimate_response_time(routing, complexity)
        
        # Generate cache key for potential caching
        cache_key = self._generate_cache_key(query_lower, intent)
        
        # Suggest template
        suggested_template = self.response_templates.get(intent, "conversational")
        
        # Update performance stats
        processing_time = asyncio.get_event_loop().time() - start_time
        self._update_stats(processing_time)
        
        logger.info(f"⚡ Query analyzed in {processing_time*1000:.1f}ms: {intent.value} -> {routing.value}")
        
        return QueryAnalysis(
            intent=intent,
            routing=routing,
            confidence=confidence,
            keywords=keywords,
            complexity=complexity,
            estimated_response_time=estimated_time,
            cache_key=cache_key,
            suggested_template=suggested_template
        )
    
    async def _detect_intent_fast(self, query: str) -> QueryIntent:
        """Fast regex-based intent detection."""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        # Fallback to fuzzy matching for unmatched queries
        return await self._detect_intent_fuzzy(query)
    
    async def _detect_intent_fuzzy(self, query: str) -> QueryIntent:
        """Fuzzy matching fallback for intent detection."""
        # Create intent examples for fuzzy matching
        intent_examples = {
            QueryIntent.GREETING: ["hello", "hi there", "good morning"],
            QueryIntent.CONTACT: ["contact me", "email address", "get in touch"],
            QueryIntent.SKILLS: ["technical skills", "programming languages", "what technologies"],
            QueryIntent.EXPERIENCE: ["work experience", "professional background", "career history"],
            QueryIntent.PROJECTS: ["projects you built", "what have you created", "portfolio"],
            QueryIntent.EDUCATION: ["education background", "degree", "university"]
        }
        
        best_match = None
        best_score = 0
        
        for intent, examples in intent_examples.items():
            for example in examples:
                score = fuzz.partial_ratio(query, example)
                if score > best_score and score > 60:  # Threshold for fuzzy matching
                    best_score = score
                    best_match = intent
        
        return best_match if best_match else QueryIntent.CONVERSATIONAL
    
    async def _extract_keywords_fuzzy(self, query: str) -> List[str]:
        """Extract relevant keywords using fuzzy matching."""
        extracted_keywords = []
        
        for category, keywords in self.keyword_database.items():
            matches = process.extract(query, keywords, limit=3, score_cutoff=70)
            for match, score in matches:
                extracted_keywords.append(match)
        
        return list(set(extracted_keywords))  # Remove duplicates
    
    def _assess_complexity(self, query: str, keywords: List[str]) -> str:
        """Assess query complexity for routing decisions."""
        # Simple heuristics for complexity assessment
        word_count = len(query.split())
        keyword_count = len(keywords)
        
        if word_count <= 5 and keyword_count <= 2:
            return "simple"
        elif word_count <= 15 and keyword_count <= 5:
            return "medium"
        else:
            return "complex"
    
    def _determine_routing(self, intent: QueryIntent, complexity: str, keywords: List[str]) -> RoutingDecision:
        """Determine optimal routing strategy."""
        # Simple queries can bypass gRPC
        if complexity == "simple" and intent in [QueryIntent.GREETING, QueryIntent.CONTACT]:
            return RoutingDecision.DIRECT_GEMINI
        
        # Resume/skills queries often benefit from caching
        if intent in [QueryIntent.RESUME, QueryIntent.SKILLS, QueryIntent.EDUCATION]:
            return RoutingDecision.CACHE_LOOKUP
        
        # Complex queries need full pipeline
        if complexity == "complex" or intent == QueryIntent.COMPLEX_ANALYSIS:
            return RoutingDecision.ENHANCED_PIPELINE
        
        # Default to enhanced pipeline for medium complexity
        return RoutingDecision.ENHANCED_PIPELINE
    
    def _calculate_confidence(self, intent: QueryIntent, keywords: List[str], complexity: str) -> float:
        """Calculate confidence score for the analysis."""
        base_confidence = 0.7
        
        # Boost confidence for clear intent matches
        if intent != QueryIntent.CONVERSATIONAL and intent != QueryIntent.UNKNOWN:
            base_confidence += 0.2
        
        # Boost for relevant keywords
        keyword_boost = min(len(keywords) * 0.05, 0.2)
        
        # Adjust for complexity
        complexity_adjustment = {
            "simple": 0.1,
            "medium": 0.0,
            "complex": -0.1
        }.get(complexity, 0.0)
        
        return min(base_confidence + keyword_boost + complexity_adjustment, 1.0)
    
    def _estimate_response_time(self, routing: RoutingDecision, complexity: str) -> float:
        """Estimate response time based on routing and complexity."""
        base_times = {
            RoutingDecision.STATIC_RESPONSE: 0.1,
            RoutingDecision.DIRECT_GEMINI: 0.5,
            RoutingDecision.CACHE_LOOKUP: 0.3,
            RoutingDecision.ENHANCED_PIPELINE: 1.5
        }
        
        complexity_multiplier = {
            "simple": 0.8,
            "medium": 1.0,
            "complex": 1.3
        }.get(complexity, 1.0)
        
        return base_times.get(routing, 1.5) * complexity_multiplier
    
    def _generate_cache_key(self, query: str, intent: QueryIntent) -> str:
        """Generate cache key for query caching."""
        # Create a normalized cache key
        words = query.split()[:5]  # First 5 words
        normalized = "_".join(sorted(words))
        return f"{intent.value}_{normalized}"
    
    def _update_stats(self, processing_time: float):
        """Update performance statistics."""
        self._performance_stats["total_queries"] += 1
        current_avg = self._performance_stats["avg_time"]
        total = self._performance_stats["total_queries"]
        
        # Running average
        self._performance_stats["avg_time"] = (
            (current_avg * (total - 1)) + processing_time
        ) / total
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get preprocessing performance statistics."""
        return {
            **self._performance_stats,
            "avg_time_ms": self._performance_stats["avg_time"] * 1000
        }


# Factory function for easy initialization
async def create_preprocessor() -> EnhancedPreprocessor:
    """Create and initialize the enhanced preprocessor."""
    logger.info("🚀 Initializing Enhanced Preprocessing Layer...")
    preprocessor = EnhancedPreprocessor()
    logger.info("✅ Enhanced Preprocessing Layer ready")
    return preprocessor 