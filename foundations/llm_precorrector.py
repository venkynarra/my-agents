"""
LLM Pre-Corrector for AI Assistant
Enhances user queries with grammar fixes, keyword addition, and intent classification.
Target: <300ms processing time with improved query quality
"""
import asyncio
import logging
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from llama_index.llms.gemini import Gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Enhanced query intent classifications."""
    GREETING = "greeting"
    CONTACT_INFO = "contact_info"
    TECHNICAL_SKILLS = "technical_skills"
    WORK_EXPERIENCE = "work_experience"
    PROJECT_DETAILS = "project_details"
    EDUCATION = "education"
    CAREER_GOALS = "career_goals"
    HIRING_PITCH = "hiring_pitch"
    COMPANY_SPECIFIC = "company_specific"
    CONVERSATIONAL = "conversational"
    COMPLEX_ANALYSIS = "complex_analysis"

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"           # Direct, single concept
    MODERATE = "moderate"       # Multiple concepts, clear intent
    COMPLEX = "complex"         # Multiple intents, requires analysis
    AMBIGUOUS = "ambiguous"     # Unclear intent, needs clarification

@dataclass
class EnhancedQuery:
    """Enhanced query with corrections and metadata."""
    original_query: str
    corrected_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    added_keywords: List[str]
    confidence: float
    suggested_followups: List[str]
    processing_notes: str
    estimated_response_time: float

class LLMPreCorrector:
    """
    Advanced LLM-powered query enhancement system.
    Uses Gemini 1.5 Flash for fast query preprocessing and enhancement.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.llm: Optional[Gemini] = None
        self.keyword_database = self._build_keyword_database()
        self.intent_patterns = self._build_intent_patterns()
        self.performance_stats = {"total_queries": 0, "avg_time": 0.0}
        
    async def initialize(self) -> bool:
        """Initialize the LLM pre-corrector."""
        try:
            if not GEMINI_AVAILABLE:
                logger.error("❌ Gemini LLM not available")
                return False
                
            if not self.api_key:
                logger.error("❌ Gemini API key not found")
                return False
            
            logger.info("🚀 Initializing LLM Pre-Corrector...")
            
            # Initialize Gemini with optimized settings for speed
            self.llm = Gemini(
                model_name="gemini-1.5-flash",
                api_key=self.api_key,
                temperature=0.1,  # Low temperature for consistent corrections
                max_tokens=500    # Limit for faster responses
            )
            
            logger.info("✅ LLM Pre-Corrector initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM Pre-Corrector: {e}")
            return False
    
    def _build_keyword_database(self) -> Dict[str, List[str]]:
        """Build comprehensive keyword database for query enhancement."""
        return {
            "technical_skills": [
                "python", "java", "javascript", "typescript", "react", "angular",
                "fastapi", "django", "flask", "spring boot", "node.js", "express",
                "aws", "docker", "kubernetes", "ci/cd", "microservices",
                "ai", "ml", "machine learning", "tensorflow", "pytorch", "langchain",
                "llm", "rag", "retrieval augmented generation", "vector database",
                "postgresql", "mysql", "mongodb", "redis", "elasticsearch"
            ],
            "experience_keywords": [
                "software engineer", "developer", "full-stack", "backend", "frontend",
                "veritis group", "tcs", "tata consultancy", "virtusa", 
                "clinical api", "testing agent", "loan origination", "chat platform",
                "years of experience", "professional background", "career journey"
            ],
            "project_keywords": [
                "built", "developed", "created", "implemented", "designed", "architected",
                "reduced", "improved", "optimized", "achieved", "delivered",
                "performance", "scalability", "efficiency", "automation"
            ],
            "education_keywords": [
                "george mason university", "gmu", "computer science", "masters", "ms",
                "bachelor", "degree", "graduation", "academic", "coursework"
            ],
            "contact_keywords": [
                "email", "phone", "linkedin", "github", "contact", "reach out",
                "get in touch", "schedule", "meeting", "calendly", "connect"
            ]
        }
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build patterns for intent detection."""
        return {
            QueryIntent.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "nice to meet", "introduce yourself", "who are you"
            ],
            QueryIntent.CONTACT_INFO: [
                "contact", "email", "phone", "linkedin", "github", "reach",
                "schedule", "meeting", "calendly", "get in touch"
            ],
            QueryIntent.TECHNICAL_SKILLS: [
                "skills", "technical", "programming", "languages", "frameworks",
                "technologies", "tools", "expertise", "proficient", "stack"
            ],
            QueryIntent.WORK_EXPERIENCE: [
                "experience", "work", "job", "career", "employment", "professional",
                "companies", "roles", "positions", "background", "industry"
            ],
            QueryIntent.PROJECT_DETAILS: [
                "projects", "built", "developed", "created", "portfolio",
                "achievements", "accomplishments", "implemented", "designed"
            ],
            QueryIntent.EDUCATION: [
                "education", "degree", "university", "college", "graduation",
                "academic", "masters", "bachelor", "coursework"
            ],
            QueryIntent.HIRING_PITCH: [
                "why hire", "hire you", "best candidate", "why choose",
                "what makes you", "convince", "persuade", "right fit"
            ],
            QueryIntent.COMPANY_SPECIFIC: [
                "veritis", "tcs", "virtusa", "george mason", "gmu",
                "specific company", "tell me about", "experience at"
            ]
        }
    
    async def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance user query with corrections, keywords, and intent classification.
        
        Args:
            query: Original user query
            
        Returns:
            EnhancedQuery with all improvements
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Basic preprocessing
            cleaned_query = self._clean_query(query)
            
            # Step 2: Fast intent detection
            detected_intent = await self._detect_intent_hybrid(cleaned_query)
            
            # Step 3: LLM-powered enhancement
            enhanced_result = await self._llm_enhance_query(cleaned_query, detected_intent)
            
            # Step 4: Add relevant keywords
            added_keywords = self._add_context_keywords(enhanced_result["corrected_query"], detected_intent)
            
            # Step 5: Assess complexity
            complexity = self._assess_query_complexity(enhanced_result["corrected_query"], detected_intent)
            
            # Step 6: Generate follow-up suggestions
            followups = self._generate_followup_suggestions(detected_intent, enhanced_result["corrected_query"])
            
            # Step 7: Estimate processing time
            estimated_time = self._estimate_processing_time(complexity, detected_intent)
            
            # Create enhanced query object
            enhanced_query = EnhancedQuery(
                original_query=query,
                corrected_query=enhanced_result["corrected_query"],
                intent=detected_intent,
                complexity=complexity,
                added_keywords=added_keywords,
                confidence=enhanced_result["confidence"],
                suggested_followups=followups,
                processing_notes=enhanced_result["notes"],
                estimated_response_time=estimated_time
            )
            
            # Update performance stats
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_performance_stats(processing_time)
            
            logger.info(f"✨ Query enhanced in {processing_time:.1f}ms: {detected_intent.value}")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            # Return minimal enhancement on error
            return self._create_fallback_enhancement(query)
    
    def _clean_query(self, query: str) -> str:
        """Basic query cleaning and normalization."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Fix common typos without LLM
        typo_fixes = {
            r'\bteh\b': 'the',
            r'\byou\b': 'you',
            r'\bprogramming\b': 'programming',
            r'\bexperience\b': 'experience'
        }
        
        for pattern, replacement in typo_fixes.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    async def _detect_intent_hybrid(self, query: str) -> QueryIntent:
        """Fast hybrid intent detection using patterns + LLM."""
        query_lower = query.lower()
        
        # Pattern-based detection (fast)
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return intent
        
        # LLM-based detection for unclear cases
        try:
            intent_prompt = f"""
Classify this query intent in one word:
Query: "{query}"

Options: greeting, contact_info, technical_skills, work_experience, project_details, education, hiring_pitch, company_specific, conversational, complex_analysis

Intent:"""
            
            response = await self.llm.acomplete(intent_prompt)
            intent_str = str(response).strip().lower()
            
            # Map to enum
            intent_mapping = {
                "greeting": QueryIntent.GREETING,
                "contact_info": QueryIntent.CONTACT_INFO,
                "technical_skills": QueryIntent.TECHNICAL_SKILLS,
                "work_experience": QueryIntent.WORK_EXPERIENCE,
                "project_details": QueryIntent.PROJECT_DETAILS,
                "education": QueryIntent.EDUCATION,
                "hiring_pitch": QueryIntent.HIRING_PITCH,
                "company_specific": QueryIntent.COMPANY_SPECIFIC,
                "conversational": QueryIntent.CONVERSATIONAL,
                "complex_analysis": QueryIntent.COMPLEX_ANALYSIS
            }
            
            return intent_mapping.get(intent_str, QueryIntent.CONVERSATIONAL)
            
        except Exception as e:
            logger.warning(f"LLM intent detection failed: {e}")
            return QueryIntent.CONVERSATIONAL
    
    async def _llm_enhance_query(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """Use LLM to enhance query with corrections and improvements."""
        try:
            enhancement_prompt = f"""
You are a professional query enhancer for a software engineer's career assistant.

Task: Improve this query while preserving the original meaning.

Original Query: "{query}"
Detected Intent: {intent.value}

Instructions:
1. Fix grammar and spelling errors
2. Make the query more specific and professional
3. Add relevant context keywords if missing (but don't change the core question)
4. Keep it concise and clear

Response format (JSON):
{{
    "corrected_query": "enhanced version here",
    "confidence": 0.85,
    "notes": "brief explanation of changes"
}}

Response:"""
            
            response = await self.llm.acomplete(enhancement_prompt)
            response_text = str(response).strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(response_text)
                return {
                    "corrected_query": result.get("corrected_query", query),
                    "confidence": float(result.get("confidence", 0.7)),
                    "notes": result.get("notes", "Enhanced by LLM")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "corrected_query": response_text.replace('"', '').strip(),
                    "confidence": 0.6,
                    "notes": "Basic LLM enhancement"
                }
                
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return {
                "corrected_query": query,
                "confidence": 0.5,
                "notes": "No enhancement applied"
            }
    
    def _add_context_keywords(self, query: str, intent: QueryIntent) -> List[str]:
        """Add relevant context keywords based on intent."""
        added_keywords = []
        query_lower = query.lower()
        
        # Intent-specific keyword additions
        intent_keywords = {
            QueryIntent.TECHNICAL_SKILLS: self.keyword_database["technical_skills"],
            QueryIntent.WORK_EXPERIENCE: self.keyword_database["experience_keywords"],
            QueryIntent.PROJECT_DETAILS: self.keyword_database["project_keywords"],
            QueryIntent.EDUCATION: self.keyword_database["education_keywords"],
            QueryIntent.CONTACT_INFO: self.keyword_database["contact_keywords"]
        }
        
        relevant_keywords = intent_keywords.get(intent, [])
        
        # Add keywords that are contextually relevant but missing
        for keyword in relevant_keywords:
            if keyword not in query_lower and len(added_keywords) < 3:
                # Use simple relevance heuristics
                if self._is_keyword_relevant(query_lower, keyword, intent):
                    added_keywords.append(keyword)
        
        return added_keywords
    
    def _is_keyword_relevant(self, query: str, keyword: str, intent: QueryIntent) -> bool:
        """Determine if a keyword is relevant to add to the query."""
        # Simple relevance scoring based on query content
        relevance_indicators = {
            QueryIntent.TECHNICAL_SKILLS: ["skill", "tech", "program", "develop"],
            QueryIntent.WORK_EXPERIENCE: ["work", "job", "experience", "career"],
            QueryIntent.PROJECT_DETAILS: ["project", "built", "create", "develop"]
        }
        
        indicators = relevance_indicators.get(intent, [])
        return any(indicator in query for indicator in indicators)
    
    def _assess_query_complexity(self, query: str, intent: QueryIntent) -> QueryComplexity:
        """Assess query complexity for processing optimization."""
        word_count = len(query.split())
        question_marks = query.count('?')
        compound_indicators = len(re.findall(r'\b(and|or|but|however|also|additionally)\b', query, re.IGNORECASE))
        
        # Complexity scoring
        complexity_score = 0
        
        if word_count <= 5:
            complexity_score += 1
        elif word_count <= 15:
            complexity_score += 2
        else:
            complexity_score += 3
        
        if question_marks > 1:
            complexity_score += 1
        
        if compound_indicators > 0:
            complexity_score += 1
        
        # Intent-based complexity adjustment
        complex_intents = [QueryIntent.HIRING_PITCH, QueryIntent.COMPLEX_ANALYSIS, QueryIntent.CAREER_GOALS]
        if intent in complex_intents:
            complexity_score += 1
        
        # Map score to complexity
        if complexity_score <= 2:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 4:
            return QueryComplexity.MODERATE
        elif complexity_score <= 6:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.AMBIGUOUS
    
    def _generate_followup_suggestions(self, intent: QueryIntent, query: str) -> List[str]:
        """Generate relevant follow-up question suggestions."""
        followup_templates = {
            QueryIntent.TECHNICAL_SKILLS: [
                "What specific frameworks have you used?",
                "How many years of experience do you have with each technology?",
                "Can you describe a challenging technical problem you solved?"
            ],
            QueryIntent.WORK_EXPERIENCE: [
                "What was your biggest achievement at each company?",
                "How did you grow professionally in each role?",
                "What technologies did you learn at each position?"
            ],
            QueryIntent.PROJECT_DETAILS: [
                "What challenges did you face in this project?",
                "What technologies did you use for implementation?",
                "What was the impact of this project on the business?"
            ],
            QueryIntent.HIRING_PITCH: [
                "What specific role are you interested in?",
                "What value can you bring to our team?",
                "What are your salary expectations?"
            ]
        }
        
        return followup_templates.get(intent, [
            "Can you tell me more about that?",
            "What specific aspects interest you most?",
            "Would you like additional details?"
        ])[:3]  # Limit to 3 suggestions
    
    def _estimate_processing_time(self, complexity: QueryComplexity, intent: QueryIntent) -> float:
        """Estimate total processing time for the enhanced query."""
        base_times = {
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MODERATE: 1.2,
            QueryComplexity.COMPLEX: 2.0,
            QueryComplexity.AMBIGUOUS: 2.5
        }
        
        intent_multipliers = {
            QueryIntent.GREETING: 0.5,
            QueryIntent.CONTACT_INFO: 0.3,
            QueryIntent.TECHNICAL_SKILLS: 1.0,
            QueryIntent.WORK_EXPERIENCE: 1.2,
            QueryIntent.PROJECT_DETAILS: 1.3,
            QueryIntent.HIRING_PITCH: 1.5,
            QueryIntent.COMPLEX_ANALYSIS: 2.0
        }
        
        base_time = base_times.get(complexity, 1.5)
        multiplier = intent_multipliers.get(intent, 1.0)
        
        return base_time * multiplier
    
    def _create_fallback_enhancement(self, query: str) -> EnhancedQuery:
        """Create minimal enhancement when errors occur."""
        return EnhancedQuery(
            original_query=query,
            corrected_query=query,
            intent=QueryIntent.CONVERSATIONAL,
            complexity=QueryComplexity.MODERATE,
            added_keywords=[],
            confidence=0.5,
            suggested_followups=["Can you tell me more about that?"],
            processing_notes="Fallback enhancement - no LLM processing",
            estimated_response_time=1.0
        )
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics."""
        self.performance_stats["total_queries"] += 1
        current_avg = self.performance_stats["avg_time"]
        total = self.performance_stats["total_queries"]
        
        self.performance_stats["avg_time"] = (
            (current_avg * (total - 1)) + processing_time
        ) / total
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get pre-corrector performance statistics."""
        return {
            "total_queries_processed": self.performance_stats["total_queries"],
            "average_processing_time_ms": self.performance_stats["avg_time"],
            "llm_available": self.llm is not None,
            "keyword_categories": len(self.keyword_database),
            "total_keywords": sum(len(keywords) for keywords in self.keyword_database.values())
        }


# Factory function
async def create_llm_precorrector(api_key: Optional[str] = None) -> Optional[LLMPreCorrector]:
    """Create and initialize the LLM pre-corrector."""
    logger.info("🚀 Creating LLM Pre-Corrector...")
    
    precorrector = LLMPreCorrector(api_key)
    success = await precorrector.initialize()
    
    if success:
        logger.info("✅ LLM Pre-Corrector ready")
        return precorrector
    else:
        logger.error("❌ Failed to initialize LLM Pre-Corrector")
        return None 