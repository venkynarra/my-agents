"""
Optimized Prompt Template System for AI Assistant
Provides intent-based template selection with conversational, descriptive, and action-oriented formats.
Target: Minimal prompt size with maximum effectiveness
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class PromptTemplate(Enum):
    """Available prompt templates."""
    CONVERSATIONAL = "conversational"
    DESCRIPTIVE = "descriptive" 
    ACTION_ORIENTED = "action_oriented"

class ResponseStyle(Enum):
    """Response style preferences."""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    STORYTELLING = "storytelling"

@dataclass
class PromptConfig:
    """Configuration for prompt optimization."""
    template: PromptTemplate
    style: ResponseStyle
    max_tokens: int
    include_examples: bool
    include_context: bool
    include_metrics: bool
    word_limit: Optional[int] = None

@dataclass
class OptimizedPrompt:
    """Optimized prompt with metadata."""
    prompt: str
    template_used: PromptTemplate
    style_used: ResponseStyle
    estimated_tokens: int
    context_included: List[str]
    optimization_notes: str

class PromptOptimizer:
    """
    Intelligent prompt template system that optimizes prompts based on query intent.
    Reduces bloat while maximizing response quality.
    """
    
    def __init__(self):
        self.templates = self._build_templates()
        self.style_modifiers = self._build_style_modifiers()
        self.context_banks = self._build_context_banks()
        self.optimization_stats = {"total_prompts": 0, "avg_tokens": 0}
        
    def _build_templates(self) -> Dict[PromptTemplate, str]:
        """Build optimized prompt templates."""
        return {
            PromptTemplate.CONVERSATIONAL: """You are Venkatesh Narra, a friendly and approachable Software Development Engineer. You're having a natural conversation with {audience_type}.

{context_section}

Question: {query}

Respond naturally and conversationally, as if you're talking to a colleague. Keep it engaging and personal while being informative.{style_guidance}{word_limit}

Response:""",

            PromptTemplate.DESCRIPTIVE: """You are Venkatesh Narra, a skilled Software Development Engineer with expertise in {primary_expertise}. Provide a comprehensive and detailed response.

{context_section}

Query: {query}

Provide a detailed, well-structured response that covers all relevant aspects. Include specific examples, metrics, and technical details where appropriate.{style_guidance}{word_limit}

Response:""",

            PromptTemplate.ACTION_ORIENTED: """You are Venkatesh Narra, a results-driven Software Development Engineer. Focus on actionable insights and clear outcomes.

{context_section}

Objective: {query}

Provide a focused, action-oriented response that emphasizes achievements, impacts, and concrete value. Be direct and compelling.{style_guidance}{word_limit}

Response:"""
        }
    
    def _build_style_modifiers(self) -> Dict[ResponseStyle, str]:
        """Build style-specific guidance."""
        return {
            ResponseStyle.CASUAL: " Use a relaxed, friendly tone with natural language.",
            ResponseStyle.PROFESSIONAL: " Maintain a professional, polished tone suitable for business contexts.",
            ResponseStyle.TECHNICAL: " Focus on technical details, methodologies, and implementation specifics.",
            ResponseStyle.STORYTELLING: " Tell engaging stories with specific scenarios and outcomes."
        }
    
    def _build_context_banks(self) -> Dict[str, Dict[str, str]]:
        """Build optimized context banks for different scenarios."""
        return {
            "technical_skills": {
                "brief": """Technical Expertise:
- Programming: Python, Java, JavaScript, TypeScript, C++
- Frameworks: FastAPI, Django, React, Angular, Spring Boot
- Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
- AI/ML: TensorFlow, PyTorch, LangChain, RAG systems""",
                
                "detailed": """Technical Expertise & Experience:
- **Programming Languages**: Python (4+ years), Java (3+ years), JavaScript/TypeScript (3+ years), C++
- **Backend Frameworks**: FastAPI, Django, Flask, Spring Boot - built APIs handling 25,000+ daily requests
- **Frontend Technologies**: React, Angular, Node.js - created responsive, scalable user interfaces
- **Cloud & DevOps**: AWS (EC2, S3, Lambda, SQS), Docker, Kubernetes - architected scalable infrastructure
- **AI/ML Stack**: TensorFlow, PyTorch, LangChain, LlamaIndex, RAG systems - integrated AI into production systems
- **Databases**: PostgreSQL, MySQL, MongoDB, Vector DBs (FAISS, Chroma) - optimized for performance"""
            },
            
            "experience": {
                "brief": """Professional Background:
- Current: Software Development Engineer at Veritis Group Inc (Jan 2023 - Present)
- Previous: Full-Stack Developer at TCS (2021-2022), Junior Engineer at Virtusa (2020-2021)
- 4+ years developing scalable systems with measurable business impact""",
                
                "detailed": """Professional Journey:

**Veritis Group Inc - Software Development Engineer (Jan 2023 - Present)**
- Built AI-powered testing agent reducing manual QA by 60%
- Developed clinical APIs handling 25,000+ daily inferences with <200ms latency
- Created multi-modal chat platform with RAG implementation
- Optimized healthcare systems achieving significant performance improvements

**TCS - Full-Stack Developer (Feb 2021 - Jun 2022)**
- Built loan origination platform reducing approval time by 40%
- Automated document processing and risk scoring for major banking client
- Migrated legacy systems to AWS-based microservices architecture

**Virtusa - Junior Software Engineer (May 2020 - Jan 2021)**
- Developed responsive web applications using React and Node.js
- Participated in agile development and CI/CD implementation
- Focused on database optimization and performance tuning"""
            },
            
            "projects": {
                "brief": """Key Projects:
- AI Testing Agent: 60% reduction in manual QA
- Clinical API: 25,000+ daily inferences, <200ms latency
- Loan Platform: 40% faster approval process
- Chat Platform: Multi-modal with RAG implementation""",
                
                "detailed": """Signature Projects & Achievements:

**AI-Powered Testing Agent**
- **Impact**: Reduced manual QA efforts by 60%
- **Technologies**: Python, Google Gemini, OpenAI, automation frameworks
- **Innovation**: Automated test generation using AI for comprehensive API validation

**Clinical Decision Support Tool**
- **Scale**: 25,000+ daily inferences with <200ms latency
- **Technologies**: FastAPI, AWS, real-time processing, HIPAA compliance
- **Impact**: Reduced diagnostic errors by 15% for healthcare providers

**Loan Origination Platform (TCS)**
- **Business Impact**: 40% reduction in loan approval time
- **Technologies**: Python, Machine Learning, AWS, document processing
- **Innovation**: ML-powered risk scoring with automated document processing

**Multi-Modal Chat Platform**
- **Technical Achievement**: RAG implementation with intelligent fallback
- **Technologies**: LangChain, Vector DBs, NLP, multiple LLMs (Gemini, Cohere, Mistral)
- **Innovation**: Advanced conversational AI with knowledge retrieval capabilities"""
            },
            
            "education": {
                "brief": """Education: MS Computer Science - George Mason University (2022-2024)
Previous: B.Tech Computer Science - GITAM University (2018-2022)""",
                
                "detailed": """Educational Background:

**Master of Science in Computer Science**
George Mason University (2022-2024)
- Specialization: AI/ML, Distributed Systems, Software Engineering
- Relevant Coursework: Machine Learning, Cloud Computing, System Design

**Bachelor of Technology in Computer Science**
GITAM University (2018-2022)
- Graduated with Honors
- Foundation in algorithms, data structures, software development"""
            }
        }
    
    def optimize_prompt(self, 
                       query: str,
                       intent: str,
                       context_categories: List[str],
                       audience_type: str = "potential employer",
                       max_tokens: int = 1000,
                       include_metrics: bool = True) -> OptimizedPrompt:
        """
        Create optimized prompt based on query intent and requirements.
        
        Args:
            query: User's question
            intent: Detected query intent
            context_categories: Relevant context categories to include
            audience_type: Who the user is (employer, colleague, etc.)
            max_tokens: Maximum tokens for the prompt
            include_metrics: Whether to include specific metrics
        """
        try:
            # Select optimal template and style
            template, style = self._select_template_and_style(intent, query)
            
            # Build context section
            context_section = self._build_context_section(context_categories, max_tokens, include_metrics)
            
            # Determine primary expertise area
            primary_expertise = self._determine_primary_expertise(context_categories, query)
            
            # Build style guidance
            style_guidance = self.style_modifiers.get(style, "")
            
            # Determine word limit
            word_limit = self._calculate_word_limit(max_tokens, template)
            word_limit_text = f" Keep response under {word_limit} words." if word_limit else ""
            
            # Build the optimized prompt
            template_str = self.templates[template]
            optimized_prompt = template_str.format(
                query=query,
                context_section=context_section,
                audience_type=audience_type,
                primary_expertise=primary_expertise,
                style_guidance=style_guidance,
                word_limit=word_limit_text
            )
            
            # Estimate token count
            estimated_tokens = self._estimate_tokens(optimized_prompt)
            
            # Update stats
            self._update_optimization_stats(estimated_tokens)
            
            optimization_notes = f"Template: {template.value}, Style: {style.value}, Context: {len(context_categories)} categories"
            
            logger.info(f"🎯 Prompt optimized: {template.value} template, ~{estimated_tokens} tokens")
            
            return OptimizedPrompt(
                prompt=optimized_prompt,
                template_used=template,
                style_used=style,
                estimated_tokens=estimated_tokens,
                context_included=context_categories,
                optimization_notes=optimization_notes
            )
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {e}")
            return self._create_fallback_prompt(query, context_categories)
    
    def _select_template_and_style(self, intent: str, query: str) -> Tuple[PromptTemplate, ResponseStyle]:
        """Select optimal template and style based on intent and query characteristics."""
        intent_lower = intent.lower()
        query_lower = query.lower()
        
        # Template selection logic
        if any(word in intent_lower for word in ['greeting', 'conversational', 'casual']):
            template = PromptTemplate.CONVERSATIONAL
        elif any(word in intent_lower for word in ['hiring_pitch', 'why_hire', 'convince']):
            template = PromptTemplate.ACTION_ORIENTED
        elif any(word in intent_lower for word in ['technical', 'skills', 'projects', 'experience', 'education']):
            template = PromptTemplate.DESCRIPTIVE
        else:
            # Default based on query characteristics
            if len(query.split()) <= 8:
                template = PromptTemplate.CONVERSATIONAL
            elif any(word in query_lower for word in ['why', 'how', 'what makes', 'convince']):
                template = PromptTemplate.ACTION_ORIENTED
            else:
                template = PromptTemplate.DESCRIPTIVE
        
        # Style selection logic
        if any(word in query_lower for word in ['technical', 'implement', 'architecture', 'framework']):
            style = ResponseStyle.TECHNICAL
        elif any(word in query_lower for word in ['story', 'example', 'time when', 'experience with']):
            style = ResponseStyle.STORYTELLING
        elif any(word in query_lower for word in ['professional', 'formal', 'business']):
            style = ResponseStyle.PROFESSIONAL
        else:
            style = ResponseStyle.PROFESSIONAL  # Default professional tone
        
        return template, style
    
    def _build_context_section(self, categories: List[str], max_tokens: int, include_metrics: bool) -> str:
        """Build optimized context section."""
        if not categories:
            return ""
        
        context_parts = []
        available_tokens = max_tokens * 0.4  # Reserve 40% of tokens for context
        
        # Prioritize categories based on relevance
        prioritized_categories = self._prioritize_categories(categories)
        
        for category in prioritized_categories:
            if category in self.context_banks:
                # Choose brief or detailed based on available space
                context_type = "detailed" if available_tokens > 300 else "brief"
                
                if context_type in self.context_banks[category]:
                    context_content = self.context_banks[category][context_type]
                    
                    # Filter metrics if not needed
                    if not include_metrics:
                        context_content = self._remove_metrics(context_content)
                    
                    context_parts.append(context_content)
                    available_tokens -= self._estimate_tokens(context_content)
                    
                    if available_tokens <= 100:  # Stop if running low on tokens
                        break
        
        return "\n\n".join(context_parts)
    
    def _prioritize_categories(self, categories: List[str]) -> List[str]:
        """Prioritize context categories by importance."""
        priority_order = [
            "technical_skills",
            "experience", 
            "projects",
            "education"
        ]
        
        # Sort categories by priority
        prioritized = []
        for priority_cat in priority_order:
            if priority_cat in categories:
                prioritized.append(priority_cat)
        
        # Add any remaining categories
        for cat in categories:
            if cat not in prioritized:
                prioritized.append(cat)
        
        return prioritized
    
    def _determine_primary_expertise(self, categories: List[str], query: str) -> str:
        """Determine primary expertise area for the prompt."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['ai', 'ml', 'machine learning', 'artificial intelligence']):
            return "AI/ML and Software Development"
        elif any(word in query_lower for word in ['cloud', 'aws', 'docker', 'kubernetes']):
            return "Cloud Architecture and DevOps"
        elif any(word in query_lower for word in ['full-stack', 'fullstack', 'backend', 'frontend']):
            return "Full-Stack Development"
        elif "technical_skills" in categories:
            return "Software Engineering and AI/ML"
        else:
            return "Software Development"
    
    def _calculate_word_limit(self, max_tokens: int, template: PromptTemplate) -> Optional[int]:
        """Calculate appropriate word limit for response."""
        # Rough conversion: 1 token ≈ 0.75 words
        available_tokens = max_tokens * 0.6  # Reserve 60% for response
        word_limit = int(available_tokens * 0.75)
        
        # Template-specific adjustments
        if template == PromptTemplate.CONVERSATIONAL:
            return min(word_limit, 200)  # Keep conversational responses concise
        elif template == PromptTemplate.ACTION_ORIENTED:
            return min(word_limit, 250)  # Action-oriented can be slightly longer
        else:  # DESCRIPTIVE
            return min(word_limit, 300)  # Descriptive responses can be longest
    
    def _remove_metrics(self, content: str) -> str:
        """Remove specific metrics from content if not needed."""
        # Remove patterns like "25,000+", "60%", "<200ms", etc.
        metric_patterns = [
            r'\d+,?\d*\+?\s*(daily|requests|inferences|ms|seconds)',
            r'\d+%\s*(reduction|improvement|increase)',
            r'<\d+ms',
            r'\d+\+?\s*years?'
        ]
        
        cleaned_content = content
        for pattern in metric_patterns:
            cleaned_content = re.sub(pattern, '[specific metrics available]', cleaned_content, flags=re.IGNORECASE)
        
        return cleaned_content
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _create_fallback_prompt(self, query: str, context_categories: List[str]) -> OptimizedPrompt:
        """Create a basic fallback prompt when optimization fails."""
        fallback_prompt = f"""You are Venkatesh Narra, a Software Development Engineer.

Question: {query}

Please provide a helpful and professional response based on your background in software development, AI/ML, and cloud technologies.

Response:"""
        
        return OptimizedPrompt(
            prompt=fallback_prompt,
            template_used=PromptTemplate.CONVERSATIONAL,
            style_used=ResponseStyle.PROFESSIONAL,
            estimated_tokens=self._estimate_tokens(fallback_prompt),
            context_included=context_categories,
            optimization_notes="Fallback prompt - optimization failed"
        )
    
    def _update_optimization_stats(self, tokens: int):
        """Update optimization statistics."""
        self.optimization_stats["total_prompts"] += 1
        current_avg = self.optimization_stats["avg_tokens"]
        total = self.optimization_stats["total_prompts"]
        
        self.optimization_stats["avg_tokens"] = (
            (current_avg * (total - 1)) + tokens
        ) / total
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get prompt optimization statistics."""
        return {
            "total_prompts_optimized": self.optimization_stats["total_prompts"],
            "average_token_count": self.optimization_stats["avg_tokens"],
            "available_templates": len(self.templates),
            "available_styles": len(self.style_modifiers),
            "context_banks": list(self.context_banks.keys())
        }
    
    def get_template_preview(self, template: PromptTemplate) -> str:
        """Get a preview of a specific template."""
        return self.templates.get(template, "Template not found")


# Utility functions
def create_prompt_config(template: str = "auto",
                        style: str = "professional", 
                        max_tokens: int = 1000,
                        include_examples: bool = True,
                        include_context: bool = True,
                        include_metrics: bool = True,
                        word_limit: Optional[int] = None) -> PromptConfig:
    """Create a prompt configuration with validation."""
    
    # Map string inputs to enums
    template_mapping = {
        "conversational": PromptTemplate.CONVERSATIONAL,
        "descriptive": PromptTemplate.DESCRIPTIVE,
        "action_oriented": PromptTemplate.ACTION_ORIENTED,
        "auto": PromptTemplate.DESCRIPTIVE  # Default
    }
    
    style_mapping = {
        "casual": ResponseStyle.CASUAL,
        "professional": ResponseStyle.PROFESSIONAL,
        "technical": ResponseStyle.TECHNICAL,
        "storytelling": ResponseStyle.STORYTELLING
    }
    
    return PromptConfig(
        template=template_mapping.get(template, PromptTemplate.DESCRIPTIVE),
        style=style_mapping.get(style, ResponseStyle.PROFESSIONAL),
        max_tokens=max_tokens,
        include_examples=include_examples,
        include_context=include_context,
        include_metrics=include_metrics,
        word_limit=word_limit
    )


# Factory function
def create_prompt_optimizer() -> PromptOptimizer:
    """Create and initialize the prompt optimizer."""
    logger.info("🚀 Creating Prompt Optimizer...")
    optimizer = PromptOptimizer()
    logger.info("✅ Prompt Optimizer ready")
    return optimizer 