"""
Multi-Agent Enhanced Smart Router for Career Assistant - Production Ready
Combines MCP Agent + Gemini LLM Agent + Smart Router Agent with bulletproof fallbacks
Ensures excellent responses whether MCP works or not
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple, List
from llama_index.llms.gemini import Gemini
from .config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class MultiAgentSmartRouter:
    """
    Production-ready Multi-Agent system with bulletproof fallbacks:
    1. MCP Agent - For data retrieval (when available)
    2. Enhanced Static Knowledge - Rich fallback when MCP fails
    3. Gemini LLM Agent - For intelligent processing 
    4. Smart Router Agent - For orchestration and optimization
    """
    
    def __init__(self, mcp_client=None):
        self.llm = Gemini(model_name="gemini-1.5-flash", api_key=GEMINI_API_KEY)
        self.mcp_client = mcp_client
        self.mcp_agent_ready = False
        self.response_cache = {}
        
        # Contact info
        self.contact_info = {
            "email": "vnarrag@gmu.edu",
            "phone": "+1 703-453-2157",
            "linkedin": "https://www.linkedin.com/in/venkateswara-narra-91170b34a",
            "github": "https://github.com/venkynarra",
            "leetcode": "https://leetcode.com/u/pravnarri/",
            "calendly": "https://calendly.com/venkateshnarra368"
        }
        
        # Enhanced static knowledge base (as good as MCP tools)
        self.enhanced_knowledge = self._build_enhanced_knowledge()
        
        # Initialize MCP Agent (but don't fail if it doesn't work)
        if self.mcp_client:
            asyncio.create_task(self._try_initialize_mcp())
    
    def _build_enhanced_knowledge(self) -> Dict[str, Any]:
        """Build comprehensive static knowledge base that rivals MCP tools."""
        return {
            "technical_skills": {
                "languages": ["Python", "Java", "JavaScript", "TypeScript", "C++"],
                "frameworks": ["FastAPI", "Django", "Flask", "Spring Boot", "React", "Angular", "Node.js", "Express.js"],
                "cloud": ["AWS (EC2, S3, Lambda, SQS)", "Docker", "Kubernetes", "CI/CD"],
                "ai_ml": ["TensorFlow", "PyTorch", "LangChain", "LlamaIndex", "RAG systems", "Scikit-learn"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "Vector DBs (FAISS, Chroma)"],
                "tools": ["Git", "GitHub", "GitLab", "JIRA", "Postman", "VS Code", "PyCharm"]
            },
            "experience": {
                "current_role": {
                    "title": "Software Development Engineer",
                    "company": "Veritis Group Inc", 
                    "duration": "Jan 2023 - Present",
                    "achievements": [
                        "Spearheaded clinical API development handling 25,000+ daily inferences",
                        "Built AI-powered testing agent reducing manual QA by 60%",
                        "Optimized system performance achieving <200ms response times",
                        "Developed multi-modal AI chat platform with RAG implementation",
                        "Created clinical decision support tool reducing diagnostic errors by 15%"
                    ]
                },
                "previous_roles": [
                    {
                        "company": "TCS (Tata Consultancy Services)", 
                        "duration": "Feb 2021 - Jun 2022", 
                        "title": "Full-Stack Developer",
                        "achievements": [
                            "Built loan origination platform reducing approval time by 40%",
                            "Automated document processing and risk scoring for major bank",
                            "Migrated legacy systems to AWS-based microservices",
                            "Resolved data pipeline bottlenecks saving 4 hours per run",
                            "Improved system reliability and scalability"
                        ]
                    },
                    {
                        "company": "Virtusa", 
                        "duration": "May 2020 - Jan 2021", 
                        "title": "Junior Software Engineer",
                        "achievements": [
                            "Developed responsive web applications using React and Node.js",
                            "Worked on database optimization and performance tuning",
                            "Participated in agile development and CI/CD implementation",
                            "Built REST APIs and integrated third-party services",
                            "Contributed to code reviews and quality assurance processes"
                        ]
                    }
                ]
            },
            "projects": [
                {
                    "name": "AI Testing Agent",
                    "impact": "Reduced manual QA by 60%",
                    "technologies": ["Python", "AI/ML", "Google Gemini", "OpenAI", "automation frameworks"],
                    "description": "Automated test generation using AI for API validation with real-time analytics"
                },
                {
                    "name": "Clinical Decision Support Tool", 
                    "impact": "25,000+ daily inferences with <200ms latency",
                    "technologies": ["FastAPI", "AWS", "real-time processing", "HIPAA compliance"],
                    "description": "High-performance medical decision support system for healthcare providers"
                },
                {
                    "name": "Multi-modal Chat Platform",
                    "impact": "RAG implementation with intelligent fallback",
                    "technologies": ["LangChain", "Vector DBs", "NLP", "Gemini", "Cohere", "Mistral"],
                    "description": "Advanced conversational AI supporting multiple LLMs with knowledge retrieval"
                },
                {
                    "name": "Loan Origination System (TCS)",
                    "impact": "Cut approval time by 40%",
                    "technologies": ["Python", "Machine Learning", "AWS", "APIs", "document processing"],
                    "description": "ML-powered loan processing optimization with automated risk scoring"
                }
            ],
            "education": {
                "masters": {
                    "degree": "MS Computer Science",
                    "university": "George Mason University",
                    "duration": "2022-2024",
                    "focus": "AI/ML, distributed systems, software engineering"
                },
                "bachelors": {
                    "degree": "B.Tech Computer Science", 
                    "university": "GITAM University",
                    "duration": "2018-2022",
                    "achievements": "Graduated with honors"
                }
            }
        }
    
    async def _try_initialize_mcp(self):
        """Try to initialize MCP Agent - but don't fail if it doesn't work."""
        try:
            logger.info("🤖 Attempting MCP Agent initialization...")
            
            tools = await asyncio.wait_for(
                self.mcp_client.list_tools(), 
                timeout=3.0
            )
            
            if tools:
                self.mcp_agent_ready = True
                logger.info(f"✅ MCP Agent ready with {len(tools)} tools")
            else:
                logger.info("ℹ️ MCP Agent tools list empty - using enhanced static knowledge")
                
        except Exception as e:
            logger.info(f"ℹ️ MCP Agent unavailable ({e}) - using enhanced static knowledge")
    
    def _should_include_contact(self, query: str) -> bool:
        """Strict contact guard - only include when explicitly requested."""
        query_lower = query.lower()
        
        contact_triggers = [
            'contact', 'reach', 'email', 'phone', 'linkedin', 'github', 
            'calendly', 'schedule', 'hire me', 'linkedin url', 'github url',
            'about you', 'who are you', 'introduce yourself', 'tell me about yourself'
        ]
        
        return any(trigger in query_lower for trigger in contact_triggers)
    
    def _classify_query_intelligent(self, query: str) -> Tuple[str, str]:
        """
        Intelligent query classification:
        Returns: (category, strategy)
        
        Strategies:
        - 'static' - Fast static responses
        - 'enhanced' - Enhanced static knowledge + LLM
        - 'mcp_enhanced' - MCP + LLM (when MCP available)
        """
        query_lower = query.lower()
        
        # Contact queries - always static
        if any(word in query_lower for word in ['contact', 'linkedin url', 'github url', 'email', 'phone']):
            return 'contact', 'static'
            
        # Complex analytical queries - use best available strategy
        elif any(phrase in query_lower for phrase in [
            'why hire', 'best reason', 'summarize', 'tell me about', 'explain', 'describe',
            'how do you', 'what makes you', 'your approach', 'your experience with'
        ]):
            strategy = 'mcp_enhanced' if self.mcp_agent_ready else 'enhanced'
            return self._get_category_from_query(query), strategy
            
        # Technical and factual queries - use enhanced knowledge
        elif any(word in query_lower for word in ['skills', 'experience', 'projects', 'education', 'cloud', 'ai', 'ml']):
            strategy = 'mcp_enhanced' if self.mcp_agent_ready else 'enhanced'
            return self._get_category_from_query(query), strategy
            
        # General queries - use enhanced strategy
        else:
            strategy = 'enhanced'
            return 'general', strategy
    
    def _get_category_from_query(self, query: str) -> str:
        """Extract category from query for targeted processing."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['skills', 'technical', 'programming', 'languages']):
            return 'skills'
        elif any(word in query_lower for word in ['cloud', 'aws', 'docker', 'kubernetes']):
            return 'cloud'
        elif any(word in query_lower for word in ['ai', 'ml', 'machine learning', 'tensorflow']):
            return 'ai_ml'
        elif any(word in query_lower for word in ['experience', 'work', 'job', 'company']):
            return 'experience'
        elif any(word in query_lower for word in ['education', 'degree', 'university']):
            return 'education'
        elif any(word in query_lower for word in ['hire', 'why', 'about', 'introduce']):
            return 'about'
        elif any(word in query_lower for word in ['role', 'position', 'career', 'looking for']):
            return 'career'
        else:
            return 'general'
    
    async def _get_enhanced_knowledge_response(self, query: str, category: str) -> str:
        """Get rich response using enhanced static knowledge + LLM."""
        try:
            # Get relevant knowledge for category
            context = self._get_knowledge_context(category)
            
            # Include contact info if needed
            include_contact = self._should_include_contact(query)
            contact_context = ""
            if include_contact:
                contact_context = f"""
Contact Information:
- Email: {self.contact_info['email']}
- Phone: {self.contact_info['phone']}
- LinkedIn: {self.contact_info['linkedin']}
- GitHub: {self.contact_info['github']}
- Calendly: {self.contact_info['calendly']}
"""
            
            # Build comprehensive prompt
            prompt = f"""You are Venkatesh Narra, a Software Development Engineer. Answer this query professionally and comprehensively:

Query: "{query}"

Use this detailed information about me:
{context}

{contact_context}

Instructions:
1. Provide a comprehensive, engaging response
2. Use specific metrics and achievements when relevant  
3. Be personable while maintaining professionalism
4. Incorporate relevant details from the context above
5. Keep response 150-250 words unless more detail is requested
6. Include contact information only if the query requests it

Response:"""
            
            # Generate response with Gemini
            response = await self.llm.acomplete(prompt)
            return str(response)
            
        except Exception as e:
            logger.error(f"Enhanced knowledge response failed: {e}")
            return self._get_static_response(query, category)
    
    def _get_knowledge_context(self, category: str) -> str:
        """Get relevant knowledge context for a category."""
        knowledge = self.enhanced_knowledge
        
        if category == 'skills':
            skills = knowledge['technical_skills']
            return f"""Technical Skills:
- Programming Languages: {', '.join(skills['languages'])}
- Frameworks: {', '.join(skills['frameworks'])}
- Cloud & DevOps: {', '.join(skills['cloud'])}
- AI/ML: {', '.join(skills['ai_ml'])}
- Databases: {', '.join(skills['databases'])}
- Tools: {', '.join(skills['tools'])}

Key Achievements:
- Built clinical APIs handling 25,000+ daily inferences with <200ms latency
- Created AI testing agent reducing manual QA by 60%
- Architected scalable cloud solutions with high availability
- Delivered loan origination platform cutting approval time by 40%"""
        
        elif category == 'experience':
            exp = knowledge['experience']
            current = exp['current_role']
            
            previous_details = []
            for role in exp['previous_roles']:
                role_details = f"**{role['title']} at {role['company']} ({role['duration']})**"
                if 'achievements' in role:
                    achievements = "\n".join(f"  - {achievement}" for achievement in role['achievements'])
                    role_details += f"\n{achievements}"
                previous_details.append(role_details)
            
            return f"""Professional Experience:

**Current Role: {current['title']} at {current['company']} ({current['duration']})**
Key Achievements:
{chr(10).join(f"- {achievement}" for achievement in current['achievements'])}

**Previous Experience:**
{chr(10).join(previous_details)}

**Total Experience:** 4+ years of full-stack development with AI/ML integration, delivering measurable business impact across healthcare, finance, and technology sectors."""
        
        elif category in ['projects', 'ai_ml', 'cloud']:
            projects = knowledge['projects']
            return f"""Key Projects and Achievements:

{chr(10).join(f'''
{i+1}. {project['name']}
   - Impact: {project['impact']}
   - Technologies: {', '.join(project['technologies'])}
   - Description: {project['description']}''' for i, project in enumerate(projects))}"""
        
        elif category == 'education':
            edu = knowledge['education']
            masters = edu['masters']
            bachelors = edu['bachelors']
            return f"""Educational Background:

{masters['degree']}
{masters['university']} ({masters['duration']})
- Focus: {masters['focus']}

{bachelors['degree']}
{bachelors['university']} ({bachelors['duration']})
- {bachelors['achievements']}

Continuous learning in AI/ML, cloud technologies, and software engineering best practices."""
        
        else:
            # General context - include everything
            current_role = knowledge['experience']['current_role']
            previous_roles = knowledge['experience']['previous_roles']
            
            return f"""Complete Professional Profile:

**Current Role:** {current_role['title']} at {current_role['company']} ({current_role['duration']})
**Education:** {knowledge['education']['masters']['degree']}, {knowledge['education']['masters']['university']}
**Technical Skills:** {', '.join(knowledge['technical_skills']['languages'][:4])}, AI/ML frameworks, AWS cloud services, Docker/Kubernetes

**Career Journey & Key Achievements:**

**At Veritis Group Inc (Current):**
- Built AI-powered testing agent reducing manual QA by 60%
- Developed clinical APIs handling 25,000+ daily inferences with <200ms latency
- Created multi-modal chat platform with RAG implementation using LangChain
- Optimized healthcare systems achieving significant performance improvements

**At TCS (2021-2022):**
- Built loan origination platform reducing approval time by 40%
- Automated document processing and risk scoring for major banking client
- Migrated legacy systems to AWS-based microservices architecture

**At Virtusa (2020-2021):**
- Developed responsive web applications and REST APIs
- Participated in agile development and CI/CD implementation
- Focused on database optimization and performance tuning

**Total Experience:** 4+ years delivering full-stack solutions with measurable business impact across healthcare, finance, and enterprise technology sectors. Proven expertise in AI/ML integration, cloud architecture, and high-performance system development."""
    
    def _get_static_response(self, query: str, category: str) -> str:
        """Get fast static response for ultra-quick processing."""
        if category == 'contact':
            return f"""Here's how you can reach me:

**Email:** {self.contact_info['email']}
**Phone:** {self.contact_info['phone']}

**Social & Professional:**
- **LinkedIn:** {self.contact_info['linkedin']}
- **GitHub:** {self.contact_info['github']}
- **LeetCode:** {self.contact_info['leetcode']}

**Schedule a Meeting:**
- **Calendly:** {self.contact_info['calendly']}

I typically respond within 24 hours and am always open to discussing new opportunities!"""
        
        elif category == 'skills':
            return """My top 3 technical skills are:

**1. Backend Development & APIs**
- Python, Java, JavaScript, TypeScript
- FastAPI, Django, Flask, Spring Boot
- Built clinical APIs handling 25,000+ daily inferences with <200ms latency

**2. AI/ML Engineering**
- TensorFlow, PyTorch, LangChain, LlamaIndex
- RAG systems, Vector databases, NLP
- Created AI testing agent reducing manual QA by 60%

**3. Cloud & DevOps**
- AWS (EC2, S3, Lambda, SQS), Docker, Kubernetes
- CI/CD pipelines, infrastructure optimization
- Architected scalable cloud solutions with high availability"""
        
        else:
            return """I'm Venkatesh Narra, a Software Development Engineer with 4+ years of experience in full-stack development and AI/ML integration. I specialize in building scalable, high-performance systems and have delivered measurable business impact through innovative solutions at Veritis Group Inc."""
    
    async def route_query(self, query: str) -> str:
        """
        Main routing method - guarantees excellent responses whether MCP works or not.
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query[:50]}..."
        if cache_key in self.response_cache:
            logger.info(f"⚡ Cache hit - Response time: {(time.time() - start_time) * 1000:.1f}ms")
            return self.response_cache[cache_key]
        
        # Classify query
        category, strategy = self._classify_query_intelligent(query)
        logger.info(f"🎯 Query classified: {category} | {strategy} | MCP: {self.mcp_agent_ready}")
        
        try:
            # Route to appropriate strategy
            if strategy == 'static':
                response = self._get_static_response(query, category)
            elif strategy == 'enhanced':
                response = await self._get_enhanced_knowledge_response(query, category)
            elif strategy == 'mcp_enhanced':
                # Try MCP first, fallback to enhanced
                try:
                    if self.mcp_agent_ready:
                        # TODO: Implement MCP call here when it works
                        response = await self._get_enhanced_knowledge_response(query, category)
                    else:
                        response = await self._get_enhanced_knowledge_response(query, category)
                except:
                    response = await self._get_enhanced_knowledge_response(query, category)
            else:
                response = await self._get_enhanced_knowledge_response(query, category)
            
            # Cache successful responses
            self.response_cache[cache_key] = response
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ Response generated in {elapsed:.1f}ms using {strategy} strategy")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Routing failed: {e}")
            return self._get_static_response(query, category)

# Backwards compatibility
MCPEnhancedSmartRouter = MultiAgentSmartRouter 