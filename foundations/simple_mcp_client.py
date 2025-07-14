"""
Simple MCP Client - Direct Implementation
Provides the same functionality as MCP tools without stdio communication issues.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class SimpleMCPClient:
    """Simple MCP client that provides tools functionality directly."""
    
    def __init__(self, server_name: str):
        self.server_name = server_name
        self._initialized = False
        self.knowledge_base = self._load_knowledge_base()
        logger.info(f"✅ Simple MCP client initialized for {server_name}")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from files."""
        knowledge = {}
        
        # Load profile data
        try:
            profile_path = Path("agent_knowledge/profile.md")
            if profile_path.exists():
                with open(profile_path, 'r', encoding='utf-8') as f:
                    knowledge['profile'] = f.read()
        except Exception as e:
            logger.warning(f"Could not load profile: {e}")
        
        # Load experience data
        try:
            exp_path = Path("agent_knowledge/experience_and_projects.md")
            if exp_path.exists():
                with open(exp_path, 'r', encoding='utf-8') as f:
                    knowledge['experience'] = f.read()
        except Exception as e:
            logger.warning(f"Could not load experience: {e}")
        
        # Load resume data
        try:
            resume_path = Path("agent_knowledge/resume.md")
            if resume_path.exists():
                with open(resume_path, 'r', encoding='utf-8') as f:
                    knowledge['resume'] = f.read()
        except Exception as e:
            logger.warning(f"Could not load resume: {e}")
        
        # Load FAQ data
        try:
            faq_path = Path("agent_knowledge/faq.md")
            if faq_path.exists():
                with open(faq_path, 'r', encoding='utf-8') as f:
                    knowledge['faq'] = f.read()
        except Exception as e:
            logger.warning(f"Could not load FAQ: {e}")
        
        return knowledge
    
    async def initialize(self) -> bool:
        """Initialize the client."""
        if not self._initialized:
            self._initialized = True
            logger.info("✅ Simple MCP client initialized successfully")
        return True
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        tools = [
            {
                "name": "get_profile_summary",
                "description": "Get a comprehensive profile summary including skills, experience, and background",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Optional focus area (skills, experience, education, etc.)"
                        }
                    }
                }
            },
            {
                "name": "get_technical_skills",
                "description": "Get detailed technical skills including programming languages, frameworks, and tools",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category (languages, frameworks, cloud, ai_ml, etc.)"
                        }
                    }
                }
            },
            {
                "name": "get_experience_details",
                "description": "Get work experience details including roles, companies, and achievements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "Optional company name to filter experience"
                        }
                    }
                }
            },
            {
                "name": "get_project_details",
                "description": "Get project details including technologies used and achievements",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_type": {
                            "type": "string",
                            "description": "Optional project type (ai, web, mobile, etc.)"
                        }
                    }
                }
            },
            {
                "name": "get_education_details",
                "description": "Get education background and certifications",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "search_knowledge",
                "description": "Search through all knowledge base content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
        
        logger.info(f"✅ Retrieved {len(tools)} tools from simple MCP client")
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool and return the result."""
        try:
            if name == "get_profile_summary":
                return await self._get_profile_summary(arguments)
            elif name == "get_technical_skills":
                return await self._get_technical_skills(arguments)
            elif name == "get_experience_details":
                return await self._get_experience_details(arguments)
            elif name == "get_project_details":
                return await self._get_project_details(arguments)
            elif name == "get_education_details":
                return await self._get_education_details(arguments)
            elif name == "search_knowledge":
                return await self._search_knowledge(arguments)
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return {"error": str(e)}
    
    async def _get_profile_summary(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get profile summary."""
        result = {
            "name": "Venkatesh Narra",
            "title": "Software Development Engineer",
            "experience_years": "4+",
            "current_role": "Software Development Engineer at Veritis Group Inc",
            "education": "MS Computer Science - George Mason University",
            "key_skills": [
                "Full-stack development",
                "AI/ML integration",
                "Cloud architecture",
                "System optimization"
            ],
            "achievements": [
                "AI testing agent reducing manual QA by 60%",
                "Clinical APIs handling 25,000+ daily inferences",
                "Multi-modal chat platform with RAG implementation",
                "Loan origination system cutting approval time by 40%"
            ],
            "contact": {
                "email": "vnarrag@gmu.edu",
                "phone": "+1 703-453-2157",
                "linkedin": "https://www.linkedin.com/in/venkateswara-narra-91170b34a",
                "github": "https://github.com/venkynarra"
            }
        }
        
        return {"result": result}
    
    async def _get_technical_skills(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get technical skills."""
        result = {
            "programming_languages": [
                "Python", "Java", "JavaScript", "TypeScript", "C++"
            ],
            "backend_frameworks": [
                "FastAPI", "Django", "Flask", "Spring Boot"
            ],
            "frontend_technologies": [
                "React", "Angular", "Node.js", "Express.js"
            ],
            "cloud_platforms": [
                "AWS (EC2, S3, Lambda, SQS)", "Docker", "Kubernetes"
            ],
            "ai_ml_frameworks": [
                "TensorFlow", "PyTorch", "LangChain", "LlamaIndex", "Scikit-learn"
            ],
            "databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Vector DBs (FAISS, Chroma)"
            ],
            "tools": [
                "Git", "GitHub", "GitLab", "JIRA", "Postman", "VS Code"
            ]
        }
        
        category = arguments.get("category")
        if category:
            if category in result:
                result = {category: result[category]}
        
        return {"result": result}
    
    async def _get_experience_details(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get experience details."""
        result = {
            "current_role": {
                "title": "Software Development Engineer",
                "company": "Veritis Group Inc",
                "duration": "Jan 2023 - Present",
                "achievements": [
                    "Spearheaded clinical API development handling 25,000+ daily inferences",
                    "Built AI-powered testing agent reducing manual QA by 60%",
                    "Optimized system performance achieving <200ms response times"
                ],
                "technologies": ["Python", "FastAPI", "AWS", "AI/ML", "Docker"]
            },
            "previous_roles": [
                {
                    "title": "Full-Stack Developer",
                    "company": "TCS",
                    "duration": "Feb 2021 - Jun 2022",
                    "achievements": [
                        "Developed scalable web applications",
                        "Implemented microservices architecture",
                        "Collaborated with cross-functional teams"
                    ]
                },
                {
                    "title": "Junior Software Engineer",
                    "company": "Virtusa",
                    "duration": "May 2020 - Jan 2021",
                    "achievements": [
                        "Built responsive web interfaces",
                        "Worked on database optimization",
                        "Participated in agile development"
                    ]
                }
            ]
        }
        
        return {"result": result}
    
    async def _get_project_details(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get project details."""
        result = {
            "projects": [
                {
                    "name": "AI Testing Agent",
                    "type": "AI/ML",
                    "impact": "Reduced manual QA by 60%",
                    "technologies": ["Python", "AI/ML", "automation frameworks"],
                    "description": "Automated test generation using AI to improve QA efficiency"
                },
                {
                    "name": "Clinical Decision Support Tool",
                    "type": "Web/API",
                    "impact": "25,000+ daily inferences with <200ms latency",
                    "technologies": ["FastAPI", "AWS", "real-time processing"],
                    "description": "High-performance medical decision support system"
                },
                {
                    "name": "Multi-modal Chat Platform",
                    "type": "AI/NLP",
                    "impact": "RAG implementation with intelligent fallback",
                    "technologies": ["LangChain", "Vector DBs", "NLP"],
                    "description": "Advanced conversational AI with knowledge retrieval"
                },
                {
                    "name": "Loan Origination System",
                    "type": "ML/Finance",
                    "impact": "Cut approval time by 40%",
                    "technologies": ["Python", "Machine Learning", "APIs"],
                    "description": "ML-powered loan processing optimization"
                }
            ]
        }
        
        return {"result": result}
    
    async def _get_education_details(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get education details."""
        result = {
            "education": [
                {
                    "degree": "MS Computer Science",
                    "university": "George Mason University",
                    "duration": "2022-2024",
                    "focus": "AI/ML, distributed systems, software engineering"
                },
                {
                    "degree": "B.Tech Computer Science",
                    "university": "GITAM University",
                    "duration": "2018-2022",
                    "achievements": "Graduated with honors"
                }
            ],
            "certifications": [
                "AWS Certified Developer",
                "Python Professional Certificate",
                "Machine Learning Specialization"
            ]
        }
        
        return {"result": result}
    
    async def _search_knowledge(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Search through knowledge base."""
        query = arguments.get("query", "").lower()
        
        if not query:
            return {"error": "Query parameter is required"}
        
        results = []
        
        # Search through all knowledge base content
        for source, content in self.knowledge_base.items():
            if query in content.lower():
                # Extract relevant snippets
                lines = content.split('\n')
                relevant_lines = []
                for i, line in enumerate(lines):
                    if query in line.lower():
                        # Get surrounding context
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        context = '\n'.join(lines[start:end])
                        relevant_lines.append(context)
                
                if relevant_lines:
                    results.append({
                        "source": source,
                        "matches": relevant_lines[:3]  # Top 3 matches
                    })
        
        return {"result": {"query": query, "results": results}}
    
    def close(self):
        """Close the client."""
        logger.info("✅ Simple MCP client closed")


async def create_simple_mcp_client(server_name: str) -> SimpleMCPClient:
    """Create a simple MCP client without stdio communication."""
    logger.info(f"🚀 Creating simple MCP client for server: {server_name}")
    
    client = SimpleMCPClient(server_name)
    success = await client.initialize()
    
    if success:
        logger.info("✅ Simple MCP client created successfully")
        return client
    else:
        logger.error("❌ Simple MCP client initialization failed")
        return None 