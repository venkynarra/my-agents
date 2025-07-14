"""
MCP Server Implementation for Venkatesh Narra Career Assistant
Provides tools for resume data, project information, skills, and more.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os
from pathlib import Path

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    CallToolRequest, CallToolResult, ListResourcesRequest, ListResourcesResult,
    ListToolsRequest, ListToolsResult, ReadResourceRequest, ReadResourceResult
)
import mcp.types as types
from pydantic import AnyUrl

# Import the RAG engine components
from .rag_engine import build_knowledge_index, query_knowledge_index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("career-assistant-mcp")

# Initialize the MCP server
server = Server("career-assistant")

# Knowledge base paths
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "agent_knowledge"
INDEX_PATH = Path(__file__).parent / "rag_index"

# Global variables for lazy initialization
rag_engine: Optional[Any] = None # Using Any to avoid circular import issues with LlamaIndex types
career_manager: Optional["CareerDataManager"] = None
initialization_lock = asyncio.Lock()

async def _ensure_initialized():
    """Ensure the data manager is initialized quickly without RAG timeouts."""
    global rag_engine, career_manager
    async with initialization_lock:
        if career_manager is None:
            logger.info("🚀 Quick initializing MCP Server data manager...")
            
            # Skip RAG engine to avoid timeouts - use static data instead
            rag_engine = None
            career_manager = CareerDataManager(None)
            logger.info("✅ CareerDataManager initialized quickly without RAG")

class CareerDataManager:
    """Manages career data and provides structured access to information."""
    
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.resume_data = self._load_resume_data()
        self.projects_data = self._load_projects_data()
        self.skills_data = self._load_skills_data()
        self.profile_data = self._load_profile_data()
        self.faq_data = self._load_faq_data()
    
    def _load_file_content(self, filename: str) -> str:
        """Load content from a knowledge base file."""
        try:
            file_path = KNOWLEDGE_BASE_PATH / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return ""
    
    def _load_resume_data(self) -> Dict[str, Any]:
        """Parse and structure resume data."""
        content = self._load_file_content("resume.md")
        return {
            "raw_content": content,
            "education": self._extract_education(content),
            "experience": self._extract_experience(content),
            "skills": self._extract_skills(content),
            "certifications": self._extract_certifications(content),
            "summary": self._extract_summary(content)
        }
    
    def _load_projects_data(self) -> Dict[str, Any]:
        """Parse and structure projects data."""
        content = self._load_file_content("experience_and_projects.md")
        return {
            "raw_content": content,
            "projects": self._extract_projects(content),
            "achievements": self._extract_achievements(content)
        }
    
    def _load_skills_data(self) -> Dict[str, Any]:
        """Parse and structure skills data."""
        content = self._load_file_content("resume.md")
        return {
            "technical_skills": self._extract_technical_skills(content),
            "languages": self._extract_languages(content),
            "frameworks": self._extract_frameworks(content),
            "tools": self._extract_tools(content)
        }
    
    def _load_profile_data(self) -> Dict[str, Any]:
        """Parse and structure profile data."""
        content = self._load_file_content("profile.md")
        return {
            "raw_content": content,
            "contact_info": self._extract_contact_info(content),
            "social_links": self._extract_social_links(content)
        }
    
    def _load_faq_data(self) -> Dict[str, Any]:
        """Parse and structure FAQ data."""
        content = self._load_file_content("faq.md")
        return {
            "raw_content": content,
            "qa_pairs": self._extract_qa_pairs(content)
        }
    
    def _extract_education(self, content: str) -> List[Dict[str, str]]:
        """Extract education information from resume content."""
        lines = content.split('\n')
        education = []
        in_education = False
        
        for line in lines:
            if "## Education" in line:
                in_education = True
                continue
            elif line.startswith("##") and in_education:
                break
            elif in_education and line.strip().startswith("- "):
                education.append(line.strip()[2:])
        
        return education
    
    def _extract_experience(self, content: str) -> List[Dict[str, Any]]:
        """Extract work experience from resume content."""
        # Implementation for extracting structured experience data
        return []
    
    def _extract_skills(self, content: str) -> List[str]:
        """Extract skills from resume content."""
        # Implementation for extracting skills
        return []
    
    def _extract_certifications(self, content: str) -> List[str]:
        """Extract certifications from resume content."""
        # Implementation for extracting certifications
        return []
    
    def _extract_summary(self, content: str) -> str:
        """Extract professional summary from resume content."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "## Summary" in line and i + 1 < len(lines):
                return lines[i + 1].strip()
        return ""
    
    def _extract_projects(self, content: str) -> List[Dict[str, Any]]:
        """Extract project information."""
        # Implementation for extracting projects
        return []
    
    def _extract_achievements(self, content: str) -> List[str]:
        """Extract achievements from content."""
        # Implementation for extracting achievements
        return []
    
    def _extract_technical_skills(self, content: str) -> Dict[str, List[str]]:
        """Extract categorized technical skills."""
        # Implementation for extracting technical skills
        return {}
    
    def _extract_languages(self, content: str) -> List[str]:
        """Extract programming languages."""
        # Implementation for extracting languages
        return []
    
    def _extract_frameworks(self, content: str) -> List[str]:
        """Extract frameworks and libraries."""
        # Implementation for extracting frameworks
        return []
    
    def _extract_tools(self, content: str) -> List[str]:
        """Extract tools and technologies."""
        # Implementation for extracting tools
        return []
    
    def _extract_contact_info(self, content: str) -> Dict[str, str]:
        """Extract contact information."""
        return {
            "email": "venkateshnarra368@gmail.com",
            "linkedin": "https://www.linkedin.com/in/venkateswara-narra-91170b34a/",
            "github": "https://github.com/venkynarra",
            "leetcode": "https://leetcode.com/u/pravnarri/"
        }
    
    def _extract_social_links(self, content: str) -> Dict[str, str]:
        """Extract social media links."""
        return {}
    
    def _extract_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """Extract FAQ Q&A pairs."""
        # Implementation for extracting Q&A pairs
        return []

# Initialize career data manager (will be initialized lazily)
career_manager: Optional[CareerDataManager] = None

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available career data resources."""
    return [
        Resource(
            uri=AnyUrl("career://resume"),
            name="Resume Data",
            description="Complete resume information including education, experience, and skills",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("career://projects"),
            name="Projects Data",
            description="Detailed project information and achievements",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("career://skills"),
            name="Technical Skills",
            description="Comprehensive technical skills and competencies",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("career://profile"),
            name="Professional Profile",
            description="Professional profile and contact information",
            mimeType="application/json",
        ),
        Resource(
            uri=AnyUrl("career://faq"),
            name="FAQ Data",
            description="Frequently asked questions and answers",
            mimeType="application/json",
        )
    ]

@server.read_resource()
async def handle_read_resource(request: ReadResourceRequest) -> ReadResourceResult:
    """Read career data resources."""
    uri = str(request.uri)
    
    if uri == "career://resume":
        content = json.dumps(career_manager.resume_data, indent=2)
        return ReadResourceResult(contents=[TextContent(type="text", text=content)])
    
    elif uri == "career://projects":
        content = json.dumps(career_manager.projects_data, indent=2)
        return ReadResourceResult(contents=[TextContent(type="text", text=content)])
    
    elif uri == "career://skills":
        content = json.dumps(career_manager.skills_data, indent=2)
        return ReadResourceResult(contents=[TextContent(type="text", text=content)])
    
    elif uri == "career://profile":
        content = json.dumps(career_manager.profile_data, indent=2)
        return ReadResourceResult(contents=[TextContent(type="text", text=content)])
    
    elif uri == "career://faq":
        content = json.dumps(career_manager.faq_data, indent=2)
        return ReadResourceResult(contents=[TextContent(type="text", text=content)])
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available MCP tools for career assistant."""
    return [
        Tool(
            name="get_resume_summary",
            description="Get a structured summary of resume information",
            inputSchema={
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": ["all", "education", "experience", "skills", "summary"],
                        "description": "Which section of the resume to retrieve"
                    }
                },
                "required": ["section"]
            },
        ),
        Tool(
            name="get_project_details",
            description="Get detailed information about specific projects",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of the project to get details for (optional)"
                    },
                    "company": {
                        "type": "string",
                        "description": "Company name to filter projects (optional)"
                    }
                },
                "required": []
            },
        ),
        Tool(
            name="get_technical_skills",
            description="Get categorized technical skills. Use 'all' to get all skills, or specific categories like 'languages', 'frameworks', 'tools', 'cloud', 'ai_ml'",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["all", "languages", "frameworks", "tools", "cloud", "ai_ml"],
                        "description": "Must be one of: 'all' (for all skills), 'languages' (programming languages), 'frameworks' (web frameworks), 'tools' (development tools), 'cloud' (cloud platforms), 'ai_ml' (AI/ML technologies)"
                    }
                },
                "required": ["category"]
            },
        ),
        Tool(
            name="search_knowledge_base",
            description="Search through all career knowledge base files for detailed information about projects, experience, and skills",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant information about projects, experience, skills, or achievements"
                    }
                },
                "required": ["query"]
            },
        ),
        Tool(
            name="get_contact_info",
            description="Get professional contact information and social links",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        ),
        Tool(
            name="get_experience_by_company",
            description="Get work experience details for a specific company",
            inputSchema={
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "enum": ["TCS", "Virtusa", "Veritis", "all"],
                        "description": "Company name to get experience for"
                    }
                },
                "required": ["company"]
            },
        ),
        Tool(
            name="get_education_details",
            description="Get detailed educational background and certifications",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls for career assistant functionality."""
    # Ensure the data manager and RAG engine are loaded before proceeding.
    await _ensure_initialized()
    
    if request.name == "get_resume_summary":
        section = request.arguments.get("section", "all")
        
        if section == "all":
            result = career_manager.resume_data
        elif section == "education":
            result = {
                "education": career_manager.resume_data.get("education", []),
                "certifications": career_manager.resume_data.get("certifications", [])
            }
        elif section == "experience":
            result = {"experience": career_manager.resume_data.get("experience", [])}
        elif section == "skills":
            result = {"skills": career_manager.resume_data.get("skills", [])}
        elif section == "summary":
            result = {"summary": career_manager.resume_data.get("summary", "")}
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "get_project_details":
        project_name = request.arguments.get("project_name")
        company = request.arguments.get("company")
        
        result = {
            "projects": career_manager.projects_data.get("projects", []),
            "achievements": career_manager.projects_data.get("achievements", []),
            "filter_applied": {
                "project_name": project_name,
                "company": company
            }
        }
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "get_technical_skills":
        category = request.arguments.get("category", "all")
        
        # Validate category parameter
        valid_categories = ["all", "languages", "frameworks", "tools", "cloud", "ai_ml"]
        if category not in valid_categories:
            error_result = {
                "error": f"Invalid category '{category}'. Must be one of: {', '.join(valid_categories)}",
                "valid_categories": valid_categories,
                "suggestion": "Use 'all' to get all skills"
            }
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(error_result, indent=2))]
            )
        
        if category == "all":
            result = {
                "languages": ["Python", "Java", "JavaScript", "TypeScript", "C++"],
                "frameworks": ["FastAPI", "Django", "Flask", "Spring Boot", "React", "Angular", "Node.js", "Express.js"],
                "cloud": ["AWS (EC2, S3, Lambda, SQS)", "Docker", "Kubernetes", "CI/CD"],
                "ai_ml": ["TensorFlow", "PyTorch", "Scikit-learn", "LangChain", "LlamaIndex", "RAG systems"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "Vector DBs (FAISS, Chroma)"],
                "tools": ["Git", "GitHub", "GitLab", "JIRA", "Postman", "VS Code", "PyCharm"]
            }
        else:
            result = {category: career_manager.skills_data.get(category, [])}
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "search_knowledge_base":
        query = request.arguments.get("query", "")
        if not query:
            raise ValueError("A query must be provided for search_knowledge_base.")
            
        logger.info(f"MCP Tool: Searching knowledge base for query: '{query}'")
        
        # Use static knowledge responses when RAG is not available
        if career_manager.rag_engine is None:
            response_text = _get_static_knowledge_response(query)
        else:
            # Use the RAG engine if available
            response_text = await query_knowledge_index(career_manager.rag_engine, query)
        
        result = {
            "query": query,
            "response": response_text,
            "source": "static_knowledge" if career_manager.rag_engine is None else "rag_engine"
        }
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "get_contact_info":
        result = career_manager.profile_data.get("contact_info", {})
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "get_experience_by_company":
        company = request.arguments.get("company", "all")
        
        # Extract experience for specific company from raw content
        content = career_manager.resume_data.get("raw_content", "")
        
        result = {
            "company": company,
            "experience_details": f"Experience information for {company} from resume content",
            "raw_content": content
        }
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    elif request.name == "get_education_details":
        result = {
            "education": career_manager.resume_data.get("education", []),
            "certifications": career_manager.resume_data.get("certifications", []),
            "details": "B.Tech in Computer Science, GITAM University Vizag (2018-2022)"
        }
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    
    else:
        raise ValueError(f"Unknown tool: {request.name}")

def _get_static_knowledge_response(query: str) -> str:
    """Generate static knowledge responses for common queries when RAG is unavailable."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['skills', 'technical', 'programming']):
        return """Technical Skills Summary:
- Programming Languages: Python, Java, JavaScript, TypeScript, C++
- Backend Frameworks: FastAPI, Django, Flask, Spring Boot
- Frontend: React, Angular, Node.js, Express.js
- Cloud & DevOps: AWS (EC2, S3, Lambda, SQS), Docker, Kubernetes
- AI/ML: TensorFlow, PyTorch, LangChain, LlamaIndex, RAG systems
- Databases: PostgreSQL, MySQL, MongoDB, Vector DBs (FAISS, Chroma)

Key Achievements:
- Built clinical APIs handling 25,000+ daily inferences with <200ms latency
- Created AI testing agent reducing manual QA by 60%
- Architected scalable cloud solutions with high availability"""
    
    elif any(word in query_lower for word in ['experience', 'work', 'job']):
        return """Professional Experience:
        
Current Role: Software Development Engineer at Veritis Group Inc (Jan 2023 - Present)
- Spearheaded clinical API development handling 25,000+ daily inferences
- Built AI-powered testing agent reducing manual QA by 60%
- Optimized system performance achieving <200ms response times

Previous Experience:
- TCS (Feb 2021 - Jun 2022): Full-Stack Developer
- Virtusa (May 2020 - Jan 2021): Junior Software Engineer

Key Projects:
- Clinical decision support tool with real-time inference
- AI testing agent with automated test generation
- Multi-modal chat platform with RAG implementation
- Loan origination system cutting approval time by 40%"""
    
    elif any(word in query_lower for word in ['education', 'degree', 'university']):
        return """Educational Background:
        
Master of Science, Computer Science
George Mason University (2022-2024)
- Advanced coursework in AI/ML, distributed systems, and software engineering
- Thesis work on intelligent systems and language models

Bachelor of Technology, Computer Science
GITAM University (2018-2022)
- Strong foundation in algorithms, data structures, and software development
- Graduated with honors, active in technical projects and competitions"""
    
    elif any(word in query_lower for word in ['projects', 'achievements']):
        return """Key Projects and Achievements:
        
1. AI Testing Agent (Veritis Group Inc)
   - Reduced manual QA by 60% using automated test generation
   - Technologies: Python, AI/ML, automation frameworks
   
2. Clinical Decision Support Tool (Veritis Group Inc)
   - Handles 25,000+ daily inferences with <200ms latency
   - Technologies: FastAPI, AWS, real-time processing
   
3. Multi-modal Chat Platform
   - RAG implementation with intelligent fallback
   - Technologies: LangChain, Vector DBs, NLP
   
4. Loan Origination System
   - Cut approval time by 40% through ML optimization
   - Technologies: Python, Machine Learning, APIs"""
    
    elif any(word in query_lower for word in ['cloud', 'aws', 'devops']):
        return """Cloud & DevOps Experience:
        
AWS Services:
- EC2 for compute instances and auto-scaling
- S3 for object storage and data lakes
- Lambda for serverless functions
- SQS for message queuing and async processing

Real-world Applications:
- Clinical decision support tool: EC2 + S3 + Lambda handling 25,000+ daily inferences
- AI testing agent: Used EC2 for model training, S3 for datasets, Lambda for processing
- Optimized resource utilization and cost-effectiveness across environments

DevOps & Orchestration:
- Docker containerization for consistent deployments
- Kubernetes for container orchestration and scaling
- CI/CD pipelines for automated testing and deployment"""
    
    elif any(word in query_lower for word in ['ai', 'ml', 'machine learning']):
        return """AI/ML Experience:
        
Frameworks & Technologies:
- TensorFlow, PyTorch for deep learning models
- LangChain, LlamaIndex for LLM applications
- Vector databases (FAISS, Chroma) for RAG systems
- Scikit-learn for traditional ML algorithms

Key Projects:
- AI Testing Agent: Reduced manual QA by 60% using automated test generation
- Clinical Decision Support: 25,000+ daily inferences with <200ms latency
- Multi-modal Chat Platform: RAG implementation with intelligent fallback
- Loan Origination: ML models cutting approval time by 40%

Specializations:
- Natural Language Processing (NLP)
- Retrieval-Augmented Generation (RAG)
- Model optimization and deployment
- Real-time inference systems"""
    
    elif any(phrase in query_lower for phrase in ['hire', 'why', 'best reason']):
        return """Why You Should Hire Me:
        
1. Proven Impact: Delivered measurable business results
   - 60% reduction in manual QA through AI automation
   - 40% faster loan approvals through ML optimization
   - 25,000+ daily inferences with <200ms latency

2. Technical Excellence: Full-stack expertise with modern technologies
   - 4+ years of hands-on development experience
   - Proficient in Python, Java, JavaScript, AI/ML, AWS
   - Strong foundation in scalable system architecture

3. Innovation & Problem-Solving: Forward-thinking approach
   - Created AI-powered solutions for complex business challenges
   - Optimized system performance and reliability
   - Experience with cutting-edge technologies like RAG and LLMs

4. Professional Growth: Continuous learning and adaptation
   - MS Computer Science from George Mason University
   - Active contributor to technical projects and innovations
   - Passionate about emerging technologies and best practices"""
    
    else:
        return f"""Based on your query about "{query}", here's relevant information:

I'm Venkatesh Narra, a Software Development Engineer with 4+ years of experience in full-stack development and AI/ML integration.

Current Role: Software Development Engineer at Veritis Group Inc
Background: MS Computer Science from George Mason University

Key Achievements:
- AI testing agent reducing manual QA by 60%
- Clinical APIs handling 25,000+ daily inferences with <200ms latency
- Multi-modal chat platform with RAG implementation
- Loan origination system cutting approval time by 40%

Technical Expertise: Python, Java, JavaScript, AI/ML, AWS, Docker, React, FastAPI, Django

For more specific information, please feel free to ask targeted questions about my skills, experience, projects, or career goals."""

async def main():
    """Main function to run the MCP server."""
    logger.info("MCP Server starting...")
    
    # Pre-initialize immediately for faster startup
    global career_manager
    career_manager = CareerDataManager(None)
    logger.info("✅ CareerDataManager pre-initialized")
    logger.info("🚀 MCP Server ready for connections")
    
    from mcp.server.stdio import stdio_server
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="career-assistant",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"❌ MCP Server failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Set up Windows-specific event loop policy
    if sys.platform == "win32":
        # Use WindowsProactorEventLoopPolicy for better subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🚫 MCP Server interrupted by user")
    except Exception as e:
        logger.error(f"❌ MCP Server failed with error: {e}")
        sys.exit(1) 