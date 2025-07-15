import asyncio
import time
from typing import Dict, List, Any
import sys
from pathlib import Path
import json

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.logging import logger

class RAGEngine:
    """High-performance RAG engine for fast knowledge retrieval"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.vector_cache = {}  # In-memory vector cache
        self.query_cache = {}   # Query result cache
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load and structure knowledge base from resume and other sources"""
        return {
            "personal_info": {
                "name": "Venkateswara Rao Narra",
                "title": "Full Stack Python Developer",
                "experience_years": "4+ years",
                "phone": "+1 703-453-2157",
                "email": "venkateswaran533@gmail.com",
                "linkedin": "LinkedIn",
                "github": "GITHUB"
            },
            "professional_summary": "Full Stack Python Developer with 4+ years of experience building real-world web apps and ML-based features from scratch. I work mainly with FastAPI, Django, and Flask for complex APIs, and use React + TypeScript on the frontend. Comfortable handling everything from backend logic and model deployment to CI/CD and cloud setup on AWS or Azure.",
            "work_experience": {
                "current": {
                    "title": "Software Development Engineer",
                    "company": "Veritis Group Inc",
                    "location": "Dallas, TX",
                    "period": "Jan 2023 - Present",
                    "achievements": [
                        "Developed real-time prediction APIs using FastAPI for clinical decision support tool",
                        "Built and deployed diagnosis risk scoring models for hospital intake prioritization",
                        "Created event-driven pipeline using AWS Lambda + SQS",
                        "Set up GitLab CI/CD workflows for automated deployment",
                        "Integrated Prometheus and Grafana dashboards for monitoring"
                    ]
                },
                "previous": [
                    {
                        "title": "Full Stack Developer",
                        "company": "TCS",
                        "location": "India",
                        "period": "Feb 2021 - June 2022",
                        "achievements": [
                            "Built full-stack loan platform using Django and React with TypeScript",
                            "Designed ML scoring APIs for loan risk classification",
                            "Developed real-time dashboards using Redux and TypeScript"
                        ]
                    },
                    {
                        "title": "Junior Software Engineer",
                        "company": "Virtusa",
                        "location": "India",
                        "period": "May 2020 - Jan 2021",
                        "achievements": [
                            "Developed backend features using Python and Flask",
                            "Automated data ingestion pipelines using Pandas and Azure Functions"
                        ]
                    }
                ]
            },
            "skills": {
                "languages": ["C", "Java", "Python", "C#", "HTML", "CSS", "NodeJS", "JavaScript", "TypeScript"],
                "frameworks": ["Angular", "React-Redux", "FastAPI", "Django", "Flask", "Bootstrap"],
                "databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis"],
                "cloud_devops": ["AWS (EC2, Lambda, S3, SQS, CloudWatch)", "Azure (App Services, Blob, SQL)", "Docker", "Kubernetes", "Terraform", "Jenkins", "GitLab CI/CD"],
                "ml_data": ["Scikit-learn", "Pandas", "NumPy", "TensorFlow", "Matplotlib", "LSTM"]
            },
            "certifications": [
                "Advanced Learning Algorithms (Stanford)",
                "Artificial Intelligence I (IBM)",
                "Deep Learning Specialization (Coursera)"
            ],
            "education": {
                "degree": "Master of Computer Science",
                "university": "George Mason University",
                "location": "Fairfax",
                "period": "Aug 2022 - May 2024",
                "gpa": "3.47/4.00"
            },
            "projects": [
                {
                    "name": "Clinical Decision Support Tool",
                    "tech": ["FastAPI", "React", "AWS", "ML Models"],
                    "description": "Real-time prediction APIs for clinical decision support"
                },
                {
                    "name": "Loan Platform",
                    "tech": ["Django", "React", "TypeScript", "Azure"],
                    "description": "Full-stack loan processing platform with ML scoring"
                },
                {
                    "name": "Music Streaming Service",
                    "tech": ["Node.js", "Express", "React"],
                    "description": "Complete music streaming application"
                },
                {
                    "name": "Stock Market Prediction",
                    "tech": ["LSTM", "Monte Carlo Simulation", "Python"],
                    "description": "ML-based stock market prediction system"
                }
            ]
        }
    
    async def retrieve_relevant_context(self, query: str, max_results: int = 3) -> str:
        """Fast context retrieval with caching"""
        start_time = time.time()
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.query_cache:
            logger.info(f"RAG cache hit: {time.time() - start_time:.3f}s")
            return self.query_cache[cache_key]
        
        try:
            # Ultra-fast direct context based on query keywords
            context = self._get_direct_context(query)
            
            # Cache the result
            self.query_cache[cache_key] = context
            
            retrieval_time = time.time() - start_time
            logger.info(f"RAG retrieval: {retrieval_time:.3f}s")
            
            return context
            
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return self._get_fallback_context(query)
    
    def _get_direct_context(self, query: str) -> str:
        """Ultra-fast direct context retrieval"""
        query_lower = query.lower()
        
        # Skills query
        if any(word in query_lower for word in ['skill', 'technology', 'tech', 'programming']):
            return "I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. My technical skills include Python, Java, JavaScript, TypeScript, React, Angular, FastAPI, Django, Flask, AWS, Azure, Docker, Kubernetes, PostgreSQL, MongoDB, Redis, Git, CI/CD, Scikit-learn, Pandas, and TensorFlow."
        
        # Experience query
        if any(word in query_lower for word in ['experience', 'work', 'job', 'career']):
            return "I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. Currently working as Software Development Engineer at Veritis Group Inc (2023-Present), previously Full Stack Developer at TCS (2021-2022), and Junior Software Engineer at Virtusa (2020-2021). I've worked on clinical decision support tools, loan platforms, and retail reporting systems."
        
        # Project query
        if any(word in query_lower for word in ['project', 'build', 'create', 'develop']):
            return "I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. My key projects include: Clinical Decision Support Tool (FastAPI/React), Loan Platform (Django/React), Real-time ML Prediction APIs, Patient Risk Overview Dashboard, Music Streaming Service, and Stock Market Prediction using LSTM."
        
        # Certification query
        if any(word in query_lower for word in ['certification', 'cert', 'certified']):
            return "I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience. My certifications include: Advanced Learning Algorithms (Stanford), Artificial Intelligence I (IBM), and Deep Learning Specialization (Coursera). I also have a Master of Computer Science from George Mason University with a 3.47/4.00 GPA."
        
        # General query
        return "I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web apps and ML-based features. I work mainly with FastAPI, Django, and Flask for complex APIs, and use React + TypeScript on the frontend. Comfortable handling everything from backend logic and model deployment to CI/CD and cloud setup on AWS or Azure."
    
    def _get_fallback_context(self, query: str) -> str:
        """Fallback context for unknown queries"""
        return f"I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web apps and ML-based features."

# Global RAG engine instance
rag_engine = RAGEngine() 