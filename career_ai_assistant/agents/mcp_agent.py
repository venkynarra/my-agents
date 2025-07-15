import asyncio
import time
from typing import Dict
import sys
from pathlib import Path
import google.generativeai as genai
import os

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from monitoring.logging import logger
from core.rag_engine import rag_engine

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

async def get_mcp_response(query: str) -> Dict:
    """
    Enhanced MCP response using RAG engine and optimized processing.
    Target: Sub-3 second responses with high quality and detailed answers.
    """
    start_time = time.time()
    
    try:
        # Step 1: Fast RAG context retrieval
        context = await asyncio.wait_for(
            rag_engine.retrieve_relevant_context(query),
            timeout=0.3
        )
        
        # Step 2: Query analysis for response style
        query_analysis = _analyze_query_for_response_style(query)
        
        # Step 3: Try enhanced response first, fallback if it fails
        try:
            response = await _generate_enhanced_response(query, context, query_analysis)
            if response and len(response.strip()) > 50:  # Ensure we got a good response
                response_time = time.time() - start_time
                logger.info(f"Fast enhanced MCP response: {response_time:.2f}s")
                return {
                    'response': response,
                    'context': context,
                    'source': 'enhanced_mcp',
                    'response_time': response_time,
                    'fallback_used': False
                }
        except Exception as e:
            logger.warning(f"Enhanced response failed, using fallback: {e}")
        
        # Step 4: Fallback to static response
        fallback_response = _get_static_fallback_response(query)
        response_time = time.time() - start_time
        
        logger.info(f"Using static fallback response: {response_time:.2f}s")
        return {
            'response': fallback_response,
            'context': context,
            'source': 'static_fallback',
            'response_time': response_time,
            'fallback_used': True
        }
        
    except Exception as e:
        logger.error(f"Enhanced MCP error: {e}")
        response_time = time.time() - start_time
        
        # Ultimate fallback
        fallback_response = _get_static_fallback_response(query)
        
        return {
            'response': fallback_response,
            'context': 'Error fallback',
            'source': 'ultimate_fallback',
            'response_time': response_time,
            'fallback_used': True
        }

def _analyze_query_for_response_style(query: str) -> Dict:
    """Analyze query to determine response style and length"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['skill', 'technology', 'tech', 'programming']):
        return {
            'style': 'technical_detailed',
            'max_tokens': 1000,
            'focus': 'skills_and_examples'
        }
    elif any(word in query_lower for word in ['project', 'build', 'create', 'develop', 'application']):
        return {
            'style': 'project_detailed',
            'max_tokens': 1200,
            'focus': 'projects_and_architecture'
        }
    elif any(word in query_lower for word in ['experience', 'work', 'job', 'career']):
        return {
            'style': 'experience_detailed',
            'max_tokens': 900,
            'focus': 'work_history_and_achievements'
        }
    elif any(word in query_lower for word in ['certification', 'cert', 'certified', 'education']):
        return {
            'style': 'certification_detailed',
            'max_tokens': 700,
            'focus': 'education_and_certifications'
        }
    elif any(word in query_lower for word in ['hire', 'recruit', 'interview', 'why', 'special']):
        return {
            'style': 'hiring_detailed',
            'max_tokens': 1200,
            'focus': 'value_proposition_and_achievements'
        }
    else:
        return {
            'style': 'general_detailed',
            'max_tokens': 800,
            'focus': 'general_expertise'
        }

async def _generate_enhanced_response(query: str, context: str, analysis: Dict) -> str:
    """Generate elaborate response using context and analysis"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Simplified but effective prompt
        prompt = f"""You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

Context: {context}

User Query: {query}

Respond as Venkatesh in first person. Provide detailed, specific answers with examples from your real experience. Include technical details, projects, and achievements. Be comprehensive and helpful.

Target: {analysis['max_tokens']} words maximum."""
        
        logger.info(f"Attempting enhanced response generation for: {query[:50]}...")
        
        # Generate response with reasonable QKV values
        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=min(analysis['max_tokens'], 800),  # Cap at 800 for reliability
                    candidate_count=1
                )
            ),
            timeout=2.0  # Reduced timeout
        )
        
        response_text = response.text.strip()
        logger.info(f"Enhanced response generated successfully: {len(response_text)} characters")
        return response_text
        
    except asyncio.TimeoutError as e:
        logger.error(f"Enhanced response timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"Enhanced response generation error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

def _build_enhanced_prompt(query: str, context: str, analysis: Dict) -> str:
    """Build comprehensive prompt with context and analysis"""
    
    # Base system prompt
    system_prompt = """You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web applications and ML-based features.

ROLE: You are a senior software engineer who specializes in:
- Full-stack development with Python, React, TypeScript
- Machine Learning and AI applications
- Cloud platforms (AWS, Azure)
- Microservices and scalable systems
- Real-time data processing and APIs

PERSONALITY: You are:
- Confident and knowledgeable about your expertise
- Enthusiastic about technology and problem-solving
- Professional yet approachable
- Detail-oriented with concrete examples
- Always speaking in first person ("I", "my", "me")

RESPONSE STYLE: Provide:
- Detailed, comprehensive answers
- Specific examples from your real experience
- Technical depth when appropriate
- Code examples when relevant
- Quantifiable achievements and impacts
- Professional insights and recommendations"""

    # Query-specific instructions
    query_instructions = _get_query_specific_instructions(query, analysis)
    
    # Final prompt
    prompt = f"""{system_prompt}

**USER QUERY:** {query}

**QUERY ANALYSIS:** {analysis['style']} - {analysis['focus']}

**CONTEXT INFORMATION:**
{context}

**SPECIFIC INSTRUCTIONS:**
{query_instructions}

**RESPONSE REQUIREMENTS:**
- Respond as Venkatesh in first person
- Use the provided context information
- Provide specific, detailed answers with examples
- Include technical depth and code examples when relevant
- Mention real projects, technologies, and achievements
- Be comprehensive but well-structured
- Target length: {analysis['max_tokens']} words

Now provide a detailed, comprehensive response to: {query}"""

    return prompt

def _get_query_specific_instructions(query: str, analysis: Dict) -> str:
    """Get specific instructions based on query type"""
    query_lower = query.lower()
    
    if analysis['style'] == 'technical_detailed':
        return """- Detail your technical skills with proficiency levels
- Provide specific examples of how you've used each technology
- Include code snippets or architecture examples
- Mention certifications and ongoing learning
- Explain your approach to learning new technologies
- Quantify your experience (e.g., "4+ years", "10+ projects")"""
    
    elif analysis['style'] == 'project_detailed':
        return """- Describe projects in detail with technical architecture
- Include code examples and implementation details
- Explain the business impact and user benefits
- Describe challenges faced and solutions implemented
- Mention technologies used and why they were chosen
- Quantify results and achievements"""
    
    elif analysis['style'] == 'experience_detailed':
        return """- Provide detailed work history with achievements
- Quantify your impact and contributions
- Describe the technologies and methodologies used
- Explain challenges overcome and lessons learned
- Connect experience to current capabilities
- Mention specific companies and roles"""
    
    elif analysis['style'] == 'certification_detailed':
        return """- Detail your educational background and certifications
- Explain how each certification helps in your work
- Connect education to practical applications
- Mention ongoing learning and professional development
- Quantify academic achievements (GPA, etc.)"""
    
    elif analysis['style'] == 'hiring_detailed':
        return """- Explain why you're a strong candidate
- Highlight unique skills and experiences
- Provide specific examples of achievements
- Demonstrate problem-solving abilities
- Show passion for technology and continuous learning
- Quantify your value proposition"""
    
    else:
        return """- Provide comprehensive, detailed answers
- Use specific examples from your experience
- Include technical details when relevant
- Be professional and confident
- Show enthusiasm for the topic
- Quantify achievements where possible"""

async def _generate_fallback_response(query: str) -> str:
    """Generate fallback response when enhanced processing fails"""
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""You are Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience.

User asked: "{query}"

Provide a detailed response as Venkatesh. Include specific examples from your experience. Be conversational and helpful. Keep it under 400 words."""

        logger.info(f"Attempting fallback response generation for: {query[:50]}...")

        response = await asyncio.wait_for(
            asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=30,
                    max_output_tokens=400
                )
            ),
            timeout=1.5
        )
        
        response_text = response.text.strip()
        logger.info(f"Fallback response generated successfully: {len(response_text)} characters")
        return response_text
        
    except asyncio.TimeoutError as e:
        logger.error(f"Fallback response timeout: {e}")
        raise
    except Exception as e:
        logger.error(f"Fallback response error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise

def _get_static_fallback_response(query: str) -> str:
    """Static fallback response with detailed information and code examples"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['code', 'example', 'snippet', 'programming']):
        return """Here are some code examples from my real projects:

**FastAPI ML Prediction API (Clinical Decision Support Tool):**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('clinical_model.pkl')

class PatientData(BaseModel):
    age: int
    symptoms: list
    vital_signs: dict

@app.post("/predict")
async def predict_diagnosis(patient: PatientData):
    features = preprocess_patient_data(patient)
    prediction = model.predict_proba([features])[0]
    return {"risk_score": float(prediction[1]), "confidence": 0.95}
```

**React TypeScript Component (Patient Dashboard):**
```typescript
import React, { useState, useEffect } from 'react';

interface Patient {
  id: string;
  name: string;
  riskScore: number;
  priority: 'high' | 'medium' | 'low';
}

const PatientDashboard: React.FC = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  
  useEffect(() => {
    fetchPatients();
  }, []);
  
  const fetchPatients = async () => {
    const response = await fetch('/api/patients');
    const data = await response.json();
    setPatients(data);
  };
  
  return (
    <div className="dashboard">
      {patients.map(patient => (
        <PatientCard key={patient.id} patient={patient} />
      ))}
    </div>
  );
};
```

**ML Pipeline with Scikit-learn (Loan Risk Model):**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def build_loan_risk_model(data_path: str):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = model.score(X_test, y_test)
    return model, accuracy
```

These are real code examples from my projects at Veritis Group Inc and TCS. I've built production systems using these patterns."""
    
    elif any(word in query_lower for word in ['skill', 'technology', 'tech']):
        return """My technical skills are quite comprehensive. I'm an expert in Python, JavaScript, and TypeScript, with 4+ years of production experience. For backend development, I specialize in FastAPI, Django, and Flask, having built scalable APIs that handle thousands of requests per minute. On the frontend, I'm proficient with React, Angular, and modern state management with Redux.

For cloud and DevOps, I have extensive experience with AWS (EC2, Lambda, S3, SQS, CloudWatch) and Azure (App Services, Blob Storage, SQL Database). I'm also skilled with Docker, Kubernetes, Terraform, and CI/CD pipelines using Jenkins and GitLab.

In the database realm, I work with PostgreSQL, MySQL, MongoDB, and Redis, having designed and optimized database schemas for high-traffic applications. My ML/AI expertise includes Scikit-learn, Pandas, NumPy, and TensorFlow, with experience building production ML models for real-time predictions.

What sets me apart is my ability to work across the entire stack - from database design to frontend optimization, from ML model development to cloud deployment. I've consistently delivered projects that not only meet technical requirements but also provide measurable business value."""
    
    elif any(word in query_lower for word in ['experience', 'work', 'job', 'career']):
        return """I have 4+ years of progressive experience as a software engineer, currently working as a Software Development Engineer at Veritis Group Inc in Dallas, TX since January 2023. In this role, I've developed real-time prediction APIs using FastAPI for clinical decision support tools, built and deployed diagnosis risk scoring models that help hospitals prioritize patient intake, and created event-driven pipelines using AWS Lambda and SQS for scalable processing.

Previously, I was a Full Stack Developer at TCS from February 2021 to June 2022, where I built a comprehensive loan platform using Django and React with TypeScript. I designed ML scoring APIs for loan risk classification and developed real-time dashboards that processed thousands of applications daily.

My journey started as a Junior Software Engineer at Virtusa from May 2020 to January 2021, where I developed backend features using Python and Flask, and automated data ingestion pipelines using Pandas and Azure Functions.

Throughout my career, I've consistently delivered projects that combine technical excellence with business impact. I've worked on clinical decision support tools, loan platforms, retail reporting systems, and various ML-based applications. My experience spans the entire development lifecycle, from requirements gathering to production deployment and monitoring."""
    
    elif any(word in query_lower for word in ['project', 'build', 'create', 'develop']):
        return """My most significant project is the Clinical Decision Support Tool I built at Veritis Group Inc. This is a comprehensive healthcare system that helps hospitals prioritize patient intake using ML models. I developed real-time prediction APIs using FastAPI that process patient data and provide risk scores for various diagnoses. The frontend is built with React and TypeScript, providing an intuitive interface for healthcare professionals.

The system uses an event-driven architecture with AWS Lambda and SQS for scalable processing, and I implemented CI/CD pipelines using GitLab for automated deployment. The ML models I developed achieve 85% accuracy in predicting diagnosis risks, which has improved patient prioritization by 40% and reduced wait times significantly.

Another notable project is the Loan Platform I built at TCS. This full-stack application uses Django for the backend and React with TypeScript for the frontend. I designed ML scoring APIs for loan risk classification and developed real-time dashboards that processed over 10,000 loan applications with 95% accuracy.

I also built a Stock Market Prediction System using LSTM and Monte Carlo simulation, achieving 78% prediction accuracy on historical data. This project involved time series analysis, risk assessment, and portfolio optimization algorithms."""
    
    elif any(word in query_lower for word in ['certification', 'cert', 'certified']):
        return """I hold several prestigious certifications that complement my practical experience. I have a Master of Computer Science from George Mason University with a 3.47/4.00 GPA, where I focused on Machine Learning and Software Engineering.

My professional certifications include Advanced Learning Algorithms from Stanford University (2023), which covers neural networks and deep learning techniques. I also completed Artificial Intelligence I from IBM (2022), focusing on AI fundamentals and machine learning principles.

Additionally, I earned a Deep Learning Specialization from Coursera (2021), which covered neural networks, CNN, RNN, and LSTM architectures. I'm currently pursuing AWS Solutions Architect and Kubernetes Administration certifications to further enhance my cloud and DevOps expertise.

These certifications, combined with my 4+ years of hands-on experience, provide me with both theoretical knowledge and practical skills needed for complex software development projects."""
    
    else:
        return f"""Hi! I'm Venkateswara Rao Narra, a Full Stack Python Developer with 4+ years of experience building real-world web applications and ML-based features. I specialize in creating scalable, high-performance systems that solve real business problems.

I'd be happy to help you with '{query}'! I have extensive experience in full-stack development, machine learning, cloud platforms, and building production-ready applications. My expertise spans from backend API development with FastAPI and Django to frontend applications with React and TypeScript, and includes ML model development and cloud deployment on AWS and Azure.

What specific aspect would you like to know more about - my technical skills, project experience, or how I can contribute to your team?""" 