"""
Response Guardrails System for Career Assistant
Ensures perfect, comprehensive responses with validation and enhancement
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from llama_index.llms.gemini import Gemini
from .config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class ResponseGuardrails:
    """
    Guardrails system that ensures perfect responses by:
    1. Validating response completeness
    2. Enhancing responses with missing information
    3. Ensuring consistent formatting
    4. Checking for accuracy
    5. Adding context when needed
    """
    
    def __init__(self):
        self.llm = Gemini(model_name="gemini-1.5-flash", api_key=GEMINI_API_KEY)
        self.quality_thresholds = {
            "min_length": 150,
            "max_length": 2000,
            "required_elements": ["specific_info", "context", "professionalism"],
            "forbidden_elements": ["[", "]", "TODO", "placeholder", "TBD"]
        }
        
        # Expected information for different query types
        self.expected_content = {
            "skills": {
                "required": ["programming_languages", "frameworks", "technologies", "experience_years"],
                "preferred": ["specific_projects", "achievements", "certifications", "tools"]
            },
            "experience": {
                "required": ["current_role", "company", "duration", "achievements"],
                "preferred": ["previous_roles", "career_progression", "measurable_impact", "technologies_used"]
            },
            "education": {
                "required": ["degrees", "universities", "graduation_dates", "gpa"],
                "preferred": ["specializations", "relevant_coursework", "academic_achievements", "locations"]
            },
            "projects": {
                "required": ["project_names", "technologies", "impact", "descriptions"],
                "preferred": ["github_links", "live_demos", "team_size", "duration"]
            },
            "contact": {
                "required": ["email", "phone", "professional_profiles"],
                "preferred": ["calendly", "availability", "response_time", "preferred_contact_method"]
            }
        }
    
    def _classify_response_type(self, query: str) -> str:
        """Classify the type of response expected"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['skill', 'technology', 'programming', 'technical']):
            return 'skills'
        elif any(keyword in query_lower for keyword in ['experience', 'work', 'job', 'career']):
            return 'experience'
        elif any(keyword in query_lower for keyword in ['education', 'degree', 'university', 'study']):
            return 'education'
        elif any(keyword in query_lower for keyword in ['project', 'built', 'developed', 'portfolio']):
            return 'projects'
        elif any(keyword in query_lower for keyword in ['contact', 'email', 'phone', 'reach']):
            return 'contact'
        else:
            return 'general'
    
    def _validate_response_structure(self, response: str) -> Dict[str, Any]:
        """Validate response structure and content"""
        validation_results = {
            "passes_validation": True,
            "issues": [],
            "suggestions": [],
            "quality_score": 0
        }
        
        # Check length
        if len(response) < self.quality_thresholds["min_length"]:
            validation_results["issues"].append(f"Response too short ({len(response)} chars, min: {self.quality_thresholds['min_length']})")
            validation_results["passes_validation"] = False
        
        if len(response) > self.quality_thresholds["max_length"]:
            validation_results["issues"].append(f"Response too long ({len(response)} chars, max: {self.quality_thresholds['max_length']})")
            validation_results["suggestions"].append("Consider condensing information while maintaining detail")
        
        # Check for forbidden elements
        for forbidden in self.quality_thresholds["forbidden_elements"]:
            if forbidden in response:
                validation_results["issues"].append(f"Contains forbidden element: {forbidden}")
                validation_results["passes_validation"] = False
        
        # Check for professional tone
        if not self._has_professional_tone(response):
            validation_results["issues"].append("Response lacks professional tone")
            validation_results["suggestions"].append("Add professional language and structure")
        
        # Check for specific information
        if not self._has_specific_information(response):
            validation_results["issues"].append("Response lacks specific information")
            validation_results["suggestions"].append("Add specific metrics, dates, and concrete examples")
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(validation_results["issues"]) * 20
        quality_score -= len(validation_results["suggestions"]) * 10
        validation_results["quality_score"] = max(0, quality_score)
        
        return validation_results
    
    def _has_professional_tone(self, response: str) -> bool:
        """Check if response has professional tone"""
        professional_indicators = [
            "experience", "expertise", "skilled", "proficient", "accomplished",
            "developed", "implemented", "achieved", "delivered", "optimized"
        ]
        
        return any(indicator in response.lower() for indicator in professional_indicators)
    
    def _has_specific_information(self, response: str) -> bool:
        """Check if response contains specific information"""
        specific_patterns = [
            r'\d+\+?\s*(years?|months?)',  # Experience duration
            r'\d+%',  # Percentages
            r'\$\d+',  # Dollar amounts
            r'\d+,\d+',  # Numbers with commas
            r'\d{4}',  # Years
            r'GPA:\s*\d+\.\d+',  # GPA
        ]
        
        return any(re.search(pattern, response) for pattern in specific_patterns)
    
    def _check_content_completeness(self, response: str, query: str) -> Dict[str, Any]:
        """Check if response contains expected content for query type"""
        response_type = self._classify_response_type(query)
        
        if response_type not in self.expected_content:
            return {"complete": True, "missing_elements": []}
        
        expected = self.expected_content[response_type]
        missing_required = []
        missing_preferred = []
        
        response_lower = response.lower()
        
        # Check required elements
        for element in expected["required"]:
            if not self._element_present(element, response_lower):
                missing_required.append(element)
        
        # Check preferred elements
        for element in expected["preferred"]:
            if not self._element_present(element, response_lower):
                missing_preferred.append(element)
        
        return {
            "complete": len(missing_required) == 0,
            "missing_required": missing_required,
            "missing_preferred": missing_preferred
        }
    
    def _element_present(self, element: str, response_lower: str) -> bool:
        """Check if a specific element is present in the response"""
        element_keywords = {
            "programming_languages": ["python", "java", "javascript", "typescript", "c++"],
            "frameworks": ["react", "django", "fastapi", "spring", "flask", "angular"],
            "technologies": ["aws", "docker", "kubernetes", "postgresql", "mongodb"],
            "experience_years": ["years", "experience", "4+", "years of"],
            "current_role": ["software development engineer", "current", "veritis"],
            "company": ["veritis", "tcs", "virtusa"],
            "duration": ["2023", "2024", "2021", "2022"],
            "achievements": ["built", "developed", "achieved", "reduced", "optimized"],
            "degrees": ["master", "bachelor", "ms", "b.tech", "computer science"],
            "universities": ["george mason", "gitam", "university"],
            "graduation_dates": ["2024", "2021", "2017"],
            "gpa": ["gpa", "3.8", "3.7"],
            "project_names": ["ai career", "clinical", "chat platform", "vision"],
            "impact": ["25,000", "80%", "60%", "99.9%", "reduced", "improved"],
            "email": ["vnarrag@gmu.edu", "email"],
            "phone": ["703-453-2157", "phone"],
            "professional_profiles": ["linkedin", "github", "leetcode"]
        }
        
        if element not in element_keywords:
            return True  # Unknown element, assume present
        
        return any(keyword in response_lower for keyword in element_keywords[element])
    
    async def _enhance_response(self, original_response: str, query: str, validation_results: Dict[str, Any]) -> str:
        """Enhance response based on validation results"""
        if validation_results["passes_validation"] and validation_results["quality_score"] > 80:
            return original_response
        
        response_type = self._classify_response_type(query)
        completeness = self._check_content_completeness(original_response, query)
        
        enhancement_prompt = f"""
You are an expert career assistant enhancing a response about Venkatesh Narra, a Software Development Engineer. 

Original Query: "{query}"
Original Response: "{original_response}"

Issues Found:
{chr(10).join(f"- {issue}" for issue in validation_results["issues"])}

Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in validation_results["suggestions"])}

Missing Required Elements: {completeness.get("missing_required", [])}
Missing Preferred Elements: {completeness.get("missing_preferred", [])}

**Key Information about Venkatesh:**
- Current: Software Development Engineer at Veritis Group Inc (2023-Present)
- Education: MS Computer Science, George Mason University, Virginia, USA (GPA: 3.8/4.0); B.Tech Computer Science, GITAM Deemed University, Visakhapatnam, India (GPA: 3.7/4.0)
- Experience: 4+ years in AI/ML and full-stack development
- Key Technologies: Python, Java, JavaScript, React, FastAPI, AWS, Docker, TensorFlow, LangChain
- Major Achievements: Built AI systems handling 25,000+ daily inferences, reduced manual QA by 80%, achieved 99.9% uptime
- Contact: vnarrag@gmu.edu, +1 703-453-2157, LinkedIn: https://www.linkedin.com/in/venkateswara-narra-91170b34a

**Enhancement Instructions:**
1. Fix all identified issues
2. Add missing required elements
3. Include preferred elements when relevant
4. Ensure professional tone with specific metrics
5. Use proper formatting with emojis and structure
6. Keep response comprehensive but concise (200-500 words)
7. Remove any brackets, placeholders, or incomplete information
8. Add specific achievements and measurable impact
9. Include relevant contact information if query suggests it

Enhanced Response:"""
        
        try:
            enhanced_response = await self.llm.acomplete(enhancement_prompt)
            
            # Extract text properly
            if hasattr(enhanced_response, 'text'):
                return enhanced_response.text
            elif hasattr(enhanced_response, 'content'):
                return enhanced_response.content
            else:
                return str(enhanced_response).strip('[]"\'').strip()
                
        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            return original_response
    
    def _add_context_if_needed(self, response: str, query: str) -> str:
        """Add context if response seems incomplete"""
        query_lower = query.lower()
        
        # Add contact context for introduction queries
        if any(keyword in query_lower for keyword in ['intro', 'yourself', 'who are you', 'about you']):
            if 'contact' not in response.lower():
                response += "\n\n📞 **Contact Information:**\n• Email: vnarrag@gmu.edu\n• Phone: +1 703-453-2157\n• LinkedIn: https://www.linkedin.com/in/venkateswara-narra-91170b34a"
        
        # Add availability context for hiring queries
        if any(keyword in query_lower for keyword in ['hire', 'opportunity', 'job', 'position']):
            if 'available' not in response.lower():
                response += "\n\n🌟 **Availability:** Open to discussing new opportunities and excited to contribute to innovative projects!"
        
        return response
    
    def _final_formatting_check(self, response: str) -> str:
        """Final formatting and cleanup"""
        # Remove any remaining brackets or placeholders
        response = re.sub(r'\[.*?\]', '', response)
        response = re.sub(r'TODO:.*', '', response)
        response = re.sub(r'TBD.*', '', response)
        
        # Ensure proper spacing
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()
        
        # Ensure response ends properly
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response
    
    async def process_response(self, original_response: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process response through guardrails system"""
        logger.info("🛡️ Processing response through guardrails...")
        
        # Validate response structure
        validation_results = self._validate_response_structure(original_response)
        
        # Check content completeness
        completeness = self._check_content_completeness(original_response, query)
        
        # Enhance response if needed
        enhanced_response = original_response
        if not validation_results["passes_validation"] or validation_results["quality_score"] < 80:
            enhanced_response = await self._enhance_response(original_response, query, validation_results)
        
        # Add context if needed
        enhanced_response = self._add_context_if_needed(enhanced_response, query)
        
        # Final formatting check
        final_response = self._final_formatting_check(enhanced_response)
        
        # Final validation
        final_validation = self._validate_response_structure(final_response)
        
        processing_results = {
            "original_quality_score": validation_results["quality_score"],
            "final_quality_score": final_validation["quality_score"],
            "enhancement_applied": enhanced_response != original_response,
            "completeness": completeness,
            "validation_passed": final_validation["passes_validation"]
        }
        
        logger.info(f"🛡️ Guardrails processing complete. Quality: {validation_results['quality_score']}→{final_validation['quality_score']}")
        
        return final_response, processing_results

# Global instance
response_guardrails = ResponseGuardrails()

async def ensure_perfect_response(response: str, query: str) -> str:
    """Convenience function to ensure perfect response"""
    perfect_response, _ = await response_guardrails.process_response(response, query)
    return perfect_response 