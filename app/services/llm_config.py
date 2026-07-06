import os
from typing import Dict, Any
import groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SafeStructuredRunnable:
    def __init__(self, runnable, schema):
        self._runnable = runnable
        self._schema = schema

    def invoke(self, *args, **kwargs):
        try:
            return self._runnable.invoke(*args, **kwargs)
        except Exception as e:
            # Let's traverse the causal chain of the exception to find a groq.APIStatusError
            err = e
            api_status_err = None
            while err is not None:
                if isinstance(err, groq.APIStatusError):
                    api_status_err = err
                    break
                err = getattr(err, "__cause__", None)

            if api_status_err and api_status_err.status_code == 400:
                body = api_status_err.body
                if isinstance(body, dict) and "error" in body:
                    err_detail = body["error"]
                    failed_gen = err_detail.get("failed_generation")
                    if failed_gen:
                        # Extract the JSON block
                        import re
                        import json
                        json_match = re.search(r'\{.*\}', failed_gen, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            # Clean up invalid quote escaping
                            cleaned_json_str = json_str.replace("\\'", "'").replace("\'", "'")
                            try:
                                parsed = json.loads(cleaned_json_str)
                                if hasattr(self._schema, "model_validate"):
                                    return self._schema.model_validate(parsed)
                                return parsed
                            except Exception:
                                pass # If rescue parsing fails, fall through to raise standard formatted error

            # Traverse to find any GroqError for user-friendly raising
            err = e
            groq_err = None
            while err is not None:
                if isinstance(err, groq.GroqError):
                    groq_err = err
                    break
                err = getattr(err, "__cause__", None)

            if groq_err:
                if isinstance(groq_err, groq.AuthenticationError):
                    raise ValueError(f"Groq API Error: The provided GROQ_API_KEY is invalid. Please check your .env file. Details: {groq_err}") from e
                elif isinstance(groq_err, groq.RateLimitError):
                    raise RuntimeError(f"Groq API Error: Rate limit exceeded. Please try again later. Details: {groq_err}") from e
                elif isinstance(groq_err, (groq.APIConnectionError, groq.APITimeoutError)):
                    raise ConnectionError(f"Groq API Error: Network/connection error occurred while contacting Groq API. Please check your network connection. Details: {groq_err}") from e
                elif isinstance(groq_err, groq.APIStatusError):
                    raise RuntimeError(f"Groq API Error: Groq API returned status code {groq_err.status_code}. Details: {groq_err}") from e
                else:
                    raise RuntimeError(f"Groq API Error: An unexpected error occurred: {groq_err}") from e
            
            raise RuntimeError(f"Groq API Error: An unexpected error occurred: {e}") from e

    def __getattr__(self, name):
        return getattr(self._runnable, name)

class SafeChatGroq(ChatGroq):
    def invoke(self, *args, **kwargs):
        try:
            return super().invoke(*args, **kwargs)
        except Exception as e:
            # Traverse to find any GroqError for user-friendly raising
            err = e
            groq_err = None
            while err is not None:
                if isinstance(err, groq.GroqError):
                    groq_err = err
                    break
                err = getattr(err, "__cause__", None)

            if groq_err:
                if isinstance(groq_err, groq.AuthenticationError):
                    raise ValueError(f"Groq API Error: The provided GROQ_API_KEY is invalid. Please check your .env file. Details: {groq_err}") from e
                elif isinstance(groq_err, groq.RateLimitError):
                    raise RuntimeError(f"Groq API Error: Rate limit exceeded. Please try again later. Details: {groq_err}") from e
                elif isinstance(groq_err, (groq.APIConnectionError, groq.APITimeoutError)):
                    raise ConnectionError(f"Groq API Error: Network/connection error occurred while contacting Groq API. Please check your network connection. Details: {groq_err}") from e
                elif isinstance(groq_err, groq.APIStatusError):
                    raise RuntimeError(f"Groq API Error: Groq API returned status code {groq_err.status_code}. Details: {groq_err}") from e
                else:
                    raise RuntimeError(f"Groq API Error: An unexpected error occurred: {groq_err}") from e
            
            raise RuntimeError(f"Groq API Error: An unexpected error occurred: {e}") from e

    def with_structured_output(self, schema, *args, **kwargs):
        structured_runnable = super().with_structured_output(schema, *args, **kwargs)
        return SafeStructuredRunnable(structured_runnable, schema)

def get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please supply a valid Groq API key.")
    return SafeChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=api_key
    )


def format_proposal_summary(proposal: Dict[str, Any], max_words: int = 250) -> str:
    if not proposal:
        return "No proposal summary available."
    
    keys = [
        ("Startup Name", "startup_name"),
        ("Industry/Sector", "industry"),
        ("Core Problem", "problem"),
        ("Proposed Solution", "solution"),
        ("Business Model", "business_model"),
        ("Revenue Model", "revenue_model"),
        ("Target Market", "target_market"),
        ("Technology Stack", "technology"),
        ("Unique Value Proposition (USP)", "usp"),
        ("Funding Required", "funding_required")
    ]
    
    parts = []
    for label, key in keys:
        val = proposal.get(key)
        if val and val not in ("Not Specified", "Not specified in the uploaded proposal."):
            parts.append(f"- {label}: {val}")
            
    summary_str = "\n".join(parts)
    
    # Enforce word limit
    words = summary_str.split()
    if len(words) > max_words:
        summary_str = " ".join(words[:max_words]) + "..."
        
    return summary_str


def format_research_summary(research: Dict[str, Any], max_words: int = 300) -> str:
    if not research:
        return "No research summary available."
        
    parts = []
    
    # Try active fields first
    market_size = research.get("market_size")
    if market_size and market_size not in ("Not Specified", "Not specified in the uploaded proposal."):
        parts.append(f"Market Size: {market_size}")
        
    trends = research.get("trends") or research.get("market_trends")
    if trends and isinstance(trends, list):
        trends = [t for t in trends if t and t not in ("Not Specified", "Not specified in the uploaded proposal.")]
        if trends:
            parts.append("Market & Industry Trends:\n" + "\n".join(f"- {t}" for t in trends))
            
    competitors = research.get("competitors")
    if competitors and isinstance(competitors, list):
        competitors = [c for c in competitors if c and c not in ("Not Specified", "Not specified in the uploaded proposal.")]
        if competitors:
            parts.append("Key Competitors:\n" + "\n".join(f"- {c}" for c in competitors))
            
    regulations = research.get("regulations")
    if regulations and regulations not in ("Not Specified", "Not specified in the uploaded proposal."):
        parts.append(f"Regulations & Compliance: {regulations}")
        
    funding_trends = research.get("funding_trends") or research.get("recent_funding")
    if funding_trends and funding_trends not in ("Not Specified", "Not specified in the uploaded proposal."):
        parts.append(f"Funding Trends: {funding_trends}")
        
    # If everything is empty, try industry_overview
    if not parts:
        industry_overview = research.get("industry_overview")
        if industry_overview and industry_overview not in ("Not Specified", "Not specified in the uploaded proposal."):
            parts.append(f"Industry Overview: {industry_overview}")
            
    research_str = "\n\n".join(parts)
    
    # Enforce word limit
    words = research_str.split()
    if len(words) > max_words:
        research_str = " ".join(words[:max_words]) + "..."
        
    return research_str


