import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary

class FinanceAnalysisSchema(BaseModel):
    funding_analysis: str = Field(description="Analysis of the requested funding and estimated allocation efficiency")
    revenue_analysis: str = Field(description="Monetization model, pricing strength, and scalability analysis")
    financial_sustainability: str = Field(description="Runway sustainability, burn-rate risks, and breakeven indicators")
    roi: str = Field(description="Estimated Return on Investment (ROI) profile and VC return multiple likelihood")
    financial_risks: List[str] = Field(description="Key financial risks (e.g. high burn rate, high customer acquisition costs)")
    score: int = Field(description="Financial Score, an integer between 30 and 95 based strictly on proposal finances and market trends")
    recommendations: List[str] = Field(description="Actionable recommendations to improve financial health or unit economics")

def analyze_finance_fallback(proposal: Dict[str, Any]) -> Dict[str, Any]:
    score = 65 if proposal.get('funding_required') != 'Not Specified' else 50
    return {
        'funding_analysis': 'Diligence pending OpenAI API setup.',
        'revenue_analysis': 'Diligence pending OpenAI API setup.',
        'financial_sustainability': 'Diligence pending OpenAI API setup.',
        'roi': 'Diligence pending OpenAI API setup.',
        'financial_risks': ['General runway risk'],
        'score': score,
        'reasoning': 'Rule-based fallback due to API failure or configuration missing.',
        'recommendations': ['Provide a detailed breakdown of runway allocation and target milestones.']
    }

def analyze_finance(proposal: Dict[str, Any], research: Dict[str, Any], chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = []
        
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(FinanceAnalysisSchema)
        
        proposal_context = "\n".join([c.get('content', '') for c in chunks]) if chunks else "No relevant document chunks found."
        
        startup_name = proposal.get('startup_name', 'Not Specified')
        industry = proposal.get('industry', 'Not Specified')
        business_model = proposal.get('business_model', 'Not Specified')
        
        prompt = (
            "You are a startup financial diligence expert specializing in unit economics, revenue architectures, and runway calculations.\n"
            f"Perform a comprehensive financial evaluation for {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "Your output must satisfy the following instructions:\n"
            "1. INVENT NOTHING AND TAILOR EVERYTHING: Avoid generic financial advice "
            "(e.g., do NOT tell them to 'monitor cash flow', 'optimize unit economics', 'maintain a low burn rate', or 'track operational expenses'). "
            "Every sentence must focus strictly on the specific revenue model, pricing structures, and financial projections of this business.\n"
            "2. QUOTE SPECIFIC FACTS: You must quote specific financial facts, figures, pricing points, runway estimates, "
            "or target metrics directly from the retrieved pitch deck chunks.\n"
            "3. MANDATORY MENTIONS: Natural integration of the company name, sector, business model, and funding requirements "
            "into the prose is required. Do NOT use boilerplate introductory sentences.\n"
            "4. ELIMINATE REPETITIVE WORDING AND PATTERNS: Do not start consecutive sentences with identical words or constructs "
            "(e.g. avoid beginning multiple sentences with 'The company...', 'Funding will...', 'Based on...'). Use varied sentence formats. "
            "Do not include cliché transition words such as 'Additionally', 'Furthermore', 'Moreover', 'Consequently', or 'Indeed'.\n"
            "5. SECTOR-APPROPRIATE TERMINOLOGY: Adapt terminology (e.g., CAC, LTV, ARR, margins, transaction fees) specifically to fit "
            f"the {industry} sector and '{business_model}' business model of {startup_name} to generate a highly professional and unique analysis.\n\n"
            "Generate a structured financial analysis. Assign a score between 30 and 95 based strictly on proposal finances and market trends."
        )
        res = structured_llm.invoke(prompt)
        return res.model_dump()
    except Exception:
        return analyze_finance_fallback(proposal)
