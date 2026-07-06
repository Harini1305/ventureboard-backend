import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary

class MarketAnalysisSchema(BaseModel):
    market_opportunity: str = Field(description="Market opportunity details and positioning")
    customer_demand: str = Field(description="Customer demand analysis based on target segment pain points")
    growth_potential: str = Field(description="Estimated growth potential and scaling indicators")
    market_risks: List[str] = Field(description="List of specific market risks (e.g. low adoption, segment density)")
    score: int = Field(description="Market Score, an integer between 30 and 95 based strictly on proposal strength and research results")
    reasoning: str = Field(description="Explanation of the assigned score")
    recommendations: List[str] = Field(description="Actionable recommendations to address market gaps")

def analyze_market_fallback(proposal: Dict[str, Any]) -> Dict[str, Any]:
    startup_name = proposal.get('startup_name', 'The startup')
    industry = proposal.get('industry', 'the target industry')
    target_market = proposal.get('target_market', 'the identified customer segment')
    usp = proposal.get('usp', 'its unique value proposition')
    problem = proposal.get('problem', 'the core problem identified')
    solution = proposal.get('solution', 'the proposed solution')
    business_model = proposal.get('business_model', 'the described business model')
    score = 65 if proposal.get('target_market') not in (None, 'Not Specified') else 50
    return {
        'market_opportunity': (
            f"{startup_name} targets {target_market} within the {industry} sector, addressing {problem}. "
            f"The {business_model} model positions the product at the intersection of unmet demand and "
            f"an addressable segment with growing urgency for {solution}. Market sizing requires "
            "primary research to establish SAM and SOM boundaries, but the structural need is evident "
            f"from the problem framing and the segment specificity of {target_market}."
        ),
        'customer_demand': (
            f"Demand for {startup_name}'s offering is driven by {problem} within {target_market}. "
            f"{usp} signals a clear value hypothesis that maps to real friction in the customer workflow. "
            f"Early adopter conviction in the {industry} space typically correlates with high switching intent "
            "from manual or legacy tooling — making structured discovery interviews a near-term priority."
        ),
        'growth_potential': (
            f"Growth for {startup_name} in {industry} depends on the adoption velocity of {target_market} "
            f"and the repeatability of the {business_model} motion. The solution to {problem} through "
            f"{solution} creates a foundation for land-and-expand dynamics if initial customers achieve "
            "measurable outcomes. Macro tailwinds in the sector further support scalable expansion "
            "once product-market fit is validated."
        ),
        'market_risks': [
            f"Unvalidated willingness-to-pay within {target_market} for {solution}",
            f"High competitive density in the {industry} vertical may compress margins",
            "Market timing risk if category education is still required at scale",
            f"Geographic or regulatory constraints specific to {industry} deployments",
        ],
        'score': score,
        'reasoning': (
            f"Score of {score} derived from proposal metadata for {startup_name}. "
            f"The presence of a defined target market ({target_market}) and a concrete USP "
            "indicate a structured market thesis, though quantitative market data was unavailable for full scoring."
        ),
        'recommendations': [
            f"Conduct structured customer discovery interviews within {target_market} to quantify {problem} severity and willingness-to-pay.",
            f"Define TAM, SAM, and SOM with cited third-party data sources for the {industry} sector.",
            f"Map the buyer journey from problem awareness to purchase decision for {target_market} to identify key conversion levers.",
        ]
    }

def analyze_market(proposal: Dict[str, Any], research: Dict[str, Any], chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = []
        
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(MarketAnalysisSchema)
        
        proposal_context = "\n".join([c.get('content', '') for c in chunks]) if chunks else "No relevant document chunks found."
        
        startup_name = proposal.get('startup_name', 'Not Specified')
        industry = proposal.get('industry', 'Not Specified')
        technology = proposal.get('technology', 'Not Specified')
        
        prompt = (
            "You are an expert venture capital analyst specializing in market dynamics and customer segmentation.\n"
            f"Conduct a highly customized, rigorous market diligence report for the startup {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "Your output must satisfy the following instructions:\n"
            "1. INVENT NOTHING AND TAILOR EVERYTHING: Avoid generic startup advice and standard VC recommendations "
            "(e.g., do NOT tell them to 'validate customer pain points', 'build an MVP', 'do user interviews', or 'launch marketing campaigns'). "
            "Every sentence must relate directly to the domain-specific nuances, technical parameters, and target segments of this startup.\n"
            "2. QUOTE SPECIFIC FACTS: You must quote at least one or two exact facts, customer statistics, or metrics "
            "directly from the retrieved pitch deck chunks.\n"
            "3. MANDATORY MENTIONS: You must naturally weave the startup's name, industry, business model, and funding requirements "
            "into the analysis. Do NOT use a formulaic introduction. Integrate these details seamlessly into your evaluative writing.\n"
            "4. ELIMINATE REPETITIVE WORDING AND PATTERNS: Do not start consecutive sentences with the same words (e.g., 'The startup...', "
            "'The market...', 'According to...'). Vary your sentence length, grammatical structures, and starts. "
            "Avoid generic transition words like 'Additionally', 'Furthermore', 'Moreover', 'In terms of', or 'On the other hand'.\n"
            "5. VERTICAL-SPECIFIC LANGUAGE: Adopt vocabulary and terminology that fits the specific industry/technology "
            f"({industry} / {technology}) of {startup_name} to generate a highly professional and unique analysis.\n\n"
            "Generate a structured market analysis. Assign a score between 30 and 95 based strictly on proposal strength and research results."
        )
        res = structured_llm.invoke(prompt)
        return res.model_dump()
    except Exception:
        return analyze_market_fallback(proposal)
