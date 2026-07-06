import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary

class CompetitionAnalysisSchema(BaseModel):
    direct_competitors: List[str] = Field(description="List of direct startup or enterprise competitors")
    indirect_competitors: List[str] = Field(description="List of indirect competitors, legacy tools, or workarounds")
    competitive_advantage: str = Field(description="Analysis of the startup's defensive competitive advantage or moat")
    differentiation: str = Field(description="Details on how the product/service differentiates itself from competitors")
    score: int = Field(description="Competition Score, an integer between 30 and 95 based strictly on USP and competitor landscape")
    recommendations: List[str] = Field(description="Actionable recommendations to build a defensive market positioning")

def analyze_competition_fallback(proposal: Dict[str, Any]) -> Dict[str, Any]:
    startup_name = proposal.get('startup_name', 'The startup')
    industry = proposal.get('industry', 'the target industry')
    usp = proposal.get('usp', 'its unique value proposition')
    solution = proposal.get('solution', 'the proposed solution')
    business_model = proposal.get('business_model', 'the described business model')
    target_market = proposal.get('target_market', 'the identified customer segment')
    score = 65 if proposal.get('usp') not in (None, 'Not Specified') else 50
    return {
        'direct_competitors': [
            f"Established players in the {industry} sector with overlapping product offerings",
            f"Venture-backed startups targeting {target_market} with similar positioning",
        ],
        'indirect_competitors': [
            f"Manual workflows and legacy tools currently used by {target_market}",
            "Spreadsheet-based workarounds and in-house custom solutions",
        ],
        'competitive_advantage': (
            f"{startup_name}'s primary moat lies in {usp}. Within the {industry} vertical, "
            f"a {business_model} model that directly addresses {solution} creates a defensible positioning "
            f"relative to incumbents who rely on generic or fragmented tooling for {target_market}. "
            "Switching costs will deepen as customer workflows integrate the product more deeply."
        ),
        'differentiation': (
            f"Unlike established competitors in {industry}, {startup_name} differentiates through {usp}. "
            f"The focus on {target_market} with a purpose-built {solution} reduces the feature-bloat "
            f"and workflow friction typical of legacy alternatives. The {business_model} model further "
            "reinforces retention by aligning product value directly with customer outcomes."
        ),
        'score': score,
        'recommendations': [
            f"Produce a competitive matrix mapping {startup_name}'s capabilities against the top three named incumbents in {industry}.",
            f"Quantify switching costs: define the workflow depth and integration touchpoints that lock {target_market} customers in.",
            f"Articulate the '10x better' claim for {usp} with measurable benchmarks relative to the closest direct competitor.",
        ]
    }

def analyze_competition(proposal: Dict[str, Any], research: Dict[str, Any], chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = []
        
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(CompetitionAnalysisSchema)
        
        proposal_context = "\n".join([c.get('content', '') for c in chunks]) if chunks else "No relevant document chunks found."
        
        startup_name = proposal.get('startup_name', 'Not Specified')
        industry = proposal.get('industry', 'Not Specified')
        business_model = proposal.get('business_model', 'Not Specified')
        
        prompt = (
            "You are an expert competitive market strategist specializing in defensive moats, USP differentiation, and barrier-to-entry mechanics.\n"
            f"Analyze the competitive landscape and defensive positioning for {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "Your output must satisfy the following instructions:\n"
            "1. INVENT NOTHING AND TAILOR EVERYTHING: Avoid generic competitive advice "
            "(e.g., do NOT tell them to 'focus on marketing', 'provide better support', 'build a brand', or 'move faster'). "
            "Every statement must address the concrete differentiators, customer acquisition channels, or tech-driven moats relative to specific named incumbents and indirect workarounds.\n"
            "2. QUOTE SPECIFIC FACTS: You must quote specific competitive facts, unique features, or competitor names "
            "directly from the retrieved pitch deck chunks.\n"
            "3. MANDATORY MENTIONS: Natural integration of the company name, sector, business model, and funding requirements "
            "into the analysis prose is required. Do NOT use formulaic introductory sentences.\n"
            "4. ELIMINATE REPETITIVE WORDING AND PATTERNS: Do not start consecutive sentences with identical words or constructs "
            "(e.g. avoid beginning multiple sentences with 'The competitor...', 'Moats are...', 'Based on...'). Use varied sentence structures. "
            "Do not include cliché transition words such as 'Additionally', 'Furthermore', 'Moreover', 'Consequently', or 'Indeed'.\n"
            "5. DOMAIN-SPECIFIC VOCABULARY: Leverage specific competitive dynamics terms (e.g. high switching costs, network effects, distribution channels, feature parity) "
            f"customized to the {industry} and '{business_model}' model of {startup_name} to generate a highly professional and unique analysis.\n\n"
            "Generate a structured competition analysis. Assign a score between 30 and 95 based strictly on USP and competitor landscape."
        )
        res = structured_llm.invoke(prompt)
        return res.model_dump()
    except Exception:
        return analyze_competition_fallback(proposal)
