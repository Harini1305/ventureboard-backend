import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary

class TechnologyAnalysisSchema(BaseModel):
    innovation: str = Field(description="Evaluation of technical innovation and IP defensibility")
    scalability: str = Field(description="Analysis of architecture scalability, capacity, and operational overhead")
    technical_feasibility: str = Field(description="Development feasibility, complexity, and timeline risks")
    technology_risks: List[str] = Field(description="Key technology risks (e.g. data lock-in, latency, model drift, API dependency)")
    score: int = Field(description="Technology Score, an integer between 30 and 95 based strictly on proposal technology stack and research trends")
    recommendations: List[str] = Field(description="Actionable recommendations to scale technology or secure system architectures")

def analyze_technology_fallback(proposal: Dict[str, Any]) -> Dict[str, Any]:
    startup_name = proposal.get('startup_name', 'The startup')
    industry = proposal.get('industry', 'the target industry')
    technology = proposal.get('technology', 'the described technology stack')
    solution = proposal.get('solution', 'the proposed solution')
    usp = proposal.get('usp', 'its unique value proposition')
    business_model = proposal.get('business_model', 'the described business model')
    target_market = proposal.get('target_market', 'the identified customer segment')
    score = 65 if proposal.get('technology') not in (None, 'Not Specified') else 50
    return {
        'innovation': (
            f"{startup_name} deploys {technology} to deliver {solution}, with differentiation anchored in {usp}. "
            f"The innovation claim hinges on whether the {technology} configuration introduces a genuinely novel "
            f"capability within the {industry} space or represents a competent application of existing tooling. "
            "IP defensibility will require a detailed review of any proprietary algorithms, data pipelines, "
            f"or integration architectures specific to the {industry} deployment context."
        ),
        'scalability': (
            f"The {technology} stack underpinning {startup_name} must be evaluated against the load profile "
            f"of {target_market} at projected growth volumes. A {business_model} model delivering {solution} "
            "requires horizontal scaling capabilities and defined SLA targets before institutional growth. "
            f"Infrastructure overhead in the {industry} vertical — including data residency, throughput, "
            "and latency constraints — must be explicitly mapped to the current architecture."
        ),
        'technical_feasibility': (
            f"Building {solution} on {technology} is technically achievable for a well-resourced team with "
            f"domain expertise in {industry}. Critical unknowns include integration depth with existing "
            f"{target_market} toolchains, build-versus-buy decisions for non-core components, and the "
            "engineering timeline to production-grade reliability. A phased delivery roadmap with clearly "
            "defined milestones would significantly de-risk the development execution."
        ),
        'technology_risks': [
            f"Third-party dependency risk on external components within the {technology} stack",
            f"Implementation complexity for {solution} at the scale required by {target_market}",
            f"Talent acquisition risk in specialized {technology} domains within the {industry} market",
            "Technical debt accumulation risk if early architecture decisions prioritize speed over modularity",
        ],
        'score': score,
        'recommendations': [
            f"Publish a detailed architecture diagram mapping {technology} components to the core {solution} workflow and data flows.",
            f"Define SLA targets, latency budgets, and failure-mode recovery plans applicable to the {industry} deployment environment.",
            f"Conduct a build-versus-buy analysis for non-differentiating infrastructure components to optimize the {startup_name} engineering runway.",
        ]
    }

def analyze_technology(proposal: Dict[str, Any], research: Dict[str, Any], chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if chunks is None:
        chunks = []
        
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(TechnologyAnalysisSchema)
        
        proposal_context = "\n".join([c.get('content', '') for c in chunks]) if chunks else "No relevant document chunks found."
        
        startup_name = proposal.get('startup_name', 'Not Specified')
        industry = proposal.get('industry', 'Not Specified')
        technology = proposal.get('technology', 'Not Specified')
        
        prompt = (
            "You are an expert technical startup evaluator and chief architect evaluating codebase feasibility, innovation, and scalability.\n"
            f"Evaluate the technology stack and architecture for {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "Your output must satisfy the following instructions:\n"
            "1. INVENT NOTHING AND TAILOR EVERYTHING: Avoid generic technical advice "
            "(e.g., do NOT tell them to 'document the code', 'choose a scalable cloud provider', 'use microservices', 'write unit tests', or 'implement CI/CD'). "
            "Every statement must deal directly with the unique engineering stack, framework structures, databases, API architectures, or model topologies specified.\n"
            "2. QUOTE SPECIFIC FACTS: You must quote specific technological facts, libraries, system components, or parameters "
            "directly from the retrieved pitch deck chunks.\n"
            "3. MANDATORY MENTIONS: Incorporate the company name, sector, business model, and funding requirements "
            "seamlessly into the analysis text. Do NOT use boilerplate introductory sentences.\n"
            "4. ELIMINATE REPETITIVE WORDING AND PATTERNS: Do not start consecutive sentences with identical words or formats "
            "(e.g. avoid beginning multiple sentences with 'The tech...', 'This technology...', 'Based on...'). Use varied sentence structures. "
            "Do not include cliché transition words such as 'Additionally', 'Furthermore', 'Moreover', 'Consequently', or 'Indeed'.\n"
            "5. VERTICAL-SPECIFIC LANGUAGE: Use technical vocabulary and concepts (e.g. consensus engines, vector embeddings, latency metrics, synchronization) "
            f"that naturally match the specific industry/technology ({industry} / {technology}) of {startup_name} to generate a highly professional, unique analysis.\n\n"
            "Generate a structured technology analysis. Assign a score between 30 and 95 based strictly on proposal technology stack and research trends."
        )
        res = structured_llm.invoke(prompt)
        return res.model_dump()
    except Exception:
        return analyze_technology_fallback(proposal)
