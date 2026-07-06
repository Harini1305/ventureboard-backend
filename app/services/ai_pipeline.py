from typing import Dict, Any
from app.services.langgraph_orchestrator import run_agents


def run_diligence_pipeline(text: str, filename: str, vector_store=None, audit_trail=None) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("The document text is empty.")

    orchestrated = run_agents(text, vector_store=vector_store, audit_trail=audit_trail)
    committee = orchestrated["investment_committee"]

    return {
        "document_name": filename,
        "proposal_summary": orchestrated["proposal_summary"],
        "research_summary": orchestrated["research_summary"],
        "market_analysis": orchestrated["market_analysis"],
        "finance_analysis": orchestrated["finance_analysis"],
        "technology_analysis": orchestrated["technology_analysis"],
        "competition_analysis": orchestrated["competition_analysis"],
        "risk_assessment": orchestrated["risk_assessment"],
        "investment_committee": committee,
        # Top-level convenience fields used by PDF generator and frontend
        "executive_summary": committee.get("executive_summary", ""),
        "investment_recommendation": committee.get("investment_recommendation", "Pending"),
        "investment_score": committee.get("investment_score", 0),
        "confidence_score": committee.get("confidence_score", 0),
        # New fields
        "recommendation_rationale": committee.get("recommendation_rationale", {}),
        "score_breakdown": committee.get("score_breakdown", {}),
        "vector_matches": orchestrated.get("vector_matches", []),
        "audit_trail": orchestrated.get("audit_trail", [])
    }
