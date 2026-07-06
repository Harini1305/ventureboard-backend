import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm

class SimulationResultSchema(BaseModel):
    investment_score: int = Field(
        description="The estimated new overall investment score after the scenario is applied (integer from 0 to 100)."
    )
    overall_risk: str = Field(
        description="The estimated new risk level. Must be exactly one of: 'Low', 'Moderate', 'High', or 'Critical'."
    )
    investment_recommendation: str = Field(
        description="The estimated new investment recommendation. Must be exactly one of: 'Proceed', 'Proceed with caution', or 'Defer'."
    )
    confidence_score: int = Field(
        description="The estimated new confidence score (integer from 0 to 100)."
    )
    executive_summary: str = Field(
        description="A short executive summary (2-3 sentences) explaining the key impact of this scenario on the startup."
    )
    recommendation_changes: str = Field(
        description="A clear description of how the recommendation/stance changes compared to the original recommendation."
    )
    key_benefits: List[str] = Field(
        description="2-4 key benefits/upsides introduced by this scenario."
    )
    new_risks: List[str] = Field(
        description="2-4 new risks/challenges introduced by this scenario."
    )
    why_score_changed: str = Field(
        description="A detailed explanation (2-3 sentences) of exactly which business assumptions changed and why the investment score changed."
    )

def run_scenario_simulation(report: Dict[str, Any], scenario_type: str, scenario_value: str | None = None) -> Dict[str, Any]:
    """
    Simulates a hypothetical change to the startup business assumptions.
    Reuses existing proposal and research metadata, and requests the LLM
    to generate updated score, risks, recommendation, confidence and key changes.
    """
    proposal = report.get("proposal_summary") or {}
    research = report.get("research_summary") or {}
    
    # Extract original downstream agent summaries / scores if available
    market_score = (report.get("market_analysis") or {}).get("score", "N/A")
    finance_score = (report.get("finance_analysis") or {}).get("score", "N/A")
    tech_score = (report.get("technology_analysis") or {}).get("score", "N/A")
    comp_score = (report.get("competition_analysis") or {}).get("score", "N/A")
    risk_score = (report.get("risk_assessment") or {}).get("risk_score", "N/A")

    original_score = report.get("investment_score") or (report.get("investment_committee") or {}).get("investment_score", 0)
    original_risk = (report.get("risk_assessment") or {}).get("overall_risk") or "Moderate"
    original_rec = report.get("investment_recommendation") or (report.get("investment_committee") or {}).get("investment_recommendation", "Pending")
    original_conf = report.get("confidence_score") or (report.get("investment_committee") or {}).get("confidence_score", 0)
    original_summary = report.get("executive_summary") or (report.get("investment_committee") or {}).get("executive_summary", "")

    # Setup prompt
    prompt = (
        "You are an expert venture capitalist and senior startup diligence analyst. "
        "Your job is to simulate how a hypothetical business scenario would affect the startup's metrics and diligence assessment. "
        "Keep all original findings unchanged, and focus on evaluating the net-new change.\n\n"
        "=== STARTUP ORIGINAL PROFILE ===\n"
        f"- Startup Name: {proposal.get('startup_name', 'Not Specified')}\n"
        f"- Industry/Sector: {proposal.get('industry', 'Not Specified')}\n"
        f"- Core Problem: {proposal.get('problem', 'Not Specified')}\n"
        f"- Solution: {proposal.get('solution', 'Not Specified')}\n"
        f"- Unique Value Proposition (USP): {proposal.get('usp', 'Not Specified')}\n"
        f"- Business Model: {proposal.get('business_model', 'Not Specified')}\n"
        f"- Revenue Model: {proposal.get('revenue_model', 'Not Specified')}\n"
        f"- Technology Stack: {proposal.get('technology', 'Not Specified')}\n"
        f"- Funding Required (Original): {proposal.get('funding_required', 'Not Specified')}\n"
        f"- Target Market: {proposal.get('target_market', 'Not Specified')}\n\n"
        "=== ORIGINAL DILIGENCE METRICS ===\n"
        f"- Overall Investment Score: {original_score}/100\n"
        f"- Risk Level: {original_risk}\n"
        f"- Recommendation: {original_rec}\n"
        f"- Confidence Score: {original_conf}%\n"
        f"- Executive Summary: {original_summary}\n\n"
        "=== DOWNSTREAM AGENT METRICS ===\n"
        f"- Market Analysis Score: {market_score}/100\n"
        f"- Finance Analysis Score: {finance_score}/100\n"
        f"- Technology Analysis Score: {tech_score}/100\n"
        f"- Competition Analysis Score: {comp_score}/100\n"
        f"- Risk Score: {risk_score}/100\n\n"
        "=== HYPOTHETICAL SCENARIO TO EVALUATE ===\n"
        f"- Scenario Type: {scenario_type}\n"
        f"- Value/Details: {scenario_value or 'Not Specified'}\n\n"
        "=== SIMULATION RULES ===\n"
        "1. Assess the scenario objectively. For example:\n"
        "   - 'Increase Funding': May boost cash runway, improve finance score, and accelerate R&D/GTM, but might increase execution risks or governance complexity if the team lacks experience.\n"
        "   - 'Decrease Funding': Increases runway risk, lowers finance score, but could force team lean efficiency.\n"
        "   - 'Reduce Customer Acquisition Cost': Directly improves financial margins, unit economics, scaling potential, and therefore improves scores.\n"
        "   - 'Improve Team Experience': Reduces risk levels and boosts the investment score and execution confidence significantly.\n"
        "   - 'Expand to New Market': Increases addressable market opportunities, but increases competition risk and operating expenses.\n"
        "2. Keep the updated investment_score an integer between 0 and 100.\n"
        "3. Keep the overall_risk exactly one of: 'Low', 'Moderate', 'High', 'Critical'.\n"
        "4. Keep the investment_recommendation exactly one of: 'Proceed', 'Proceed with caution', 'Defer'.\n"
        "5. Fill out all other fields in the schema in detail.\n\n"
        "Compute the simulation results and generate the JSON response matching the required schema."
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(SimulationResultSchema)
    result = structured_llm.invoke(prompt)
    return result.model_dump()
