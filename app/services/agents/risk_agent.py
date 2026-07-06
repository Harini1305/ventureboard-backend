import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary


class RiskAssessmentSchema(BaseModel):
    overall_risk: str = Field(
        description="Overall Risk profile rating (must be exactly 'Low', 'Moderate', or 'High')"
    )
    risk_score: int = Field(
        description=(
            "Numerical inverse-risk score from 0 to 100 reflecting how manageable the risk profile is. "
            "High risk → low score (0-49). Moderate risk → mid score (50-74). Low risk → high score (75-100)."
        )
    )
    critical_risks: List[str] = Field(
        description=(
            "List of 4-6 specific critical risk factors extracted exclusively from this proposal. "
            "Each entry must name the exact risk category (e.g., supply-chain concentration, "
            "regulatory approval timeline, SaaS customer churn, data-privacy liability) AND "
            "tie it to a concrete detail found in the proposal (company name, sector, model, funding, tech stack)."
        )
    )
    mitigation_strategies: List[str] = Field(
        description=(
            "List of mitigation strategies numbered to match critical_risks (Risk 1 → Mitigation 1, etc.). "
            "Each mitigation must directly address the corresponding risk with a specific, actionable "
            "countermeasure that is feasible given the startup's stage, funding, and business model. "
            "Do not reuse generic advice such as 'diversify revenue' or 'monitor progress'."
        )
    )


def assess_risk_fallback(
    market: Dict[str, Any],
    finance: Dict[str, Any],
    technology: Dict[str, Any],
    competition: Dict[str, Any],
) -> Dict[str, Any]:
    avg = (
        market.get("score", 50)
        + finance.get("score", 50)
        + technology.get("score", 50)
        + competition.get("score", 50)
    ) / 4
    if avg >= 75:
        risk_level = "Low"
        risk_score = int(round(avg))
    elif avg >= 50:
        risk_level = "Moderate"
        risk_score = int(round(avg))
    else:
        risk_level = "High"
        risk_score = int(round(avg))

    return {
        "overall_risk": risk_level,
        "risk_score": risk_score,
        "critical_risks": ["General startup execution and market-adoption risk."],
        "mitigation_strategies": [
            "Establish measurable traction milestones and a clear technical architecture blueprint."
        ],
    }


def assess_risk(
    market: Dict[str, Any],
    finance: Dict[str, Any],
    technology: Dict[str, Any],
    competition: Dict[str, Any],
    chunks: List[Dict[str, Any]] | None = None,
    proposal: Dict[str, Any] | None = None,
    research: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if chunks is None:
        chunks = []

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(RiskAssessmentSchema)

        proposal_context = (
            "\n".join([c.get("content", "") for c in chunks])
            if chunks
            else "No relevant document chunks found."
        )

        if proposal is None:
            proposal = {}
        if research is None:
            research = {}

        startup_name = proposal.get("startup_name", "Not Specified")
        industry = proposal.get("industry", "Not Specified")
        business_model = proposal.get("business_model", "Not Specified")
        funding = proposal.get("funding_required", "Not Specified")
        target_market = proposal.get("target_market", "Not Specified")
        prop_tech = proposal.get("technology", "Not Specified")
        usp = proposal.get("usp", "Not Specified")

        prompt = (
            "You are a senior investment risk officer specialising in early-stage venture risk profiles, "
            "operational vulnerability analysis, and mitigation architecture.\n\n"
            f"TASK: Produce a rigorous, proposal-specific risk assessment for {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "=== DOWNSTREAM AGENT SCORES ===\n"
            f"  * Market Analysis Score: {market.get('score', 'N/A')}/100\n"
            f"  * Financial Analysis Score: {finance.get('score', 'N/A')}/100\n"
            f"  * Technology Analysis Score: {technology.get('score', 'N/A')}/100\n"
            f"  * Competition Analysis Score: {competition.get('score', 'N/A')}/100\n\n"
            "=== MANDATORY INSTRUCTIONS ===\n"
            f"1. EXTRACT RISKS FROM THIS PROPOSAL ONLY. Every critical risk must be derived from {startup_name}'s "
            f"actual industry ({industry}), business model ({business_model}), technology ({prop_tech}), "
            f"funding situation ({funding}), operations, and target market ({target_market}). "
            "Do NOT generate risks that could apply to any startup universally.\n\n"
            "   Industry-specific risk categories to consider (select what is relevant; do NOT list all):\n"
            f"   - If {industry} involves physical goods/food: supply-chain concentration, ingredient cost inflation, "
            "     food-safety compliance, cold-chain logistics, distribution fragility.\n"
            f"   - If {industry} is SaaS/software: customer churn rate, CAC vs LTV imbalance, cybersecurity liability, "
            "     scalability bottlenecks, platform dependency risk.\n"
            f"   - If {industry} is healthcare/medtech: regulatory approval timelines (FDA/CE/local), clinical "
            "     validation requirements, HIPAA/GDPR compliance exposure, reimbursement uncertainty.\n"
            f"   - If {industry} is fintech/payments: AML/KYC compliance, payment processing partner risk, "
            "     fraud exposure, licensing requirements.\n"
            f"   - If {industry} is marketplace/platform: liquidity (supply-demand balance), take-rate compression, "
            "     disintermediation, network-effect dependency.\n"
            "   - Cross-sector: funding runway given stated burn vs. funding ask, key-person dependency, "
            "     IP protection gaps, competitive displacement risk.\n\n"
            "2. PAIR MITIGATIONS 1-TO-1: For each numbered critical risk, provide a mitigation strategy with the "
            "same number that directly addresses that specific risk. The mitigation must be actionable and feasible "
            f"for a startup at {startup_name}'s stage and funding level.\n\n"
            "3. QUOTE SPECIFIC FACTS: Each risk and mitigation entry must reference at least one concrete detail "
            f"from the proposal — e.g., the technology stack ({prop_tech}), the funding quantum ({funding}), "
            f"the target segment ({target_market}), or the USP ({usp}).\n\n"
            "4. FORBIDDEN GENERIC PHRASES: Do not write any of the following:\n"
            "   - 'validate market demand', 'hire a stronger team', 'diversify revenue', 'establish metrics',\n"
            "   - 'monitor progress', 'general execution risk', 'typical startup risks'.\n\n"
            "5. VARIED SENTENCE STRUCTURE: Do not start consecutive items with the same word pattern. "
            "Avoid filler transitions such as 'Additionally', 'Furthermore', 'Moreover'.\n\n"
            "6. RISK SCORE: Assign a risk_score integer (0-100). "
            "Low risk → 75-100. Moderate risk → 50-74. High risk → 0-49. "
            "Base this score on the severity and number of risks identified across market, finance, technology, and competition findings.\n\n"
            f"Generate a structured risk assessment for {startup_name} with overall_risk level "
            "('Low', 'Moderate', or 'High'), risk_score (0-100), critical_risks (4-6 items), "
            "and mitigation_strategies (same count, paired by number to the risks)."
        )

        res = structured_llm.invoke(prompt)
        return res.model_dump()

    except Exception:
        return assess_risk_fallback(market, finance, technology, competition)
