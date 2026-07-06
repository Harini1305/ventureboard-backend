import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary, format_research_summary


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class SWOTSchema(BaseModel):
    strengths: List[str] = Field(
        description=(
            "3-5 internal strengths drawn EXCLUSIVELY from this proposal. "
            "Each entry must reference a concrete advantage specific to this startup: "
            "e.g., a named technology or IP, a specific team credential, a stated customer traction metric, "
            "or a differentiated pricing/distribution strategy. "
            "FORBIDDEN generic entries: 'Core technology stack', 'Identified market opportunity', "
            "'Strong team', 'First-mover advantage' unless tied to verifiable proposal detail."
        )
    )
    weaknesses: List[str] = Field(
        description=(
            "3-5 internal weaknesses drawn from gaps, omissions, or concerns visible in this proposal. "
            "Cover at least two of: missing financial projections, operational limitations, funding gaps, "
            "team/execution risks, or unclear GTM strategy. "
            "Each entry must be specific to this company's situation — not universally true of all startups."
        )
    )
    opportunities: List[str] = Field(
        description=(
            "3-5 external opportunities relevant to this startup's industry and market. "
            "Include industry trends, expansion vectors, strategic partnership potential, "
            "or emerging customer demand that this startup is positioned to capture. "
            "Must be specific to the sector, geography, or business model described in the proposal."
        )
    )
    threats: List[str] = Field(
        description=(
            "3-5 external threats facing this startup. "
            "Cover at least two of: named or category competitors, regulatory headwinds, "
            "macro-economic factors, or industry-specific disruption risks. "
            "FORBIDDEN generic entries: 'Competitive startup landscape', 'Regulatory compliance standards' "
            "unless tied to a named regulation or a named competitor."
        )
    )


class RecommendationRationaleSchema(BaseModel):
    positive_factors: List[str] = Field(
        description=(
            "Top 3 specific strengths from the proposal that most increased the investment score. "
            "Each must reference a concrete detail (metric, technology, market size, traction datum). "
            "Do not list generic factors."
        )
    )
    concerns: List[str] = Field(
        description=(
            "Top 3 specific weaknesses or risks from the proposal that most reduced the investment score. "
            "Each must reference a concrete gap or vulnerability from the proposal."
        )
    )
    reason_for_score: str = Field(
        description=(
            "A 2-4 sentence explanation of why the proposal received its specific investment score. "
            "Reference the individual agent scores (market, finance, technology, competition, risk) "
            "and explain what drove the highest and lowest sub-scores."
        )
    )
    reason_for_recommendation: str = Field(
        description=(
            "A 2-4 sentence explanation of why the recommendation is 'Proceed', 'Proceed with caution', "
            "or 'Defer'. Must directly tie the recommendation to the combined evidence — not just restate the score."
        )
    )


class InvestmentCommitteeSchema(BaseModel):
    executive_summary: str = Field(
        description="Executive diligence summary synthesising all agent findings into a single cohesive narrative."
    )
    investment_recommendation: str = Field(
        description=(
            "Final recommendation rating. "
            "Must be exactly one of: 'Proceed', 'Proceed with caution', or 'Defer'."
        )
    )
    confidence_score: int = Field(
        description="Diligence confidence rating as an integer from 0 to 100 based on completeness of proposal information."
    )
    investment_score: int = Field(
        description="Investment score as an integer from 0 to 100, derived as the weighted average of all agent scores."
    )
    swot_analysis: SWOTSchema = Field(description="SWOT Analysis matrix — all entries must be proposal-specific.")
    action_plan: List[str] = Field(
        description="4-6 specific next steps for the diligence/investment process tailored to this startup."
    )
    recommendation_rationale: RecommendationRationaleSchema = Field(
        description="Structured rationale explaining the recommendation and score."
    )


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------

def draft_committee_fallback(
    proposal: Dict[str, Any],
    market: Dict[str, Any],
    finance: Dict[str, Any],
    technology: Dict[str, Any],
    competition: Dict[str, Any],
    risk: Dict[str, Any],
) -> Dict[str, Any]:
    m_score = market.get("score", 50)
    f_score = finance.get("score", 50)
    t_score = technology.get("score", 50)
    c_score = competition.get("score", 50)
    r_score = risk.get("risk_score", 50)

    investment_score = int(round((m_score + f_score + t_score + c_score + r_score) / 5))

    valid = sum(1 for v in proposal.values() if v and v != "Not Specified")
    confidence_score = int(round(40 + (valid / max(len(proposal), 1)) * 50))

    if investment_score >= 80:
        recommendation = "Proceed"
    elif investment_score >= 60:
        recommendation = "Proceed with caution"
    else:
        recommendation = "Defer"

    score_breakdown = {
        "market_analysis": m_score,
        "finance_analysis": f_score,
        "technology_analysis": t_score,
        "competition_analysis": c_score,
        "risk_assessment": r_score,
        "overall_investment_score": investment_score,
    }

    return {
        "executive_summary": (
            f"Diligence completed for {proposal.get('startup_name', 'the startup')} "
            f"with an investment score of {investment_score}/100."
        ),
        "investment_recommendation": recommendation,
        "confidence_score": confidence_score,
        "investment_score": investment_score,
        "swot_analysis": {
            "strengths": ["Proposal-specific strengths could not be extracted (LLM unavailable)."],
            "weaknesses": ["Proposal-specific weaknesses could not be extracted (LLM unavailable)."],
            "opportunities": ["Proposal-specific opportunities could not be extracted (LLM unavailable)."],
            "threats": ["Proposal-specific threats could not be extracted (LLM unavailable)."],
        },
        "action_plan": [
            "Formulate measurable traction indicators aligned with stated revenue model.",
            "Conduct technical architecture audit of stated technology stack.",
        ],
        "recommendation_rationale": {
            "positive_factors": ["Score could not be computed (LLM unavailable)."],
            "concerns": ["Detailed concerns require LLM analysis."],
            "reason_for_score": f"The investment score of {investment_score}/100 is a simple average of all agent scores.",
            "reason_for_recommendation": f"Recommendation '{recommendation}' is based on the computed average score threshold.",
        },
        "score_breakdown": score_breakdown,
    }


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def draft_committee(
    proposal: Dict[str, Any],
    research: Dict[str, Any],
    market: Dict[str, Any],
    finance: Dict[str, Any],
    technology: Dict[str, Any],
    competition: Dict[str, Any],
    risk: Dict[str, Any],
    chunks: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if chunks is None:
        chunks = []

    # ------------------------------------------------------------------
    # Compute score_breakdown deterministically from actual agent outputs.
    # This is done in Python — never delegated to the LLM — to guarantee
    # accuracy and avoid hallucination.
    # ------------------------------------------------------------------
    m_score = market.get("score", 50)
    f_score = finance.get("score", 50)
    t_score = technology.get("score", 50)
    c_score = competition.get("score", 50)
    r_score = risk.get("risk_score", 50)
    overall = int(round((m_score + f_score + t_score + c_score + r_score) / 5))

    score_breakdown = {
        "market_analysis": m_score,
        "finance_analysis": f_score,
        "technology_analysis": t_score,
        "competition_analysis": c_score,
        "risk_assessment": r_score,
        "overall_investment_score": overall,
    }

    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(InvestmentCommitteeSchema)

        proposal_context = (
            "\n".join([c.get("content", "") for c in chunks])
            if chunks
            else "No relevant document chunks found."
        )

        startup_name = proposal.get("startup_name", "Not Specified")
        industry = proposal.get("industry", "Not Specified")
        business_model = proposal.get("business_model", "Not Specified")
        funding = proposal.get("funding_required", "Not Specified")

        prompt = (
            "You are the expert Investment Committee Chair leading a professional venture-fund diligence review.\n"
            f"Synthesise a final structured investment memorandum for {startup_name}.\n\n"
            "=== PROPOSAL SUMMARY ===\n"
            f"{format_proposal_summary(proposal)}\n\n"
            "=== RETRIEVED PITCH DECK CHUNKS ===\n"
            f"{proposal_context}\n\n"
            "=== WEB RESEARCH SUMMARY ===\n"
            f"{format_research_summary(research)}\n\n"
            "=== DOWNSTREAM AGENT SCORES ===\n"
            f"  * Market Analysis Score: {m_score}/100\n"
            f"  * Finance Analysis Score: {f_score}/100\n"
            f"  * Technology Analysis Score: {t_score}/100\n"
            f"  * Competition Analysis Score: {c_score}/100\n"
            f"  * Risk Assessment Score: {r_score}/100\n\n"
            f"=== PRE-COMPUTED SCORE BREAKDOWN ===\n"
            f"Market: {m_score} | Finance: {f_score} | Technology: {t_score} | "
            f"Competition: {c_score} | Risk: {r_score} | Overall: {overall}\n"
            f"(Your investment_score field MUST equal {overall})\n\n"
            "=== MANDATORY INSTRUCTIONS ===\n\n"
            "EXECUTIVE SUMMARY:\n"
            f"  - Open with a statement specific to {startup_name}, its sector ({industry}), and what distinguishes it.\n"
            f"  - Integrate the business model ({business_model}), revenue model ({proposal.get('revenue_model', 'Not Specified')}), "
            f"and funding requirement ({funding}) naturally into the narrative.\n"
            "  - Reference concrete findings from the agent reports — quote specific scores, metrics, or technical details.\n"
            "  - Do NOT use boilerplate openers or clichéd transitions ('Additionally', 'Furthermore', 'Moreover', 'Consequently').\n\n"
            "SWOT ANALYSIS — STRICT RULES:\n"
            "  Strengths: List 3-5 internal strengths UNIQUE to this startup. Reference actual proposal facts.\n"
            "    FORBIDDEN: 'Core technology stack', 'Identified market opportunity', 'Strong team', 'First-mover advantage'\n"
            "    unless supported by a verifiable proposal detail (e.g., name the technology, state the team's credential).\n"
            "  Weaknesses: List 3-5 internal weaknesses visible in THIS proposal's gaps or limitations.\n"
            "    Must be specific to this company — not generic 'early-stage risks'.\n"
            "  Opportunities: List 3-5 external opportunities for THIS startup in its specific industry and geography.\n"
            "    Reference actual trends from the research summary and agent reports.\n"
            "  Threats: List 3-5 external threats. Name competitors or name specific regulations.\n"
            "    FORBIDDEN: 'Competitive startup landscape', 'Regulatory compliance standards' as standalone entries.\n\n"
            "RECOMMENDATION RATIONALE:\n"
            "  positive_factors: The top 3 specific strengths that most increased the investment score.\n"
            "    Each must reference a concrete metric, technology, or market fact from the proposal.\n"
            "  concerns: The top 3 specific weaknesses or risks that most reduced the score.\n"
            "    Each must reference a concrete gap or vulnerability from the proposal.\n"
            "  reason_for_score: In 2-4 sentences, explain why the overall score is {overall}/100.\n"
            "    Reference which sub-scores were highest and lowest and why.\n"
            "  reason_for_recommendation: In 2-4 sentences, explain why the recommendation is 'Proceed', "
            "'Proceed with caution', or 'Defer'. Tie it directly to the combined evidence.\n\n"
            "ACTION PLAN:\n"
            f"  - Provide 4-6 specific next steps relevant to {startup_name}'s stage, model, and identified risks.\n"
            "  - Each step must be actionable and unique to this startup.\n"
            "  FORBIDDEN: 'Validate product-market fit', 'Strengthen executive team', "
            "'Conduct pilot tests', 'Refine the strategy'.\n\n"
            "GENERAL STYLE RULES:\n"
            f"  - Use professional {industry} and '{business_model}' investment vocabulary.\n"
            "  - Do not reference the platform name 'VentureBoard AI'.\n"
            "  - Vary sentence structure; no consecutive sentences starting with the same word.\n"
            f"  - Every section must contain wording unique to {startup_name}.\n\n"
            "Generate the full structured report now."
        )

        res = structured_llm.invoke(prompt)
        output = res.model_dump()

        # Always inject the Python-computed score_breakdown (override LLM if it diverges)
        output["score_breakdown"] = score_breakdown
        # Ensure investment_score matches our computed overall
        output["investment_score"] = overall

        return output

    except Exception:
        return draft_committee_fallback(proposal, market, finance, technology, competition, risk)
