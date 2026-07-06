import re
from typing import Dict, Any
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm

NOT_SPECIFIED = 'Not specified in the uploaded proposal.'


class ProposalSummarySchema(BaseModel):
    startup_name: str = Field(description="Name of the startup or company")
    industry: str = Field(description="Industry or sector the startup operates in")
    problem: str = Field(description="The primary customer problem or market need being addressed")
    solution: str = Field(description="The product, service, or solution being offered")
    business_model: str = Field(description="Business model description (e.g. B2B, B2C, Enterprise, SaaS, Marketplace)")
    revenue_model: str = Field(description="How the company makes money (e.g. subscription pricing, transaction fees, freemium)")
    funding_required: str = Field(description="Total capital or funding required by the startup")
    target_market: str = Field(description="Target market segment and customer profile")
    technology: str = Field(description="Core technology stack, frameworks, or models leveraged")
    usp: str = Field(description="Unique Value Proposition or key differentiation from competitors")


def _normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip())


def _find_sentence(text: str, keywords: list[str]) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            return sentence.strip()
    return ''


def _extract_title(text: str) -> str:
    for line in [line.strip() for line in text.splitlines() if line.strip()][:8]:
        cleaned = re.sub(r'[^A-Za-z0-9\s&./-]', ' ', line).strip()
        if 2 <= len(cleaned.split()) <= 8 and len(cleaned) <= 80:
            lowered = cleaned.lower()
            if any(term in lowered for term in ['startup', 'company', 'platform', 'labs', 'ai', 'product', 'venture']):
                return cleaned
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        cleaned = re.sub(r'[^A-Za-z0-9\s&./-]', ' ', sentence).strip()
        if 2 <= len(cleaned.split()) <= 10:
            return cleaned
    return ''


def summarize_proposal_fallback(proposal_text: str) -> Dict[str, Any]:
    text = _normalize_text(proposal_text)
    if not text:
        return {
            'startup_name': NOT_SPECIFIED,
            'industry': NOT_SPECIFIED,
            'problem': NOT_SPECIFIED,
            'solution': NOT_SPECIFIED,
            'business_model': NOT_SPECIFIED,
            'revenue_model': NOT_SPECIFIED,
            'funding_required': NOT_SPECIFIED,
            'target_market': NOT_SPECIFIED,
            'technology': NOT_SPECIFIED,
            'usp': NOT_SPECIFIED,
        }

    startup_name = _extract_title(text) or NOT_SPECIFIED
    problem = _find_sentence(text, ['problem', 'challenge', 'pain', 'need', 'difficulty', 'hard', 'struggle']) or NOT_SPECIFIED
    solution = _find_sentence(text, ['solution', 'platform', 'product', 'helps', 'enables', 'automates', 'provides', 'offers']) or NOT_SPECIFIED
    business_model = _find_sentence(text, ['subscription', 'saas', 'license', 'freemium', 'marketplace', 'enterprise', 'b2b', 'b2c']) or NOT_SPECIFIED
    revenue_model = _find_sentence(text, ['subscription', 'license', 'fee', 'fees', 'commission', 'pricing', 'revenue', 'pay']) or NOT_SPECIFIED

    funding_match = re.search(r'\$\s?\d+(?:[.,]\d+)?(?:\s?(?:m|million|k|thousand))?', text, re.IGNORECASE)
    funding_required = funding_match.group(0) if funding_match else NOT_SPECIFIED

    target_market = _find_sentence(text, ['target market', 'customers', 'users', 'enterprises', 'small businesses', 'investors', 'teams', 'organizations']) or NOT_SPECIFIED
    technology = _find_sentence(text, ['ai', 'ml', 'machine learning', 'llm', 'api', 'cloud', 'blockchain', 'mobile', 'saas', 'automation', 'agents']) or NOT_SPECIFIED
    usp = _find_sentence(text, ['unique', 'differentiated', 'differentiation', 'advantage', 'better', 'faster', 'superior', 'disruptive', 'first']) or solution

    return {
        'startup_name': startup_name,
        'industry': _find_sentence(text, ['ai', 'health', 'fintech', 'saas', 'enterprise', 'marketplace', 'consumer']) or NOT_SPECIFIED,
        'problem': problem,
        'solution': solution,
        'business_model': business_model,
        'revenue_model': revenue_model,
        'funding_required': funding_required,
        'target_market': target_market,
        'technology': technology,
        'usp': usp,
    }


def summarize_proposal(proposal_text: str) -> Dict[str, Any]:
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ProposalSummarySchema)
        prompt = (
            "You are a startup diligence expert. Analyze the uploaded startup pitch deck or business plan text. "
            "Extract the requested structured metadata. If a field is missing or cannot be identified from the text, "
            "output 'Not specified in the uploaded proposal.' for that field. Never invent values.\n\n"
            "Keep each field extremely concise (1-2 sentences maximum, under 25 words per field). "
            "The entire summary must be under 250 words in total.\n\n"
            f"Document Content:\n{proposal_text}"
        )
        res = structured_llm.invoke(prompt)
        return res.model_dump()
    except Exception as e:
        return summarize_proposal_fallback(proposal_text)
