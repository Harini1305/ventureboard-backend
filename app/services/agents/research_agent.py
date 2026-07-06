import os
import requests
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from app.services.llm_config import get_llm, format_proposal_summary

NOT_SPECIFIED = 'Not specified in the uploaded proposal.'


class ResearchSummarySchema(BaseModel):
    market_size: str = Field(description="Estimated market size and addressable segment details based on search results")
    trends: List[str] = Field(description="List of current key market and industry trends based on search results")
    competitors: List[str] = Field(description="Key industry competitors identified based on search results")
    regulations: str = Field(description="Relevant industry regulations, compliance standards or frameworks")
    funding_trends: str = Field(description="Recent venture funding trends, VC activity, or valuation metrics in this space")
    
    # Legacy fields for backward compatibility
    industry_overview: str = Field(description="Overview of the industry space based on search results")
    market_trends: List[str] = Field(description="List of current key market trends")
    recent_funding: str = Field(description="Recent venture funding trends in this space")
    technology_trends: List[str] = Field(description="Key technology trends driving this space")


def research_market_fallback(proposal_summary: Dict[str, Any]) -> Dict[str, Any]:
    industry = proposal_summary.get('industry', 'general software')
    problem = proposal_summary.get('problem', 'unspecified inefficiencies')
    target_market = proposal_summary.get('target_market', 'unspecified segments')
    technology = proposal_summary.get('technology', 'digital workflow tools')
    funding = proposal_summary.get('funding_required', 'early-stage seed funding')

    # Fallback to smart rule-based templates
    industry_overview = (
        f"The {industry} sector is undergoing active transformation as organizations seek "
        f"solutions to resolve the core problem: '{problem}'."
    )
    
    market_size = (
        f"The addressable market segment spans {target_market} in the {industry} space."
    )

    market_trends = [
        f"Integration of {technology} to automate legacy workflows",
        f"Growing focus on solving the challenge: '{problem}'"
    ]

    competitors = [
        "Traditional manual processes and legacy spreadsheets",
        f"Incumbent software providers operating in the {industry} domain"
    ]

    recent_funding = (
        f"Private capital is actively financing {industry} companies seeking around {funding}."
    )

    regulations = (
        f"Companies utilizing {technology} in {industry} must align with standard data protection rules."
    )

    technology_trends = [
        f"Leveraging {technology} for increased integration efficiency"
    ]

    return {
        'search_queries': [],
        'market_size': market_size,
        'trends': market_trends,
        'competitors': competitors,
        'regulations': regulations,
        'funding_trends': recent_funding,
        'industry_overview': industry_overview,
        'market_trends': market_trends,
        'recent_funding': recent_funding,
        'technology_trends': technology_trends,
    }


def research_market(proposal_text: str | Dict[str, Any] | None = None, proposal_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if isinstance(proposal_text, dict):
        proposal_summary = proposal_text
        proposal_text = None

    if proposal_summary is None:
        proposal_summary = {}

    startup_name = proposal_summary.get('startup_name', '')
    if not startup_name or startup_name == NOT_SPECIFIED:
        startup_name = 'the startup'
        
    industry = proposal_summary.get('industry', '')
    if not industry or industry == NOT_SPECIFIED:
        industry = 'general software'
        
    problem = proposal_summary.get('problem', '')
    if not problem or problem == NOT_SPECIFIED:
        problem = 'unspecified market inefficiencies'
        
    target_market = proposal_summary.get('target_market', '')
    if not target_market or target_market == NOT_SPECIFIED:
        target_market = 'unspecified target segments'
        
    technology = proposal_summary.get('technology', '')
    if not technology or technology == NOT_SPECIFIED:
        technology = 'digital workflow tools'
        
    funding = proposal_summary.get('funding_required', '')
    if not funding or funding == NOT_SPECIFIED:
        funding = 'early-stage funding'

    # Step 1: Generate dynamic search queries using requested parameters
    search_queries = []
    try:
        llm = get_llm()
        query_prompt = (
            "You are a market research query generator. Based on the startup proposal details, "
            "generate exactly 3 distinct, specific search queries to research the industry, competitors, "
            "and market trends on the web.\n\n"
            "Output each query on a new line. Do not add numbers, bullets, or quotes.\n\n"
            f"Startup Name: {startup_name}\n"
            f"Industry: {industry}\n"
            f"Problem: {problem}\n"
            f"Target Market: {target_market}\n"
            f"Technology: {technology}\n"
            f"Funding: {funding}\n"
        )
        res = llm.invoke(query_prompt)
        queries = [line.strip().replace('"', '').replace("'", "") for line in res.content.splitlines() if line.strip()]
        search_queries = queries[:3]
    except Exception:
        # Fallback query list
        search_queries = [
            f"Market size and growth rate for {industry} industry",
            f"Key competitors solving: {problem[:50]}",
            f"Venture funding trends and regulations for {industry} {technology}"
        ]

    # Step 2: Execute search queries
    search_results = []
    for query in search_queries:
        result = ""
        # 1. Try Tavily
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            try:
                url = "https://api.tavily.com/search"
                payload = {"api_key": tavily_key, "query": query, "max_results": 2}
                res = requests.post(url, json=payload, timeout=8)
                res.raise_for_status()
                t_results = res.json().get("results", [])
                result = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('content')}\nURL: {r.get('url')}" for r in t_results])
            except Exception:
                pass
        
        # 2. Try DuckDuckGo
        if not result:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(query, max_results=2))
                    result = "\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}" for r in ddg_results])
            except Exception as e:
                result = f"Search failed for query '{query}': {str(e)}"
        
        search_results.append(f"Query: {query}\nSearch Results:\n{result}")

    # Step 3: Compile structured research summary via LLM
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ResearchSummarySchema)
        summary_prompt = (
            "You are a professional market research analyst. Synthesize a structured market research report "
            "using the startup's details and the web search results provided. Focus on extracting market size, "
            "market and industry trends, competitors, regulations, and recent funding trends.\n\n"
            "Never invent details. If facts or statistics are missing, write that they are not specified or not found.\n\n"
            "Keep each field extremely concise (1-2 sentences maximum). "
            "The entire synthesized research report must be under 300 words in total.\n\n"
            f"Startup Details:\n{format_proposal_summary(proposal_summary)}\n\n"
            f"Web Search Results:\n" + "\n\n".join(search_results)
        )
        res = structured_llm.invoke(summary_prompt)
        output = res.model_dump()
        output['search_queries'] = search_queries
        return output
    except Exception:
        fallback_res = research_market_fallback(proposal_summary)
        fallback_res['search_queries'] = search_queries
        return fallback_res
