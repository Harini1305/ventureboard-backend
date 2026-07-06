import re
import time
from datetime import datetime
from typing import Dict, Any, List

from app.services.agents.proposal_summary_agent import summarize_proposal
from app.services.agents.research_agent import research_market
from app.services.agents.market_agent import analyze_market
from app.services.agents.finance_agent import analyze_finance
from app.services.agents.technology_agent import analyze_technology
from app.services.agents.competition_agent import analyze_competition
from app.services.agents.risk_agent import assess_risk
from app.services.agents.investment_committee_agent import draft_committee
from app.services.vector_store import SimpleVectorStore


class AgentState(dict):
    """Represents the shared workflow state."""
    
    def __init__(self, initial_data: Dict[str, Any] | None = None):
        super().__init__()
        self['proposal_text'] = ''
        self['vector_store'] = None
        self['proposal_summary'] = {}
        self['research_summary'] = {}
        self['market_analysis'] = {}
        self['finance_analysis'] = {}
        self['technology_analysis'] = {}
        self['competition_analysis'] = {}
        self['risk_assessment'] = {}
        self['investment_committee'] = {}
        
        if initial_data:
            self.update(initial_data)


class StateGraph:
    def __init__(self, state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.entry_point = None

    def add_node(self, name: str, func):
        self.nodes[name] = func

    def add_edge(self, start: str, end: str):
        self.edges.append((start, end))

    def set_entry_point(self, name: str):
        self.entry_point = name

    def compile(self):
        return CompiledGraph(self)


class CompiledGraph:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = AgentState(initial_state)
        # Execute nodes sequentially corresponding to the pipeline structure
        sequence = [
            'proposal_summary',
            'research',
            'market',
            'finance',
            'technology',
            'competition',
            'risk',
            'committee'
        ]
        
        node_display_names = {
            'proposal_summary': 'Proposal Summary Agent',
            'research': 'Research Agent',
            'market': 'Market Agent',
            'finance': 'Finance Agent',
            'technology': 'Technology Agent',
            'competition': 'Competition Agent',
            'risk': 'Risk Agent',
            'committee': 'Investment Committee'
        }
        
        audit_trail = state.get('audit_trail')
        if audit_trail is None:
            audit_trail = []
            state['audit_trail'] = audit_trail

        for node_name in sequence:
            if node_name in self.graph.nodes:
                agent_name = node_display_names.get(node_name, node_name)
                start_time = time.time()
                success = False
                status = "Failed"
                try:
                    updates = self.graph.nodes[node_name](state)
                    if updates:
                        state.update(updates)
                    success = True
                    status = "Completed"
                except Exception as e:
                    exec_time = time.time() - start_time
                    audit_trail.append({
                        'timestamp': datetime.now().isoformat(),
                        'agent_name': agent_name,
                        'status': status,
                        'execution_time': f"{exec_time:.2f}s",
                        'success': success
                    })
                    raise e
                
                exec_time = time.time() - start_time
                audit_trail.append({
                    'timestamp': datetime.now().isoformat(),
                    'agent_name': agent_name,
                    'status': status,
                    'execution_time': f"{exec_time:.2f}s",
                    'success': success
                })
        return state


def proposal_summary_node(state: AgentState) -> Dict[str, Any]:
    proposal_text = state.get('proposal_text', '')
    summary = summarize_proposal(proposal_text)
    return {'proposal_summary': summary}


def research_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research = research_market(proposal_summary)
    return {'research_summary': research}


def market_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    vector_store = state.get('vector_store')
    
    market_chunks = []
    if vector_store:
        market_chunks = vector_store.search("market customer audience segment scale size", limit=3)
        
    market = analyze_market(proposal_summary, research_summary, market_chunks)
    return {'market_analysis': market}


def finance_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    vector_store = state.get('vector_store')
    
    finance_chunks = []
    if vector_store:
        finance_chunks = vector_store.search("finance revenue cost funding price expense runway metrics", limit=3)
        
    finance = analyze_finance(proposal_summary, research_summary, finance_chunks)
    return {'finance_analysis': finance}


def technology_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    vector_store = state.get('vector_store')
    
    tech_chunks = []
    if vector_store:
        tech_chunks = vector_store.search("technology stack platform software AI framework code database", limit=3)
        
    technology = analyze_technology(proposal_summary, research_summary, tech_chunks)
    return {'technology_analysis': technology}


def competition_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    vector_store = state.get('vector_store')
    
    comp_chunks = []
    if vector_store:
        comp_chunks = vector_store.search("competition competitor USP moat advantage landscape defense alternative", limit=3)
        
    competition = analyze_competition(proposal_summary, research_summary, comp_chunks)
    return {'competition_analysis': competition}


def risk_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    market = state.get('market_analysis', {})
    finance = state.get('finance_analysis', {})
    technology = state.get('technology_analysis', {})
    competition = state.get('competition_analysis', {})
    vector_store = state.get('vector_store')
    
    risk_chunks = []
    if vector_store:
        risk_chunks = vector_store.search("risk challenge threat vulnerability failure limitation", limit=3)
        
    risk = assess_risk(
        market,
        finance,
        technology,
        competition,
        risk_chunks,
        proposal=proposal_summary,
        research=research_summary
    )
    return {'risk_assessment': risk}


def committee_node(state: AgentState) -> Dict[str, Any]:
    proposal_summary = state.get('proposal_summary', {})
    research_summary = state.get('research_summary', {})
    market = state.get('market_analysis', {})
    finance = state.get('finance_analysis', {})
    technology = state.get('technology_analysis', {})
    competition = state.get('competition_analysis', {})
    risk = state.get('risk_assessment', {})
    vector_store = state.get('vector_store')
    
    committee_chunks = []
    if vector_store:
        committee_chunks = vector_store.search("investment decision score SWOT strengths recommendations action", limit=3)
        
    committee = draft_committee(
        proposal_summary, research_summary, market, finance, technology, competition, risk, committee_chunks
    )
    return {'investment_committee': committee}


def run_agents(proposal_text: str, vector_store: SimpleVectorStore | None = None, audit_trail: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if vector_store is None:
        vector_store = SimpleVectorStore()

    # Pre-populate ChromaDB with proposal chunks for matching
    chunks = [chunk.strip() for chunk in re.split(r'(?<=[.!?])\s+', proposal_text or '') if chunk.strip()]
    documents = [
        {'source': 'proposal_chunk', 'content': chunk, 'metadata': {'type': 'proposal'}} for chunk in chunks
    ]
    vector_store.add_documents(documents)

    # Initialize StateGraph workflow
    builder = StateGraph(AgentState)
    builder.add_node('proposal_summary', proposal_summary_node)
    builder.add_node('research', research_node)
    builder.add_node('market', market_node)
    builder.add_node('finance', finance_node)
    builder.add_node('technology', technology_node)
    builder.add_node('competition', competition_node)
    builder.add_node('risk', risk_node)
    builder.add_node('committee', committee_node)

    graph = builder.compile()

    initial_state = {
        'proposal_text': proposal_text,
        'vector_store': vector_store,
        'proposal_summary': {},
        'research_summary': {},
        'market_analysis': {},
        'finance_analysis': {},
        'technology_analysis': {},
        'competition_analysis': {},
        'risk_assessment': {},
        'investment_committee': {},
        'audit_trail': audit_trail
    }

    final_state = graph.invoke(initial_state)

    summary_content = final_state['proposal_summary'].get('problem', '')
    research_content = final_state['research_summary'].get('industry_overview', '')
    committee_content = final_state['investment_committee'].get('executive_summary', '')

    vector_store.add_documents([
        {'source': 'proposal_summary', 'content': summary_content, 'metadata': {'type': 'proposal'}},
        {'source': 'research_summary', 'content': research_content, 'metadata': {'type': 'research'}},
        {'source': 'investment_committee', 'content': committee_content, 'metadata': {'type': 'committee'}},
    ])

    return {
        'proposal_summary': final_state['proposal_summary'],
        'research_summary': final_state['research_summary'],
        'market_analysis': final_state['market_analysis'],
        'finance_analysis': final_state['finance_analysis'],
        'technology_analysis': final_state['technology_analysis'],
        'competition_analysis': final_state['competition_analysis'],
        'risk_assessment': final_state['risk_assessment'],
        'investment_committee': final_state['investment_committee'],
        'vector_matches': vector_store.search(proposal_text, limit=3),
        'audit_trail': final_state.get('audit_trail', [])
    }

