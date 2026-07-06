import json
import time
import fitz
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
from typing import List
import pytest

class MockStructuredLLM:
    def __init__(self, schema):
        self.schema = schema
    def invoke(self, prompt):
        data = {}
        for name, field in self.schema.model_fields.items():
            annotation = field.annotation
            if hasattr(annotation, "__origin__"):
                origin = annotation.__origin__
                if origin is list or origin is List:
                    data[name] = ["Mock Bullet 1", "Mock Bullet 2"]
                else:
                    data[name] = "Mock String"
            elif annotation == int:
                data[name] = 78
            elif annotation == str:
                if name == "overall_risk":
                    data[name] = "Moderate"
                elif name == "investment_recommendation":
                    data[name] = "Proceed with caution"
                else:
                    data[name] = f"Mock {name.replace('_', ' ').capitalize()}"
            elif hasattr(annotation, "model_fields"):
                nested_data = {}
                for n_name, n_field in annotation.model_fields.items():
                    nested_data[n_name] = ["Mock Bullet 1", "Mock Bullet 2"]
                data[name] = annotation(**nested_data)
            else:
                data[name] = "Mock Value"
        return self.schema(**data)

class MockLLM:
    def invoke(self, prompt):
        return MagicMock(content="Mock Search Query 1\nMock Search Query 2\nMock Search Query 3")
    def with_structured_output(self, schema):
        return MockStructuredLLM(schema)

# Apply patch at module load time so all subsequently imported agent modules use the mock get_llm
get_llm_patcher = patch('app.services.llm_config.get_llm', return_value=MockLLM())
get_llm_patcher.start()

from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import SimpleVectorStore
from app.services.agents.proposal_summary_agent import summarize_proposal
from app.services.agents.research_agent import research_market
from app.services.agents.market_agent import analyze_market
from app.services.agents.finance_agent import analyze_finance
from app.services.agents.technology_agent import analyze_technology
from app.services.agents.competition_agent import analyze_competition
from app.services.agents.risk_agent import assess_risk
from app.services.agents.investment_committee_agent import draft_committee
from app.services.langgraph_orchestrator import run_agents

@pytest.fixture(autouse=True)
def mock_llm_and_search():
    with patch('app.services.agents.research_agent.requests.post') as mock_post:
        mock_post.return_value.json.return_value = {"results": [{"title": "Mock Title", "content": "Mock Content", "url": "http://mock.com"}]}
        with patch('duckduckgo_search.DDGS') as mock_ddg:
            mock_ddg.return_value.__enter__.return_value.text.return_value = [{"title": "Mock DDG Title", "body": "Mock DDG Snippet", "href": "http://mockddg.com"}]
            yield

client = TestClient(app)


def test_proposal_summary_agent_returns_expected_structure():
    result = summarize_proposal('AI startup helping investors analyze pitch decks with agents.')
    assert result['startup_name']
    assert 'industry' in result
    assert 'problem' in result


def test_research_agent_returns_research_pack():
    result = research_market('AI startup helping investors analyze pitch decks with agents.')
    assert result['industry_overview']
    assert isinstance(result['market_trends'], list)


def test_agent_modules_return_valid_shapes():
    proposal = summarize_proposal('AI startup helping investors analyze pitch decks with agents.')
    research = research_market('AI startup helping investors analyze pitch decks with agents.')
    market = analyze_market(proposal, research)
    finance = analyze_finance(proposal, research)
    technology = analyze_technology(proposal, research)
    competition = analyze_competition(proposal, research)
    risk = assess_risk(market, finance, technology, competition)
    committee = draft_committee(proposal, research, market, finance, technology, competition, risk)

    assert isinstance(market['score'], int)
    assert isinstance(finance['score'], int)
    assert isinstance(technology['score'], int)
    assert isinstance(competition['score'], int)
    assert risk['overall_risk']
    assert committee['investment_score'] >= 0


def test_langgraph_orchestrator_completes_pipeline():
    report = run_agents('AI startup helping investors analyze pitch decks with agents.')
    assert 'proposal_summary' in report
    assert 'investment_committee' in report


def test_api_health_and_analysis_endpoints():
    health = client.get('/api/health')
    assert health.status_code == 200

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), 'VentureBoard AI demo pitch deck for investors evaluating startups with autonomous agents and subscription pricing')
    pdf_bytes = document.write()
    document.close()

    response = client.post('/api/analyze', files={'file': ('demo.pdf', pdf_bytes, 'application/pdf')})
    assert response.status_code == 200
    assert response.json()['ok'] is True

    report_response = client.get('/api/report')
    assert report_response.status_code == 200
    assert report_response.json()['proposal_summary']['startup_name']


def test_report_pdf_endpoint_returns_400_when_no_report():
    app.state.report = {}
    response = client.get('/api/report/pdf')
    assert response.status_code == 400
    assert "No diligence report" in response.json()['detail']


def test_vector_store_handles_concurrent_writes(tmp_path):
    store = SimpleVectorStore(collection_name='test-store', data_dir=str(tmp_path))

    original_dump = json.dump

    def delayed_dump(obj, file_handle, *args, **kwargs):
        time.sleep(0.01)
        return original_dump(obj, file_handle, *args, **kwargs)

    def worker(index: int) -> None:
        store.add_documents([{'source': f'source-{index}', 'content': f'doc-{index}'}])

    with patch('app.services.vector_store.json.dump', side_effect=delayed_dump):
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(worker, range(12)))

    docs = store.search('doc', limit=50)
    assert len(docs) == 12


def test_document_validation_rejects_resumes_and_contracts():
    # Mocking resume content
    resume_doc = fitz.open()
    page = resume_doc.new_page()
    page.insert_text((72, 72), 'Curriculum Vitae of John Doe. Work Experience at Google and Microsoft. Skills: Python, FastAPI.')
    resume_bytes = resume_doc.write()
    resume_doc.close()
    
    response = client.post('/api/analyze', files={'file': ('resume.pdf', resume_bytes, 'application/pdf')})
    assert response.status_code == 400
    assert 'Resume or CV' in response.json()['detail']

    # Mocking contract content
    contract_doc = fitz.open()
    page = contract_doc.new_page()
    page.insert_text((72, 72), 'Non-Disclosure Agreement. This agreement between the parties hereto is governed by the laws of California.')
    contract_bytes = contract_doc.write()
    contract_doc.close()

    response = client.post('/api/analyze', files={'file': ('contract.pdf', contract_bytes, 'application/pdf')})
    assert response.status_code == 400
    assert 'Legal Contract' in response.json()['detail']


def test_pdf_report_contains_real_flowable_data():
    # Build a valid deck
    deck_doc = fitz.open()
    page = deck_doc.new_page()
    page.insert_text((72, 72), 'Pitch deck for SolVenture startup. Target market is retail store owners. Business model: subscription pricing. Technology stack: AI agent automation. Funding required: $2 million.')
    deck_bytes = deck_doc.write()
    deck_doc.close()

    # Upload valid deck
    response = client.post('/api/analyze', files={'file': ('pitch.pdf', deck_bytes, 'application/pdf')})
    assert response.status_code == 200

    # Request PDF download
    pdf_response = client.get('/api/report/pdf')
    assert pdf_response.status_code == 200
    assert pdf_response.headers['content-type'] == 'application/pdf'
    assert len(pdf_response.content) > 1000  # Verify it is non-empty and has realistic PDF size


def test_post_report_pdf_endpoint_returns_valid_pdf():
    # Fetch existing report first
    report_response = client.get('/api/report')
    assert report_response.status_code == 200
    report_data = report_response.json()
    
    # Clean in-memory report to test stateless POST
    app.state.report = {}
    
    # POST the report data to generate PDF
    response = client.post('/api/report/pdf', json=report_data)
    assert response.status_code == 200
    assert response.headers['content-type'] == 'application/pdf'
    assert len(response.content) > 1000
    
    # Test POST fails with empty data
    err_response = client.post('/api/report/pdf', json={})
    assert err_response.status_code == 400


def test_simulation_endpoint():
    app.state.report = {
        'proposal_summary': {
            'startup_name': 'SolVenture',
            'industry': 'CleanTech',
            'problem': 'Missing clean energy systems.',
            'solution': 'Solar energy platform.',
            'usp': 'Peer-to-peer solar trading.',
            'business_model': 'B2C subscription.',
            'revenue_model': 'SaaS subscription.',
            'technology': 'React, Python.',
            'target_market': 'North America.',
            'funding_required': '$2 million.'
        },
        'research_summary': {
            'industry_overview': 'CleanTech is expanding rapidly.'
        },
        'investment_committee': {
            'investment_score': 75,
            'investment_recommendation': 'Proceed with caution',
            'confidence_score': 85,
            'executive_summary': 'Solid startup with minor tech risks.'
        }
    }

    response = client.post('/api/simulate', json={
        'scenario_type': 'Increase Funding',
        'value': '$5M'
    })

    assert response.status_code == 200
    res_data = response.json()
    assert res_data['ok'] is True
    assert 'simulation' in res_data
    sim = res_data['simulation']
    assert sim['scenario_type'] == 'Increase Funding'
    assert sim['value'] == '$5M'
    assert 'result' in sim
    result = sim['result']
    assert 'investment_score' in result
    assert 'overall_risk' in result
    assert 'investment_recommendation' in result
    assert 'executive_summary' in result
    assert 'why_score_changed' in result
    assert 'key_benefits' in result
    assert 'new_risks' in result

    assert app.state.report['investment_committee']['investment_score'] == 75

    response2 = client.post('/api/simulate', json={
        'scenario_type': 'Reduce Customer Acquisition Cost',
        'value': '20% reduction'
    })
    assert response2.status_code == 200
    res_data2 = response2.json()
    assert res_data2['ok'] is True
    assert len(app.state.simulations) == 2



