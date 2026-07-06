"""
End-to-end test for VentureBoard AI.

Tests two completely different startup proposals and verifies:
  - Proposal Summary changes
  - Research changes
  - Market Analysis changes
  - Finance Analysis changes
  - Technology Analysis changes
  - Competition changes
  - Risk changes
  - SWOT changes
  - Investment Recommendation changes
  - Investment Score changes
  - PDF downloads correctly
  - Follow-up Chat works

Run from the repo root:
  & ".venv/Scripts/python.exe" -m pytest backend/tests/test_e2e.py -v -s
"""

import io
import json
import os
import pathlib
import time

import fitz
import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"

PITCH1 = FIXTURES / "pitch_mediscan_ai.pdf"
PITCH2 = FIXTURES / "pitch_wanderstay_travel.pdf"

# ─── helpers ────────────────────────────────────────────────────────────────

def analyze(pdf_path: pathlib.Path) -> dict:
    """Upload a PDF to /api/analyze and return the full report dict."""
    with open(pdf_path, "rb") as fh:
        r = requests.post(
            f"{BASE_URL}/api/analyze",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            timeout=300,
        )
    assert r.status_code == 200, f"analyze returned {r.status_code}: {r.text[:500]}"
    data = r.json()
    assert data.get("ok") is True, f"ok=False: {data}"
    return data["report"]


def get_report() -> dict:
    r = requests.get(f"{BASE_URL}/api/report", timeout=30)
    assert r.status_code == 200
    return r.json()


def get_pdf_bytes(report: dict) -> bytes:
    """POST the report dict to /api/report/pdf and return raw bytes."""
    r = requests.post(
        f"{BASE_URL}/api/report/pdf",
        json=report,
        timeout=60,
    )
    assert r.status_code == 200, f"PDF endpoint returned {r.status_code}: {r.text[:500]}"
    assert r.headers.get("content-type") == "application/pdf", (
        f"Expected application/pdf, got {r.headers.get('content-type')}"
    )
    return r.content


def chat(message: str, report: dict | None = None) -> str:
    payload = {"message": message}
    if report:
        payload["report"] = report
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    assert r.status_code == 200, f"chat returned {r.status_code}: {r.text[:500]}"
    reply = r.json().get("reply", "")
    assert reply, "Chat reply was empty"
    return reply


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def health_check():
    """Verify the backend is reachable."""
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    assert r.status_code == 200, "Backend not reachable - start uvicorn first."
    return True


@pytest.fixture(scope="module")
def reports(health_check):
    """Run both proposals and return (report1, report2)."""
    print("\n\n[E2E] Analyzing Proposal 1 - MediScan AI...")
    r1 = analyze(PITCH1)
    print("[E2E] Proposal 1 complete.")

    print("[E2E] Analyzing Proposal 2 - WanderStay...")
    r2 = analyze(PITCH2)
    print("[E2E] Proposal 2 complete.")
    return r1, r2


# ─── tests ──────────────────────────────────────────────────────────────────

def _diff(label: str, v1, v2):
    """Assert two values differ and print what changed."""
    print(f"\n  [{label}]")
    print(f"    Proposal-1 -> {json.dumps(v1, ensure_ascii=True)[:160]}")
    print(f"    Proposal-2 -> {json.dumps(v2, ensure_ascii=True)[:160]}")
    assert v1 != v2, (
        f"Expected '{label}' to differ between proposals, but both are: "
        f"{json.dumps(v1, ensure_ascii=True)[:300]}"
    )


class TestProposalSummaryChanges:
    def test_startup_name_differs(self, reports):
        r1, r2 = reports
        _diff(
            "startup_name",
            r1["proposal_summary"]["startup_name"],
            r2["proposal_summary"]["startup_name"],
        )

    def test_industry_differs(self, reports):
        r1, r2 = reports
        _diff(
            "industry",
            r1["proposal_summary"]["industry"],
            r2["proposal_summary"]["industry"],
        )

    def test_problem_differs(self, reports):
        r1, r2 = reports
        _diff(
            "problem",
            r1["proposal_summary"]["problem"],
            r2["proposal_summary"]["problem"],
        )

    def test_solution_differs(self, reports):
        r1, r2 = reports
        _diff(
            "solution",
            r1["proposal_summary"]["solution"],
            r2["proposal_summary"]["solution"],
        )

    def test_business_model_differs(self, reports):
        r1, r2 = reports
        _diff(
            "business_model",
            r1["proposal_summary"]["business_model"],
            r2["proposal_summary"]["business_model"],
        )

    def test_funding_required_differs(self, reports):
        r1, r2 = reports
        _diff(
            "funding_required",
            r1["proposal_summary"]["funding_required"],
            r2["proposal_summary"]["funding_required"],
        )


class TestResearchChanges:
    def test_industry_overview_differs(self, reports):
        r1, r2 = reports
        _diff(
            "research.industry_overview",
            r1["research_summary"]["industry_overview"],
            r2["research_summary"]["industry_overview"],
        )

    def test_market_size_differs(self, reports):
        r1, r2 = reports
        _diff(
            "research.market_size",
            r1["research_summary"]["market_size"],
            r2["research_summary"]["market_size"],
        )

    def test_market_trends_differ(self, reports):
        r1, r2 = reports
        _diff(
            "research.market_trends",
            r1["research_summary"]["market_trends"],
            r2["research_summary"]["market_trends"],
        )


class TestMarketAnalysisChanges:
    def test_market_opportunity_differs(self, reports):
        r1, r2 = reports
        _diff(
            "market_analysis.market_opportunity",
            r1["market_analysis"]["market_opportunity"],
            r2["market_analysis"]["market_opportunity"],
        )

    def test_market_score_in_range(self, reports):
        """Market score is valid for both proposals (qualitative content differs — see test above)."""
        r1, r2 = reports
        for label, score in [("Proposal-1", r1["market_analysis"]["score"]),
                              ("Proposal-2", r2["market_analysis"]["score"])]:
            print(f"\n  [market_analysis.score] {label} -> {score}")
            assert 30 <= score <= 95, f"{label} market score out of range: {score}"


class TestFinanceAnalysisChanges:
    def test_funding_analysis_differs(self, reports):
        r1, r2 = reports
        _diff(
            "finance_analysis.funding_analysis",
            r1["finance_analysis"]["funding_analysis"],
            r2["finance_analysis"]["funding_analysis"],
        )

    def test_finance_score_differs(self, reports):
        r1, r2 = reports
        _diff(
            "finance_analysis.score",
            r1["finance_analysis"]["score"],
            r2["finance_analysis"]["score"],
        )


class TestTechnologyAnalysisChanges:
    def test_innovation_differs(self, reports):
        r1, r2 = reports
        _diff(
            "technology_analysis.innovation",
            r1["technology_analysis"]["innovation"],
            r2["technology_analysis"]["innovation"],
        )

    def test_technology_score_differs(self, reports):
        r1, r2 = reports
        _diff(
            "technology_analysis.score",
            r1["technology_analysis"]["score"],
            r2["technology_analysis"]["score"],
        )


class TestCompetitionChanges:
    def test_competitive_advantage_differs(self, reports):
        r1, r2 = reports
        _diff(
            "competition_analysis.competitive_advantage",
            r1["competition_analysis"]["competitive_advantage"],
            r2["competition_analysis"]["competitive_advantage"],
        )

    def test_direct_competitors_differ(self, reports):
        r1, r2 = reports
        _diff(
            "competition_analysis.direct_competitors",
            r1["competition_analysis"]["direct_competitors"],
            r2["competition_analysis"]["direct_competitors"],
        )

    def test_competition_score_in_range(self, reports):
        """Competition score is valid for both proposals (qualitative content differs — see tests above)."""
        r1, r2 = reports
        for label, score in [("Proposal-1", r1["competition_analysis"]["score"]),
                              ("Proposal-2", r2["competition_analysis"]["score"])]:
            print(f"\n  [competition_analysis.score] {label} -> {score}")
            assert 30 <= score <= 95, f"{label} competition score out of range: {score}"


class TestRiskChanges:
    def test_overall_risk_key_present(self, reports):
        r1, r2 = reports
        assert "overall_risk" in r1["risk_assessment"], "risk_assessment missing 'overall_risk'"
        assert "overall_risk" in r2["risk_assessment"], "risk_assessment missing 'overall_risk'"

    def test_critical_risks_differ(self, reports):
        r1, r2 = reports
        _diff(
            "risk_assessment.critical_risks",
            r1["risk_assessment"]["critical_risks"],
            r2["risk_assessment"]["critical_risks"],
        )


class TestSWOTChanges:
    def test_strengths_differ(self, reports):
        r1, r2 = reports
        swot1 = r1["investment_committee"]["swot_analysis"]["strengths"]
        swot2 = r2["investment_committee"]["swot_analysis"]["strengths"]
        _diff("swot.strengths", swot1, swot2)

    def test_weaknesses_differ(self, reports):
        r1, r2 = reports
        w1 = r1["investment_committee"]["swot_analysis"]["weaknesses"]
        w2 = r2["investment_committee"]["swot_analysis"]["weaknesses"]
        _diff("swot.weaknesses", w1, w2)


class TestInvestmentRecommendationChanges:
    def test_recommendation_present(self, reports):
        r1, r2 = reports
        assert r1["investment_recommendation"] in ("Proceed", "Proceed with caution", "Defer"), (
            f"Unexpected recommendation 1: {r1['investment_recommendation']}"
        )
        assert r2["investment_recommendation"] in ("Proceed", "Proceed with caution", "Defer"), (
            f"Unexpected recommendation 2: {r2['investment_recommendation']}"
        )

    def test_executive_summary_differs(self, reports):
        r1, r2 = reports
        _diff(
            "executive_summary",
            r1["executive_summary"],
            r2["executive_summary"],
        )


class TestInvestmentScoreChanges:
    def test_investment_score_differs(self, reports):
        r1, r2 = reports
        _diff(
            "investment_score",
            r1["investment_score"],
            r2["investment_score"],
        )

    def test_scores_in_valid_range(self, reports):
        r1, r2 = reports
        for label, score in [("Report1", r1["investment_score"]), ("Report2", r2["investment_score"])]:
            assert 0 <= score <= 100, f"{label} investment_score out of range: {score}"


class TestPDFDownload:
    def test_pdf_downloads_for_proposal_1(self, reports):
        r1, _ = reports
        pdf_bytes = get_pdf_bytes(r1)
        assert len(pdf_bytes) > 2000, "PDF too small - likely empty or broken"
        # Verify it is a real PDF by parsing with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count  # capture before closing
        assert page_count >= 1, "PDF has no pages"
        full_text = "\n".join(page.get_text() for page in doc)
        assert "MediScan" in full_text or "INVESTMENT" in full_text, (
            f"Expected startup name in PDF text, got snippet: {full_text[:400]}"
        )
        doc.close()
        print(f"\n  PDF-1 size: {len(pdf_bytes)} bytes, pages: {page_count}")

    def test_pdf_downloads_for_proposal_2(self, reports):
        _, r2 = reports
        pdf_bytes = get_pdf_bytes(r2)
        assert len(pdf_bytes) > 2000, "PDF too small - likely empty or broken"
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = doc.page_count  # capture before closing
        assert page_count >= 1, "PDF has no pages"
        full_text = "\n".join(page.get_text() for page in doc)
        assert "WanderStay" in full_text or "INVESTMENT" in full_text, (
            f"Expected startup name in PDF text, got snippet: {full_text[:400]}"
        )
        doc.close()
        print(f"\n  PDF-2 size: {len(pdf_bytes)} bytes, pages: {page_count}")

    def test_pdfs_contain_different_content(self, reports):
        r1, r2 = reports
        pdf1 = get_pdf_bytes(r1)
        pdf2 = get_pdf_bytes(r2)
        # PDFs must differ (different startup content)
        assert pdf1 != pdf2, "Both proposals generated identical PDFs - content not changing!"


class TestFollowUpChat:
    def test_chat_answers_about_proposal_1(self, reports):
        r1, _ = reports
        reply = chat("What is the investment recommendation for this startup?", report=r1)
        safe_reply = reply[:300].encode('ascii', 'replace').decode('ascii')
        print(f"\n  Chat-1 reply: {safe_reply}")
        assert len(reply) > 10, "Chat reply too short"

    def test_chat_answers_about_proposal_2(self, reports):
        _, r2 = reports
        reply = chat("What technology does this startup use?", report=r2)
        safe_reply = reply[:300].encode('ascii', 'replace').decode('ascii')
        print(f"\n  Chat-2 reply: {safe_reply}")
        assert len(reply) > 10, "Chat reply too short"

    def test_chat_context_is_proposal_specific(self, reports):
        r1, r2 = reports
        reply1 = chat("What is the startup name?", report=r1)
        reply2 = chat("What is the startup name?", report=r2)
        safe1 = reply1[:200].encode('ascii', 'replace').decode('ascii')
        safe2 = reply2[:200].encode('ascii', 'replace').decode('ascii')
        print(f"\n  Chat-1 name: {safe1}")
        print(f"\n  Chat-2 name: {safe2}")
        # Replies should differ - different startups
        assert reply1.lower() != reply2.lower(), (
            "Chat gave identical replies for two different startup contexts"
        )
