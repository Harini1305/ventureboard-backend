"""
Create two distinct startup pitch deck PDFs for end-to-end testing.
"""
import fitz
import os

# ─────────────────────────────────────────────────────────────────
# PITCH DECK 1 – AI Healthcare startup
# ─────────────────────────────────────────────────────────────────
PITCH_1 = """MediScan AI – Transforming Medical Imaging Diagnosis

MediScan AI is an AI-powered diagnostic platform built for radiologists and hospital networks.

Problem:
Radiologists worldwide face extreme workload pressure, with burnout rates exceeding 60%. Manual image review is slow and prone to missed diagnoses. Hospitals lose an estimated $28 billion annually due to diagnostic delays and misdiagnoses.

Solution:
MediScan AI deploys deep learning convolutional neural network models trained on 50 million anonymized medical images to detect cancer, fractures, and cardio-pulmonary conditions with 98.2% accuracy — outperforming average radiologist accuracy of 88.4%.

Business Model:
Enterprise SaaS B2B targeting hospital networks, radiology centers, and telemedicine providers. Annual contract licensing with per-scan pricing tiers.

Revenue Model:
Subscription licensing starting at $120,000 per hospital per year, plus $0.80 per scan processed. Projected Year 1 revenue of $4.2 million from 35 signed hospital contracts.

Target Market:
Primary customers are hospital networks with 200+ beds in the US, EU, and Southeast Asia. Total addressable market of $38 billion in medical imaging AI by 2028.

Technology:
Built on PyTorch and custom transformer vision models. DICOM-compatible integration with existing PACS systems. HIPAA-compliant cloud infrastructure on AWS with on-premise deployment options. FDA 510(k) clearance obtained for 3 imaging modalities.

Unique Value Proposition:
MediScan AI is the only radiology AI with FDA clearance across CT, MRI, and X-ray simultaneously. Integration with existing PACS reduces deployment time to under 72 hours versus 6 months for competitors.

Competitors:
Aidoc, Zebra Medical Vision, Enlitic, and traditional PACS vendors like Philips IntelliSpace.

Funding Required:
$12 million Series A to fund clinical deployment in 200 new hospital sites, expand FDA clearance to 5 additional imaging modalities, and grow the engineering team from 18 to 45 engineers.

Market Traction:
35 signed hospital contracts generating $1.8M ARR. 2.3 million scans processed to date. 94% customer retention after Year 1. Partnerships with Johns Hopkins and Mayo Clinic for clinical validation.
"""

# ─────────────────────────────────────────────────────────────────
# PITCH DECK 2 – Airbnb-like Travel startup
# ─────────────────────────────────────────────────────────────────
PITCH_2 = """WanderStay – Peer-to-Peer Boutique Travel Accommodations Marketplace

WanderStay is a peer-to-peer travel marketplace connecting travelers with unique local home stays and boutique accommodations.

Problem:
Traditional hotel chains lack local cultural experiences and are overpriced, while current home-sharing platforms face rising cleaning fees, inconsistent quality, and regulatory crackdowns in major travel destinations.

Solution:
WanderStay operates a curated peer-to-peer home accommodation marketplace. All listings are verified by local community ambassadors, with transparent pricing (zero hidden cleaning fees) and a smart insurance contract protecting hosts and guests automatically.

Business Model:
Two-sided marketplace model. WanderStay charges a 12% booking commission from guests and a 3% service fee from hosts on every booking.

Revenue Model:
Commission-based transaction model. With $220 million in booking volume projected for Year 2, platform revenue reaches $33 million. Additional revenue from WanderStay experiences and tour partnerships.

Target Market:
Millennial and Gen-Z travelers seeking authentic travel experiences in urban and coastal destinations. TAM of $140 billion in boutique travel lodging.

Technology:
React and React Native marketplace apps. AI-powered search and dynamic pricing matching algorithms. Secure payments integration via Stripe. Automated host compliance checklist based on local municipal rules.

Unique Value Proposition:
WanderStay is the only home-sharing platform with 100% verified listings by local ambassadors, guaranteeing quality, and a strict no-hidden-fees booking policy.

Competitors:
Airbnb, Booking.com, Vrbo, Sonder, and traditional hotel booking sites.

Funding Required:
$8 million seed round to expand host recruitment in 15 major European and US cities, obtain local operator licensing, and expand marketing campaigns to reach 300,000 active guests.

Market Traction:
Active pilot in Lisbon and Barcelona with 850 hosts and 14,000 completed bookings. Average guest rating of 4.85 stars. Positive unit economics achieved with 34% repeat guest bookings.
"""

OUT_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(OUT_DIR, exist_ok=True)


def make_pdf(text: str, path: str):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    # wrap text into lines
    x, y, w = 50, 60, 495
    fontsize = 10
    leading = 14
    for raw_line in text.strip().splitlines():
        words = raw_line.split()
        line_buf = ""
        for word in words:
            test = (line_buf + " " + word).strip()
            if fitz.get_text_length(test, fontname="helv", fontsize=fontsize) <= w:
                line_buf = test
            else:
                if y > 800:
                    page = doc.new_page(width=595, height=842)
                    y = 60
                page.insert_text((x, y), line_buf, fontname="helv", fontsize=fontsize)
                y += leading
                line_buf = word
        if line_buf:
            if y > 800:
                page = doc.new_page(width=595, height=842)
                y = 60
            page.insert_text((x, y), line_buf, fontname="helv", fontsize=fontsize)
            y += leading
        y += 2  # paragraph gap
    doc.save(path)
    doc.close()
    print(f"Created: {path}")


if __name__ == "__main__":
    p1 = os.path.join(OUT_DIR, "pitch_mediscan_ai.pdf")
    p2 = os.path.join(OUT_DIR, "pitch_wanderstay_travel.pdf")
    make_pdf(PITCH_1, p1)
    make_pdf(PITCH_2, p2)
    print("Done.")
