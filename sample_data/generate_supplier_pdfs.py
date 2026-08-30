"""
generate_supplier_pdfs.py
Generates four fictional supplier RFP response PDFs used for testing and
demoing the app. Run: python sample_data/generate_supplier_pdfs.py
No real/confidential supplier data is used -- entirely synthetic.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

OUT_DIR = os.path.join(os.path.dirname(__file__), "supplier_pdfs")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()


def section(title, body_paragraphs, story):
    story.append(Paragraph(title, styles["Heading2"]))
    for p in body_paragraphs:
        story.append(Paragraph(p, styles["Normal"]))
        story.append(Spacer(1, 6))
    story.append(Spacer(1, 10))


def price_table(rows, story):
    data = [["Item", "Detail", "Cost (USD)"]] + rows
    t = Table(data, colWidths=[1.8 * inch, 3 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))


def build_pdf(filename, title, sections, prices):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 14)]
    for sec_title, paras in sections:
        section(sec_title, paras, story)
    story.append(Paragraph("Price Table", styles["Heading2"]))
    price_table(prices, story)
    doc.build(story)
    print(f"Created {path}")


# ---------------------------------------------------------------------------
# Apex Systems: strong technical + security, higher price, moderate schedule
# ---------------------------------------------------------------------------
build_pdf(
    "Apex_Systems.pdf",
    "RFP Response: Apex Systems",
    [
        ("Executive Summary", [
            "Apex Systems proposes a modular, cloud-native platform architecture designed "
            "for high scalability and long-term maintainability, addressing the client's "
            "requirement for a resilient procurement automation system.",
        ]),
        ("Proposed Solution & Architecture", [
            "Our architecture uses a microservices design with independent scaling of "
            "the ingestion, scoring, and reporting layers. Integrations are exposed via "
            "documented REST APIs with OAuth2, supporting both synchronous and event-driven "
            "consumption patterns.",
            "The system is built for horizontal scalability using container orchestration, "
            "with load-tested throughput exceeding 500 requests/second at P95 latency under 200ms.",
        ]),
        ("Timeline, Team Structure & Milestones", [
            "Delivery is organized into four phases across 16 weeks: Discovery (2 weeks), "
            "Core Build (8 weeks), Integration & Hardening (4 weeks), and Go-Live (2 weeks). "
            "A dedicated team of 6 (1 architect, 3 engineers, 1 QA, 1 PM) is staffed for the "
            "full engagement, with weekly milestone checkpoints and a documented risk log.",
        ]),
        ("Security, Compliance & Risk Controls", [
            "Apex maintains SOC 2 Type II and ISO 27001 certifications. All data is encrypted "
            "at rest (AES-256) and in transit (TLS 1.3). Role-based access control, audit "
            "logging, and quarterly third-party penetration testing are standard. A named "
            "compliance officer is assigned to every engagement for auditability.",
        ]),
        ("Support Model, Experience & References", [
            "24/7 tiered support (L1-L3) with a 1-hour SLA for critical incidents. Apex has "
            "delivered 3 comparable procurement-automation platforms for enterprise clients "
            "in the last 4 years, with references available on request.",
        ]),
        ("Risks", [
            "Primary risk is the 16-week schedule being tight if client-side integration "
            "credentials are delayed beyond week 2; mitigation includes a buffer sprint.",
        ]),
    ],
    prices=[
        ["Platform License", "Annual, includes updates", "58,000"],
        ["Implementation", "One-time, phases 1-4", "42,000"],
        ["Support (Year 1)", "24/7 tiered, included Y1", "0 (bundled)"],
        ["Total (Year 1)", "License + implementation", "100,000"],
    ],
)

# ---------------------------------------------------------------------------
# BrightPath Tech: lowest price, fastest timeline, weak compliance/experience
# ---------------------------------------------------------------------------
build_pdf(
    "BrightPath_Tech.pdf",
    "RFP Response: BrightPath Tech",
    [
        ("Executive Summary", [
            "BrightPath Tech offers a lean, cost-effective solution that can be delivered "
            "quickly, ideal for teams wanting to move fast without a large upfront budget.",
        ]),
        ("Proposed Solution & Approach", [
            "We will build a single-service web application connecting to the client's "
            "existing SQL database. The system will expose a basic REST endpoint for "
            "data upload and a dashboard for results.",
        ]),
        ("Timeline, Team & Milestones", [
            "We can deliver the full solution in 6 weeks using a small team of 2 developers. "
            "Milestones: Week 2 - basic upload flow; Week 4 - scoring logic; Week 6 - dashboard "
            "and handover.",
        ]),
        ("Security & Compliance", [
            "Standard HTTPS is used for all traffic. Passwords are hashed. We are happy to "
            "discuss additional compliance requirements if the client can specify them.",
        ]),
        ("Support & Experience", [
            "Email support is available during business hours (9am-6pm). This would be our "
            "first project specifically in the procurement-evaluation space, though we have "
            "built several internal dashboards for other clients.",
        ]),
        ("Pricing Notes", [
            "Pricing assumes the client provides sample data in a ready-to-use CSV/JSON "
            "format; additional data-cleaning work would be billed separately at $80/hour.",
        ]),
    ],
    prices=[
        ["Development", "One-time, 6-week build", "18,000"],
        ["Hosting", "Annual", "2,400"],
        ["Support (Year 1)", "Business hours, email only", "3,000"],
        ["Total (Year 1)", "Dev + hosting + support", "23,400"],
    ],
)

# ---------------------------------------------------------------------------
# NexaWorks: balanced, strongest implementation plan and support model
# ---------------------------------------------------------------------------
build_pdf(
    "NexaWorks.pdf",
    "RFP Response: NexaWorks",
    [
        ("Executive Summary", [
            "NexaWorks proposes a balanced solution that combines solid technical design "
            "with a highly structured implementation approach and a support model built "
            "around long-term partnership.",
        ]),
        ("Proposed Solution & Architecture", [
            "The platform uses a service-oriented architecture with a document-processing "
            "pipeline, a rules engine for deterministic scoring, and a reporting layer. "
            "Integrations are available via REST API and scheduled batch export.",
        ]),
        ("Implementation Plan", [
            "Our implementation methodology follows a detailed 5-phase plan: Discovery (1 wk), "
            "Design Sign-off (1 wk), Build Sprint 1-3 (6 wks), UAT (2 wks), Go-Live & Hypercare "
            "(2 wks) — 12 weeks total. Each phase has named owners, entry/exit criteria, and a "
            "RAID log reviewed weekly with the client's steering committee. Staffing includes "
            "1 delivery lead, 2 engineers, 1 QA, and a dedicated client success manager from day one.",
        ]),
        ("Security & Compliance", [
            "Data is encrypted in transit (TLS 1.2+) and at rest. We follow a documented "
            "internal security checklist aligned to common industry frameworks and can "
            "pursue formal certification alongside the client if required.",
        ]),
        ("Support Model, Experience & References", [
            "Dedicated named support contact plus a ticketing portal, 4-hour response SLA "
            "during business hours and next-business-day for non-critical items. NexaWorks "
            "has delivered 5 similar vendor-scoring and evaluation platforms across logistics "
            "and healthcare clients in the past 3 years; two references are available.",
        ]),
        ("Risks", [
            "Integration risk exists if client legacy systems lack a documented API; a "
            "discovery-phase spike is included specifically to de-risk this.",
        ]),
    ],
    prices=[
        ["Platform Build", "One-time, 12-week plan", "36,000"],
        ["Annual Support", "4-hour SLA, dedicated CSM", "9,600"],
        ["Hosting", "Annual, managed cloud", "3,600"],
        ["Total (Year 1)", "Build + support + hosting", "49,200"],
    ],
)

# ---------------------------------------------------------------------------
# Orbit Digital: strong experience/references, vague integration, medium price
# ---------------------------------------------------------------------------
build_pdf(
    "Orbit_Digital.pdf",
    "RFP Response: Orbit Digital",
    [
        ("Executive Summary", [
            "Orbit Digital brings deep domain experience in procurement technology, having "
            "delivered similar systems for a range of enterprise and mid-market clients over "
            "the past decade.",
        ]),
        ("Proposed Solution", [
            "We will build a web-based evaluation tool that connects to the client's data "
            "sources and produces supplier rankings. Specific integration mechanisms will "
            "be finalized during the discovery phase based on the client's existing systems.",
        ]),
        ("Timeline & Team", [
            "Estimated delivery is 10-14 weeks depending on discovery findings. Our senior "
            "delivery team has an average of 8 years of experience in enterprise procurement "
            "software.",
        ]),
        ("Security & Compliance", [
            "Orbit Digital is ISO 27001 certified at the company level. Standard encryption "
            "and access-control practices are applied to all client engagements; certificate "
            "and audit documentation is available on request.",
        ]),
        ("Support Model, Experience & References", [
            "Orbit Digital has completed 7 comparable supplier-evaluation projects over the "
            "past 9 years, including for two Fortune 500 procurement teams. Three client "
            "references with contact details are available on request. Support is offered "
            "via a dedicated account manager with a 24-hour response SLA.",
        ]),
        ("Risks", [
            "Because the exact integration approach depends on discovery-phase findings, "
            "final architecture details and timeline may shift after week 2.",
        ]),
    ],
    prices=[
        ["Discovery & Design", "2-3 weeks", "12,000"],
        ["Build & Integration", "8-11 weeks, scope TBD post-discovery", "38,000"],
        ["Annual Support", "24-hour SLA, dedicated AM", "7,200"],
        ["Total (Year 1, est.)", "Discovery + build + support", "57,200"],
    ],
)

print("All 4 synthetic supplier PDFs generated.")
