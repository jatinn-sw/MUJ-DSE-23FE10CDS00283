import json
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import Counter
from app.config import get_config
from app.utils import load_prompt, generate_id
from app.claim_extractor import Claim
from app.llm_analyzer import AnalysisResult
from app.pdf_parser import Document
from app.academic_search import AcademicSource
from app.web_search import WebSource


@dataclass
class AuditReport:
    document: Document
    claims: List[Claim]
    analyses: List[AnalysisResult]
    external_sources: List[AcademicSource]
    web_sources: List[WebSource]
    internal_evidence: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def to_markdown(self) -> str:
        verdict_counts = Counter(a.verdict for a in self.analyses)
        
        md = []
        md.append(f"# ClaimGuard AI - Evidence Audit Report")
        md.append(f"\n**Document:** {self.document.filename}")
        md.append(f"**Pages:** {self.document.num_pages} | **Words:** {self.document.word_count:,}")
        md.append(f"**Claims Analyzed:** {len(self.claims)} | **Sources:** {len(self.external_sources) + len(self.web_sources)}")
        
        md.append("\n## Executive Summary")
        total = len(self.analyses)
        denom = max(1, total)
        supported = verdict_counts.get("SUPPORTED", 0)
        partial = verdict_counts.get("PARTIALLY_SUPPORTED", 0)
        contradicted = verdict_counts.get("CONTRADICTED", 0)
        overstated = verdict_counts.get("OVERSTATED", 0)
        insufficient = verdict_counts.get("INSUFFICIENT_EVIDENCE", 0)
        
        md.append(f"This audit analyzed {total} claims extracted from the document.")
        md.append(f"- **Supported:** {supported} ({supported/denom*100:.0f}%)")
        md.append(f"- **Partially Supported:** {partial} ({partial/denom*100:.0f}%)")
        md.append(f"- **Contradicted:** {contradicted} ({contradicted/denom*100:.0f}%)")
        md.append(f"- **Overstated:** {overstated} ({overstated/denom*100:.0f}%)")
        md.append(f"- **Insufficient Evidence:** {insufficient} ({insufficient/denom*100:.0f}%)")
        
        md.append("\n## Document Statistics")
        md.append(f"- Pages: {self.document.num_pages}")
        md.append(f"- Word Count: {self.document.word_count:,}")
        md.append(f"- Sections: {', '.join(self.document.sections)}")
        md.append(f"- Claims Extracted: {len(self.claims)}")
        md.append(f"- External Sources Found: {len(self.external_sources)}")
        md.append(f"- Web Sources Found: {len(self.web_sources)}")
        
        md.append("\n## Claim Overview")
        md.append("| Verdict | Count | Percentage |")
        md.append("|---------|-------|------------|")
        for verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"]:
            count = verdict_counts.get(verdict, 0)
            md.append(f"| {verdict} | {count} | {count/denom*100:.1f}% |")
        
        for verdict in ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "OVERSTATED", "INSUFFICIENT_EVIDENCE"]:
            verdict_claims = [(c, a) for c, a in zip(self.claims, self.analyses) if a.verdict == verdict]
            if verdict_claims:
                md.append(f"\n## {verdict} Claims")
                for claim, analysis in verdict_claims:
                    md.append(f"\n### {claim.claim_id}: {claim.text[:100]}...")
                    md.append(f"- **Category:** {claim.category}")
                    md.append(f"- **Confidence:** {analysis.confidence:.0%}")
                    md.append(f"- **Pages:** {', '.join(str(s.page) for s in self.document.sentences if s.sentence_id in claim.sentence_ids)}")
                    md.append(f"- **Reasoning:** {analysis.reasoning}")
                    if analysis.suggested_revision:
                        md.append(f"- **Suggested Revision:** {analysis.suggested_revision}")
        
        md.append("\n## Evidence Conflicts")
        conflicts = self._detect_conflicts()
        if conflicts:
            for conflict in conflicts:
                md.append(f"\n### Claim: {conflict['claim'].text[:100]}")
                md.append(f"- Supporting: {conflict['supporting']}")
                md.append(f"- Contradicting: {conflict['contradicting']}")
        else:
            md.append("No significant evidence conflicts detected.")
        
        md.append("\n## Source Analysis")
        tier_counts = Counter(src.tier for src in self.external_sources)
        md.append("### Source Quality Distribution")
        for tier in ["A", "B", "C"]:
            md.append(f"- **Tier {tier}:** {tier_counts.get(tier, 0)} sources")
        
        year_counts = Counter(src.year for src in self.external_sources if src.year)
        if year_counts:
            md.append("\n### Publication Years")
            for year in sorted(year_counts.keys(), reverse=True):
                md.append(f"- {year}: {year_counts[year]} sources")
        
        md.append("\n## Methodological Concerns")
        md.append("1. This audit is based on available evidence at the time of analysis.")
        md.append("2. External search is limited to indexed academic sources and web search APIs.")
        md.append("3. LLM reasoning is constrained to retrieved evidence; internal knowledge is not used as evidence.")
        md.append("4. 'Insufficient Evidence' does not mean a claim is false.")
        md.append("5. 'Supported' means supported by available evidence, not proven true.")
        
        md.append("\n## Recommendations")
        md.append("1. **For Authors:** Revise overstated claims to match evidence strength.")
        md.append("2. **For Reviewers:** Pay attention to contradicted and overstated claims.")
        md.append("3. **For Readers:** Trace claims to original evidence using page references.")
        
        md.append("\n---\n*Report generated by ClaimGuard AI*")
        md.append("*Verify the claim. Trace the evidence. Challenge the conclusion.*")
        
        return "\n".join(md)

    def _detect_conflicts(self) -> List[Dict[str, Any]]:
        conflicts = []
        for claim, analysis in zip(self.claims, self.analyses):
            supporting = len(analysis.supporting_evidence)
            contradicting = len(analysis.contradicting_evidence)
            if supporting > 0 and contradicting > 0:
                conflicts.append({
                    "claim": claim,
                    "supporting": supporting,
                    "contradicting": contradicting,
                })
        return conflicts

    def to_pdf(self) -> bytes:
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40,
        )
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#0f172a"),
            alignment=0,
        )
        h2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6,
        )
        normal_style = ParagraphStyle(
            "DocNormal",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#334155"),
        )
        claim_style = ParagraphStyle(
            "ClaimText",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f172a"),
        )

        elements = []
        elements.append(Paragraph("ClaimGuard AI — Evidence Audit Report", title_style))
        elements.append(Paragraph("<i>Verify the claim. Trace the evidence. Challenge the conclusion.</i>", normal_style))
        elements.append(Spacer(1, 8))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

        # Doc stats
        stats_text = (
            f"<b>Document:</b> {self.document.filename} &nbsp;|&nbsp; "
            f"<b>Pages:</b> {self.document.num_pages} &nbsp;|&nbsp; "
            f"<b>Words:</b> {self.document.word_count:,} &nbsp;|&nbsp; "
            f"<b>Claims Audited:</b> {len(self.claims)}"
        )
        elements.append(Paragraph(stats_text, normal_style))
        elements.append(Spacer(1, 10))

        # Verdict Overview Table
        elements.append(Paragraph("Audit Summary", h2_style))
        verdict_counts = Counter(a.verdict for a in self.analyses)
        total = max(1, len(self.analyses))
        
        table_data = [["Verdict", "Count", "Percentage", "Status Definition"]]
        verdict_rows = [
            ("SUPPORTED", verdict_counts.get("SUPPORTED", 0), "Supported by internal/external empirical evidence"),
            ("PARTIALLY_SUPPORTED", verdict_counts.get("PARTIALLY_SUPPORTED", 0), "Supported under narrower conditions or caveats"),
            ("OVERSTATED", verdict_counts.get("OVERSTATED", 0), "Wording is stronger than empirical findings justify"),
            ("CONTRADICTED", verdict_counts.get("CONTRADICTED", 0), "Reliable evidence conflicts with the stated claim"),
            ("INSUFFICIENT_EVIDENCE", verdict_counts.get("INSUFFICIENT_EVIDENCE", 0), "Not enough evidence to confirm or refute"),
        ]
        for v, cnt, desc in verdict_rows:
            table_data.append([v, str(cnt), f"{cnt/total*100:.1f}%", desc])

        summary_table = Table(table_data, colWidths=[130, 45, 65, 290])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 14))

        # Claims Breakdown
        elements.append(Paragraph("Detailed Claim Audits", h2_style))
        for claim, analysis in zip(self.claims[:12], self.analyses[:12]):
            vcolor = {
                "SUPPORTED": "#16a34a",
                "PARTIALLY_SUPPORTED": "#d97706",
                "OVERSTATED": "#e11d48",
                "CONTRADICTED": "#dc2626",
                "INSUFFICIENT_EVIDENCE": "#64748b",
            }.get(analysis.verdict, "#0f172a")

            elements.append(Paragraph(f"• <b>{claim.claim_id}</b> [{claim.category}]: {claim.text}", claim_style))
            audit_detail = (
                f"<b>Verdict:</b> <font color='{vcolor}'><b>{analysis.verdict}</b></font> "
                f"({analysis.confidence:.0%} confidence)<br/>"
                f"<b>Reasoning:</b> {analysis.reasoning}"
            )
            if analysis.suggested_revision:
                audit_detail += f"<br/><b>Suggested Revision:</b> <i>{analysis.suggested_revision}</i>"
            elements.append(Paragraph(audit_detail, normal_style))
            elements.append(Spacer(1, 6))

        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
        elements.append(Paragraph("<i>ClaimGuard AI Evidence Audit Platform — Generated for academic review</i>", normal_style))

        doc.build(elements)
        return buffer.getvalue()

    def to_json(self) -> Dict[str, Any]:
        return {
            "document": {
                "filename": self.document.filename,
                "num_pages": self.document.num_pages,
                "word_count": self.document.word_count,
                "sections": self.document.sections,
            },
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "category": c.category,
                    "importance": c.importance,
                    "sentence_ids": c.sentence_ids,
                }
                for c in self.claims
            ],
            "analyses": [
                {
                    "claim_id": a.claim_id,
                    "verdict": a.verdict,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning,
                    "suggested_revision": a.suggested_revision,
                }
                for a in self.analyses
            ],
            "summary": {
                "total_claims": len(self.claims),
                "verdict_distribution": dict(Counter(a.verdict for a in self.analyses)),
            }
        }


def generate_report(
    document: Document,
    claims: List[Claim],
    analyses: List[AnalysisResult],
    external_sources: List[AcademicSource],
    web_sources: List[WebSource],
    internal_evidence: Dict[str, Any] = None,
    processing_time: float = 0.0,
) -> AuditReport:
    return AuditReport(
        document=document,
        claims=claims,
        analyses=analyses,
        external_sources=external_sources,
        web_sources=web_sources,
        internal_evidence=internal_evidence or {},
        processing_time=processing_time,
    )