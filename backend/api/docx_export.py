"""Pure-Python OpenXML (.docx) clinical consultation & prescription note generator."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

from backend.api.schemas import PredictResponse

_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""

_ROOT_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""

_DOC_RELS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

_STYLES_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="19"/>
      <w:color w:val="1E293B"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def _p(
    text: str,
    *,
    bold: bool = False,
    size_half_pt: int = 19,
    color: str = "1E293B",
    space_after: int = 60,
    space_before: int = 0,
) -> str:
    b_tag = "<w:b/>" if bold else ""
    return (
        "<w:p>"
        f"<w:pPr><w:spacing w:before=\"{space_before}\" w:after=\"{space_after}\"/></w:pPr>"
        "<w:r>"
        f"<w:rPr>{b_tag}<w:color w:val=\"{color}\"/><w:sz w:val=\"{size_half_pt}\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def _label_value_p(label: str, value: str, *, color: str = "1E293B") -> str:
    return (
        "<w:p>"
        "<w:pPr><w:spacing w:before=\"0\" w:after=\"50\"/></w:pPr>"
        "<w:r><w:rPr><w:b/><w:color w:val=\"0F172A\"/><w:sz w:val=\"18\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(label)} </w:t></w:r>"
        f"<w:r><w:rPr><w:color w:val=\"{color}\"/><w:sz w:val=\"18\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{escape(value)}</w:t></w:r>"
        "</w:p>"
    )


def _writing_box(title: str, min_height_twips: int) -> str:
    """Create a spacious single-cell bordered box for doctors to write notes, tests, or Rx."""
    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblW w:w=\"10466\" w:type=\"dxa\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"CBD5E1\"/>"
        "<w:left w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"CBD5E1\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"CBD5E1\"/>"
        "<w:right w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"CBD5E1\"/>"
        "</w:tblBorders>"
        "<w:tblCellMar>"
        "<w:top w:w=\"90\" w:type=\"dxa\"/><w:left w:w=\"140\" w:type=\"dxa\"/>"
        "<w:bottom w:w=\"90\" w:type=\"dxa\"/><w:right w:w=\"140\" w:type=\"dxa\"/>"
        "</w:tblCellMar>"
        "</w:tblPr>"
        "<w:tblGrid><w:gridCol w:w=\"10466\"/></w:tblGrid>"
        "<w:tr>"
        f"<w:trPr><w:trHeight w:val=\"{min_height_twips}\" w:hRule=\"atLeast\"/></w:trPr>"
        "<w:tc>"
        "<w:tcPr><w:tcW w:w=\"10466\" w:type=\"dxa\"/></w:tcPr>"
        + _p(title, bold=True, size_half_pt=20, color="0F6E75", space_after=120)
        + _p("", size_half_pt=20, space_after=0)
        + "</w:tc>"
        "</w:tr>"
        "</w:tbl>"
    )


def build_consultation_docx(
    result: PredictResponse,
    symptom_display_names: list[str],
    patient_name: str = "",
) -> bytes:
    """Build a 1-page A4 Word document with compact triage header and ~70% open doctor writing area."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    patient_line = (
        f"Patient: {patient_name.strip()}"
        if patient_name.strip()
        else "Patient Name / ID: ____________________________"
    )
    meta_line = f"{patient_line}     Age / Sex: __________     BP / HR / Temp / SpO2: ____________________"

    top = result.predictions[0] if result.predictions else None
    top_text = (
        f"{top.display_name} ({top.probability * 100:.1f}%)"
        if top is not None
        else "Pending assessment"
    )
    others = [
        f"{c.display_name} ({c.probability * 100:.1f}%)"
        for c in result.predictions[1:3]
    ]
    diff_line = f"{top_text}" + (f"   |   Differentials: {', '.join(others)}" if others else "")

    urgency = result.triage.urgency.upper()
    urgency_color = (
        "B91C1C"
        if urgency == "CRITICAL"
        else "B45309"
        if urgency == "HIGH"
        else "1D4ED8"
        if urgency == "MODERATE"
        else "15803D"
    )
    red_flag_txt = (
        f"   |   RED FLAG: {'; '.join(result.triage.red_flags)}"
        if result.triage.red_flags
        else ""
    )

    summary_box = (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblW w:w=\"10466\" w:type=\"dxa\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"0F6E75\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"0F6E75\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"0F6E75\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"0F6E75\"/>"
        "</w:tblBorders>"
        "<w:tblCellMar>"
        "<w:top w:w=\"80\" w:type=\"dxa\"/><w:left w:w=\"130\" w:type=\"dxa\"/>"
        "<w:bottom w:w=\"80\" w:type=\"dxa\"/><w:right w:w=\"130\" w:type=\"dxa\"/>"
        "</w:tblCellMar>"
        "</w:tblPr>"
        "<w:tblGrid><w:gridCol w:w=\"10466\"/></w:tblGrid>"
        "<w:tr><w:tc>"
        "<w:tcPr><w:tcW w:w=\"10466\" w:type=\"dxa\"/><w:shd w:val=\"clear\" w:fill=\"F8FAFC\"/></w:tcPr>"
        + _label_value_p(
            "Triage Urgency & Referral:",
            f"{urgency}  —  {result.triage.specialist}{red_flag_txt}",
            color=urgency_color,
        )
        + _label_value_p("Presenting Symptoms:", ", ".join(symptom_display_names))
        + _label_value_p("VitalAI Assessment:", diff_line)
        + _label_value_p("Clinical Rationale:", result.primary_reasoning)
        + "</w:tc></w:tr></w:tbl>"
    )

    body_xml = (
        _p(
            f"VitalAI Clinical Consultation & Prescription Note          Date: {timestamp}",
            bold=True,
            size_half_pt=24,
            color="0F6E75",
            space_after=70,
        )
        + _p(meta_line, size_half_pt=18, color="334155", space_after=90)
        + summary_box
        + _p("", size_half_pt=10, space_after=90)
        + _writing_box("Investigations / Lab Tests Ordered:", min_height_twips=2150)
        + _p("", size_half_pt=10, space_after=90)
        + _writing_box("Rx  —  Medications, Dosage & Duration:", min_height_twips=5250)
        + _p("", size_half_pt=10, space_after=90)
        + _writing_box("Doctor's Clinical Notes & Follow-Up Plan:", min_height_twips=2000)
        + _p(
            "Attending Physician Signature & Stamp: ________________________________________     Follow-Up Date: ____________",
            bold=True,
            size_half_pt=18,
            color="334155",
            space_before=140,
            space_after=40,
        )
        + _p(
            result.disclaimer,
            size_half_pt=15,
            color="64748B",
            space_after=0,
        )
    )

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body>"
        + body_xml
        + "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"640\" w:left=\"720\"/>"
        "</w:sectPr>"
        "</w:body>"
        "</w:document>"
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
        zf.writestr("_rels/.rels", _ROOT_RELS_XML)
        zf.writestr("word/_rels/document.xml.rels", _DOC_RELS_XML)
        zf.writestr("word/styles.xml", _STYLES_XML)
        zf.writestr("word/document.xml", document_xml.encode("utf-8"))
    return buf.getvalue()
