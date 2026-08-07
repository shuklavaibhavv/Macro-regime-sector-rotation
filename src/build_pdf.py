import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

md_filepath = '/Users/vaibhavshukla/.gemini/antigravity/brain/79e420a3-1a60-45a5-ba38-2044c77c540c/jpmc_interview_qa_guide.md'
pdf_filepath = '/Users/vaibhavshukla/Desktop/JPMC/macro_regime_project/JPMC_Quantitative_Interview_100_QA.pdf'

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont('Helvetica', 8)
        self.setFillColor(colors.HexColor('#555555'))
        
        # Top Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 755, 'J.P. Morgan Quantitative Research & Analytics Interview Guide')
            self.setStrokeColor(colors.HexColor('#002060'))
            self.setLineWidth(0.75)
            self.line(54, 747, 612 - 54, 747)
            
        # Bottom Footer (all pages)
        page_text = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(612 - 54, 30, page_text)
        self.drawString(54, 30, 'CONFIDENTIAL — MACRO REGIME SECTOR ROTATION STRATEGY (JPMC PREP)')
        self.setStrokeColor(colors.HexColor('#CCCCCC'))
        self.setLineWidth(0.5)
        self.line(54, 42, 612 - 54, 42)
        
        self.restoreState()

def format_ans_block(raw_text):
    # Escape XML special characters first
    text = raw_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Re-enable ReportLab HTML tags
    text = re.sub(r'\*\*(.*?)\*\*', lambda m: f'<b>{m.group(1)}</b>', text)
    text = re.sub(r'\*(.*?)\*', lambda m: f'<i>{m.group(1)}</i>', text)
    text = re.sub(r'`(.*?)`', lambda m: f'<font face="Courier" color="#800000"><b>{m.group(1)}</b></font>', text)
    text = text.replace('\n', '<br/>')
    return text

def main():
    with open(md_filepath, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        pdf_filepath,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#002060'),
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'ModuleHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#002060'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#C00000'), # RED font for questions
        spaceBefore=10,
        spaceAfter=3,
        keepWithNext=True
    )

    answer_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#000000'), # BLACK font for answers
        spaceBefore=2,
        spaceAfter=6
    )

    story = []
    story.append(Paragraph('J.P. Morgan (JPMC) Quantitative Interview Preparation Guide', title_style))
    story.append(Paragraph('Macro Regime Classification & Sector Rotation Strategy — 100 Questions & Answers', subtitle_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#002060'), spaceAfter=10))

    lines = md_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('# J.P. Morgan') or line.startswith('## Macro Regime') or line == '---':
            i += 1
            continue
            
        if line.startswith('## Module'):
            mod_title = line.replace('## ', '').strip()
            story.append(Spacer(1, 6))
            story.append(Paragraph(mod_title, h1_style))
            story.append(HRFlowable(width='100%', thickness=0.75, color=colors.HexColor('#002060'), spaceAfter=6))
            i += 1
            continue

        if line.startswith('### Q'):
            q_text = line.replace('### ', '').strip()
            q_text_clean = q_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            q_text_clean = re.sub(r'\*\*(.*?)\*\*', lambda m: f'<b>{m.group(1)}</b>', q_text_clean)
            story.append(Paragraph(q_text_clean, question_style))
            i += 1
            continue

        if line.startswith('**Answer:**') or line.startswith('Answer:'):
            ans_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].startswith('### Q') and not lines[i].startswith('## Module') and not lines[i].startswith('---'):
                ans_lines.append(lines[i])
                i += 1
            
            ans_block = '\n'.join(ans_lines).strip()
            formatted_html = format_ans_block(ans_block)
            story.append(Paragraph(formatted_html, answer_style))
            continue

        i += 1

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {pdf_filepath}")

if __name__ == '__main__':
    main()
