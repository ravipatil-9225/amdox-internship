"""
Export utilities for Streamlit dashboard.
Provides PDF and Excel export functionality.
"""
import io
import pandas as pd
from fpdf import FPDF
from datetime import datetime


def export_to_excel(dataframes: dict, filename: str = "neuralretail_report.xlsx") -> bytes:
    """
    Export multiple DataFrames to an Excel workbook with separate sheets.
    
    Args:
        dataframes: dict of {sheet_name: DataFrame}
        filename: output filename
    Returns:
        bytes of the Excel file
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=True)  # Excel max 31 chars
    return output.getvalue()


def export_to_pdf(title: str, sections: list, filename: str = "neuralretail_report.pdf") -> bytes:
    """
    Generate a professional PDF report.
    
    Args:
        title: Report title
        sections: list of dicts with 'heading', 'content' (str), and optional 'table' (DataFrame)
    Returns:
        bytes of the PDF file
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 28)
    pdf.cell(0, 60, '', ln=True)
    pdf.cell(0, 15, 'NeuralRetail', ln=True, align='C')
    pdf.set_font('Helvetica', '', 16)
    pdf.cell(0, 10, 'AI Sales Intelligence Platform', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
    pdf.cell(0, 10, 'Amdox Technologies | Data Science & Analytics', ln=True, align='C')
    pdf.cell(0, 10, title, ln=True, align='C')
    
    # Content pages
    for section in sections:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 12, section.get('heading', ''), ln=True)
        pdf.ln(5)
        
        # Text content
        if 'content' in section:
            pdf.set_font('Helvetica', '', 11)
            for line in section['content'].split('\n'):
                pdf.cell(0, 7, line.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        
        # Table
        if 'table' in section and section['table'] is not None:
            df = section['table']
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 9)
            
            # Calculate column widths
            cols = list(df.columns)
            col_width = min(180 / len(cols), 45)
            
            # Header
            for col in cols:
                pdf.cell(col_width, 8, str(col)[:20], border=1, align='C')
            pdf.ln()
            
            # Rows (limit to 50 for PDF)
            pdf.set_font('Helvetica', '', 8)
            for idx, row in df.head(50).iterrows():
                for col in cols:
                    val = str(row[col])[:18]
                    pdf.cell(col_width, 7, val, border=1, align='C')
                pdf.ln()
    
    return bytes(pdf.output())
