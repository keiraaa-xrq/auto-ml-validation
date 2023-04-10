from docx import Document
from typing import *
from .util import *


def generate_report(
    m_result,
    bm_result: Optional[dict] = None,
    report_path: str = 'report.docx'
):
    try:
        generate_charts(m_result, False)
        if bm_result:
            generate_charts(bm_result, True)
    except:
        print("Error in saving images to local")

    doc = Document()
    doc.add_heading('Model Validation Report', 0)

    try:
        doc.add_heading('Model Performance Metrics', 1)
        doc.add_paragraph(
            'This section includes generic model performance metrics.')
        generate_chart_table(doc, 'dist', m_result, bm_result != None)
        generate_chart_table(doc, 'confusion', m_result, bm_result != None)

        for m in ['accuracy', 'precision', 'recall', 'f1_score']:
            generate_metric_table(doc, m, m_result, bm_result)

        generate_chart_auc_table(doc, 'roc', m_result, bm_result)
        generate_chart_auc_table(doc, 'pr', m_result, bm_result)
        generate_chart_table(doc, 'lift', m_result, bm_result != None)
        generate_gini_table(doc, m_result, bm_result)
    except:
        print("Error in generating model performance report")

    doc.add_heading('Statistical Metrics', 1)
    doc.add_paragraph("This section includes statistical metrics.")
    try:
        generate_psi_table(doc, m_result, bm_result)
    except Exception as e:
        print(f'Error in generating PSI: {e}')
    try:
        generate_csi_table(doc, m_result)
    except Exception as e:
        print(f'Error in generating PSI: {e}')
    try:
        generate_ks_table(doc, m_result, bm_result)
    except Exception as e:
        print(f'Error in generating KS test: ({e}')
    try:
        generate_feature_gini_table(doc, m_result, bm_result)
    except Exception as e:
        print(f'Error in generating GINI: {e}')

    try:
        doc.add_heading('Transparency Metrics', 1)
        doc.add_paragraph(
            "This section includes LIME and SHAP interpretability under the framework of MAS's FEAT metrics.")
        doc.add_paragraph(interp_map['local_']['exp'])
        doc.add_paragraph(interp_map['global_']['exp'])

        generate_trans_table(doc, 'lime', m_result, bm_result)
        generate_trans_table(doc, 'shap', m_result, bm_result)
    except:
        print("Error in generating transparency report")

    doc.save(report_path)
