from io import BytesIO
from html import escape
import re
from urllib.parse import urlparse
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def report_pdf(report):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(595, 842), rightMargin=45, leftMargin=45)
    styles = getSampleStyleSheet()
    story = [Paragraph('Verification report', styles['Title']),
             Paragraph('AI-Powered Fake News Detection and Verification System', styles['Normal']), Spacer(1, 18)]
    def p(text, style='Normal'):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 8))
    def p_with_citations(text, sources, style='Normal'):
        source_urls = {}
        for index, item in enumerate(sources or [], 1):
            source_id = item.get('id', index)
            url = item.get('url', '')
            parsed = urlparse(url)
            if type(source_id) is int and parsed.scheme == 'https' and parsed.hostname:
                source_urls[source_id] = url
        parts, cursor = [], 0
        for match in re.finditer(r'\[(\d+)\]', str(text)):
            parts.append(escape(str(text)[cursor:match.start()]))
            source_id = int(match.group(1))
            url = source_urls.get(source_id)
            label = escape(match.group(0))
            parts.append(f'<link href="{escape(url, quote=True)}" color="#11594e">{label}</link>' if url else label)
            cursor = match.end()
        parts.append(escape(str(text)[cursor:]))
        story.append(Paragraph(''.join(parts), styles[style]))
        story.append(Spacer(1, 8))
    p('Report ID: ' + report['id'])
    p('Created: ' + report['created_at'])
    p('Content type: ' + report['type'])
    original = report.get('content', '')
    if report.get('type') in ('image', 'video'):
        if report.get('filename'):
            p('Uploaded media file: ' + report['filename'])
        digest = (report.get('result') or {}).get('sha256')
        if digest:
            p('Uploaded media SHA-256: ' + digest)
        if original:
            p('Submitted claim / caption', 'Heading2')
            p(original)
    elif original and not original.isascii():
        p('Original content includes Unicode text. Read the full original in the web report.', 'Heading2')
    else:
        p(report.get('title', ''), 'Heading2')
    result = report.get('result') or {}
    media_only = (result.get('report_summary') or {}).get('mode') == 'media_only'
    rows = [['Measure', 'Result']]
    metric_rows = ([('AI-generated estimate', 'fake_percentage'),
                    ('Likely real estimate', 'real_percentage')]
                   if report.get('type') in ('image', 'video') else
                   [('Fake / manipulated content', 'fake_percentage'), ('Real / authentic class', 'real_percentage'),
                    ('AI-generated', 'ai_percentage'), ('Model confidence', 'confidence')])
    for name, key in metric_rows:
        val = result.get(key)
        rows.append([name, 'Not assessed' if val is None else f'{val:.1f}%'])
    table = Table(rows, colWidths=[320, 175])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11594e')),
                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                              ('TOPPADDING', (0, 0), (-1, -1), 12),
                              ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#d9e5e1'))]))
    story.extend([table, Spacer(1, 18)])
    text_model = result.get('text_model_assessment') or {}
    if report.get('type') == 'text' and text_model:
        p('Local text classifier output', 'Heading2')
        p('Status: ' + str(text_model.get('status', 'not assessed')))
        if text_model.get('verdict') and text_model['verdict'] != 'not_assessed':
            p('Classifier signal: ' + str(text_model['verdict']).replace('_', ' ').upper())
        p(text_model.get('reason', 'The text classifier did not produce a prediction.'))
        evaluation = text_model.get('evaluation') or {}
        if evaluation.get('accuracy') is not None:
            count = evaluation.get('test_count')
            suffix = f" on {count} examples" if count is not None else ''
            p(f"Held-out dataset accuracy: {evaluation['accuracy'] * 100:.1f}%{suffix}. This is not claim-verification accuracy.")
    verification = result.get('verification', {})
    summary = result.get('report_summary') or {}
    if summary:
        p('Visual media assessment' if media_only else 'Concise evidence assessment', 'Heading2')
        if not media_only:
            p('Verdict: ' + str(summary.get('verdict', 'unverified')).replace('_', ' ').upper(), 'Heading3')
        p(summary.get('headline', 'Media analysis complete.' if media_only else 'The claim could not be verified.'), 'Heading3')
        summary_sources = summary.get('sources') or verification.get('evidence', [])
        p_with_citations(summary.get('summary', 'No source-grounded summary is available.'), summary_sources)
        if summary.get('reason'):
            p_with_citations('Reason: ' + str(summary['reason']), summary_sources)
        if summary.get('provider'):
            p('Summary method: ' + summary['provider'])
        if summary.get('summary_source_ids') or summary.get('source_ids'):
            ids = summary.get('summary_source_ids') or summary.get('source_ids')
            p('Sources used for the summary: ' + ', '.join(f'[{number}]' for number in ids))
        for point in summary.get('key_points', []):
            ids = ', '.join(f'[{number}]' for number in point.get('source_ids', []))
            p_with_citations(f"- {point.get('text', '')}" + (f' Sources: {ids}' if ids else ''), summary_sources)
        for claim in summary.get('claims', []):
            p('Claim assessment: ' + str(claim.get('verdict', 'unverified')).upper(), 'Heading3')
            p(claim.get('text', ''))
            p_with_citations(claim.get('reason', ''), summary_sources)
            if claim.get('supporting_evidence'):
                p('Supporting source IDs: ' + ', '.join(f'[{number}]' for number in claim['supporting_evidence']))
            if claim.get('contradicting_evidence'):
                p('Contradicting source IDs: ' + ', '.join(f'[{number}]' for number in claim['contradicting_evidence']))
            if claim.get('missing_evidence'):
                p('Evidence still needed: ' + '; '.join(claim['missing_evidence']))
        if summary.get('model_signal'):
            p('Separate model signal: ' + summary['model_signal'])
        gemini_model = summary.get('gemini_model_assessment') or {}
        if report.get('type') == 'text' and gemini_model:
            p('Gemini model-only response (not source verified)', 'Heading2')
            p('Model-only verdict: ' + str(gemini_model.get('verdict', 'unverified')).upper(), 'Heading3')
            if gemini_model.get('answer'):
                p(gemini_model['answer'])
            p(gemini_model.get('reason', 'No separate Gemini model-only response was returned.'))
            p(gemini_model.get('basis', 'This response is not based on retrieved source evidence.'))
        retrieval = summary.get('rag_analysis') or {}
        if retrieval:
            p('Retrieval summary', 'Heading3')
            p(f"Method: {retrieval.get('method', 'Not recorded')}")
            p(f"Provider: {retrieval.get('provider', 'Not configured')} | status: {retrieval.get('status', 'unknown')} | excerpts: {retrieval.get('retrieved_count', 0)}")
            p(retrieval.get('scope', ''))
        for warning in summary.get('technical_warnings', []):
            p('Technical warning: ' + warning)
        for limitation in summary.get('limitations', []):
            p('Limitation: ' + limitation)
        if not media_only:
            p('The verdict is limited to the retrieved sources listed in this report. An unverified result means the available evidence did not settle the claim.')
    ai = result.get('ai_assessment', {})
    if ai.get('status') == 'experimental':
        p('Experimental AI-generation estimate', 'Heading2')
        if ai.get('scope') == 'text':
            p('This local classifier was trained on historical human and ChatGPT answers. Its benchmark does not establish accuracy on current generators, news writing or other languages, and it cannot prove authorship or factual truth.')
        else:
            p('This is an uncalibrated pretrained image-model score. Independent accuracy has not been measured for this project. It cannot prove AI authorship or factual truth.')
        if ai.get('scope') == 'sampled_video_frames':
            p(f"The score averages {ai.get('sample_count')} sampled video frames. Audio and generated motion are not assessed by this model.")
    if not media_only:
        p('Verification and evidence', 'Heading2')
        p('Evidence provider: ' + verification.get('provider', 'Not configured'))
        if verification.get('attribution_url'):
            p('Data attribution: ' + verification['attribution_url'])
        if verification.get('search_query'):
            p('Evidence search terms: ' + verification['search_query'])
        for provider in verification.get('providers', []):
            if isinstance(provider, dict):
                p(f"Source service: {provider.get('name', 'Unknown')} | {provider.get('status', 'unknown')}")
        p('Retrieved source count: ' + str(len(verification.get('evidence', []))))
        p('Status: ' + verification.get('status', 'unverified'))
        p(verification.get('explanation', ''))
        p('Checked at: ' + verification.get('checked_at', 'Not checked'))
    if result.get('metric_note'):
        p(result['metric_note'], 'Heading3')
    if result.get('observations'):
        p('Content observations', 'Heading2')
        for observation in result['observations']:
            p(f"{observation.get('label', 'Observation')}: {observation.get('value', '')}")
    if result.get('frames'):
        p('Sampled video AI-origin estimates', 'Heading2')
        for frame in result['frames']:
            p(f"At {frame.get('seconds', '?')} seconds: AI-generated estimate "
              + (f"{frame['ai_percentage']:.1f}%" if frame.get('ai_percentage') is not None else 'not available'))
    for source_number, item in enumerate(verification.get('evidence', []), 1):
        p(f"Source [{source_number}]: {item.get('title', '')}", 'Heading3')
        for key in ['source_type', 'claim', 'publisher', 'reliability', 'rating', 'published_at',
                    'reviewed_at', 'seen_at', 'retrieved_at', 'retrieval_status',
                    'snippet', 'relationship']:
            p(f'{key}: {item.get(key, "")}')
        source_url = item.get('url', '')
        if urlparse(source_url).scheme == 'https' and urlparse(source_url).hostname:
            safe_url = escape(source_url, quote=True)
            story.append(Paragraph(f'<link href="{safe_url}" color="#11594e">Open original source: {escape(source_url)}</link>', styles['Normal']))
            story.append(Spacer(1, 8))
    if result.get('speech'):
        p('Video speech transcription', 'Heading2')
        p('Status: ' + result['speech'].get('status', 'unavailable'))
        transcript = result.get('transcript', '')
        if transcript and transcript.isascii():
            p(transcript)
        elif transcript:
            p('The transcript includes Unicode text. Read the complete transcript in the web report.')
        assessment = result.get('speech_text_assessment', {})
        if assessment.get('fake_percentage') is not None:
            p(f"Spoken-text fake-class score: {assessment['fake_percentage']:.1f}%. This is separate from the visual score.")
        p('Automatic transcripts may contain errors. Review the words against the recording.')
    p('Model and interpretation', 'Heading2')
    p(result.get('scope', 'No model assessment available.'))
    for item in result.get('limitations', []):
        p(item)
    for item in result.get('models', []):
        p(f"{item.get('name')} | version {item.get('version')} | {item.get('status')}")
        if item.get('source_url'):
            p('Model source: ' + item['source_url'])
        for benchmark, external in item.get('external_evaluations', {}).items():
            p(f"External benchmark ({benchmark}): {external['accuracy'] * 100:.2f}% accuracy on {external['test_count']} claims. Historical article accuracy does not establish claim-verification accuracy.")
        evaluation = item.get('evaluation')
        if evaluation:
            p(f"Held-out accuracy: {evaluation['accuracy'] * 100:.2f}% on {evaluation['test_count']} test examples. This only describes that dataset.")
            if 'f1' in evaluation:
                p(f"F1: {evaluation['f1']:.4f}")
    if report.get('type') == 'text' and original and original.isascii():
        p('Original submitted text', 'Heading2')
        p(original)
    if report.get('type') in ('image', 'video'):
        p('Visual-origin scores are experimental estimates. They do not establish whether an event or caption is true, and they are not calibrated confidence.')
        inspection = result.get('media_inspection') or {}
        origin = result.get('media_origin') or {}
        verdict_labels = {'ai_generated': 'Likely AI-generated', 'likely_real': 'Likely real / not flagged as AI-generated',
                          'classification_unavailable': 'Origin classification unavailable'}
        if origin.get('verdict'):
            p('AI-origin classification: ' + verdict_labels.get(origin['verdict'], 'Classification unavailable'))
        if inspection.get('description'):
            p('Content description: ' + inspection['description'])
        if inspection.get('visual_artifacts'):
            p('Visible details to review: ' + inspection['visual_artifacts'])
        ocr_text = str(result.get('ocr_text') or inspection.get('visible_text') or '').strip()
        if ocr_text:
            p('Text read from the media (OCR)', 'Heading2')
            p(ocr_text)
    else:
        p('Confidence is a model probability, not measured accuracy or a guarantee of truth. Fake and real are complementary model classes; AI generation is a separate assessment.')
    if report.get('language') in ('hi', 'te'):
        p('This PDF uses an English summary. See the web report for the original Unicode content.')
    doc.build(story)
    return buffer.getvalue()
