import React, { useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, CircleCheck, CircleHelp, Download, FileText,
  Fingerprint, Image as ImageIcon, ScanLine, Search, ShieldCheck, Sparkles } from 'lucide-react';
import { date } from '../lib/common.js';
import { api } from '../lib/api.js';
import { analysisErrorMessage } from '../lib/analysisErrors.js';
import { IconBadge } from './ui.jsx';

function Metric({ title, value, icon: Icon, t, tone }) {
  const assessed = typeof value === 'number' && Number.isFinite(value);
  return <div className={`metric ${tone} ${assessed ? '' : 'metric-unavailable'}`}>
    <div className="metric-label"><Icon size={17}/><span>{title}</span></div>
    <strong>{!assessed ? <span className="not-assessed">{t.notAssessed}</span> : <>{value.toFixed(1)}<small>%</small></>}</strong>
    <div className="metric-track" aria-hidden="true">{assessed && <span style={{ width: `${Math.max(0, Math.min(100, value))}%` }}/>}</div>
  </div>;
}

const VERDICT_LABELS = {
  true: 'TRUE',
  false: 'FALSE',
  unverified: 'UNVERIFIED',
};

function sourceLink(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'https:' ? parsed.href : null;
  } catch {
    return null;
  }
}

function ReportSummary({ summary, t }) {
  const verdict = VERDICT_LABELS[summary?.verdict] ? summary.verdict : 'unverified';
  const Icon = verdict === 'true' ? CircleCheck : verdict === 'false' ? AlertTriangle : CircleHelp;
  const tone = verdict === 'true' ? 'green' : verdict === 'false' ? 'coral' : 'amber';
  const evidence = Array.isArray(summary?.sources) ? summary.sources : [];
  const citationIds = Array.isArray(summary?.summary_source_ids) ? summary.summary_source_ids : [];
  const citedSources = citationIds.map(id => evidence.find(item => item.id === id)).filter(Boolean);
  const claims = Array.isArray(summary?.claims) ? summary.claims : [];
  const limitations = [...new Set(summary?.limitations || [])].slice(0, 6);
  return <section className={`report-summary summary-${verdict}`} aria-label="Concise evidence summary">
    <div className="summary-heading">
      <div>
        <span className="eyebrow"><Sparkles size={14}/> {summary?.gemini_analysis?.status === 'completed' ? 'Gemini evidence assessment' : 'Evidence-based assessment'}</span>
        <h4>{summary?.headline || 'The claim could not be verified'}</h4>
        {summary?.provider && <small className="summary-method">Summary method: {summary.provider}</small>}
      </div>
      <span className={`summary-verdict ${verdict}`}><IconBadge icon={Icon} tone={tone}/>{VERDICT_LABELS[verdict]}</span>
    </div>
    <p className="summary-body">{summary?.summary || 'A concise summary is unavailable for this saved report. Run a new analysis to create one.'}</p>
    {summary?.reason && <p className="summary-reason"><strong>Reason:</strong> {summary.reason}</p>}
    {citedSources.length > 0 && <div className="summary-citations" aria-label="Sources cited in the summary">
      <strong>Evidence cited:</strong>{' '}
      {citedSources.map((item, index) => {
        const url = sourceLink(item.url);
        return url ? <a key={item.id} href={url} target="_blank" rel="noopener noreferrer">
          [{item.id}] {item.publisher || item.title || `Source ${index + 1}`}
        </a> : <span key={item.id}>[{item.id}] {item.publisher || 'Source record'}</span>;
      })}
    </div>}
    <p className="summary-footnote">This report separates source evidence from model probabilities. Model probabilities are not proof of truth. {t.fullEvidenceInPdf || 'Full retrieval details and any available source links are in the downloaded PDF.'}</p>
    {(claims.length > 0 || evidence.length > 0 || limitations.length > 0) && <details className="report-evidence-details">
      <summary>View claim reasoning, evidence, and limitations</summary>
      {claims.map((claim, index) => <section className="claim-assessment" key={`${index}-${claim.text}`}>
        <div className="claim-assessment-heading"><strong>Claim {index + 1}</strong><span className={`claim-verdict ${claim.verdict || 'unverified'}`}>{VERDICT_LABELS[claim.verdict] || 'UNVERIFIED'}</span></div>
        <p className="claim-assessment-text">{claim.text}</p>
        <p>{claim.reason}</p>
        {claim.supporting_evidence?.length > 0 && <small>Supporting evidence: {claim.supporting_evidence.map(id => `[${id}]`).join(', ')}</small>}
        {claim.contradicting_evidence?.length > 0 && <small>Contradicting evidence: {claim.contradicting_evidence.map(id => `[${id}]`).join(', ')}</small>}
        {claim.missing_evidence?.length > 0 && <small>Still needed: {claim.missing_evidence.join(' ')}</small>}
      </section>)}
      {evidence.length > 0 && <div className="report-evidence-list"><h5>Retrieved source excerpts</h5>
        {evidence.map((item, index) => {
          const id = item.id || index + 1;
          const url = sourceLink(item.url);
          const support = summary.supporting_evidence?.includes(id);
          const contradict = summary.contradicting_evidence?.includes(id);
          return <article className="report-evidence-item" key={`${id}-${item.url || index}`}>
            <div><strong>[{id}] {item.title || 'Source record'}</strong>{(support || contradict) && <span className={`evidence-relation ${support ? 'supports' : 'contradicts'}`}>{support ? 'Supports verdict' : 'Contradicts verdict'}</span>}</div>
            <small>{item.publisher || 'Publisher not recorded'}{item.published_at ? ` · Published ${item.published_at}` : item.reviewed_at ? ` · Reviewed ${item.reviewed_at}` : item.seen_at ? ` · Seen ${item.seen_at}` : ''}</small>
            {item.rating && <small>Fact-check rating: {item.rating}</small>}
            {item.reliability && <small>Source class: {item.reliability.replaceAll('_', ' ')}</small>}
            {item.snippet && <p>{item.snippet}</p>}
            {item.relationship && <small>{item.relationship}</small>}
            {item.retrieval_status && item.retrieval_status === 'metadata_only' && <small>Article text was unavailable; this record is context only.</small>}
            {url && <a href={url} target="_blank" rel="noopener noreferrer">Open original source <ArrowRight size={13}/></a>}
          </article>;
        })}
      </div>}
      {summary.rag_analysis && <p className="retrieval-note"><strong>Retrieval:</strong> {summary.rag_analysis.method} {summary.rag_analysis.retrieved_count} relevant excerpt(s) returned across {summary.rag_analysis.claim_search_count || 0} claim search(es). {summary.rag_analysis.scope}</p>}
      {summary.gemini_analysis && <p className="retrieval-note"><strong>Gemini:</strong> {summary.gemini_analysis.role} Status: {summary.gemini_analysis.status}.
        {summary.gemini_diagnostics?.http_status ? ` HTTP ${summary.gemini_diagnostics.http_status}.` : ''}
        {summary.gemini_diagnostics?.finish_reason ? ` Finish reason: ${summary.gemini_diagnostics.finish_reason}.` : ''}
        {summary.gemini_diagnostics?.candidate_content_returned === true ? ' Candidate text was returned.' : ''}
        {summary.gemini_diagnostics?.error_category ? ` Category: ${summary.gemini_diagnostics.error_category}.` : ''}
        {summary.gemini_diagnostics?.validation_error ? ` Validation: ${summary.gemini_diagnostics.validation_error.replaceAll('_', ' ')}.` : ''}</p>}
      {limitations.length > 0 && <div className="report-limitations"><h5>Model limitations</h5><ul>{limitations.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}
      {summary.technical_warnings?.map((item, index) => <p className="report-warning" key={index}><AlertTriangle size={14}/>{item}</p>)}
    </details>}
  </section>;
}

function TextModelAssessment({ assessment, metricNote }) {
  if (!assessment && !metricNote) return null;
  const assessed = assessment?.status === 'assessed';
  const label = assessed
    ? (assessment.verdict === 'fake' ? 'Fake / manipulated signal' : 'Real / authentic signal')
    : 'Not assessed';
  const accuracy = assessment?.evaluation?.accuracy;
  const count = assessment?.evaluation?.test_count;
  return <section className={`text-model-assessment ${assessed ? 'assessed' : 'unassessed'}`} aria-label="Text classifier output">
    <div className="text-model-heading">
      <span className="eyebrow"><ScanLine size={14}/> Local text classifier</span>
      <strong>{label}</strong>
    </div>
    <p>{assessment?.reason || metricNote || 'The text classifier did not return a prediction for this report.'}</p>
    {assessed && <p className="text-model-score">Fake/manipulated score: {assessment.fake_percentage.toFixed(1)}% · Model confidence: {assessment.confidence.toFixed(1)}%</p>}
    {assessment?.model && <small>Model: {assessment.model}{assessment.version ? ` · ${assessment.version}` : ''}</small>}
    {Number.isFinite(accuracy) && <small>Held-out dataset accuracy: {(accuracy * 100).toFixed(1)}%{Number.isFinite(count) ? ` on ${count} examples` : ''}. This measures that dataset, not fact-checking accuracy.</small>}
  </section>;
}

function GeminiModelAssessment({ assessment }) {
  if (!assessment || assessment.status === 'not_available') return null;
  const answered = assessment.status === 'completed' && assessment.answer;
  const verdict = VERDICT_LABELS[assessment.verdict] ? assessment.verdict : 'unverified';
  const Icon = verdict === 'true' ? CircleCheck : verdict === 'false' ? AlertTriangle : CircleHelp;
  const tone = verdict === 'true' ? 'green' : verdict === 'false' ? 'coral' : 'amber';
  return <section className="gemini-model-assessment" aria-label="Gemini model-only response">
    <div className="gemini-model-heading">
      <span className="eyebrow"><Sparkles size={14}/> Gemini model-only response</span>
      <span className={`summary-verdict ${verdict}`}><IconBadge icon={Icon} tone={tone}/>{answered ? `MODEL: ${VERDICT_LABELS[verdict]}` : 'NO ANSWER'}</span>
    </div>
    <small className="gemini-model-label">Model knowledge only · not source verified</small>
    {answered && <p className="gemini-model-answer">{assessment.answer}</p>}
    <p>{assessment.reason || 'Gemini did not return a model-only response for this report.'}</p>
  </section>;
}

function MediaInspection({ inspection, origin }) {
  const visualLabels = {
    photograph: 'Photograph',
    cartoon_or_illustration: 'Cartoon or illustration',
    screenshot_or_document: 'Screenshot or document',
    other: 'Other visual content',
    unclear: 'Visual type not identified',
  };
  const verdictLabels = {
    ai_generated: 'Likely AI-generated',
    likely_real: 'Likely real / not flagged as AI-generated',
    classification_unavailable: 'Could not classify image origin',
    likely_not_ai_generated: 'Likely real / not flagged as AI-generated',
    not_assessed: 'Could not classify image origin',
  };
  return <section className="media-inspection-card" aria-label="Image and video inspection">
    <div className="media-inspection-icon"><ImageIcon size={19}/></div>
    <div className="media-inspection-copy">
      <div className="media-inspection-heading">
        <strong>{verdictLabels[origin?.verdict] || visualLabels[inspection?.visual_type] || 'Media analyzed'}</strong>
        <span>{visualLabels[inspection?.visual_type] || 'Visual type not identified'}</span>
      </div>
      <p>{inspection?.description || (inspection?.status === 'completed'
        ? 'The uploaded media was inspected for visual content.'
        : 'OCR and scene description were unavailable; the AI-origin model result is shown below if available.')}</p>
      {inspection?.visible_text && <div className="media-visible-text">
        <strong>Text read from the media (OCR)</strong><p>{inspection.visible_text}</p>
      </div>}
      {inspection?.visual_artifacts && <div className="media-visible-text">
        <strong>Visible details to review</strong><p>{inspection.visual_artifacts}</p>
      </div>}
      {typeof origin?.score === 'number' ? <small>Experimental AI-origin estimate: {origin.score.toFixed(1)}%. This is a model score, not measured accuracy or proof.</small>
        : <small>The AI-origin classifier did not return a score for this upload.</small>}
    </div>
  </section>;
}

function MediaOnlySummary({ summary }) {
  return <section className="media-only-summary" aria-label="Media-only result">
    <span className="eyebrow"><Sparkles size={14}/> IMAGE OR VIDEO CLASSIFICATION</span>
    <h4>{summary?.headline || 'Media analyzed'}</h4>
    <p>{summary?.summary || 'The image or video was inspected for visible text, scene content, and AI-origin signals.'}</p>
  </section>;
}

function savedReportSummary(result, refreshing) {
  const verification = result?.verification || {};
  const evidence = Array.isArray(verification.evidence) ? verification.evidence : [];
  const count = evidence.length;
  return {
    schema_version: 4,
    verdict: 'unverified',
    headline: refreshing ? 'Comparing the claim with available sources' : 'No Gemini assessment was saved for this report',
    summary: refreshing
      ? 'The claim and retrieved source excerpts are being compared.'
      : count
        ? `This saved report contains ${count} source excerpt${count === 1 ? '' : 's'}, but it predates the current Gemini claim comparison. The claim remains UNVERIFIED until the sources are assessed; review the excerpts below.`
        : 'No source excerpts were saved with this report, so its stored information cannot settle whether the claim is true or false. The result remains UNVERIFIED.',
    reason: refreshing
      ? 'The report is being updated from the saved claim and evidence.'
      : count
        ? 'The saved report has source excerpts but no current Gemini verdict tied to them.'
        : 'There are no saved source excerpts to support a TRUE or FALSE verdict.',
    provider: count ? 'Saved source excerpts' : undefined,
    sources: evidence.filter(item => item && typeof item === 'object').map((item, index) => ({ ...item, id: index + 1 })),
    source_ids: [], summary_source_ids: [], supporting_evidence: [], contradicting_evidence: [],
    claims: [], key_points: [],
    limitations: ['This report has no saved Gemini comparison. The excerpts and local classifier result are shown separately.'],
    rag_analysis: { provider: verification.provider || 'Not recorded', status: verification.provider_status || verification.status || 'unknown',
      retrieved_count: count, method: 'Saved evidence records', scope: 'Only evidence already saved with this report is shown.' },
    gemini_analysis: { status: refreshing ? 'pending' : 'not_saved', role: 'No Gemini assessment is stored for this saved report.' },
  };
}

export default function Report({ report, t, lang, onNew }) {
  const [displayReport, setDisplayReport] = useState(report);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    setDisplayReport(report);
    if (!report?.id || report.status !== 'complete' || report.result?.report_summary?.schema_version >= 4) {
      setRefreshing(false);
      return undefined;
    }
    let active = true;
    setRefreshing(true);
    api(`/reports/${report.id}`).then(updated => {
      if (active) setDisplayReport(updated);
    }).catch(() => {
      // Keep the saved report visible and show a useful fallback below.
    }).finally(() => {
      if (active) setRefreshing(false);
    });
    return () => { active = false; };
  }, [report?.id, report?.status, report?.result?.report_summary]);

  if (!displayReport) return <div className="empty-result">
    <div className="empty-orbit"><Search size={45} strokeWidth={1.2}/><span><Sparkles size={18}/></span></div>
    <div className="eyebrow">{t.result}</div><h3>{t.ready}</h3><p>{t.readyBody}</p>
    <div className="empty-steps"><FileText/><span>›</span><ScanLine/><span>›</span><ShieldCheck/></div>
  </div>;
  if (['queued', 'analyzing'].includes(displayReport.status)) return <div className="empty-result loading-result" role="status" aria-live="polite">
    <div className="analysis-loader"><ScanLine size={45}/><i/></div><div className="eyebrow">{t.result}</div>
    <h3>{displayReport.status === 'queued' ? t.queued : t.processing}</h3><p>{t.pendingBody}</p>
    <div className="loader-track"><span/></div><div className="small muted">{t.reportIdLabel} · {report.id.slice(0, 10)}</div>
  </div>;
  if (displayReport.status === 'failed') return <div className="empty-result">
    <IconBadge icon={AlertTriangle} tone="amber"/><h3>{t.failed}</h3><p>{analysisErrorMessage(displayReport.error)}</p>
    <button className="button primary" onClick={onNew}>{t.retry}<ArrowRight size={18}/></button>
  </div>;

  const result = displayReport.result || {};
  const summary = result.report_summary?.schema_version >= 4
    ? result.report_summary
    : savedReportSummary(result, refreshing);
  const mediaOnly = summary.mode === 'media_only';
  const isMedia = displayReport.type === 'image' || displayReport.type === 'video';
  return <article className="report-card concise-report" aria-live="polite">
    <div className="report-heading"><span className="eyebrow">{t.result}</span><span className="saved-note"><CircleCheck size={13}/>{t.saved}</span></div>
    <div className="report-title-block">
      <p className="report-title">{displayReport.title}</p>
      <span className="report-id">#{displayReport.id.slice(0, 6)}</span>
    </div>
    {isMedia && <MediaInspection inspection={result.media_inspection} origin={result.media_origin}/>}
    {mediaOnly ? <MediaOnlySummary summary={summary}/> : <ReportSummary summary={summary} t={t}/>}
    {displayReport.type === 'text' && <GeminiModelAssessment assessment={summary.gemini_model_assessment}/>}
    {displayReport.type === 'text' && <TextModelAssessment assessment={result.text_model_assessment} metricNote={result.metric_note}/>}
    {isMedia ? <><div className="metric-grid media-metric-grid">
      <Metric title="AI-generated estimate" value={result.fake_percentage} icon={ScanLine} tone="coral" t={t}/>
      <Metric title="Likely real estimate" value={result.real_percentage} icon={ShieldCheck} tone="green" t={t}/>
    </div><p className="media-metric-note">{result.metric_note}</p></> : <div className="metric-grid">
      <Metric title={t.fake} value={result.fake_percentage} icon={ScanLine} tone="coral" t={t}/>
      <Metric title={t.authentic} value={result.real_percentage} icon={ShieldCheck} tone="green" t={t}/>
      <Metric title={t.ai} value={result.ai_percentage} icon={Sparkles} tone="violet" t={t}/>
      <Metric title={t.confidence} value={result.confidence} icon={Fingerprint} tone="blue" t={t}/>
    </div>}
    <div className="report-bottom">
      <a className="button primary" href={`/api/reports/${displayReport.id}/pdf`} download={`verification-report-${displayReport.id.slice(0,10)}.pdf`}>
        <Download size={17}/>{t.download}
      </a>
      <span>{date(displayReport.created_at, lang)}</span>
    </div>
  </article>;
}
