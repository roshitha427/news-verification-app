import { test, expect } from '@playwright/test';

for (const sample of [
  { name: 'short English text', type: 'text', score: 31, confidence: 69, words: 8 },
  { name: 'English blog article', type: 'text', score: 18, confidence: 82, words: 140, ai: 84 },
  { name: 'unsupported-language text', type: 'text', score: null, confidence: null, words: 6 },
  { name: 'image', type: 'image', score: 20, confidence: 80 },
  { name: 'image with an unvalidated high AI score', type: 'image', score: 99.9, confidence: 99.9 },
  { name: 'video', type: 'video', score: 74, confidence: 74 },
]) {
  test(`${sample.name} shows clearly named metrics without an authenticity class`, async ({ page }) => {
    const report = {
      id: 'metrics-fixture-' + sample.type, type: sample.type, status: 'complete',
      title: sample.name, content: 'A test claim for checking report metrics.',
      created_at: '2026-10-20T03:00:00Z',
      result: {
        fake_percentage: sample.score, real_percentage: sample.score === null ? null : 100 - sample.score,
        confidence: sample.confidence, ai_percentage: sample.type === 'text' ? (sample.ai ?? null) : sample.score,
        text_metrics: { word_count: sample.words, source_count: 0 },
        media_origin: sample.type === 'text' ? undefined : {
          verdict: 'inconclusive', score: sample.score,
          reason: 'The detector has not been locally validated for origin classification.',
        },
        text_model_assessment: sample.type === 'text' ? {
          status: sample.score === null ? 'unsupported_language' : 'assessed',
          experimental_input: true, fake_percentage: sample.score, confidence: sample.confidence,
          reason: sample.score === null ? 'This language is not supported by the classifier.' : 'Short-text scores are experimental, not a fact-check.',
        } : undefined,
        verification: { evidence: [], status: 'unverified' },
        report_summary: { schema_version: 5, verdict: 'unverified', headline: 'Review the evidence',
          mode: sample.type === 'text' ? undefined : 'media_only', summary: 'A report metric test fixture.' },
      },
    };
    await page.route('**/api/reports**', route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/reports') return route.fulfill({ json: { items: [report] } });
      if (pathname === '/api/reports/' + report.id) return route.fulfill({ json: report });
      return route.continue();
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Sign in', exact: true }).first().click();
    const dialog = page.getByRole('dialog');
    await dialog.getByLabel('Email address').fill('qa-admin@example.invalid');
    await dialog.getByLabel('Password', { exact: true }).fill('local-browser-qa-only-password');
    await dialog.getByRole('button', { name: 'Sign in', exact: true }).click();
    await expect(dialog).not.toBeVisible();
    await page.goto('/#history');
    await page.getByRole('button', { name: 'Open report', exact: true }).click();
    const metrics = page.locator('.report-card .metric-grid');
    await expect(metrics).toBeVisible();
    await expect(metrics).not.toContainText(/authentic|likely real/i);
    const confidence = metrics.locator('.metric').filter({ hasText: 'Model confidence' });
    if (sample.confidence === null) await expect(confidence).toContainText('Not assessed');
    else await expect(confidence.locator('strong')).toHaveText(sample.confidence.toFixed(1) + '%');
    if (sample.type === 'text') {
      await expect(metrics).toContainText('Fake-news pattern score');
      const real = metrics.locator('.metric').filter({ hasText: 'Real-news pattern score' });
      if (sample.score === null) await expect(real).toContainText('Not assessed');
      else await expect(real.locator('strong')).toHaveText((100 - sample.score).toFixed(1) + '%');
      const ai = metrics.locator('.metric').filter({ hasText: 'AI-writing likelihood' });
      if (sample.ai === undefined) await expect(ai).toContainText('Not assessed');
      else await expect(ai.locator('strong')).toHaveText(sample.ai.toFixed(1) + '%');
      await expect(page.locator('.report-card')).toContainText('not the percentage of words written by AI');
      const words = metrics.locator('.metric').filter({ hasText: 'Words analyzed' });
      await expect(words.locator('strong')).toHaveText(String(sample.words));
      await expect(metrics.locator('.metric').filter({ hasText: 'Source excerpts' }).locator('strong')).toHaveText('0');
    } else {
      await expect(metrics).toContainText('AI-generated estimate');
      await expect(page.locator('.media-inspection-heading')).toContainText('Origin inconclusive');
      await expect(page.locator('.report-card')).not.toContainText('Likely AI-generated');
    }
  });
}
