import { test, expect } from '@playwright/test';

test('live keyless fact-check sources appear in the saved report @live',async({page})=>{
 test.skip(process.env.VERITAS_LIVE_EVIDENCE!=='1','Enable explicitly with the isolated public-provider QA server.');
 await page.goto('/#verify');
 await page.getByRole('button',{name:'Sign in',exact:true}).click();
 const dialog=page.getByRole('dialog');
 await dialog.getByLabel('Email address').fill('qa-admin@example.invalid');
 await dialog.getByLabel('Password',{exact:true}).fill('local-browser-qa-only-password');
 await dialog.getByRole('button',{name:'Sign in',exact:true}).click();
 await expect(dialog).not.toBeVisible();
 await page.getByLabel('Paste a claim or news article').fill('NASA moon landing is fake.');
 await page.getByRole('button',{name:'Analyze content',exact:true}).click();
 await expect(page.getByRole('heading',{name:'Review needed',exact:true})).toBeVisible({timeout:60000});
 await expect(page.locator('.evidence-item').first()).toBeVisible();
 await expect(page.getByRole('link',{name:'Read original fact check',exact:true}).first()).toBeVisible();
 await expect(page.locator('.evidence-providers')).toContainText('Searched');
 await expect(page.locator('.report-section').filter({hasText:'A search match is not automatic verification.'})).toBeVisible();
 await page.screenshot({path:'../artifacts/live-free-verification.png',fullPage:true});
 const event=page.waitForEvent('download');
 await page.getByRole('link',{name:'Download PDF',exact:true}).click();
 await (await event).saveAs('../artifacts/live-free-verification.pdf');
});
