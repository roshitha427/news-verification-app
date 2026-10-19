import { defineConfig } from '@playwright/test';
export default defineConfig({
 testDir:'./e2e', timeout:90000, workers:1, fullyParallel:false,
 outputDir:'../artifacts/browser-results', reporter:'list',
 use:{baseURL:'http://127.0.0.1:5001',headless:true,launchOptions:{executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'},screenshot:'only-on-failure',trace:'retain-on-failure'},
});
