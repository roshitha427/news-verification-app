import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ShieldCheck, ArrowUpRight, ArrowRight, ArrowLeft, FileText, Image as ImageIcon, Video, Search, ScanLine, Check, X, UploadCloud, LockKeyhole, Globe2, Volume2, History, LogOut, Menu, ChevronRight, ChevronDown, CircleHelp, Sparkles, Fingerprint, Link2, Download, LoaderCircle, AlertTriangle, ExternalLink, Trash2, LayoutDashboard, Users, Database, Settings2, CircleCheck, Plus, RefreshCw, Play, Eye, BookOpen, Pause, Leaf } from 'lucide-react';
import { translations } from './i18n';
import '@fontsource/dm-sans/latin-400.css';
import '@fontsource/dm-sans/latin-500.css';
import '@fontsource/dm-sans/latin-600.css';
import '@fontsource/dm-sans/latin-700.css';
import '@fontsource/manrope/latin-400.css';
import '@fontsource/manrope/latin-600.css';
import '@fontsource/manrope/latin-700.css';
import '@fontsource/manrope/latin-800.css';
import './styles.css';
import './examples.css';

import { api } from './lib/api.js';
import { verificationMessage } from './lib/common.js';
import { Brand, IconBadge } from './components/ui.jsx';
import Home from './components/Home.jsx';
import Report from './components/Report.jsx';
import Workspace from './components/Workspace.jsx';
import Reports from './components/HistoryPage.jsx';
import Sources from './components/Sources.jsx';
import Admin from './components/Admin.jsx';
import AuthModal from './components/AuthModal.jsx';
import Profile from './components/Profile.jsx';
import ReferenceApp from './ReferenceApp.jsx';
import './reference.css';

function resetTokenFromLocation(){const match=location.hash.match(/^#reset(?:\/|=)([A-Za-z0-9_-]{20,})$/);return match?.[1]||''}

function SignInPrompt({t,onLogin}){return <section className="wrap section app-section access-section"><div className="empty-state"><IconBadge icon={LockKeyhole}/><h3>{t.accountRequired}</h3><p>{t.signInFirst}</p><button className="button primary" onClick={onLogin}>{t.login}<ArrowRight size={16}/></button></div></section>}

function App(){
 const [lang,setLang]=useState(localStorage.getItem('news-verification-language')||'en'),[page,setPage]=useState(()=>resetTokenFromLocation()?'home':location.hash.slice(1)||'home'),[user,setUser]=useState(null),[health,setHealth]=useState(null),[auth,setAuth]=useState(false),[resetToken,setResetToken]=useState(()=>resetTokenFromLocation()),[mobile,setMobile]=useState(false),[toast,setToast]=useState(''),[kind,setKind]=useState('text'),[report,setReport]=useState(null),[speaking,setSpeaking]=useState(false),[pendingStart,setPendingStart]=useState('');
 const t=translations[lang]||translations.en, toastTimer=useRef(null);
 function notify(message){setToast(message);clearTimeout(toastTimer.current);toastTimer.current=setTimeout(()=>setToast(''),6500)}
 useEffect(()=>{document.title='AI-Powered Fake News Detection and Verification System'},[]);
 useEffect(()=>{api('/session').then(d=>setUser(d.user)).catch(()=>notify(t.offline));api('/health').then(setHealth).catch(()=>setHealth({status:'offline'}));const change=()=>{const token=resetTokenFromLocation();setResetToken(token);setAuth(false);setPage(token?'home':location.hash.slice(1)||'home');setMobile(false);window.scrollTo({top:0,behavior:'instant'})};window.addEventListener('hashchange',change);return()=>window.removeEventListener('hashchange',change)},[]);
 useEffect(()=>{localStorage.setItem('news-verification-language',lang);document.documentElement.lang=lang;if('speechSynthesis'in window)window.speechSynthesis.cancel();setSpeaking(false)},[lang]);
 useEffect(()=>{if(!report?.id||!['queued','analyzing'].includes(report.status))return;let stopped=false,id;async function poll(){try{const r=await api(`/reports/${report.id}`);if(!stopped){setReport(r);if(['queued','analyzing'].includes(r.status))id=setTimeout(poll,1400)}}catch(e){if(!stopped){notify(e.message);id=setTimeout(poll,5000)}}}id=setTimeout(poll,700);return()=>{stopped=true;clearTimeout(id)}},[report?.id,report?.status]);
 function start(type){setKind(type);setReport(null);if(!user){setPendingStart(type);setAuth(true);return}location.hash='verify'}
 function afterLogin(account){setUser(account);if(pendingStart){setKind(pendingStart);setPendingStart('');location.hash='verify'}}
 async function open(id){try{const r=await api(`/reports/${id}`);setReport(r);setKind(r.type);location.hash='verify'}catch(e){notify(e.message)}}
 async function logout(){try{await api('/auth/logout',{method:'POST'});const d=await api('/session');setUser(null);setReport(null);location.hash='home'}catch(e){notify(e.message)}}
 function closeAuth(){setAuth(false);setPendingStart('');if(resetToken){history.replaceState(null,'',`${location.pathname}${location.search}#home`);setResetToken('')}}
 function completePasswordReset(){history.replaceState(null,'',`${location.pathname}${location.search}#home`);setResetToken('');setAuth(false)}
 function goToSection(id){setMobile(false);if(page!=='home'){setPage('home');history.replaceState(null,'',`${location.pathname}${location.search}#home`)}requestAnimationFrame(()=>requestAnimationFrame(()=>document.getElementById(id)?.scrollIntoView({behavior:'smooth',block:'start'})))}
 function read(){if(!('speechSynthesis'in window)){notify('Read aloud is not supported by this browser.');return}if(speaking){window.speechSynthesis.cancel();setSpeaking(false);return}const content=page==='home'?`${t.heroA} ${t.heroB} ${t.heroBody} ${t.step1}. ${t.step1Body} ${t.step2}. ${t.step2Body} ${t.step3}. ${t.step3Body}`:page==='verify'?`${t.verifyTitle} ${t.verifyBody} ${t.learnBody} ${report?.result?`${t[report.result.verification.status]} ${verificationMessage(report.result.verification,t)}`:''}`:`${t[page]} ${t.sourceGuideBody}`;const speech=new SpeechSynthesisUtterance(content);speech.lang={en:'en-IN',hi:'hi-IN',te:'te-IN'}[lang];speech.rate=.9;speech.onend=()=>setSpeaking(false);speech.onerror=()=>{setSpeaking(false);notify('A voice for this language may not be installed on your device.')};window.speechSynthesis.speak(speech);setSpeaking(true)}
 const nav=[['home'],['how','how-it-works'],['about','about-project'],['sources']];
 const accountNav=[['home',LayoutDashboard],['verify',ScanLine],['history',History],['profile',Fingerprint],...(user?.role==='admin'?[['admin',Settings2]]:[])];
 return <><a className="skip-link" href="#main">Skip to content</a><header className="site-header"><div className="wrap nav"><Brand/><nav className={mobile?'navigation open':'navigation'} aria-label="Main">{nav.map(([key,section])=><a key={key} className={!section&&page===key?'active':''} href={section?`#${section}`:`#${key}`} aria-current={!section&&page===key?'page':undefined} onClick={section?e=>{e.preventDefault();goToSection(section)}:()=>setMobile(false)}>{section==='how-it-works'?t.howNav:section==='about-project'?t.aboutNav:t[key]}</a>)}</nav><div className="nav-actions"><label className="language-select"><Globe2 size={16}/><select value={lang} onChange={e=>setLang(e.target.value)} aria-label={t.language}><option value="en">English</option><option value="hi">हिन्दी</option><option value="te">తెలుగు</option></select></label><button className={`listen-button ${speaking?'speaking':''}`} onClick={read} aria-label={speaking?t.stop:t.listen} title={t.listen}><Volume2 size={18}/></button>{user?<div className="account-menu"><span className="avatar" title={`${user.name}${user.email?` · ${user.email}`:''}`}>{user.name.slice(0,1).toUpperCase()}</span><a className="account-profile" href="#profile" aria-label={t.profile}>{user.name}</a><button className="round-button" onClick={logout} aria-label={t.logout}><LogOut size={16}/></button></div>:<button className="button nav-login" onClick={()=>setAuth(true)}>{t.login}<ArrowUpRight size={15}/></button>}<button className="mobile-toggle round-button" onClick={()=>setMobile(m=>!m)} aria-label="Menu" aria-expanded={mobile}>{mobile?<X size={22}/>:<Menu size={22}/>}</button></div></div></header>
 {health?.status==='offline'&&<div className="connection-notice"><AlertTriangle size={15}/>{t.offline}</div>}
 <main id="main" className={user?'app-shell':''}>{user&&<nav className="account-sidebar" aria-label={t.accountOverview}><span className="account-sidebar-title">{t.accountOverview}</span>{accountNav.map(([key,Icon])=><a key={key} href={`#${key}`} className={page===key?'active':''} aria-current={page===key?'page':undefined}><Icon size={18}/><span>{key==='history'?t.reportsAndPdfs:t[key]}</span></a>)}</nav>}<div className={user?'app-main':''}>{page==='home'?<Home t={t} start={start} user={user}/>:page==='verify'?(user?<Workspace t={t} lang={lang} user={user} onLogin={()=>setAuth(true)} kind={kind} setKind={setKind} report={report} setReport={setReport} notify={notify}/>:<SignInPrompt t={t} onLogin={()=>setAuth(true)}/>):page==='history'?(user?<Reports t={t} lang={lang} user={user} onLogin={()=>setAuth(true)} onOpen={open} notify={notify}/>:<SignInPrompt t={t} onLogin={()=>setAuth(true)}/>):page==='profile'?(user?<Profile t={t} user={user} onUpdate={setUser}/>:<SignInPrompt t={t} onLogin={()=>setAuth(true)}/>):page==='sources'?<Sources t={t} notify={notify}/>:page==='admin'?<Admin t={t} user={user} notify={notify}/>:<Home t={t} start={start} user={user}/>}</div></main>
 <footer className="site-footer"><div className="wrap footer-main"><div className="footer-intro"><Brand small/><p>{t.footer}</p></div><nav className="footer-links" aria-label={t.footerLinks}>{[['how-it-works',t.howNav],['about-project',t.aboutNav]].map(([id,label])=><a key={id} href={`#${id}`} onClick={e=>{e.preventDefault();goToSection(id)}}>{label}</a>)}<a href="#sources" onClick={()=>setMobile(false)}>{t.sources}</a></nav></div><div className="wrap footer-bottom"><span>© {new Date().getFullYear()}</span></div></footer>
  {(auth||resetToken)&&<AuthModal key={resetToken?'reset-'+resetToken:'auth'} t={t} onClose={closeAuth} onSuccess={afterLogin} notify={notify} resetToken={resetToken} onResetComplete={completePasswordReset}/>}<div className="toast-zone" aria-live="polite">{toast&&<div className="toast"><CircleHelp size={19}/><span>{toast}</span><button aria-label={t.close} onClick={()=>setToast('')}><X size={16}/></button></div>}</div></>
}
createRoot(document.getElementById('root')).render(<ReferenceApp/>);
