import React, { useEffect, useRef, useState } from 'react';
import { ShieldCheck, ArrowUpRight, ArrowRight, ArrowLeft, FileText, Image as ImageIcon, Video, Search, ScanLine, Check, X, UploadCloud, LockKeyhole, Globe2, Volume2, History, LogOut, Menu, ChevronRight, ChevronDown, CircleHelp, Sparkles, Fingerprint, Link2, Download, LoaderCircle, AlertTriangle, ExternalLink, Trash2, LayoutDashboard, Users, Database, Settings2, CircleCheck, Plus, RefreshCw, Play, Eye, BookOpen, Pause, Leaf } from 'lucide-react';
import { api } from '../lib/api.js';
import { types, date, verificationMessage, scopeMessage, limitMessage, observationLabel } from '../lib/common.js';
import { Brand, IconBadge } from './ui.jsx';

function Illustration({slide=0, t}) {
 return <div className={`illustration illustration-${slide}`} aria-hidden="true">
  <div className="orbit orbit-one"/><div className="orbit orbit-two"/>
  <span className="art-spark art-spark-a">✳</span><span className="art-spark art-spark-b">✦</span>
  <div className="floating-label art-label-top"><span className="tiny-dot"/>{['Look a little closer','Visual signals','Frame by frame','Follow the source','See the full story'][slide]}</div>
  <div className="news-illustration">
   <div className="art-browser"><div><i/><i/><i/></div><LockKeyhole size={11}/><span>THE DAILY PERSPECTIVE</span></div>
   <div className="art-paper">
    <div className="paper-kicker"><span>THE BIG PICTURE</span><span>01 / 05</span></div>
    <div className="paper-heading">There’s more<br/>to every story.</div>
    <div className={`landscape ${slide===2?'video-landscape':''}`}><div className="landscape-sun"/><div className="mountain back"/><div className="mountain front"/><div className="landscape-grid"/>{slide===2?<span className="play-art"><Play fill="currentColor" size={25}/></span>:<div className="scan-brackets"><span/><span/><span/><span/></div>}<div className="scanning-line"/></div>
    <div className="paper-lines"><i/><i/><i/></div>
   </div>
  </div>
  <div className="magnifier"><Search size={58} strokeWidth={1}/><span><Eye size={25}/></span></div>
  <div className="floating-proof"><span className="proof-icon"><ShieldCheck size={25}/></span><div><strong>Evidence matters.</strong><span>Check the source. Find the context.</span></div><ArrowUpRight size={17}/></div>
  <div className="floating-pixels"><div><i/><i/><i/><i/><i/></div><span>AI SIGNALS + HUMAN CONTEXT</span></div>
  <div className="art-caption">{t.carouselNote}</div>
 </div>
}

export default function Home({t, start, user}) {
 const [slide,setSlide]=useState(0), [manualPaused,setManualPaused]=useState(false),[hovered,setHovered]=useState(false),[focused,setFocused]=useState(false); const paused=manualPaused||hovered||focused;
 const slides=[['heroA','heroB','heroBody'],['slideText',null,'slideTextBody'],['slideImage',null,'slideImageBody'],['slideVideo',null,'slideVideoBody'],['slideAI',null,'slideAIBody'],['slideEvidence',null,'slideEvidenceBody'],['slideResults',null,'slideResultsBody']];
 const examples=['text','image','video'];
 useEffect(()=>{if(paused || window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;const id=setInterval(()=>setSlide(s=>(s+1)%slides.length),7500);return()=>clearInterval(id)},[paused]);
 return <>
  <section className="hero wrap" aria-roledescription="carousel" aria-label={t.carousel} onMouseEnter={()=>setHovered(true)} onMouseLeave={()=>setHovered(false)} onFocusCapture={()=>setFocused(true)} onBlurCapture={e=>{if(!e.currentTarget.contains(e.relatedTarget))setFocused(false)}}>
   <div className="hero-content"><div className="eyebrow"><span className="tiny-dot"/>{t.eyebrow}</div><h1>{t.heroA}<br/><em>{t.heroB}</em></h1><p className="hero-body">{t.heroBody}</p><div className="hero-actions"><button className="button primary" onClick={()=>start('text')}>{t.start}<ArrowUpRight size={19}/></button><a className="text-link" href="#how" onClick={e=>{e.preventDefault();document.getElementById('how-it-works').scrollIntoView({behavior:'smooth'})}}><span className="play-circle"><Play size={11} fill="currentColor"/></span>{t.how}</a></div><div className="hero-assurances"><span><LockKeyhole size={14}/>{t.private}</span><span><Globe2 size={14}/>{t.languages}</span></div></div>
   <Illustration slide={[0,0,1,2,4,3,4][slide]} t={t}/>
   <div className="carousel-controls"><span className="carousel-title">{t.explore}</span><div className="carousel-dots">{slides.map((_,i)=><button key={i} className={slide===i?'active':''} onClick={()=>setSlide(i)} aria-label={`${t.carousel} ${i+1}`} aria-current={slide===i?'step':undefined}/>)}</div><span className="slide-count">0{slide+1}<span> / 0{slides.length}</span></span><button className="round-button" aria-label={manualPaused?t.resume:t.pause} onClick={()=>setManualPaused(p=>!p)}>{manualPaused?<Play size={15}/>:<Pause size={15}/>}</button><button className="round-button" aria-label={t.prev} onClick={()=>setSlide((slide+slides.length-1)%slides.length)}><ArrowLeft size={16}/></button><button className="round-button" aria-label={t.next} onClick={()=>setSlide((slide+1)%slides.length)}><ArrowRight size={16}/></button></div>
  </section>
  <div className="trust-ribbon"><div className="wrap ribbon-inner"><span><ShieldCheck size={17}/>{t.assessment}</span><span>TEXT <i>✳</i> IMAGE <i>✳</i> VIDEO</span><span>{t.responsible}<ArrowUpRight size={15}/></span></div></div>
  {user&&<section className="wrap account-shortcuts" aria-label={t.accountOverview}><div className="account-summary"><span className="avatar">{user.name.slice(0,1).toUpperCase()}</span><div><span className="eyebrow">{t.signedIn}</span><strong>{user.name}</strong><small>{user.email}</small></div></div><div className="account-actions"><button className="button primary" onClick={()=>start('text')}><ScanLine size={17}/>{t.verify}</button><a className="button subtle" href="#history"><History size={17}/>{t.reportsAndPdfs}<ArrowRight size={15}/></a></div></section>}
  <section className="wrap section toolkit"><div className="section-heading"><div><div className="eyebrow">{t.toolkit}</div><h2>{t.try}<span className="serif-dot">.</span></h2></div><p>{t.verifyBody}</p></div><div className="tool-grid">{Object.entries(types).map(([key,Icon],i)=><button key={key} className={`tool-card tool-${key}`} onClick={()=>start(key)}><div className="tool-top"><IconBadge icon={Icon}/><span className="tool-no">0{i+1}</span></div><h3>{t[key]}</h3><p>{t[`${key}Desc`]}</p><div className="tool-art">{key==='text'?<><span>Aa<span>“</span></span><div className="text-art-lines"><i/><i/><i/></div></>:key==='image'?<div className="mini-photo"><div className="mini-sun"/><div className="mini-mountain"/><ScanLine size={59}/></div>:<div className="mini-video"><div/><span><Play size={26} fill="currentColor"/></span><div/></div>}</div><span className="tool-action">{t.verify}<ArrowUpRight size={19}/></span></button>)}</div></section>
  <section className="wrap section examples"><div className="section-heading"><div><div className="eyebrow">{t.examplesEyebrow}</div><h2>{t.examplesTitle}</h2></div><p>{t.examplesBody}</p></div><div className="example-grid">{examples.map((key,i)=>{const Icon=types[key],prefix=`example${key[0].toUpperCase()}${key.slice(1)}`;return <details className={`example-card example-${key}`} key={key}><summary><span className="example-card-top"><span className="example-icon"><Icon size={19}/></span><span className="example-label">{t.exampleLabel}</span><span className="example-number">0{i+1}</span></span><h3>{t[`${prefix}Title`]}</h3><p className="example-sample">{t[`${prefix}Sample`]}</p><span className="example-toggle">{t.exampleReveal}<ChevronDown size={17}/></span></summary><div className="example-checks"><p><Check size={16}/>{t[`${prefix}Check1`]}</p><p><Check size={16}/>{t[`${prefix}Check2`]}</p><button type="button" className="button subtle example-action" onClick={()=>start(key)}>{t.exampleCTA}<ArrowRight size={16}/></button><small>{t.exampleDisclaimer}</small></div></details>})}</div></section>
  <section id="how-it-works" className="how-section"><div className="wrap section"><div className="section-heading"><div><div className="eyebrow">01 → 02 → 03</div><h2>{t.built}</h2></div><p>{t.builtBody}</p></div><div className="steps">{[[UploadCloud,'step1'],[ScanLine,'step2'],[BookOpen,'step3']].map(([Icon,key],i)=><article className="step" key={key}><div className="step-graphic"><Icon size={37} strokeWidth={1.2}/><span>0{i+1}</span></div><h3>{t[key]}</h3><p>{t[`${key}Body`]}</p></article>)}</div></div></section>
  <section id="about-project" className="wrap section about-section"><div className="promise"><div className="promise-art"><Fingerprint strokeWidth={.7}/><ShieldCheck size={34}/></div><div><span className="eyebrow">{t.aboutEyebrow}</span><h2>{t.promise}</h2><p>{t.promiseBody}</p><div className="about-features">{[t.text,t.image,t.video,t.evidence,t.history].map((label,i)=><span key={i}>{label}</span>)}</div></div><button className="round-button" aria-label={t.sources} onClick={()=>location.hash='sources'}><ArrowUpRight/></button></div></section>
 </>
}
