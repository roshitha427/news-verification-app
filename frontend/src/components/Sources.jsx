import React, { useEffect, useRef, useState } from 'react';
import { ShieldCheck, ArrowUpRight, ArrowRight, ArrowLeft, FileText, Image as ImageIcon, Video, Search, ScanLine, Check, X, UploadCloud, LockKeyhole, Globe2, Volume2, History, LogOut, Menu, ChevronRight, ChevronDown, CircleHelp, Sparkles, Fingerprint, Link2, Download, LoaderCircle, AlertTriangle, ExternalLink, Trash2, LayoutDashboard, Users, Database, Settings2, CircleCheck, Plus, RefreshCw, Play, Eye, BookOpen, Pause, Leaf } from 'lucide-react';
import { api } from '../lib/api.js';
import { types, date, verificationMessage, scopeMessage, limitMessage, observationLabel } from '../lib/common.js';
import { Brand, IconBadge } from './ui.jsx';

export default function Sources({t,notify}){
 const [items,setItems]=useState([]);useEffect(()=>{api('/sources').then(d=>setItems(d.items)).catch(e=>notify(e.message))},[]);
 return <section className="wrap section app-section"><div className="page-heading"><span className="eyebrow">{t.sources}</span><h1>{t.sourcesTitle}</h1><p>{t.sourcesBody}</p></div><div className="source-grid">{items.map((s,i)=><a className="source-card" href={s.url} key={s.id} target="_blank" rel="noopener noreferrer"><span className={`source-monogram source-${i%3}`}>{s.name.split(' ').map(x=>x[0]).join('').slice(0,3)}</span><ArrowUpRight size={19}/><h3>{s.name}</h3><p>{s.description}</p><span className="small muted">{s.domain}</span></a>)}</div><div className="source-tip"><IconBadge icon={BookOpen}/><div><h3>{t.sourceGuide}</h3><p>{t.sourceGuideBody}</p></div></div></section>
}

