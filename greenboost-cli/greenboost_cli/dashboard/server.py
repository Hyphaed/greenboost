#!/usr/bin/env python3
"""
GreenBoost CLI web dashboard — localhost management UI.

Features:
  - Project second brain (prime goals, history)
  - Local RAG index management + search
  - Account management
  - Token usage stats
  - UI design pipeline launcher
  - System status (GPU, GreenBoost tier, models)
  - PDF → Markdown conversion

Run:  greenboost dashboard  (or: gb-dashboard)
Open: http://localhost:7821
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse as urlparse

from greenboost_cli.environment.settings import GB_HOME

PORT = int(os.environ.get("GB_DASHBOARD_PORT", 7821))
GLOBAL_DIR = GB_HOME

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
:root {
  /* ── TeamCity-style dark neutrals ──────────────────────────────────── */
  --bg:#111214; --surface:#1c1d20; --surface2:#252628; --surface3:#2e3033; --surface4:#38393d;
  --border:rgba(255,255,255,0.07); --border2:rgba(255,255,255,0.12); --border3:rgba(255,255,255,0.20);

  /* ── Professional muted palette ─────────────────────────────────────── */
  --teal:#3ab0a0;    --teal-glow:rgba(58,176,160,0.18);
  --coral:#d4604a;   --coral-glow:rgba(212,96,74,0.18);
  --gold:#c99a30;    --gold-glow:rgba(201,154,48,0.18);
  --lavender:#8e82c8; --lavender-glow:rgba(142,130,200,0.18);
  --pink:#c24070;    --pink-glow:rgba(194,64,112,0.18);
  --ice:#3aa8c0;     --ice-glow:rgba(58,168,192,0.18);
  --blue:#3c9cdb;    --blue-glow:rgba(60,156,219,0.20); --blue20:rgba(60,156,219,0.10);
  --cream:#d6d0c0;

  /* ── Semantic tokens ─────────────────────────────────────────────────── */
  --lime:#57a64a;    --lime-glow:rgba(87,166,74,0.20);  --lime20:rgba(87,166,74,0.10);
  --cyan:var(--ice); --cyan-glow:var(--ice-glow);       --cyan20:rgba(58,168,192,0.10);
  --amber:var(--gold); --amber-glow:var(--gold-glow);
  --red:var(--pink); --red-glow:var(--pink-glow);
  --blue-light:var(--lavender); --blue-dark:var(--teal);

  /* Tier palette */
  --t1-col:var(--teal);    --t1-glow:var(--teal-glow);
  --t2-col:var(--lavender); --t2-glow:var(--lavender-glow);
  --t3-col:var(--coral);   --t3-glow:var(--coral-glow);

  /* Status palette */
  --ok-col:var(--lime);  --warn-col:var(--gold);  --crit-col:var(--pink);

  /* ── Text ───────────────────────────────────────────────────────────── */
  --text:#c9ccd1; --text-muted:#8b8f99; --text-dim:#545760;
  --white:#c9ccd1; --gray:#8b8f99; --dim:#545760;

  /* Gradients (kept for compatibility, rarely used) */
  --grad-blue:linear-gradient(135deg,var(--blue),var(--teal));
  --grad-cyan:linear-gradient(135deg,var(--ice),var(--lavender));
  --grad-brand:linear-gradient(135deg,var(--teal),var(--blue));
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;font-size:14px;min-height:100vh;display:flex}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--border3)}
.sidebar{width:56px;min-height:100vh;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;position:fixed;left:0;top:0;bottom:0;z-index:200;transition:width 0.25s cubic-bezier(0.4,0,0.2,1);overflow:hidden}
.sidebar:hover,.sidebar.expanded{width:220px}
.sidebar-brand{height:56px;display:flex;align-items:center;padding:0 15px;border-bottom:1px solid var(--border);white-space:nowrap;overflow:hidden;flex-shrink:0}
.brand-icon{width:26px;height:26px;border-radius:7px;background:var(--accent);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:11px;color:#fff;box-shadow:0 0 12px var(--accent-glow)}
.brand-text{margin-left:12px;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:14px;color:var(--accent);opacity:0;transition:opacity 0.2s ease;white-space:nowrap}
.brand-text span{color:var(--accent)}
.sidebar:hover .brand-text,.sidebar.expanded .brand-text{opacity:1}
.sidebar-nav{flex:1;padding:8px 0;display:flex;flex-direction:column;gap:2px;overflow:hidden}
.sidebar-nav a{display:flex;align-items:center;height:42px;padding:0 16px;color:var(--text-muted);transition:color 0.15s,background 0.15s;white-space:nowrap;text-decoration:none;font-size:13px;font-weight:500}
.sidebar-nav a:hover{color:var(--text);background:var(--surface2);text-decoration:none}
.sidebar-nav a.active{color:var(--accent);background:oklch(0.62 var(--a-c) var(--a-h) / .14);border-radius:0 6px 6px 0}
.nav-icon{width:24px;height:24px;flex-shrink:0;display:flex;align-items:center;justify-content:center}
.nav-label{margin-left:12px;opacity:0;transition:opacity 0.15s ease;white-space:nowrap}
.sidebar:hover .nav-label,.sidebar.expanded .nav-label{opacity:1}
.topbar{height:52px;background:var(--topbar-bg);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;position:sticky;top:0;z-index:100}
.topbar-title{font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:14px;color:var(--text);letter-spacing:-0.1px}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:10px}
.status-pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:500;background:var(--surface2);border:1px solid var(--border);color:var(--text-muted)}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--text-dim);flex-shrink:0}
.status-dot.on{background:var(--accent);box-shadow:0 0 8px var(--accent-glow);animation:pulse 2.5s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 6px var(--accent-glow)}50%{box-shadow:0 0 16px var(--accent-glow)}}
.app-shell{display:flex;width:100%;min-height:100vh}
.main-wrap{flex:1;margin-left:56px;display:flex;flex-direction:column;min-width:0}
.page{max-width:1200px;margin:28px auto;padding:0 28px;width:100%}
h1{font-family:'Space Grotesk',sans-serif;color:var(--accent);font-size:20px;margin-bottom:28px;font-weight:700;letter-spacing:-0.3px}
h2{color:var(--text-muted);font-size:10px;text-transform:uppercase;letter-spacing:1.8px;margin:32px 0 14px;font-family:'Inter',sans-serif;font-weight:600;display:flex;align-items:center;gap:10px}
h2::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent)}
h3{font-family:'Space Grotesk',sans-serif;color:var(--text);font-size:14px;font-weight:600;margin-bottom:8px}
/* Solid card — TeamCity-style flat cards with subtle shadow */
.card{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:8px;padding:20px;margin-bottom:16px;
  position:relative;overflow:hidden;
  box-shadow:0 1px 4px rgba(0,0,0,0.28);
  transition:border-color 0.15s,box-shadow 0.15s;
}
.card:hover{
  border-color:var(--border2);
  box-shadow:0 2px 12px rgba(0,0,0,0.36);
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);opacity:0;transition:opacity 0.2s}
.card:hover::before{opacity:0.6}
.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:16px}
.card-sm{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:8px;padding:16px;
  transition:border-color 0.15s,box-shadow 0.15s;
  box-shadow:0 1px 3px rgba(0,0,0,0.22);
}
.card-sm:hover{border-color:var(--border2);box-shadow:0 2px 8px rgba(0,0,0,0.30)}
.stat-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px}
/* Solid stat tile — TeamCity style */
.stat{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:8px;padding:18px 20px;flex:1;min-width:120px;
  position:relative;overflow:hidden;
  transition:border-color 0.15s,box-shadow 0.15s;
  box-shadow:0 1px 4px rgba(0,0,0,0.24);
}
/* Top-accent per tile — unique hue per position */
.stat{border-top:2px solid var(--teal)}
.stat:nth-child(2){border-top-color:var(--lavender)}
.stat:nth-child(3){border-top-color:var(--gold)}
.stat:nth-child(4){border-top-color:var(--pink)}
.stat:nth-child(5){border-top-color:var(--ice)}
.stat:hover{border-color:var(--border2);border-top-color:inherit;box-shadow:0 2px 10px rgba(0,0,0,0.32)}
.stat .num{font-family:'Space Grotesk',sans-serif;font-size:30px;color:var(--text);font-weight:700;line-height:1;letter-spacing:-0.5px}
.stat .label{color:var(--text-dim);font-size:10px;margin-top:7px;text-transform:uppercase;letter-spacing:.8px}
table{width:100%;border-collapse:collapse}
th{color:var(--accent);text-align:left;padding:10px 14px;border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:1px;font-weight:700;background:var(--surface2)}
td{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:top;font-size:13px}
tbody tr:hover td{background:oklch(0.62 var(--a-c) var(--a-h) / .05)}
tr:last-child td{border-bottom:none}
td.mono{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-muted)}
td.score{color:var(--amber);font-size:12px}
form{display:flex;flex-direction:column;gap:10px}
.field-row{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}
.field-row input,.field-row select{flex:1;min-width:180px}
input,textarea,select{background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:8px;font-family:'Inter',sans-serif;font-size:13px;width:100%;transition:border-color 0.15s,box-shadow 0.15s,background 0.15s}
textarea{min-height:80px;resize:vertical}
input:focus,textarea:focus,select:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow);background:var(--surface3)}
label{color:var(--text-muted);font-size:12px;margin-bottom:3px;display:block}
.help{color:var(--text-dim);font-size:11px;margin-top:3px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 16px;border-radius:8px;border:1px solid var(--border);cursor:pointer;font-family:'Inter',sans-serif;font-size:13px;font-weight:500;transition:background 0.12s,border-color 0.12s,box-shadow 0.12s;text-decoration:none;background:var(--surface2);color:var(--text-muted)}
.btn:hover{border-color:var(--border2);color:var(--text);background:var(--surface3);text-decoration:none;box-shadow:0 1px 4px rgba(0,0,0,0.18)}
.btn:active{box-shadow:none;transform:none}
.btn-primary{background:var(--accent);color:#fff;border:1px solid transparent}
.btn-primary:hover{background:var(--accent-dark);color:#fff;box-shadow:0 1px 6px var(--accent-glow)}
.btn-primary:active{background:var(--accent-dark);box-shadow:none}
.btn-success{background:rgba(46,194,126,0.12);color:var(--lime);border:1px solid rgba(46,194,126,0.30)}
.btn-success:hover{background:rgba(46,194,126,0.22);color:var(--lime)}
.btn-danger{background:rgba(224,27,36,0.08);color:var(--red);border:1px solid rgba(224,27,36,0.28)}
.btn-danger:hover{background:rgba(224,27,36,0.16);color:var(--red)}
.btn-ghost{background:transparent;color:var(--text-muted);border:1px solid var(--border)}
.btn-ghost:hover{color:var(--text);border-color:var(--border2);background:var(--surface2)}
.btn-sm{padding:4px 10px;font-size:12px}
.btn-lg{padding:10px 24px;font-size:14px}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;letter-spacing:.1px}
.badge-p1{background:rgba(239,68,68,0.15);color:var(--red);border:1px solid rgba(239,68,68,0.3)}
.badge-p2{background:rgba(245,158,11,0.12);color:var(--amber);border:1px solid rgba(245,158,11,0.25)}
.badge-p3{background:rgba(163,230,53,0.1);color:var(--lime);border:1px solid rgba(163,230,53,0.25)}
.badge-other{background:var(--surface3);color:var(--text-muted);border:1px solid var(--border)}
.badge-ok{background:rgba(163,230,53,0.1);color:var(--lime);border:1px solid rgba(163,230,53,0.3)}
.badge-warn{background:rgba(245,158,11,0.1);color:var(--amber);border:1px solid rgba(245,158,11,0.3)}
.badge-err{background:rgba(239,68,68,0.1);color:var(--red);border:1px solid rgba(239,68,68,0.3)}
.cat{display:inline-block;padding:2px 7px;border-radius:5px;font-size:11px;font-weight:600}
.cat-milestone{background:rgba(163,230,53,0.15);color:var(--lime);border:1px solid rgba(163,230,53,0.3)}
.cat-decision{background:rgba(53,132,228,0.15);color:var(--cyan);border:1px solid rgba(53,132,228,0.3)}
.cat-blocker{background:rgba(239,68,68,0.12);color:var(--red);border:1px solid rgba(239,68,68,0.3)}
.cat-resolved{background:rgba(163,230,53,0.1);color:var(--lime);border:1px solid rgba(163,230,53,0.25)}
.cat-note{background:var(--surface3);color:var(--text-muted);border:1px solid var(--border)}
pre,code{font-family:'JetBrains Mono','Fira Code',monospace}
.pre-wrap{position:relative}
.copy-btn{position:absolute;top:8px;right:8px;padding:3px 8px;font-size:11px;background:var(--surface3);border:1px solid var(--border);border-radius:5px;color:var(--text-muted);cursor:pointer;font-family:'Inter',sans-serif;transition:all 0.15s}
.copy-btn:hover{color:var(--text);border-color:var(--border2)}
pre{background:rgba(18,18,18,0.9);padding:16px;border-radius:10px;overflow-x:auto;font-size:12px;color:var(--text-muted);border:1px solid var(--border);line-height:1.65}
code{font-size:12px;background:var(--surface3);padding:1px 5px;border-radius:4px}
.vram-bar-wrap{height:6px;background:var(--surface3);border-radius:3px;overflow:hidden;width:100%}
.vram-bar{height:100%;border-radius:3px;transition:width 0.5s cubic-bezier(0.4,0,0.2,1)}
.empty{color:var(--text-dim);padding:24px 0;font-size:13px;text-align:center;letter-spacing:.01em}
.kv{display:flex;gap:16px;margin-bottom:10px;align-items:flex-start}
.kv .key{color:var(--text-muted);min-width:140px;flex-shrink:0;font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;padding-top:1px}
.kv .val{color:var(--text);font-size:13px}
.sep{height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent);margin:20px 0}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font-size:11px;background:var(--surface3);color:var(--text-muted);margin:2px;border:1px solid var(--border)}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:12px}
.pill-cyan{background:rgba(34,211,238,0.08);color:var(--cyan);border:1px solid rgba(34,211,238,0.25)}
.pill-lime{background:rgba(163,230,53,0.08);color:var(--lime);border:1px solid rgba(163,230,53,0.25)}
.alert{padding:12px 16px;border-radius:10px;margin-bottom:16px;font-size:13px}
.alert-info{background:rgba(34,211,238,0.07);border:1px solid rgba(34,211,238,0.2);color:var(--cyan)}
.alert-warn{background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.2);color:var(--amber)}
.section-actions{display:flex;gap:8px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
.project-picker{display:flex;gap:8px;align-items:center;margin-bottom:20px}
.project-picker label{color:var(--text-muted);font-size:12px;white-space:nowrap}
.project-picker select{width:auto;min-width:200px}
.model-row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid var(--border)}
.model-row:last-child{border:none}
.speed-fast{color:var(--lime)}.speed-slow{color:var(--amber)}
.mono{font-family:'JetBrains Mono',monospace}
/* .text-gradient removed — gradient text is never meaningful, use a solid color */
#toast-container{position:fixed;top:16px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{padding:12px 18px;border-radius:10px;font-size:13px;font-weight:500;backdrop-filter:blur(18px);border:1px solid var(--border2);pointer-events:all;transform:translateX(12px);opacity:0;transition:opacity 0.25s,transform 0.25s;max-width:340px}
.toast.show{opacity:1;transform:translateX(0)}
.toast-success{background:rgba(36,36,36,0.97);color:var(--lime);border-color:rgba(87,227,137,0.35)}
.toast-error{background:rgba(36,36,36,0.97);color:var(--red);border-color:rgba(255,123,99,0.35)}
.toast-info{background:rgba(36,36,36,0.97);color:var(--cyan);border-color:rgba(120,174,237,0.35)}
.toast-warn{background:rgba(36,36,36,0.97);color:var(--amber);border-color:rgba(255,163,72,0.35)}
.hamburger{display:none;align-items:center;justify-content:center;width:36px;height:36px;background:transparent;border:1px solid var(--border);border-radius:8px;cursor:pointer;color:var(--text-muted);margin-right:12px;flex-shrink:0}
.flash{display:none}
.gb-tier-bar{display:flex;align-items:stretch;gap:0;border-radius:10px;overflow:hidden;height:48px;margin-bottom:4px}
.gb-tier-seg{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:10px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;transition:filter 0.2s}
.gb-tier-seg:hover{filter:brightness(1.2)}
.gb-tier-label{font-size:9px;opacity:.7;margin-top:2px}
@media(max-width:768px){
  .sidebar{transform:translateX(-100%)}
  .sidebar.mobile-open{transform:translateX(0);width:220px}
  .main-wrap{margin-left:0}
  .hamburger{display:flex!important}
  .stat-row{flex-direction:column}
  .page{padding:0 16px;margin:20px auto}
}
/* ── OKLCH accent system — default: GreenBoost teal hue 175 ─────────── */
:root {
  --a-h: 175; --a-c: 0.13;
  --accent:      oklch(0.62 var(--a-c) var(--a-h));
  --accent-glow: oklch(0.62 var(--a-c) var(--a-h) / .22);
  --accent-dark: oklch(0.50 var(--a-c) var(--a-h));
  --topbar-bg:   rgba(17,18,20,0.96);
}
/* Adwaita Next light */
.light {
  --bg:#f0f0f2; --surface:#ffffff; --surface2:#e8e8ec; --surface3:#d8d8dd;
  --surface4:#c8c8ce;
  --border:rgba(0,0,0,0.10); --border2:rgba(0,0,0,0.16); --border3:rgba(0,0,0,0.26);
  --text:#1e1f23; --text-muted:#5c5e6a; --text-dim:#9496a2;
  --white:#1e1f23; --gray:#5c5e6a; --dim:#9496a2;
  --lime:#3a8f32;  --lime-glow:rgba(58,143,50,0.18);   --lime20:rgba(58,143,50,0.10);
  --cyan:#0076a8;  --cyan-glow:rgba(0,118,168,0.16);   --cyan20:rgba(0,118,168,0.10);
  --amber:#a07820; --amber-glow:rgba(160,120,32,0.18);
  --red:#b02030;   --red-glow:rgba(176,32,48,0.16);
  --teal:#1e8070;  --teal-glow:rgba(30,128,112,0.18);
  --lavender:#5a52a8; --lavender-glow:rgba(90,82,168,0.18);
  --gold:#a07820;  --gold-glow:rgba(160,120,32,0.18);
  --ice:#007890;   --ice-glow:rgba(0,120,144,0.16);
  --pink:#a02858;  --pink-glow:rgba(160,40,88,0.16);
  --blue:#1a72b8;  --blue-glow:rgba(26,114,184,0.20);
  --ok-col:var(--teal); --warn-col:var(--amber); --crit-col:var(--red);
  --t1-col:var(--teal); --t2-col:var(--lavender); --t3-col:var(--red);
  --accent:      oklch(0.44 var(--a-c) var(--a-h));
  --accent-glow: oklch(0.44 var(--a-c) var(--a-h) / .18);
  --accent-dark: oklch(0.36 var(--a-c) var(--a-h));
  --topbar-bg:   rgba(240,240,242,0.96);
}
/* Light mode cards inherit solid style; just ensure right colors */
.light .card{box-shadow:0 1px 3px rgba(0,0,0,0.10)}
.light .card:hover{box-shadow:0 2px 8px rgba(0,0,0,0.14)}
.light .card-sm{box-shadow:0 1px 2px rgba(0,0,0,0.08)}
.light .stat{box-shadow:0 1px 3px rgba(0,0,0,0.10)}
/* Light mode sidebar active link */
.light .sidebar-nav a.active{background:oklch(0.48 var(--a-c) var(--a-h) / .14)}
/* Hide Three.js particle bg in light mode (set by JS too) */
.light #oc-bg{opacity:0!important}
/* Adwaita-style: no border-radius changes in light (keep radius var) */
/* Settings panel */
.settings-panel{position:fixed;right:0;top:0;bottom:0;width:300px;background:var(--surface);border-left:1px solid var(--border);z-index:500;transform:translateX(100%);transition:transform .28s cubic-bezier(0.4,0,0.2,1);display:flex;flex-direction:column;box-shadow:-4px 0 24px rgba(0,0,0,.4)}
.settings-panel.open{transform:translateX(0)}
.settings-backdrop{position:fixed;inset:0;background:rgba(0,0,0,0.38);z-index:499;opacity:0;pointer-events:none;transition:opacity .28s ease;backdrop-filter:blur(2px)}
.settings-backdrop.visible{opacity:1;pointer-events:all}
.sp-header{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;border-bottom:1px solid var(--border);flex-shrink:0}
.sp-header-title{font-size:14px;font-weight:600;color:var(--text);letter-spacing:-0.2px}
.sp-close{background:transparent;border:1px solid var(--border);border-radius:6px;width:30px;height:30px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:var(--text-muted);transition:all .15s}
.sp-close:hover{background:var(--surface2);color:var(--text)}
.sp-body{flex:1;overflow-y:auto;padding:16px}
.sp-section{margin-bottom:24px}
.sp-title{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);font-weight:600;margin-bottom:12px}
.sp-divider{height:1px;background:var(--border);margin:16px 0}
.style-picker{display:flex;gap:12px}
.style-card{flex:1;border:2px solid var(--border);border-radius:12px;cursor:pointer;overflow:hidden;transition:border-color .18s,box-shadow .18s;background:var(--surface2)}
.style-card:hover{border-color:var(--border2)}
.style-card.active{border-color:var(--accent,var(--lime));box-shadow:0 0 0 1px var(--accent,var(--lime))}
.style-preview{height:80px;display:flex;flex-direction:column;gap:4px;padding:8px;border-bottom:1px solid var(--border)}
.style-card.s-dark .style-preview{background:#1c1c1c}
.style-card.s-light .style-preview{background:#f6f5f4}
.style-preview-bar{height:10px;border-radius:3px}
.style-card.s-dark .style-preview-bar:nth-child(1){background:#242424;border:1px solid rgba(255,255,255,0.10)}
.style-card.s-dark .style-preview-bar:nth-child(2){background:#2e2e2e;width:70%}
.style-card.s-light .style-preview-bar:nth-child(1){background:#ffffff;border:1px solid rgba(0,0,0,0.12)}
.style-card.s-light .style-preview-bar:nth-child(2){background:#e8e7e5;width:70%}
.style-card-label{padding:8px 10px;font-size:12px;font-weight:600;color:var(--text);text-align:center}
.accent-row{background:var(--surface2);border-radius:20px;padding:8px 12px;display:flex;flex-wrap:wrap;gap:6px;border:1px solid var(--border)}
.accent-swatch{width:28px;height:28px;border-radius:50%;border:2px solid transparent;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .18s;position:relative;flex-shrink:0}
.accent-swatch:hover{transform:scale(1.18)}
.accent-swatch.active{outline:2px solid var(--text);outline-offset:2px}
.accent-swatch.active::after{content:'';position:absolute;width:8px;height:8px;background:#fff;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);box-shadow:0 1px 2px rgba(0,0,0,.4)}
.radius-row{display:flex;align-items:center;gap:10px}
.radius-row input[type=range]{flex:1;accent-color:var(--accent,var(--lime));height:4px;cursor:pointer}
/* Topbar settings btn */
.topbar-btn{width:34px;height:34px;border-radius:8px;background:transparent;border:1px solid var(--border);cursor:pointer;display:flex;align-items:center;justify-content:center;color:var(--text-muted);transition:all .15s;flex-shrink:0}
.topbar-btn:hover{background:var(--surface2);color:var(--text);border-color:var(--border2)}
/* Sidebar settings button */
.sidebar-pin{padding:8px 0;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:2px}
.sidebar-pin button{width:100%;display:flex;align-items:center;height:38px;padding:0 16px;gap:12px;background:transparent;border:none;cursor:pointer;color:var(--text-muted);font-size:13px;font-weight:500;font-family:'Inter',sans-serif;transition:color .15s,background .15s;white-space:nowrap;text-align:left}
.sidebar-pin button:hover{color:var(--text);background:var(--surface2)}
.sidebar-pin .pin-label{opacity:0;transition:opacity 0.15s ease}
.sidebar:hover .pin-label,.sidebar.expanded .pin-label{opacity:1}
/* GB diagnostic cards */
.gb-metric-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;margin-bottom:12px}
.gb-metric{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.gb-metric .gm-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:4px}
.gb-metric .gm-val{font-size:20px;font-weight:700;font-family:'JetBrains Mono','Fira Code',monospace}
.gb-metric .gm-sub{font-size:11px;color:var(--dim);margin-top:2px}
/* NVTX event table */
.nvtx-table{width:100%;border-collapse:collapse;font-family:'JetBrains Mono','Fira Code',monospace;font-size:11px}
.nvtx-table th{color:var(--dim);text-align:left;padding:4px 8px;border-bottom:1px solid var(--border);font-size:10px;text-transform:uppercase;letter-spacing:.08em;font-weight:600;font-family:'Inter',sans-serif}
.nvtx-table td{padding:4px 8px;border-bottom:1px solid var(--surface3);vertical-align:top;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
.nvtx-table tr:hover td{background:var(--surface3)}
.nvtx-table tr:last-child td{border-bottom:none}
/* NVTX event type colours — professional muted palette */
.ev-alloc{color:#3ab0a0}   /* teal   — allocs healthy */
.ev-evict{color:#c99a30}   /* gold   — evictions warning */
.ev-phase{color:#3aa8c0}   /* ice    — phase transitions */
.ev-reset{color:#d4604a}   /* coral  — resets need attention */
.ev-kv{color:#8e82c8}      /* lavender — KV cache ops */
.ev-shim{color:var(--dim)} /* dim    — shim init (noise) */
.ev-oom{color:#c24070}     /* pink   — OOM critical */
/* Alert summary bar */
.gb-alert-bar{border-radius:10px;padding:12px 16px;margin-bottom:16px;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.gb-alert-ok{background:rgba(87,227,137,0.08);border:1px solid rgba(87,227,137,0.25)}
.gb-alert-warn{background:rgba(255,163,72,0.10);border:1px solid rgba(255,163,72,0.30)}
.gb-alert-crit{background:rgba(255,123,99,0.12);border:1px solid rgba(255,123,99,0.35)}
/* Tier chart wrapper */
.tier-chart-wrap{position:relative;height:120px;margin-top:8px}
/* Phase badge */
.phase-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;background:var(--surface2);border:1px solid var(--border)}
/* ── Entrance animations (no GSAP/ScrollTrigger) ─────────────────────────── */
@keyframes _cardIn{from{opacity:0;transform:translateY(20px) scale(0.98)}to{opacity:1;transform:none}}
@keyframes _h1In{from{opacity:0;transform:translateX(-14px)}to{opacity:1;transform:none}}
.card,.card-sm,.stat{opacity:0}
.card.anim-in,.card-sm.anim-in,.stat.anim-in{animation:_cardIn .5s cubic-bezier(0.22,1,.36,1) both}
h1{opacity:0}
h1.anim-in{animation:_h1In .55s cubic-bezier(0.215,.61,.355,1) both}
@media(prefers-reduced-motion:reduce){.card,.card-sm,.stat,h1{opacity:1!important;animation:none!important}}
"""

_JS = """
function showToast(msg, type='info') {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const t = document.createElement('div');
  t.className = 'toast toast-' + type;
  t.textContent = msg;
  c.appendChild(t);
  requestAnimationFrame(() => requestAnimationFrame(() => t.classList.add('show')));
  setTimeout(() => {
    t.classList.remove('show');
    setTimeout(() => { if (t.parentNode) c.removeChild(t); }, 320);
  }, 4000);
}

function toggleSidebar() {
  const s = document.querySelector('.sidebar');
  if (!s) return;
  if (window.innerWidth <= 768) {
    s.classList.toggle('mobile-open');
  } else {
    s.classList.toggle('expanded');
    localStorage.setItem('sidebarExpanded', s.classList.contains('expanded'));
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if (localStorage.getItem('sidebarExpanded') === 'true' && window.innerWidth > 768)
    document.querySelector('.sidebar')?.classList.add('expanded');
  document.querySelectorAll('.stat .num[data-target]').forEach(el => {
    const target = parseFloat(el.dataset.target);
    if (isNaN(target)) return;
    const t0 = performance.now(), dur = 700;
    function step(now) {
      const p = Math.min((now - t0) / dur, 1);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))).toLocaleString();
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
  document.querySelectorAll('pre').forEach(pre => {
    const wrap = document.createElement('div');
    wrap.className = 'pre-wrap';
    wrap.style.position = 'relative';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    btn.onclick = () => {
      navigator.clipboard?.writeText(pre.textContent).then(() => {
        btn.textContent = 'copied!';
        showToast('Copied to clipboard', 'success');
        setTimeout(() => { btn.textContent = 'copy'; }, 2000);
      });
    };
    wrap.appendChild(btn);
  });
  document.querySelectorAll('.flash').forEach(el => {
    if (el.textContent.trim()) showToast(el.textContent.trim(), 'success');
    el.style.display = 'none';
  });
});

async function refreshStatus() {
  try {
    const d = await (await fetch('/api/status')).json();
    const g = document.getElementById('gpu-dot');
    if (g) g.className = 'status-dot' + (d.gpu ? ' on' : '');
    const gb = document.getElementById('gb-dot');
    if (gb) gb.className = 'status-dot' + (d.gb ? ' on' : '');
  } catch(e) {}
}
setInterval(refreshStatus, 5000);
refreshStatus();

// ── Particle background (Canvas 2D — zero external deps) ──────────────────
function _initBg() {
  const canvas = document.getElementById('oc-bg');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const N = 700;
  const pts = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pts[i*3]   = (Math.random() - .5) * 240;
    pts[i*3+1] = (Math.random() - .5) * 160;
    pts[i*3+2] = (Math.random() - .5) * 120;
  }
  let mx = 0, my = 0;
  window.addEventListener('mousemove', e => {
    mx = (e.clientX / window.innerWidth  - .5) * 2;
    my = (e.clientY / window.innerHeight - .5) * 2;
  });
  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  window.addEventListener('resize', resize);
  resize();
  function frame(t) {
    requestAnimationFrame(frame);
    const ry = t * .000045 + mx * .04;
    const rx = t * .000022 + my * .025;
    const cY = Math.cos(ry), sY = Math.sin(ry);
    const cX = Math.cos(rx), sX = Math.sin(rx);
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(60,156,219,0.22)';
    for (let i = 0; i < N; i++) {
      const px = pts[i*3], py = pts[i*3+1], pz = pts[i*3+2];
      const x1 = px*cY - pz*sY, z1 = px*sY + pz*cY;
      const y1 = py*cX - z1*sX, z2 = py*sX + z1*cX;
      const d = 80 + z2;
      if (d < 1) continue;
      const s = 80 / d;
      ctx.fillRect(x1*s*(w/200) + w*.5 - .5, y1*s*(h/140) + h*.5 - .5, 1, 1);
    }
  }
  requestAnimationFrame(frame);
}
// ── Card entrance animations (IntersectionObserver — no GSAP/ScrollTrigger) ─
function _initAnimations() {
  if (window.matchMedia('(prefers-reduced-motion:reduce)').matches) {
    document.querySelectorAll('.card,.card-sm,.stat,h1').forEach(el => { el.style.opacity = '1'; });
    return;
  }
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      obs.unobserve(e.target);
      e.target.classList.add('anim-in');
    });
  }, { threshold: 0.05, rootMargin: '0px 0px -10% 0px' });
  let cardIdx = 0;
  document.querySelectorAll('.card,.card-sm,.stat').forEach(el => {
    el.style.animationDelay = ((cardIdx++ % 4) * 70) + 'ms';
    obs.observe(el);
  });
  document.querySelectorAll('h1').forEach(el => obs.observe(el));
}

// ── Settings / Theme / Accent ─────────────────────────────────────────────
const GBApp = {
  KEYS: { theme: 'gb-theme', accent: 'gb-accent', radius: 'gb-radius' },
  ACCENT_PRESETS: [
    ["Lime",142],["Green",152],["Teal",175],["Cyan",195],["Sky",212],
    ["Blue",240],["Indigo",262],["Violet",280],["Purple",305],["Pink",335],
    ["Red",22],["Orange",42],["Amber",56],["Yellow",90]
  ],

  init() {
    const t = localStorage.getItem(this.KEYS.theme) || 'dark';
    const h = localStorage.getItem(this.KEYS.accent) || '220';
    const r = localStorage.getItem(this.KEYS.radius) || '8';
    this._applyTheme(t);
    this._applyAccent(h);
    this._applyRadius(r);
    setTimeout(() => document.documentElement.classList.add('theme-ready'), 50);
  },

  _applyTheme(mode) {
    const html = document.documentElement;
    if (mode === 'light') html.classList.add('light');
    else html.classList.remove('light');
    html.dataset.theme = mode;
    document.querySelectorAll('[data-theme-btn]').forEach(b => {
      b.classList.toggle('active', b.dataset.themeBtn === mode);
    });
    const bg = document.getElementById('oc-bg');
    if (bg) bg.style.opacity = mode === 'light' ? '0' : '0.45';
  },

  setTheme(mode) { this._applyTheme(mode); localStorage.setItem(this.KEYS.theme, mode); },

  _applyAccent(hue) {
    document.documentElement.style.setProperty('--a-h', hue);
    document.querySelectorAll('.accent-swatch').forEach(s => {
      s.classList.toggle('active', s.dataset.h === String(hue));
    });
  },

  setAccent(hue) { this._applyAccent(hue); localStorage.setItem(this.KEYS.accent, hue); },

  _applyRadius(r) {
    document.documentElement.style.setProperty('--radius', r + 'px');
    const sl = document.getElementById('gb-radius-slider');
    const vl = document.getElementById('gb-radius-val');
    if (sl) sl.value = r;
    if (vl) vl.textContent = r;
  },

  setRadius(r) { this._applyRadius(r); localStorage.setItem(this.KEYS.radius, r); },

  toggleSettings() {
    const p = document.getElementById('gb-settings-panel');
    const bd = document.getElementById('gb-settings-backdrop');
    if (!p) return;
    const open = p.classList.toggle('open');
    bd?.classList.toggle('visible', open);
  },

  buildAccentSwatches() {
    const row = document.getElementById('gb-accent-row');
    if (!row) return;
    row.innerHTML = this.ACCENT_PRESETS.map(([name, h]) =>
      `<button class="accent-swatch" data-h="${h}" style="background:oklch(0.62 0.22 ${h})"
        onclick="GBApp.setAccent('${h}')" title="${name}"></button>`
    ).join('');
    const saved = localStorage.getItem(this.KEYS.accent) || '220';
    this._applyAccent(saved);
  }
};
document.addEventListener('DOMContentLoaded', () => { GBApp.init(); GBApp.buildAccentSwatches(); _initBg(); _initAnimations(); });
"""

# ── Navigation ────────────────────────────────────────────────────────────────

_NAV_ITEMS = [
    ("Dashboard", "/",         "home",     '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>'),
    ("Goals",     "/goals",    "goals",    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/></svg>'),
    ("History",   "/history",  "history",  '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 8v4l3 3"/><circle cx="12" cy="12" r="9"/></svg>'),
    ("RAG",       "/rag",      "rag",      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.35-4.35"/></svg>'),
    ("Design",    "/design",   "design",   '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.86L12 17.77l-6.18 3.23L7 14.14 2 9.27l6.91-1.01z"/></svg>'),
    ("PDF",       "/pdf",      "pdf",      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'),
    ("Tokens",     "/tokens",     "tokens",     '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 1 0 0 7h5a3.5 3.5 0 1 1 0 7H6"/></svg>'),
    ("Guidelines", "/guidelines", "guidelines", '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>'),
    ("Factory",    "/factory",    "factory",    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M2 20h20M4 20V10l4-4 4 4V4l4 4v12"/><line x1="9" y1="20" x2="9" y2="14"/><line x1="15" y1="20" x2="15" y2="14"/></svg>'),
    ("GreenBoost", "/greenboost", "greenboost", '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><rect x="2" y="7" width="20" height="10" rx="2"/><path d="M6 11h4M8 9v4M14 11h4"/></svg>'),
    ("System",     "/system",     "system",     '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>'),
]


def _page(title: str, body: str, active: str = "") -> str:
    page_title = next((lbl for lbl, _, key, _ in _NAV_ITEMS if key == active), title)
    sidebar_items = "\n      ".join(
        f'<a href="{href}" class="{"active" if key == active else ""}">'
        f'<span class="nav-icon">{icon}</span>'
        f'<span class="nav-label">{label}</span></a>'
        for label, href, key, icon in _NAV_ITEMS
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — GreenBoost CLI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<canvas id="oc-bg" style="position:fixed;inset:0;z-index:0;pointer-events:none;opacity:0;transition:opacity 1.2s ease"></canvas>
<div class="app-shell" style="position:relative;z-index:1">
  <aside class="sidebar">
    <div class="sidebar-brand">
      <div class="brand-icon">GB</div>
      <span class="brand-text">Green<span>Boost</span></span>
    </div>
    <nav class="sidebar-nav">
      {sidebar_items}
    </nav>
    <div class="sidebar-pin">
      <button onclick="GBApp.toggleSettings()" title="Appearance">
        <span class="nav-icon"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg></span>
        <span class="pin-label">Appearance</span>
      </button>
    </div>
  </aside>
  <div class="main-wrap">
    <header class="topbar">
      <button class="hamburger" onclick="toggleSidebar()" aria-label="Toggle sidebar">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <span class="topbar-title">{page_title}</span>
      <div class="topbar-right">
        <div class="status-pill"><div id="gpu-dot" class="status-dot" title="GPU"></div><span>GPU</span></div>
        <div class="status-pill"><div id="gb-dot" class="status-dot" title="GreenBoost"></div><span>GB</span></div>
        <button class="topbar-btn" onclick="GBApp.toggleSettings()" title="Appearance">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>
        </button>
      </div>
    </header>
    <div class="page">
{body}
    </div>
  </div>
</div>
<div id="toast-container"></div>
<div id="browse-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;align-items:center;justify-content:center">
  <div style="background:var(--surface,#13131f);border:1px solid var(--border,#2a2a3e);border-radius:12px;padding:20px;width:min(660px,93vw);max-height:82vh;display:flex;flex-direction:column;gap:12px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <code id="browse-cwd" style="font-size:12px;word-break:break-all;color:var(--dim)"></code>
      <button type="button" onclick="closeBrowse()" style="background:none;border:none;cursor:pointer;font-size:22px;color:var(--dim);line-height:1">&times;</button>
    </div>
    <div id="browse-list" style="overflow-y:auto;border:1px solid var(--border,#2a2a3e);border-radius:6px;max-height:420px;min-height:120px"></div>
    <div style="display:flex;gap:8px">
      <input id="browse-sel" type="text" style="flex:1;font-family:monospace;font-size:13px" readonly placeholder="Click a folder/file to select, double-click dir to open">
      <button type="button" class="btn btn-primary" onclick="confirmBrowse()">Select</button>
      <button type="button" class="btn" onclick="closeBrowse()">Cancel</button>
    </div>
  </div>
</div>
<script>
var _browseTarget=null,_browseType='dir';
function openBrowse(id,type,start){{
  _browseTarget=id;_browseType=type||'dir';
  var inp=document.getElementById(id);
  var p=(inp&&inp.value)?inp.value:(start||'/');
  document.getElementById('browse-sel').value='';
  document.getElementById('browse-modal').style.display='flex';
  browseLoad(p);
}}
function closeBrowse(){{document.getElementById('browse-modal').style.display='none';}}
function browseLoad(path){{
  document.getElementById('browse-cwd').textContent=path;
  var list=document.getElementById('browse-list');
  list.innerHTML='<div style="padding:10px;color:var(--dim)">Loading…</div>';
  fetch('/api/browse?path='+encodeURIComponent(path)+'&type='+_browseType)
    .then(function(r){{return r.json();}})
    .then(function(data){{
      list.innerHTML='';
      if(data.error){{list.innerHTML='<div style="padding:10px;color:var(--red)">'+data.error+'</div>';return;}}
      document.getElementById('browse-cwd').textContent=data.path;
      data.items.forEach(function(item){{
        if(_browseType==='dir'&&item.type!=='dir')return;
        var row=document.createElement('div');
        row.style.cssText='padding:7px 12px;cursor:pointer;display:flex;align-items:center;gap:8px;border-radius:4px;font-size:13px';
        row.innerHTML=(item.type==='dir'?'&#128193;':'&#128196;')+' <span style="font-family:monospace">'+item.name+'</span>';
        row.onmouseenter=function(){{this.style.background='rgba(53,132,228,.18)';}}
        row.onmouseleave=function(){{if(this!==_browseActive)this.style.background='';}}
        row.onclick=function(){{
          if(_browseActive)_browseActive.style.background='';
          this.style.background='rgba(53,132,228,.28)';_browseActive=this;
          document.getElementById('browse-sel').value=item.path;
        }};
        if(item.type==='dir')row.ondblclick=function(){{browseLoad(item.path);}};
        list.appendChild(row);
      }});
    }})
    .catch(function(e){{list.innerHTML='<div style="padding:10px;color:var(--red)">'+e+'</div>';}});
}}
var _browseActive=null;
function confirmBrowse(){{
  var sel=document.getElementById('browse-sel').value;
  if(!sel)return;
  if(_browseTarget){{var inp=document.getElementById(_browseTarget);if(inp)inp.value=sel;}}
  closeBrowse();
}}
</script>
<script>{_JS}</script>
<!-- Settings backdrop -->
<div id="gb-settings-backdrop" class="settings-backdrop" onclick="GBApp.toggleSettings()"></div>
<!-- Settings panel — GNOME Appearance style -->
<div id="gb-settings-panel" class="settings-panel" role="dialog">
  <div class="sp-header">
    <span class="sp-header-title">Appearance</span>
    <button class="sp-close" onclick="GBApp.toggleSettings()">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="sp-body">
    <div class="sp-section">
      <div class="sp-title">Style</div>
      <div class="style-picker">
        <button class="style-card s-dark" data-theme-btn="dark" onclick="GBApp.setTheme('dark')">
          <div class="style-preview"><div class="style-preview-bar"></div><div class="style-preview-bar"></div></div>
          <div class="style-card-label">Dark</div>
        </button>
        <button class="style-card s-light" data-theme-btn="light" onclick="GBApp.setTheme('light')">
          <div class="style-preview"><div class="style-preview-bar"></div><div class="style-preview-bar"></div></div>
          <div class="style-card-label">Light</div>
        </button>
      </div>
    </div>
    <div class="sp-divider"></div>
    <div class="sp-section">
      <div class="sp-title">Accent Color</div>
      <div class="accent-row" id="gb-accent-row"></div>
    </div>
    <div class="sp-divider"></div>
    <div class="sp-section">
      <div class="sp-title">Corner Radius</div>
      <div class="radius-row">
        <input id="gb-radius-slider" type="range" min="0" max="20" value="8" oninput="GBApp.setRadius(this.value)">
        <span id="gb-radius-val" style="font-size:12px;color:var(--text-muted);min-width:28px;text-align:right;font-family:'JetBrains Mono',monospace">8</span>
      </div>
    </div>
  </div>
</div>
</body>
</html>"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_projects() -> list[str]:
    d = GLOBAL_DIR / "projects"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def _get_project(qs: dict) -> str:
    projects = _list_projects()
    req = qs.get("project", [""])[0]
    return req if req in projects else (projects[0] if projects else "")


def _project_picker(current: str, endpoint: str) -> str:
    projects = _list_projects()
    if not projects:
        return '<div class="alert alert-warn">No projects yet. Start a GreenBoost CLI session first.</div>'
    opts = "".join(
        f'<option value="{p}" {"selected" if p == current else ""}>{p}</option>'
        for p in projects
    )
    return f"""<div class="project-picker">
      <label>Project:</label>
      <select onchange="location='{endpoint}?project='+this.value">{opts}</select>
    </div>"""


def _gpu_status() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "available": True,
            "name": parts[0] if parts else "GPU",
            "mem_used": int(parts[1]) if len(parts) > 1 else 0,
            "mem_total": int(parts[2]) if len(parts) > 2 else 0,
            "temp": int(parts[3]) if len(parts) > 3 else 0,
        }
    except Exception:
        return {"available": False}


def _gb_available() -> bool:
    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats
        stats = get_tier_stats()
        return stats is not None
    except Exception:
        return False


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_dashboard() -> str:
    projects = _list_projects()
    gpu = _gpu_status()

    rag_chunks = 0
    try:
        from greenboost_cli.rag.engine import _load_store as rag_load_store
        _, meta = rag_load_store()
        rag_chunks = len(meta) if meta else 0
    except Exception:
        pass

    gb_st = {}
    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        gb_st = get_monitor().refresh().as_dict()
    except Exception:
        pass

    gb_loaded = gb_st.get("loaded", False)
    t1_gb     = round(gb_st.get("vram_physical_mb", 0) / 1024, 1)
    t2_used   = round(gb_st.get("ram_allocated_mb", 0) / 1024, 1)
    t2_total  = round(gb_st.get("ram_pool_mb", 0) / 1024, 1)
    t3_used   = round(gb_st.get("nvme_swap_used_mb", 0) / 1024, 1)
    t3_total  = round(gb_st.get("nvme_swap_total_mb", 0) / 1024, 1)
    t2_pct    = int(t2_used / t2_total * 100) if t2_total else 0
    t3_pct    = int(t3_used / t3_total * 100) if t3_total else 0
    combined  = round(gb_st.get("total_combined_mb", 0) / 1024, 1)
    t2_press  = gb_st.get("t2_pressure", 0)
    t3_press  = gb_st.get("swap_pressure", 0)
    oom       = gb_st.get("oom_active", False)
    kv_used   = gb_st.get("kv_used_mb", 0)
    tq_bits   = gb_st.get("kv_compression_bits", 0)
    gpu_name  = gb_st.get("gpu_name", "—")

    _pcol = {0: "var(--lime)", 1: "var(--amber)", 2: "var(--red)"}
    t2_col = _pcol.get(t2_press, "var(--lime)")
    t3_col = _pcol.get(t3_press, "var(--lime)")

    # GreenBoost tier bar
    if gb_loaded and t2_total > 0:
        t1_pct = int(t1_gb / combined * 100) if combined else 0
        t2_pct_w = int(t2_total / combined * 100) if combined else 0
        t3_pct_w = 100 - t1_pct - t2_pct_w
        gb_bar = f"""
        <div style="margin-bottom:20px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <span style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;color:var(--accent)">
              GreenBoost Memory Hierarchy — {combined} GB total
            </span>
            <span class="badge badge-ok">active</span>
          </div>
          <div class="gb-tier-bar">
            <div class="gb-tier-seg" style="flex:{t1_pct};background:linear-gradient(135deg,rgba(34,211,238,.25),rgba(34,211,238,.12));color:var(--cyan)">
              T1 · VRAM<div class="gb-tier-label">{t1_gb} GB</div>
            </div>
            <div class="gb-tier-seg" style="flex:{t2_pct_w};background:linear-gradient(135deg,rgba(53,132,228,.25),rgba(53,132,228,.12));color:var(--cyan)">
              T2 · RAM<div class="gb-tier-label">{t2_used}/{t2_total} GB</div>
            </div>
            <div class="gb-tier-seg" style="flex:{max(t3_pct_w,1)};background:linear-gradient(135deg,rgba(163,230,53,.18),rgba(163,230,53,.08));color:var(--lime)">
              T3 · NVMe<div class="gb-tier-label">{t3_used}/{t3_total} GB</div>
            </div>
          </div>
          <div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap">
            <span style="font-size:11px;color:var(--text-muted)">
              T2 <span style="color:{t2_col}">{t2_pct}% used</span>
            </span>
            <span style="font-size:11px;color:var(--text-muted)">
              T3 <span style="color:{t3_col}">{t3_pct}% used</span>
            </span>
            <span style="font-size:11px;color:var(--text-muted)">
              KV <span style="color:var(--text)">{kv_used} MB</span>
            </span>
            {"<span style='font-size:11px;color:var(--amber)'>TurboQuant " + str(tq_bits) + "-bit</span>" if tq_bits else ""}
            {"<span style='font-size:11px;color:var(--red);font-weight:600'>⚠ OOM ACTIVE</span>" if oom else ""}
            <a href="/greenboost" style="font-size:11px;color:var(--cyan);margin-left:auto">
              Full GreenBoost Monitor &#8594;</a>
          </div>
        </div>"""
    elif gb_loaded:
        gb_bar = '<div style="margin-bottom:20px"><span class="badge badge-ok">GreenBoost active</span> <a href="/greenboost" style="font-size:12px;color:var(--cyan)">Monitor &#8594;</a></div>'
    else:
        gb_bar = '<div style="margin-bottom:20px"><span class="badge badge-other">GreenBoost not loaded</span></div>'

    # GPU card
    gpu_html = ""
    if gpu.get("available"):
        pct = int(gpu["mem_used"] / max(gpu["mem_total"], 1) * 100)
        col = "var(--lime)" if pct < 75 else ("var(--amber)" if pct < 90 else "var(--red)")
        gpu_html = f"""
        <div class="card" style="border-color:rgba(34,211,238,0.2)">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-muted);margin-bottom:2px">GPU</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;font-size:14px;color:var(--cyan)">{gpu['name']}</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:22px;font-weight:700;color:{col};font-family:'Space Grotesk',sans-serif">{pct}%</div>
              <div style="font-size:11px;color:var(--text-muted)">VRAM</div>
            </div>
          </div>
          <div class="vram-bar-wrap"><div class="vram-bar" style="width:{pct}%;background:{col}"></div></div>
          <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:12px;color:var(--text-muted)">
            <span style="color:{col}">{gpu['mem_used']} MB used</span>
            <span>{gpu['mem_total']} MB total</span>
            <span>{gpu['temp']}°C</span>
          </div>
        </div>"""

    # Stats
    gpu_stat_val = "OK" if gpu.get("available") else "—"
    stats_html = f"""<div class="stat-row">
      <div class="stat">
        <div class="num" data-target="{len(projects)}">{len(projects)}</div>
        <div class="label">Projects</div>
      </div>
      <div class="stat">
        <div class="num" data-target="{rag_chunks}">{rag_chunks:,}</div>
        <div class="label">RAG Chunks</div>
      </div>
      <div class="stat">
        <div class="num" style="font-size:22px;padding-top:6px">{gpu_stat_val}</div>
        <div class="label">GPU Status</div>
      </div>
    </div>"""

    # Project cards
    proj_html = ""
    for p in projects[:6]:
        pdir = GLOBAL_DIR / "projects" / p
        try:
            from greenboost_cli.memory.brain import load_goals
            goals = load_goals(pdir)
            n_goals = len(goals)
            p1 = sum(1 for g in goals if g.get("priority", 5) <= 2)
        except Exception:
            n_goals = 0
            p1 = 0
        p1_badge = f'<span class="badge badge-p1" style="margin-left:4px">{p1} P1</span>' if p1 else ""
        proj_html += f"""<div class="card-sm">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <div style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                           color:var(--cyan);font-size:14px;margin-bottom:3px">{p}</div>
              <span class="badge badge-other">{n_goals} goals</span>{p1_badge}
            </div>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none"
                 stroke="var(--border2)" stroke-width="1.5" viewBox="0 0 24 24">
              <path d="M3 7h18M3 12h12M3 17h8"/></svg>
          </div>
          <div style="display:flex;gap:5px;flex-wrap:wrap">
            <a href="/goals?project={p}" class="btn btn-ghost btn-sm">Goals</a>
            <a href="/history?project={p}" class="btn btn-ghost btn-sm">History</a>
            <a href="/tokens?project={p}" class="btn btn-ghost btn-sm">Tokens</a>
            <a href="/greenboost?project={p}" class="btn btn-ghost btn-sm">GB</a>
          </div>
        </div>"""

    if not proj_html:
        proj_html = '<div class="empty" style="padding:40px 0">No projects yet — run <code>gb</code> in any project directory and it will appear here.</div>'

    # Quick actions
    actions = [
        ("/greenboost",  "GreenBoost Monitor", "Live T1/T2/T3 status, memory flow, log RAG",  "var(--lime)"),
        ("/rag",         "RAG Index",           "Semantic code search across all projects",    "var(--cyan)"),
        ("/design",      "UI Design Pipeline",  "Generate UI assets with local diffusion",     "var(--cyan)"),
        ("/goals",       "Prime Goals",         "Track project goals and priorities",          "var(--amber)"),
        ("/system",      "System Status",       "GPU, packages, services health",              "var(--text-muted)"),
        ("/tokens",      "Token Usage",         "API and local inference cost tracking",       "var(--text-muted)"),
    ]
    action_html = "".join(
        f'<a href="{href}" class="card-sm" style="text-decoration:none;display:block">'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="width:8px;height:8px;border-radius:50%;background:{col};'
        f'box-shadow:0 0 8px {col};flex-shrink:0"></div>'
        f'<div><div style="font-weight:600;font-size:13px;color:var(--text)">{label}</div>'
        f'<div style="font-size:11px;color:var(--text-muted);margin-top:2px">{desc}</div></div>'
        f'</div></a>'
        for href, label, desc, col in actions
    )

    body = f"""
    {gb_bar}
    {stats_html}
    <h2>Projects</h2>
    <div class="card-grid">{proj_html}</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px">
      <div>
        <h2>System</h2>
        <div class="card">
          <div class="kv">
            <span class="key">Active account</span>
            <span class="val"><code style="font-size:12px">{active_acc}</code></span>
          </div>
          <div class="kv">
            <span class="key">GPU</span>
            <span class="val">{gpu_name}</span>
          </div>
          <div class="kv">
            <span class="key">Dashboard URL</span>
            <span class="val"><code style="font-size:12px">http://localhost:{PORT}</code></span>
          </div>
        </div>
        {gpu_html}
      </div>
      <div>
        <h2>Navigate</h2>
        <div style="display:flex;flex-direction:column;gap:8px">
          {action_html}
        </div>
      </div>
    </div>"""
    return _page("Dashboard", body, "home")


def page_goals(qs: dict) -> str:
    project = _get_project(qs)
    picker = _project_picker(project, "/goals")
    if not project:
        return _page("Prime Goals", picker, "goals")

    pdir = GLOBAL_DIR / "projects" / project
    try:
        from greenboost_cli.memory.brain import load_goals
        goals = load_goals(pdir)
    except Exception:
        goals = []

    rows = "".join(
        f"""<tr>
          <td><span class="badge badge-p{min(g['priority'],3)}">{g['priority']}</span></td>
          <td style="color:var(--white)">{g['text']}</td>
          <td class="mono">{g.get('added_at','')[:10]}</td>
          <td><form method="post" action="/goals/remove" style="display:inline">
            <input type="hidden" name="project" value="{project}">
            <input type="hidden" name="id" value="{g['id']}">
            <button class="btn btn-danger btn-sm">x</button>
          </form></td>
        </tr>"""
        for g in goals
    ) or '<tr><td colspan="4" class="empty">No goals yet. Add one below — P1 goals are injected into every session.</td></tr>'

    body = f"""{picker}
    <div class="card">
      <table>
        <thead><tr><th>P</th><th>Goal</th><th>Added</th><th></th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Add Prime Goal</h2>
      <form method="post" action="/goals/add">
        <input type="hidden" name="project" value="{project}">
        <div class="field-row">
          <div style="flex:3"><label>Goal description</label>
            <input type="text" name="text" placeholder="Never break the public API..." required></div>
          <div style="flex:0 0 80px"><label>Priority</label>
            <select name="priority">
              {"".join(f'<option value="{i}">P{i}</option>' for i in range(1, 6))}
            </select></div>
          <div style="flex:0 0 auto;padding-top:21px">
            <button class="btn btn-primary">Add</button></div>
        </div>
        <p class="help">Priority 1 = highest. Injected into every Claude session as immutable constraints.</p>
      </form>
    </div>"""
    return _page(f"Goals — {project}", body, "goals")


def _history_entries(project: str) -> list[tuple[str, str, str]]:
    pdir = GLOBAL_DIR / "projects" / project
    hist_file = pdir / "history.md"
    entries = []
    if hist_file.exists():
        text = hist_file.read_text()
        for s in reversed(text.split("\n## ")[1:]):
            first, _, rest = s.partition("\n")
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+\[(\w+)\]", first.strip())
            ts, cat = (m.group(1), m.group(2)) if m else (first.strip(), "note")
            entries.append((ts, cat, rest.strip()))
    return entries


def page_history(qs: dict) -> str:
    project = _get_project(qs)
    picker = _project_picker(project, "/history")
    if not project:
        return _page("History", picker, "history")

    q = qs.get("q", [""])[0].strip()
    all_entries = _history_entries(project)

    if q and all_entries:
        try:
            import numpy as np
            from greenboost_cli.rag.engine import _embed
            texts = [f"{ts} [{cat}] {txt}" for ts, cat, txt in all_entries]
            embs = _embed(texts)
            q_emb = _embed([q])
            scores = (embs @ q_emb.T).flatten()
            order = list(reversed(scores.argsort().tolist()))
            entries = [(all_entries[i][0], all_entries[i][1], all_entries[i][2], float(scores[i]))
                       for i in order if float(scores[i]) >= 0.05][:20]
            show_scores = True
        except Exception:
            entries = [(ts, cat, txt, 0.0) for ts, cat, txt in all_entries[:50]]
            show_scores = False
    else:
        entries = [(ts, cat, txt, 0.0) for ts, cat, txt in all_entries[:50]]
        show_scores = False

    def _row(ts, cat, txt, score):
        score_badge = (
            f'<span class="badge" style="background:rgba(53,132,228,.2);color:var(--cyan);'
            f'margin-left:4px">{int(score*100)}%</span>'
            if show_scores and score > 0 else ""
        )
        return (
            f'<tr><td class="mono">{ts}</td>'
            f'<td><span class="cat cat-{cat}">{cat}</span>{score_badge}</td>'
            f'<td style="max-width:500px;white-space:pre-wrap">{txt[:200]}</td></tr>'
        )

    rows = "".join(_row(*e) for e in entries) or '<tr><td colspan="3" class="empty">No history entries yet. Use /history-add in the REPL or the form below.</td></tr>'
    q_val = q.replace('"', "&quot;")

    body = f"""{picker}
    <div class="card">
      <h2>Search History</h2>
      <form method="get" action="/history" style="display:flex;gap:8px;align-items:center">
        <input type="hidden" name="project" value="{project}">
        <input type="text" name="q" value="{q_val}"
               placeholder="Semantic search: blocker, auth decision, milestone…"
               style="flex:1">
        <button type="submit" class="btn">Search</button>
        {'<a href="/history?project=' + project + '" class="btn">Clear</a>' if q else ''}
      </form>
      {'<div style="color:var(--dim);font-size:12px;margin-top:6px">Showing semantic results for &ldquo;' + q_val + '&rdquo;</div>' if q else ''}
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Time</th><th>Type</th><th>Entry</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Add Entry</h2>
      <form method="post" action="/history/add">
        <input type="hidden" name="project" value="{project}">
        <textarea name="text" placeholder="What happened? Decision made? Blocker resolved?" required></textarea>
        <div class="field-row">
          <select name="category">
            <option value="note">note</option>
            <option value="decision">decision</option>
            <option value="milestone">milestone</option>
            <option value="blocker">blocker</option>
            <option value="resolved">resolved</option>
          </select>
          <button class="btn btn-primary">Add</button>
        </div>
      </form>
    </div>"""
    return _page(f"History — {project}", body, "history")


def page_rag(qs: dict) -> str:
    import html as _html
    from pathlib import Path as _Path

    folders, n_chunks, n_files, size_mb = [], 0, 0, 0.0
    try:
        from greenboost_cli.rag.engine import _load_store as rag_load_store, _load_folders
        folders = _load_folders()
        _, meta = rag_load_store()
        n_chunks = len(meta) if meta else 0
        n_files = len({m["file"] for m in meta}) if meta else 0
        db_path = GLOBAL_DIR / "rag" / "embeddings.npy"
        size_mb = db_path.stat().st_size / 1_048_576 if db_path.exists() else 0.0
    except Exception:
        pass

    flash = qs.get("flash", [""])[0]
    flash_html = f'<div class="flash" style="color:var(--lime);margin-bottom:12px">{_html.escape(flash)}</div>' if flash else ""

    def _row(f: dict) -> str:
        is_dir = False
        try:
            is_dir = _Path(f["folder"]).is_dir()
        except Exception:
            pass
        if is_dir:
            esc = _html.escape(str(f["folder"]), quote=True)
            action = (f'<form method="post" action="/rag/update" style="margin:0">'
                      f'<input type="hidden" name="folder" value="{esc}">'
                      f'<button class="btn">Update</button></form>')
        else:
            action = ""
        return (f'<tr><td style="color:var(--cyan)">{f["project"]}</td>'
                f'<td class="mono">{f["folder"]}</td>'
                f'<td>{f.get("chunk_count", 0)}</td>'
                f'<td class="mono">{f.get("last_indexed", "")[:10]}</td>'
                f'<td>{action}</td></tr>')

    folder_rows = "".join(_row(f) for f in folders) or \
        '<tr><td colspan="5" class="empty">No folders indexed yet. Add a project folder below to start building the semantic index.</td></tr>'

    body = f"""
    {flash_html}
    <div class="stat-row">
      <div class="stat"><div class="num">{n_chunks}</div><div class="label">Chunks</div></div>
      <div class="stat"><div class="num">{n_files}</div><div class="label">Files</div></div>
      <div class="stat"><div class="num">{size_mb:.1f} MB</div><div class="label">Index size</div></div>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <h2 style="margin:0">Indexed Projects</h2>
        <form method="post" action="/rag/update" style="margin:0;display:flex;gap:8px;align-items:center">
          <input type="hidden" name="all" value="1">
          <label style="font-size:12px;color:var(--muted)"><input type="checkbox" name="force"> Full rebuild</label>
          <button class="btn btn-primary">Update All</button>
        </form>
      </div>
      <table>
        <thead><tr><th>Project</th><th>Folder</th><th>Chunks</th><th>Indexed</th><th>Actions</th></tr></thead>
        <tbody>{folder_rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Add Folder</h2>
      <form method="post" action="/rag/add">
        <div class="field-row">
          <div style="flex:3"><label>Folder path</label>
            <div style="display:flex;gap:6px">
              <input type="text" name="folder" id="rag-folder-inp"
                     placeholder="/path/to/your/project" required style="flex:1">
              <button type="button" class="btn" title="Browse folders"
                      onclick="openBrowse('rag-folder-inp','dir')">&#128193;</button>
            </div></div>
          <div style="flex:1"><label>Project name (optional)</label>
            <input type="text" name="project" placeholder="auto-detected"></div>
          <div style="padding-top:21px"><button class="btn btn-primary">Index</button></div>
        </div>
      </form>
    </div>
    <div class="card">
      <h2>Semantic Search</h2>
      <form method="get" action="/rag/search">
        <div class="field-row">
          <div style="flex:3"><label>Query</label>
            <input type="text" name="q" placeholder="How does the auth flow work?" required></div>
          <div style="flex:1"><label>Project filter</label>
            <input type="text" name="project" placeholder="all"></div>
          <div style="padding-top:21px"><button class="btn btn-primary">Search</button></div>
        </div>
      </form>
    </div>
    <div class="card">
      <h2>Clear Index</h2>
      <form method="post" action="/rag/clear">
        <button class="btn btn-danger">Clear All RAG Data</button>
      </form>
    </div>"""
    return _page("RAG Index", body, "rag")


def page_rag_search(qs: dict) -> str:
    q = qs.get("q", [""])[0]
    project = qs.get("project", [""])[0] or None
    if not q:
        return page_rag(qs)

    results = []
    try:
        from greenboost_cli.rag.engine import search
        results = search(q, project=project, top_k=8)
    except Exception as e:
        import traceback
        return _page("RAG Search", f'<div class="alert alert-warn">Search error: {e}</div>', "rag")

    result_html = "".join(
        f"""<div class="card">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px">
            <code style="color:var(--cyan)">{r['file']}:{r['lines'][0]}-{r['lines'][1]}</code>
            <span class="score">score: {r['score']}</span>
          </div>
          <pre>{r['text'][:500].replace('<', '&lt;').replace('>', '&gt;')}</pre>
        </div>"""
        for r in results
    ) or '<div class="empty">No results found.</div>'

    body = f"""
    <div class="card">
      <form method="get" action="/rag/search">
        <div class="field-row">
          <input type="text" name="q" value="{q}" placeholder="Search..." required>
          <button class="btn btn-primary">Search</button>
        </div>
      </form>
    </div>
    <h2>{len(results)} results for: {q}</h2>
    {result_html}"""
    return _page(f"RAG: {q}", body, "rag")


def page_pdf(qs: dict) -> str:
    flash = qs.get("flash", [""])[0]
    result_md = qs.get("result", [""])[0]
    result_path = qs.get("path", [""])[0]

    pymupdf_ok = False
    try:
        import fitz  # noqa
        pymupdf_ok = True
    except ImportError:
        pass

    warn_html = "" if pymupdf_ok else """
    <div class="card" style="border-color:var(--amber)">
      <p style="color:var(--amber)">pymupdf not installed. Run: <code>pip install pymupdf</code></p>
    </div>"""

    flash_html = f'<div class="flash" style="color:var(--lime);margin-bottom:12px">{flash}</div>' if flash else ""

    result_html = ""
    if result_md:
        result_html = f"""
    <div class="card">
      <h2>Result{"  —  " + result_path if result_path else ""}</h2>
      <pre style="max-height:500px;overflow-y:auto;font-size:12px;line-height:1.6;white-space:pre-wrap">{result_md[:8000]}</pre>
      {"<p style='color:var(--dim);font-size:12px'>... truncated for preview</p>" if len(result_md) > 8000 else ""}
    </div>"""

    body = f"""
    {warn_html}
    {flash_html}
    <div class="card">
      <h2>Convert PDF to Markdown</h2>
      <p style="color:var(--gray);font-size:13px;margin-bottom:16px">
        Offline conversion — no API calls required.<br>
        Font-size analysis detects headings, bold, italic, code blocks, and lists.
      </p>
      <form method="post" action="/pdf/convert">
        <div><label>PDF file path</label>
          <div style="display:flex;gap:6px">
            <input name="path" id="pdf-path-inp" type="text"
                   placeholder="/home/user/paper.pdf" style="flex:1">
            <button type="button" class="btn" title="Browse files"
                    onclick="openBrowse('pdf-path-inp','file')">&#128196;</button>
          </div></div>
        <div style="display:flex;gap:12px">
          <div style="flex:1"><label>Output path <span style="color:var(--dim)">(optional)</span></label>
            <div style="display:flex;gap:6px">
              <input name="output" id="pdf-output-inp" type="text"
                     placeholder="Leave blank = same folder as PDF" style="flex:1">
              <button type="button" class="btn" title="Browse folders"
                      onclick="openBrowse('pdf-output-inp','dir')">&#128193;</button>
            </div></div>
          <div style="flex:0 0 140px"><label>Pages <span style="color:var(--dim)">(optional)</span></label>
            <input name="pages" type="text" placeholder="e.g. 1-5"></div>
        </div>
        <div style="display:flex;align-items:center;gap:16px">
          <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
            <input type="checkbox" name="page_breaks"> Insert --- between pages
          </label>
          <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
            <input type="checkbox" name="preview_only"> Preview only (don't write file)
          </label>
        </div>
        <button class="btn btn-primary" type="submit">Convert</button>
      </form>
    </div>
    <div class="card">
      <h2>Batch Convert</h2>
      <form method="post" action="/pdf/batch">
        <div><label>Folder path</label>
          <div style="display:flex;gap:6px">
            <input name="folder" id="pdf-batch-inp" type="text"
                   placeholder="/home/user/papers/" style="flex:1">
            <button type="button" class="btn" title="Browse folders"
                    onclick="openBrowse('pdf-batch-inp','dir')">&#128193;</button>
          </div></div>
        <button class="btn btn-ghost" type="submit">Batch Convert All PDFs</button>
      </form>
    </div>
    <div class="card">
      <h2>CLI Usage</h2>
      <pre>greenboost pdf2md paper.pdf                 # paper.md (same folder)
greenboost pdf2md paper.pdf -o notes.md     # custom output
greenboost pdf2md paper.pdf --pages 1-5    # first 5 pages only
greenboost pdf2md preview paper.pdf         # preview first 60 lines
greenboost pdf2md batch ~/papers/           # convert all PDFs</pre>
    </div>
    {result_html}"""
    return _page("PDF to Markdown", body, "pdf")


def page_design(qs: dict) -> str:
    models_available = False
    try:
        from greenboost_cli.diffusion.models import MODELS, auto_select_model
        models_available = True
    except Exception:
        MODELS = {}

    assets_dir = GLOBAL_DIR / "design_assets"
    recent_sessions = []
    if assets_dir.exists():
        for d in sorted(assets_dir.iterdir(), reverse=True)[:6]:
            if d.is_dir():
                files = list(d.glob("*.png")) + list(d.glob("*.webp"))
                recent_sessions.append((d.name, files))

    models_html = ""
    if models_available:
        from greenboost_cli.diffusion.models import MODELS
        for key, cfg in MODELS.items():
            req_gb = " <span class='badge badge-warn'>GreenBoost req.</span>" if cfg.get("requires_greenboost") else ""
            speed_cls = "speed-fast" if "s/" in cfg.get("speed", "") else "speed-slow"
            loras = ", ".join(cfg.get("loras", {}).keys())
            models_html += f"""<div class="model-row">
              <div>
                <strong style="color:var(--cyan)">{key}</strong>{req_gb}
                <div style="color:var(--gray);font-size:11px;margin-top:3px">{cfg.get('text_encoder', '')} · {cfg.get('quantization', '').upper()}</div>
                {f'<div style="color:var(--dim);font-size:11px">LoRAs: {loras}</div>' if loras else ''}
              </div>
              <div style="text-align:right">
                <div class="{speed_cls}">{cfg.get('speed', '—')}</div>
                <div style="color:var(--dim);font-size:11px">~{cfg.get('vram_gb', '?')}GB VRAM</div>
              </div>
            </div>"""
    else:
        models_html = '<div class="empty">diffusers not installed — run setup to install.</div>'

    design_intel_avail = False
    try:
        from greenboost_cli.design.intelligence import is_available
        design_intel_avail = is_available()
    except Exception:
        pass

    asset_html = ""
    for session_name, files in recent_sessions:
        file_tags = "".join(f'<span class="tag">{f.name}</span>' for f in files[:4])
        asset_html += f"""<div class="card-sm">
          <div style="color:var(--cyan);margin-bottom:8px">{session_name}</div>
          <div>{file_tags}</div>
          <div style="margin-top:8px;color:var(--dim);font-size:11px">{len(files)} files</div>
        </div>"""

    gb = _gb_available()
    gb_badge = '<span class="badge badge-ok">GreenBoost</span>' if gb else '<span class="badge badge-other">Standard VRAM</span>'
    di_badge = '<span class="badge badge-ok">available</span>' if design_intel_avail else '<span class="badge badge-err">unavailable</span>'

    body = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px">
      <div class="card">
        <h2>Generate UI Assets</h2>
        <form method="post" action="/design/generate">
          <div><label>Product / design description</label>
            <input type="text" name="prompt" placeholder="fintech SaaS landing page" required></div>
          <div class="field-row">
            <div><label>Model</label>
              <select name="model_key">
                <option value="klein-fp8">klein-fp8 (4-8s, recommended)</option>
                <option value="flux1-nf4">flux1-nf4 (~3min)</option>
                <option value="flux2-nf4">flux2-nf4 (GreenBoost req.)</option>
              </select></div>
            <div><label>LoRA (optional)</label>
              <select name="lora">
                <option value="">none</option>
                <option value="arcane">arcane (klein only)</option>
                <option value="sts2">sts2 (klein only)</option>
                <option value="mtg">mtg (flux1 only)</option>
              </select></div>
          </div>
          <div class="field-row">
            <div><label>Style</label>
              <input type="text" name="style" value="glassmorphism" placeholder="glassmorphism"></div>
            <div><label>Colors</label>
              <input type="text" name="colors" value="deep blue and violet" placeholder="deep blue violet"></div>
          </div>
          <div class="field-row" style="gap:16px">
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;width:auto">
              <input type="checkbox" name="hero" checked> Hero</label>
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;width:auto">
              <input type="checkbox" name="mood" checked> Mood</label>
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;width:auto">
              <input type="checkbox" name="background"> Background</label>
            <label style="display:flex;align-items:center;gap:8px;cursor:pointer;width:auto">
              <input type="checkbox" name="illustration"> Illustration</label>
          </div>
          <button class="btn btn-primary btn-lg">Generate Assets</button>
        </form>
      </div>
      <div class="card">
        <h2>Design Intelligence {di_badge}</h2>
        <form method="get" action="/design/intelligence">
          <div><label>Product description</label>
            <input type="text" name="q" placeholder="healthcare SaaS for doctors" required></div>
          <button class="btn btn-ghost">Get Design System</button>
        </form>
      </div>
    </div>

    <h2>Available Models {gb_badge}</h2>
    <div class="card">{models_html}</div>

    <h2>Recent Design Sessions</h2>
    {"<div class='card-grid'>" + asset_html + "</div>" if asset_html else '<div class="empty">No design sessions yet.</div>'}
    """
    return _page("UI Design Pipeline", body, "design")


def page_design_intelligence(qs: dict) -> str:
    q = qs.get("q", [""])[0]
    if not q:
        return page_design(qs)
    try:
        from greenboost_cli.design.intelligence import generate_design_system, format_design_system
        ds = generate_design_system(q)
        md = format_design_system(ds)
    except Exception as e:
        return _page("Design Intelligence", f'<div class="alert alert-warn">Error: {e}</div>', "design")

    body = f"""
    <div class="card">
      <h2>Design System for: {q}</h2>
      <pre>{md}</pre>
    </div>
    <a href="/design" class="btn btn-ghost">Back</a>"""
    return _page("Design Intelligence", body, "design")


def page_guidelines(qs: dict) -> str:
    project = _get_project(qs)
    picker  = _project_picker(project, "/guidelines")

    flash = ""
    if "flash" in qs:
        msg = qs["flash"][0] if isinstance(qs["flash"], list) else qs["flash"]
        flash = f'<div class="alert alert-info">{msg}</div>'

    if not project:
        return _page("UI Guidelines", picker + flash, "guidelines")

    try:
        from greenboost_cli.memory.ui_guidelines import list_guidelines
        guidelines = list_guidelines(project)
    except Exception:
        guidelines = []

    def _row(g: dict) -> str:
        name    = g["name"]
        active  = g.get("active", True)
        src     = g.get("source") or "inline"
        added   = g.get("added_at", "")[:10]
        badge   = '<span class="badge badge-ok">active</span>' if active else '<span class="badge badge-err">disabled</span>'
        toggle_action = "disable" if active else "enable"
        toggle_label  = "Disable" if active else "Enable"
        return f"""<tr>
  <td><strong style="color:var(--cyan)">{name}</strong></td>
  <td class="mono">{src}</td>
  <td>{added}</td>
  <td>{badge}</td>
  <td>
    <form method="post" action="/guidelines/{toggle_action}" style="display:inline">
      <input type="hidden" name="project" value="{project}">
      <input type="hidden" name="name" value="{name}">
      <button class="btn btn-ghost btn-sm">{toggle_label}</button>
    </form>
    <a href="/guidelines/view?project={project}&name={name}" class="btn btn-ghost btn-sm" style="margin-left:4px">View</a>
    <form method="post" action="/guidelines/remove" style="display:inline;margin-left:4px"
          onsubmit="return confirm('Delete {name}?')">
      <input type="hidden" name="project" value="{project}">
      <input type="hidden" name="name" value="{name}">
      <button class="btn btn-danger btn-sm">Delete</button>
    </form>
  </td>
</tr>"""

    rows = "".join(_row(g) for g in guidelines) \
        or '<tr><td colspan="5" class="empty">No guidelines configured. Add one below.</td></tr>'

    # View single guideline
    view_section = ""
    view_name = qs.get("view_name", [None])[0]
    if view_name:
        try:
            from greenboost_cli.memory.ui_guidelines import get_guideline_content
            content = get_guideline_content(view_name, project)
        except Exception:
            content = ""
        if content:
            import html as _html
            view_section = f"""
<h2>Viewing: {_html.escape(view_name)}</h2>
<div class="card">
  <div class="pre-wrap"><pre style="max-height:500px;overflow-y:auto">{_html.escape(content)}</pre></div>
  <form method="post" action="/guidelines/update" style="margin-top:16px">
    <input type="hidden" name="project" value="{project}">
    <input type="hidden" name="name" value="{view_name}">
    <label>Edit content:</label>
    <textarea name="content" style="min-height:300px;font-family:monospace;font-size:12px">{_html.escape(content)}</textarea>
    <button class="btn btn-primary" style="margin-top:8px">Save Changes</button>
  </form>
</div>"""

    body = f"""{picker}{flash}
{view_section}
<h2>Active UI Guidelines</h2>
<div class="card">
  <table>
    <thead><tr>
      <th>Name</th><th>Source</th><th>Added</th><th>Status</th><th>Actions</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

<h2>Add Guideline from File</h2>
<div class="card">
  <form method="post" action="/guidelines/add-file">
    <input type="hidden" name="project" value="{project}">
    <label>File path (absolute, markdown or text)</label>
    <div class="field-row">
      <div style="display:flex;gap:6px;flex:1">
        <input type="text" name="path" id="gl-path-inp"
               placeholder="/path/to/ui-guidelines.md" style="flex:1">
        <button type="button" class="btn" title="Browse files"
                onclick="openBrowse('gl-path-inp','file')">&#128196;</button>
      </div>
      <input type="text" name="name" placeholder="Name (optional)">
      <button class="btn btn-primary">Add from File</button>
    </div>
    <p class="help">The file will be copied into the project guidelines store.</p>
  </form>
</div>

<h2>Create Inline Guideline</h2>
<div class="card">
  <form method="post" action="/guidelines/add-content">
    <input type="hidden" name="project" value="{project}">
    <label>Name</label>
    <input type="text" name="name" placeholder="e.g. glassmorphism-rules" style="margin-bottom:8px">
    <label>Content (Markdown)</label>
    <textarea name="content" placeholder="# UI Guidelines&#10;- Use glassmorphism style&#10;- Primary color: violet #7c6dff&#10;..."></textarea>
    <button class="btn btn-primary" style="margin-top:8px">Create Guideline</button>
  </form>
</div>

<h2>How It Works</h2>
<div class="card">
  <p style="color:var(--text-muted);font-size:13px">
    Active guidelines are injected into the AI system prompt on every turn.
    The AI will follow them when writing UI code, generating designs, or making stylistic decisions.
    You can have multiple active guidelines per project — they are all injected together.
  </p>
  <p style="color:var(--text-muted);font-size:13px;margin-top:8px">
    From the REPL: <code>/ui-guidelines add /path/to/file.md</code> &nbsp;|&nbsp;
    <code>/ui-guidelines list</code> &nbsp;|&nbsp; <code>/ui-guidelines show &lt;name&gt;</code>
  </p>
</div>"""

    return _page("UI Guidelines", body, "guidelines")


def page_tokens(qs: dict) -> str:
    project = _get_project(qs)
    picker = _project_picker(project, "/tokens")
    if not project:
        return _page("Tokens", picker, "tokens")

    pdir = GLOBAL_DIR / "projects" / project
    t = {"today_api": 0, "today_local": 0, "total_api": 0, "total_local": 0}
    sessions = []
    try:
        from greenboost_cli.memory.token_tracker import get_totals, _load as load_tokens
        t = get_totals(pdir)
        data = load_tokens(pdir)
        sessions = list(reversed(data.get("sessions", [])[-30:]))
    except Exception:
        pass

    def fmt(n: int) -> str:
        return f"{n:,}"

    rows = "".join(
        f'<tr><td class="mono">{s.get("date","")[:16]}</td>'
        f'<td>{fmt(s.get("api", 0))}</td>'
        f'<td>{fmt(s.get("local", 0))}</td></tr>'
        for s in sessions
    ) or '<tr><td colspan="3" class="empty">No sessions recorded yet.</td></tr>'

    chart_sessions = list(reversed(sessions))
    chart_labels = [s.get("date", "")[:10] for s in chart_sessions]
    chart_api = [s.get("api", 0) for s in chart_sessions]
    chart_local = [s.get("local", 0) for s in chart_sessions]

    chart_js = f"""
<script src="/static/chart.umd.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {{
  const ctx = document.getElementById('tokenChart');
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: {json.dumps(chart_labels)},
      datasets: [
        {{ label: 'API tokens', data: {json.dumps(chart_api)},
           borderColor: 'var(--cyan)', backgroundColor: 'rgba(167,139,250,0.08)',
           tension: 0.3, fill: true, pointRadius: 3 }},
        {{ label: 'Local tokens', data: {json.dumps(chart_local)},
           borderColor: '#22d3ee', backgroundColor: 'rgba(34,211,238,0.06)',
           tension: 0.3, fill: true, pointRadius: 3 }}
      ]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ family: 'Inter' }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#64748b', maxTicksLimit: 10 }}, grid: {{ color: '#1a2235' }} }},
        y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1a2235' }} }}
      }}
    }}
  }});
}});
</script>"""

    body = f"""{picker}
    <div class="stat-row">
      <div class="stat"><div class="num" data-target="{t['today_api']}">{fmt(t['today_api'])}</div><div class="label">API tokens today</div></div>
      <div class="stat"><div class="num" data-target="{t['today_local']}">{fmt(t['today_local'])}</div><div class="label">Local tokens today</div></div>
      <div class="stat"><div class="num" data-target="{t['total_api']}">{fmt(t['total_api'])}</div><div class="label">API total</div></div>
      <div class="stat"><div class="num" data-target="{t['total_local']}">{fmt(t['total_local'])}</div><div class="label">Local total</div></div>
    </div>
    <div class="card" style="height:260px;padding:16px">
      <canvas id="tokenChart"></canvas>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Date</th><th>API tokens</th><th>Local tokens</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    {chart_js}"""
    return _page(f"Tokens — {project}", body, "tokens")


def page_system() -> str:
    gpu = _gpu_status()

    gb_tier_html = ""
    try:
        from greenboost_cli.greenboost.monitor import get_tier_stats
        stats = get_tier_stats()
        if stats:
            tier = stats.get("tier_name", "unknown")
            freq = stats.get("boost_freq_mhz", "—")
            temp = stats.get("temp_c", "—")
            power = stats.get("power_w", "—")
            gb_tier_html = f"""
            <div class="kv"><span class="key">Tier</span><span class="val">{tier}</span></div>
            <div class="kv"><span class="key">Boost freq</span><span class="val">{freq} MHz</span></div>
            <div class="kv"><span class="key">Temperature</span><span class="val">{temp} C</span></div>
            <div class="kv"><span class="key">Power</span><span class="val">{power} W</span></div>"""
        else:
            gb_tier_html = '<div class="empty">GreenBoost hardware not detected.</div>'
    except Exception:
        gb_tier_html = '<div class="empty">GreenBoost monitor unavailable.</div>'

    py_packages = {}
    for pkg in ["torch", "sentence_transformers", "diffusers", "fitz", "yaml", "numpy"]:
        try:
            mod = __import__(pkg)
            py_packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            py_packages[pkg] = None

    def _pkg_row(pkg, ver):
        badge = f'<span class="badge badge-ok">{ver}</span>' if ver else '<span class="badge badge-err">missing</span>'
        hint = f"pip install {pkg}" if not ver else ""
        return (
            f'<tr><td class="mono">{pkg}</td>'
            f'<td>{badge}</td>'
            f'<td class="mono" style="color:var(--dim)">{hint}</td></tr>'
        )

    pkg_rows = "".join(_pkg_row(pkg, ver) for pkg, ver in py_packages.items())

    gpu_rows = ""
    if gpu.get("available"):
        pct = int(gpu["mem_used"] / max(gpu["mem_total"], 1) * 100)
        col = "var(--lime)" if pct < 75 else ("var(--amber)" if pct < 90 else "var(--red)")
        gpu_rows = f"""
        <div class="kv"><span class="key">GPU Name</span><span class="val">{gpu['name']}</span></div>
        <div class="kv"><span class="key">VRAM Used</span>
          <span class="val" style="color:{col}">{gpu['mem_used']} MB / {gpu['mem_total']} MB ({pct}%)</span>
        </div>
        <div class="vram-bar-wrap"><div class="vram-bar" style="width:{pct}%;background:{col}"></div></div>
        <div class="kv" style="margin-top:8px"><span class="key">Temperature</span><span class="val">{gpu['temp']}C</span></div>"""
    else:
        gpu_rows = '<div class="empty">No NVIDIA GPU detected.</div>'

    body = f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div>
        <div class="card">
          <h2>GPU Status</h2>
          {gpu_rows}
        </div>
        <div class="card">
          <h2>GreenBoost Tier</h2>
          {gb_tier_html}
        </div>
        <div class="card">
          <h2>Services</h2>
          <div class="kv"><span class="key">Dashboard</span>
            <span class="val"><span class="badge badge-ok">running</span> :{PORT}</span></div>
          <div class="kv"><span class="key">GB_HOME</span>
            <span class="val" style="font-family:monospace;font-size:12px">{GLOBAL_DIR}</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Python Packages</h2>
        <table>
          <thead><tr><th>Package</th><th>Version</th><th>Action</th></tr></thead>
          <tbody>{pkg_rows}</tbody>
        </table>
      </div>
    </div>"""
    return _page("System", body, "system")


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _run_gb_cmd(cmd: list[str], timeout: int = 12) -> tuple[str, float]:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = _strip_ansi(r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        output = f"[command timed out after {timeout}s]"
    except Exception as exc:
        output = f"[error: {exc}]"
    return output, round(time.time() - t0, 2)


_NVTX_LOG = Path("/run/greenboost/nvtx_events.log")
_SHIM_STATS = Path("/run/greenboost/shim_stats")
_METRICS_JSON = Path("/run/greenboost/metrics.json")
_PHASE_FILE = Path("/run/greenboost/phase")
_GB_CLUSTER_CONF = Path("/etc/greenboost/cluster.conf")
_DIFFUSER_VITALS = Path("/run/greenboost/diffuser_vitals.json")

# ── Feeder GPU vitals (SSH-collected, cached) ─────────────────────────────────
_feeder_gpu_cache: dict = {}  # ip → {"ts": float, "data": dict}
_FEEDER_GPU_TTL = 4.0


def _safe_int(s: str) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return 0


def _safe_float(s: str) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return 0.0


def _parse_cluster_conf() -> list[dict]:
    """Return list of {ip, port, host, ssh_user} from /etc/greenboost/cluster.conf."""
    feeders: list[dict] = []
    if not _GB_CLUSTER_CONF.exists():
        return feeders
    try:
        for line in _GB_CLUSTER_CONF.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            addr = parts[0]
            host = parts[1] if len(parts) > 1 else addr
            user = parts[2] if len(parts) > 2 else "root"
            ip, _, port = addr.partition(":")
            feeders.append({"ip": ip, "port": port or "9740", "host": host, "ssh_user": user})
    except Exception:
        pass
    return feeders


def _fetch_feeder_gpu(ip: str, ssh_user: str) -> dict:
    """SSH to a feeder, collect nvidia-smi + shim phase + netd status. Cached 4 s."""
    import time as _time
    cached = _feeder_gpu_cache.get(ip, {})
    if cached and (_time.time() - cached.get("ts", 0)) < _FEEDER_GPU_TTL:
        return cached["data"]

    result: dict = {
        "ip": ip, "reachable": False, "error": None,
        "gpu_name": "", "vram_used_mb": 0, "vram_total_mb": 0,
        "gpu_util_pct": 0, "temp_c": 0, "power_w": 0.0,
        "netd_running": False, "phase": "UNKNOWN",
        "local_t1_alloc_mb": 0, "kernel_dispatch_count": 0,
    }
    remote_cmd = (
        "nvidia-smi --query-gpu=name,memory.used,memory.total,"
        "utilization.gpu,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null; "
        "echo '---NETD---'; ss -tlnp 2>/dev/null | grep -c ':9740' || echo 0; "
        "echo '---SHIM---'; "
        "grep -E '^phase=|^local_t1_alloc_mb=|^kernel_dispatch_count=' "
        "/run/greenboost/shim_stats 2>/dev/null || true"
    )
    try:
        raw = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
             "-o", "StrictHostKeyChecking=no", f"{ssh_user}@{ip}", remote_cmd],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode(errors="replace").strip()

        smi_part, _, rest   = raw.partition("---NETD---")
        netd_part, _, shim_part = rest.partition("---SHIM---")

        result["reachable"] = True
        for line in smi_part.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                result["gpu_name"]     = parts[0]
                result["vram_used_mb"] = _safe_int(parts[1])
                result["vram_total_mb"] = _safe_int(parts[2])
                result["gpu_util_pct"] = _safe_int(parts[3])
                result["temp_c"]       = _safe_int(parts[4])
                result["power_w"]      = _safe_float(parts[5])
                break

        netd_n = netd_part.strip()
        result["netd_running"] = netd_n.strip() not in ("", "0")

        for line in shim_part.strip().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                if k == "phase":
                    result["phase"] = v.strip()
                elif k == "local_t1_alloc_mb":
                    result["local_t1_alloc_mb"] = _safe_int(v)
                elif k == "kernel_dispatch_count":
                    result["kernel_dispatch_count"] = _safe_int(v)

    except subprocess.TimeoutExpired:
        result["error"] = "ssh_timeout"
    except Exception as exc:
        result["error"] = str(exc)[:80]

    _feeder_gpu_cache[ip] = {"ts": _time.time(), "data": result}
    return result

# Colour-coding for NVTX event types
_NVTX_COLS = {
    "ALLOC_T1": "ev-alloc", "ALLOC_T2": "ev-alloc", "ALLOC_T3": "ev-alloc",
    "EVICT": "ev-evict", "PHASE_STEADY": "ev-phase", "PHASE_OOM": "ev-oom",
    "PHASE_": "ev-phase", "SHIM_INIT": "ev-shim", "KV_COMPRESS": "ev-kv",
    "RESET": "ev-reset",
}

def _nvtx_css(ev_type: str) -> str:
    for prefix, cls in _NVTX_COLS.items():
        if ev_type.startswith(prefix):
            return cls
    return ""


def _read_nvtx(n: int = 120) -> list[dict]:
    """Read last n NVTX events from /run/greenboost/nvtx_events.log (instant, no subprocess)."""
    if not _NVTX_LOG.exists():
        return []
    try:
        lines = _NVTX_LOG.read_text(errors="replace").splitlines()
        rows = []
        for line in reversed(lines[-n * 2:]):
            parts = line.split(None, 5)
            if len(parts) < 4:
                continue
            rows.append({
                "ts_ms": parts[0],
                "event": parts[1],
                "tier": parts[2],
                "size": parts[3],
                "detail": parts[5] if len(parts) > 5 else "",
                "css": _nvtx_css(parts[1]),
            })
            if len(rows) >= n:
                break
        return rows
    except Exception:
        return []


def _read_shim_stats() -> dict:
    """Read /run/greenboost/shim_stats (key=value, instant)."""
    result: dict = {}
    if not _SHIM_STATS.exists():
        return result
    try:
        for line in _SHIM_STATS.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                try:
                    result[k.strip()] = int(v.strip())
                except ValueError:
                    result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def _read_phase() -> dict:
    result = {"phase": "UNKNOWN", "idle_ms": 0}
    if not _PHASE_FILE.exists():
        return result
    try:
        for line in _PHASE_FILE.read_text().splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                result[k.strip()] = v.strip()
    except Exception:
        pass
    return result


def _gb_logs_fast() -> dict:
    """journalctl with 3s timeout — fast enough for live polling."""
    result: dict = {"kernel": [], "service": [], "apparmor": []}
    try:
        out = subprocess.check_output(
            ["journalctl", "-k", "--grep=greenboost", "--no-pager", "-n", "60",
             "--output=short-iso-precise"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode(errors="replace").strip()
        for line in out.splitlines():
            if not line.strip() or line.startswith("--"):
                continue
            if "apparmor" in line.lower() or "audit" in line.lower():
                result["apparmor"].append(line)
            else:
                result["kernel"].append(line)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["journalctl", "-u", "greenboost*", "--no-pager", "-n", "40",
             "--output=short-iso"],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode(errors="replace").strip()
        for line in out.splitlines():
            if line.strip() and not line.startswith("--"):
                result["service"].append(line)
    except Exception:
        pass
    return result


def _get_diffuser_vitals() -> dict:
    """Collect HuggingFace diffusers pipeline vitals from multiple sources."""
    vitals: dict = {
        "active": False, "pipeline": "", "model": "", "state": "idle",
        "vram_alloc_mb": 0, "vram_reserved_mb": 0, "vram_peak_mb": 0,
        "t2_alloc_mb": 0, "gen_step": 0, "gen_total_steps": 0,
        "last_gen_s": 0.0, "last_prompt": "", "last_image": "",
        "pid": 0, "error": "",
    }
    # 1. File-based vitals written by the pipeline
    if _DIFFUSER_VITALS.exists():
        try:
            import json as _json
            data = _json.loads(_DIFFUSER_VITALS.read_text())
            import time as _time
            age = _time.time() - data.get("ts", 0)
            if age < 120:
                vitals.update(data)
                vitals["active"] = age < 30 or data.get("state") == "loading"
                vitals["stale_s"] = int(age)
        except Exception:
            pass
    # 2. Scan GPU processes for diffuser patterns (python + flux/diffuse)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=4,
        ).strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            pid, pname, mem_mib = parts[0], parts[1], _safe_int(parts[2])
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except Exception:
                cmdline = ""
            is_diffuser = any(kw in cmdline.lower() for kw in
                              ("flux", "diffus", "stable_diffus", "gen_art", "gen_manual"))
            if is_diffuser:
                vitals["active"] = True
                vitals["pid"] = int(pid)
                vitals["vram_alloc_mb"] = max(vitals["vram_alloc_mb"], mem_mib)
                if not vitals["pipeline"]:
                    if "flux" in cmdline.lower():
                        vitals["pipeline"] = "FLUX"
                    else:
                        vitals["pipeline"] = "SD"
                break
    except Exception:
        pass
    # 3. Check T2 usage attributed to diffuser pid from shim_stats
    if vitals["pid"]:
        try:
            ss = _read_shim_stats()
            vitals["t2_alloc_mb"] = _safe_int(ss.get("tier_t2_local_cur_mb", 0))
        except Exception:
            pass
    return vitals


def page_factory(qs: dict | None = None) -> str:
    qs = qs or {}
    flash = qs.get("flash", [""])[0]
    flash_html = f'<div class="flash ok">{flash}</div>' if flash else ""

    body = f"""
{flash_html}
<div class="section-header"><h2>AI Factory</h2></div>

<div class="card" style="margin-bottom:16px">
  <div class="card-header">
    <span class="card-title">Status</span>
    <button class="btn btn-sm btn-primary" onclick="factoryAction('start')" id="btn-start">Start</button>
    <button class="btn btn-sm" onclick="factoryAction('stop')" id="btn-stop" style="margin-left:4px">Stop</button>
  </div>
  <div id="factory-status-area" style="padding:12px 0;font-family:monospace;font-size:13px;color:var(--dim)">Loading…</div>
</div>

<div class="card" style="margin-bottom:16px">
  <div class="card-header"><span class="card-title">Submit Task</span></div>
  <div style="padding:8px 0;display:flex;gap:8px">
    <input id="factory-prompt" type="text" placeholder="Describe the task…"
           style="flex:1;font-family:monospace;font-size:13px"
           onkeydown="if(event.key==='Enter')submitFactory()">
    <input id="factory-priority" type="number" value="10" min="1" max="20"
           style="width:64px;text-align:center" title="Priority (1=high)">
    <button class="btn btn-primary" onclick="submitFactory()">Submit</button>
  </div>
  <div id="factory-submit-msg" style="font-size:12px;color:var(--ok);min-height:18px"></div>
</div>

<div class="card">
  <div class="card-header">
    <span class="card-title">Recent Completions</span>
    <button class="btn btn-sm" onclick="pollFactory()" style="margin-left:auto">Refresh</button>
  </div>
  <div id="factory-history-area" style="padding:8px 0"></div>
</div>

<script>
function factoryAction(action) {{
  fetch('/api/factory/' + action).then(r => r.json()).then(pollFactory).catch(console.error);
}}

function submitFactory() {{
  var prompt = document.getElementById('factory-prompt').value.trim();
  var priority = document.getElementById('factory-priority').value || '10';
  if (!prompt) return;
  fetch('/api/factory/submit?prompt=' + encodeURIComponent(prompt) + '&priority=' + priority)
    .then(r => r.json())
    .then(function(d) {{
      document.getElementById('factory-submit-msg').textContent = d.task_id ? 'Submitted: ' + d.task_id : (d.error || 'OK');
      document.getElementById('factory-prompt').value = '';
      setTimeout(pollFactory, 800);
    }}).catch(console.error);
}}

function pollFactory() {{
  fetch('/api/factory/status').then(r => r.json()).then(function(d) {{
    var state = d.active ? '<span style="color:var(--ok)">● RUNNING</span>' : '<span style="color:var(--dim)">○ stopped</span>';
    var lines = [
      state + '  &nbsp;|&nbsp;  Queue: ' + d.queue_depth + '  &nbsp;|&nbsp;  GPU: ' + (d.gpu_ratio * 100).toFixed(1) + '%',
    ];
    if (d.agents && Object.keys(d.agents).length) {{
      lines.push('<br><b>Agents:</b>');
      Object.entries(d.agents).forEach(function([name, a]) {{
        var st = a.paused ? '<em>paused</em>' : (a.current_task || 'idle');
        lines.push('&nbsp;&nbsp;<b>' + name + '</b> &mdash; ' + st
          + '  ok=' + a.total_tasks + ' fail=' + a.failed_tasks
          + '  <span style="color:var(--dim)">' + a.model + '</span>');
      }});
    }} else {{
      lines.push('<br><span style="color:var(--dim)">No agents — use /factory start or /factory agents add</span>');
    }}
    document.getElementById('factory-status-area').innerHTML = lines.join('<br>');

    var hist = '';
    if (d.recent && d.recent.length) {{
      hist = '<table style="width:100%;font-size:12px;border-collapse:collapse">'
           + '<tr style="color:var(--dim)"><th style="text-align:left;padding:4px 8px">Status</th>'
           + '<th style="text-align:left;padding:4px 8px">Agent</th>'
           + '<th style="text-align:left;padding:4px 8px">Task</th></tr>';
      d.recent.slice(0,10).forEach(function(r) {{
        var mark = r.success ? '<span style="color:var(--ok)">✓</span>' : '<span style="color:var(--err)">✗</span>';
        hist += '<tr><td style="padding:3px 8px">' + mark + '</td>'
              + '<td style="padding:3px 8px;color:var(--dim)">' + (r.agent_name||'?') + '</td>'
              + '<td style="padding:3px 8px">' + ((r.prompt||'').substring(0,80)) + '</td></tr>';
      }});
      hist += '</table>';
    }} else {{
      hist = '<span style="color:var(--dim);font-size:12px">No completed tasks yet.</span>';
    }}
    document.getElementById('factory-history-area').innerHTML = hist;
  }}).catch(function() {{
    document.getElementById('factory-status-area').textContent = 'Factory not running — start it with /factory start';
  }});
}}
pollFactory();
setInterval(pollFactory, 5000);
</script>
"""
    return _page("AI Factory", body, "factory")


def page_greenboost(qs: dict | None = None) -> str:
    qs = qs or {}
    flash = qs.get("flash", [""])[0]

    # ── Server-side initial data ───────────────────────────────────────────
    st: dict = {}
    try:
        from greenboost_cli.greenboost.monitor import get_monitor
        st = get_monitor().refresh().as_dict()
    except Exception:
        pass

    ss = _read_shim_stats()
    ph = _read_phase()
    nvtx_rows = _read_nvtx(60)

    def _pct(used, total):
        return min(100, int(used / total * 100)) if total else 0

    t1_gb      = round(st.get("vram_physical_mb", 0) / 1024, 1)
    t2_used    = round(st.get("ram_allocated_mb", 0) / 1024, 1)
    t2_total   = round(st.get("ram_pool_mb", 0) / 1024, 1)
    t3_used    = round(st.get("nvme_swap_used_mb", 0) / 1024, 1)
    t3_total   = round(st.get("nvme_swap_total_mb", 0) / 1024, 1)
    t2_pct     = _pct(st.get("ram_allocated_mb", 0), st.get("ram_pool_mb", 1))
    t3_pct     = _pct(st.get("nvme_swap_used_mb", 0), st.get("nvme_swap_total_mb", 1))
    kv_used    = st.get("kv_used_mb", 0)
    kv_reserve = st.get("kv_reserve_mb", 2048)
    kv_comp    = st.get("kv_compressed_mb", 0)
    tq_bits    = st.get("kv_compression_bits", 0)
    oom        = st.get("oom_active", False)
    active_bufs = st.get("active_buffers", 0)
    version    = st.get("version", "2.9")
    gb_loaded  = st.get("loaded", False)
    t2_press   = st.get("t2_pressure", 0)
    t3_press   = st.get("swap_pressure", 0)
    gpu_name   = st.get("gpu_name", "—")
    combined_gb = round(st.get("total_combined_mb", 0) / 1024, 1) or (t1_gb + t2_total + t3_total)

    _pcol = {0: "var(--lime)", 1: "var(--amber)", 2: "var(--red)"}
    _plab = {0: "ok", 1: "warn", 2: "critical"}
    _pbcls_map = {0: "badge badge-ok", 1: "badge badge-warn", 2: "badge badge-err"}

    def _pbcls(p: int) -> str:
        return _pbcls_map.get(p, "badge badge-ok")

    t2_col  = _pcol.get(t2_press, "var(--lime)")
    t3_col  = _pcol.get(t3_press, "var(--lime)")

    phase_str = ph.get("phase", ss.get("phase", "UNKNOWN"))
    phase_col = {"INFERENCE": "var(--lime)", "STEADY": "var(--cyan)", "OOM": "var(--red)",
                 "INIT": "var(--dim)", "RESET": "var(--amber)"}.get(phase_str, "var(--dim)")

    virtual_vram_gb = round(ss.get("virtual_vram_mb", 0) / 1024, 1) or (t1_gb + t2_total)
    vram_headroom   = ss.get("vram_headroom_mb", 0)
    path_a_cnt = ss.get("path_a_count", 0)
    path_b_cnt = ss.get("path_b_count", 0)
    path_c_cnt = ss.get("path_c_count", 0)
    h2d_mb     = ss.get("h2d_mb", 0)
    d2h_mb     = ss.get("d2h_mb", 0)
    k_dispatch = ss.get("kernel_dispatch_count", 0)
    frag_pct   = ss.get("t2_pool_frag_pct", 0)
    dedup_hits = ss.get("kv_dedup_hits", 0)
    evict_cnt  = ss.get("cold_epoch_evict_count", 0)
    t1_peak    = ss.get("tier_t1_local_peak_mb", 0) + ss.get("tier_t1_feeder_peak_mb", 0)
    t2_peak    = ss.get("tier_t2_local_peak_mb", 0) + ss.get("tier_t2_feeder_peak_mb", 0)
    t3_peak    = ss.get("tier_t3_local_peak_mb", 0) + ss.get("tier_t3_feeder_peak_mb", 0)
    feeder_t1_cur  = ss.get("tier_t1_feeder_cur_mb",  0)
    feeder_t1_peak = ss.get("tier_t1_feeder_peak_mb", 0)
    feeder_t2_cur  = ss.get("tier_t2_feeder_cur_mb",  0)
    feeder_t2_peak = ss.get("tier_t2_feeder_peak_mb", 0)
    feeder_t3_cur  = ss.get("tier_t3_feeder_cur_mb",  0)
    feeder_t3_peak = ss.get("tier_t3_feeder_peak_mb", 0)
    remote_allocs  = ss.get("remote_alloc_count",     0)
    feeder_active  = (feeder_t1_cur + feeder_t2_cur) > 0
    feeder_status_col = "var(--lime)" if feeder_active else ("var(--amber)" if remote_allocs > 0 else "var(--dim)")
    feeder_status_txt = "active" if feeder_active else ("allocated" if remote_allocs > 0 else "idle")

    tq_label = f"{tq_bits}-bit" if tq_bits else "off"
    loaded_badge_cls = "badge-ok" if gb_loaded else "badge-err"
    loaded_badge_txt = "loaded" if gb_loaded else "not loaded"

    # Alert classification
    alerts: list[tuple[str, str]] = []
    if oom:
        alerts.append(("crit", "OOM guard ACTIVE — inference blocked"))
    if t2_press == 2:
        alerts.append(("crit", f"T2 RAM pressure: critical ({t2_used}/{t2_total} GB)"))
    elif t2_press == 1:
        alerts.append(("warn", f"T2 RAM pressure: warn ({t2_used}/{t2_total} GB)"))
    if t3_press == 2:
        alerts.append(("crit", f"T3 NVMe pressure: critical ({t3_used}/{t3_total} GB)"))
    elif t3_press == 1:
        alerts.append(("warn", f"T3 NVMe pressure: warn ({t3_used}/{t3_total} GB)"))
    if not gb_loaded:
        alerts.append(("warn", "GreenBoost kernel module not loaded"))
    if phase_str == "OOM":
        alerts.append(("crit", "Phase: OOM — system needs reset"))

    alert_cls = "gb-alert-crit" if any(a[0] == "crit" for a in alerts) else ("gb-alert-warn" if alerts else "gb-alert-ok")
    if not alerts:
        alert_items = '<span style="color:var(--lime);font-weight:600">✓ All systems nominal</span>'
    else:
        def _alert_span(sev: str, msg: str) -> str:
            col = "var(--red)" if sev == "crit" else "var(--amber)"
            icon = "✗" if sev == "crit" else "⚠"
            return f'<span style="color:{col}">{icon} {msg}</span>'
        alert_items = " &nbsp;\xb7&nbsp; ".join(_alert_span(s, m) for s, m in alerts)

    frag_col = "var(--red)" if frag_pct > 30 else ("var(--amber)" if frag_pct > 10 else "var(--lime)")
    hr_col = "var(--lime)" if vram_headroom > 500 else "var(--amber)"
    oom_badge_cls = "badge-err" if oom else "badge-ok"

    # NVTX rows HTML
    def _nvtx_row(ev: dict) -> str:
        import datetime
        ts_ms_str = ev["ts_ms"]
        try:
            ts_sec = int(ts_ms_str) // 1000
            ts_fmt = datetime.datetime.fromtimestamp(ts_sec).strftime("%H:%M:%S")
        except Exception:
            ts_fmt = ts_ms_str[:12]
        detail_esc = ev["detail"][:80].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (f'<tr><td style="color:var(--dim)">{ts_fmt}</td>'
                f'<td class="{ev["css"]}">{ev["event"]}</td>'
                f'<td style="color:var(--text-muted)">{ev["tier"]}</td>'
                f'<td style="color:var(--cyan)">{ev["size"]}</td>'
                f'<td style="color:var(--text-muted);max-width:300px;overflow:hidden;text-overflow:ellipsis">{detail_esc}</td></tr>')

    nvtx_html = "".join(_nvtx_row(r) for r in nvtx_rows) if nvtx_rows else '<tr><td colspan="5" class="empty">No NVTX events</td></tr>'

    flash_html = f'<div class="flash">{flash}</div>' if flash else ""

    body = f"""{flash_html}

<!-- ── Alert Summary ── -->
<div class="gb-alert-bar {alert_cls}" id="gb-alert-bar">
  <span style="font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;flex-shrink:0">Status</span>
  <span id="gb-alert-items" style="font-size:13px">{alert_items}</span>
  <span style="margin-left:auto;color:var(--dim);font-size:11px;display:flex;align-items:center;gap:5px">
    <span id="gb-live-dot" style="width:6px;height:6px;border-radius:50%;background:var(--lime);display:inline-block;animation:pulse 2s infinite"></span>
    live &middot; <span id="gb-tick">2s</span>
  </span>
</div>

<!-- ── Action Summary ── -->
<div class="card" id="gb-summary-card" style="margin-bottom:16px;border-color:rgba(255,255,255,0.07)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <h2 style="margin:0;display:flex;align-items:center;gap:8px">
      <span style="width:6px;height:6px;border-radius:50%;background:#00C4B4;display:inline-block"></span>
      Action Items
      <span style="font-size:10px;color:var(--dim);font-weight:400;text-transform:none;letter-spacing:0">· updates every 10s</span>
    </h2>
    <span id="gb-summary-ts" style="font-size:11px;color:var(--dim)"></span>
  </div>
  <div id="gb-summary-body">
    <div style="color:var(--dim);font-size:13px">Loading…</div>
  </div>
</div>

<!-- ── Big Metric Row ── -->
<div class="stat-row" style="margin-bottom:20px">
  <div class="stat">
    <div class="num" id="gb-combined-val">{combined_gb}</div>
    <div class="label">GB Combined Pool</div>
    <div style="font-size:11px;color:var(--dim);margin-top:4px">T1+T2+T3 virtual</div>
  </div>
  <div class="stat">
    <div class="num phase-badge" id="gb-phase-val" style="font-size:14px;color:{phase_col}">{phase_str}</div>
    <div class="label">Active Phase</div>
    <div style="font-size:11px;color:var(--dim);margin-top:4px" id="gb-version-sub">v{version}</div>
  </div>
  <div class="stat">
    <div class="num" id="gb-bufs-val" style="color:var(--blue-light)">{active_bufs}</div>
    <div class="label">Active Buffers</div>
    <div style="font-size:11px;color:var(--dim);margin-top:4px" id="gb-gpu-sub">{gpu_name}</div>
  </div>
  <div class="stat">
    <div class="num" id="gb-kv-val" style="color:var(--cyan)">{kv_used}</div>
    <div class="label">KV Cache MB</div>
    <div style="font-size:11px;color:var(--dim);margin-top:4px">TurboQuant: <span id="gb-tq-sub">{tq_label}</span></div>
  </div>
  <div class="stat">
    <div class="num" id="gb-vram-headroom" style="color:{hr_col}">{vram_headroom}</div>
    <div class="label">VRAM Headroom MB</div>
    <div style="font-size:11px;color:var(--dim);margin-top:4px">virtual: {virtual_vram_gb} GB</div>
  </div>
</div>

<!-- ── Memory Tiers + Charts ── -->
<div style="display:grid;grid-template-columns:3fr 2fr;gap:16px;margin-bottom:16px">
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <h2 style="margin:0">Memory Tiers</h2>
      <span class="badge {loaded_badge_cls}">{loaded_badge_txt}</span>
    </div>

    <div style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:12px;font-weight:700;color:var(--lime);font-family:'JetBrains Mono',monospace">T1 &middot; GPU VRAM</span>
        <span id="gb-t1-lbl" style="font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace">{t1_gb} / {t1_gb} GB (100%)</span>
      </div>
      <div class="vram-bar-wrap" style="height:8px">
        <div id="gb-bar-t1" class="vram-bar" style="width:100%;background:var(--lime)"></div>
      </div>
      <div class="tier-chart-wrap"><canvas id="chart-t1"></canvas></div>
    </div>

    <div style="margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:12px;font-weight:700;color:var(--cyan);font-family:'JetBrains Mono',monospace">T2 &middot; System RAM Pool</span>
        <span id="gb-t2-lbl" style="font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace">{t2_used} / {t2_total} GB ({t2_pct}%)</span>
      </div>
      <div class="vram-bar-wrap" style="height:8px">
        <div id="gb-bar-t2" class="vram-bar" style="width:{t2_pct}%;background:{t2_col}"></div>
      </div>
      <div class="tier-chart-wrap"><canvas id="chart-t2"></canvas></div>
    </div>

    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <span style="font-size:12px;font-weight:700;color:var(--blue-light);font-family:'JetBrains Mono',monospace">T3 &middot; NVMe Swap</span>
        <span id="gb-t3-lbl" style="font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace">{t3_used} / {t3_total} GB ({t3_pct}%)</span>
      </div>
      <div class="vram-bar-wrap" style="height:8px">
        <div id="gb-bar-t3" class="vram-bar" style="width:{t3_pct}%;background:{t3_col}"></div>
      </div>
      <div class="tier-chart-wrap"><canvas id="chart-t3"></canvas></div>
    </div>

    <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="kv" style="margin:0"><span class="key">OOM Guard</span>
        <span class="val"><span id="gb-oom-badge" class="badge {oom_badge_cls}">{"ACTIVE" if oom else "clear"}</span></span></div>
      <div class="kv" style="margin:0"><span class="key">KV Reserve</span><span class="val" id="gb-kv-rsv">{kv_reserve} MB</span></div>
      <div class="kv" style="margin:0"><span class="key">KV Compressed</span><span class="val" id="gb-kv-comp">{kv_comp} MB</span></div>
      <div class="kv" style="margin:0"><span class="key">T2 Pressure</span>
        <span class="val"><span id="gb-t2-press-badge" class="{_pbcls(t2_press)}">{_plab.get(t2_press, "ok")}</span></span></div>
    </div>
  </div>

  <div>
    <!-- Memory Flow Diagram -->
    <div class="card" style="margin-bottom:14px">
      <h2>Memory Flow</h2>
      <div style="display:flex;flex-direction:column;gap:8px">
        <div id="flow-t1-box" style="border:2px solid var(--lime);border-radius:8px;padding:10px 14px">
          <div style="font-size:10px;text-transform:uppercase;color:var(--dim)">T1 &middot; GPU VRAM</div>
          <div id="flow-t1" style="font-size:20px;font-weight:700;color:var(--lime)">{t1_gb} GB</div>
          <div style="font-size:11px;color:var(--dim)">{gpu_name}</div>
        </div>
        <div style="text-align:center;font-size:11px;color:var(--dim)">&darr; KV spill / eviction</div>
        <div id="flow-t2-box" style="border:2px solid {t2_col};border-radius:8px;padding:10px 14px">
          <div style="font-size:10px;text-transform:uppercase;color:var(--dim)">T2 &middot; RAM Pool</div>
          <div id="flow-t2" style="font-size:20px;font-weight:700;color:{t2_col}">{t2_used}/{t2_total} GB</div>
          <div class="vram-bar-wrap" style="margin:4px 0">
            <div id="flow-t2-bar" class="vram-bar" style="width:{t2_pct}%;background:{t2_col}"></div>
          </div>
        </div>
        <div style="text-align:center;font-size:11px;color:var(--dim)">&darr; cold eviction</div>
        <div id="flow-t3-box" style="border:2px solid {t3_col};border-radius:8px;padding:10px 14px">
          <div style="font-size:10px;text-transform:uppercase;color:var(--dim)">T3 &middot; NVMe Swap</div>
          <div id="flow-t3" style="font-size:20px;font-weight:700;color:{t3_col}">{t3_used}/{t3_total} GB</div>
          <div class="vram-bar-wrap" style="margin:4px 0">
            <div id="flow-t3-bar" class="vram-bar" style="width:{t3_pct}%;background:{t3_col}"></div>
          </div>
        </div>
      </div>
    </div>
    <!-- KV Cache details -->
    <div class="card">
      <h2>KV Cache</h2>
      <div class="kv"><span class="key">Used</span><span class="val" id="kv-used-val" style="color:var(--cyan)">{kv_used} MB</span></div>
      <div class="kv"><span class="key">Reserve</span><span class="val">{kv_reserve} MB</span></div>
      <div class="kv"><span class="key">T2 Spill</span><span class="val" id="kv-t2-val">{st.get("kv_t2_mb", 0)} MB</span></div>
      <div class="kv"><span class="key">Compressed</span><span class="val" id="kv-comp-val">{kv_comp} MB</span></div>
      <div class="kv"><span class="key">TurboQuant</span><span class="val" id="kv-tq-val">{tq_label}</span></div>
      <div class="kv"><span class="key">Dedup hits</span><span class="val" style="color:var(--lime)">{dedup_hits}</span></div>
      <div class="kv"><span class="key">Internal frag</span><span class="val">{ss.get("kv_internal_frag_mb", 0)} MB</span></div>
    </div>
  </div>
</div>

<!-- ── Shim Stats ── -->
<div class="card" style="margin-bottom:16px">
  <h2>Shim Statistics <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; /run/greenboost/shim_stats</span></h2>
  <div class="gb-metric-grid">
    <div class="gb-metric">
      <div class="gm-label">Active Path</div>
      <div class="gm-val" id="gm-active-path" style="font-size:16px;color:var(--cyan)">{ss.get("active_path", "—")}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Path A allocs</div>
      <div class="gm-val" id="gm-path-a" style="color:var(--lime)">{path_a_cnt:,}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Path B allocs</div>
      <div class="gm-val" id="gm-path-b" style="color:var(--amber)">{path_b_cnt:,}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Path C allocs</div>
      <div class="gm-val" id="gm-path-c" style="color:var(--red)">{path_c_cnt:,}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">H&rarr;D transfers</div>
      <div class="gm-val" id="gm-h2d">{h2d_mb:,} <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">D&rarr;H transfers</div>
      <div class="gm-val" id="gm-d2h">{d2h_mb:,} <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Kernel dispatches</div>
      <div class="gm-val" id="gm-kdispatch" style="color:var(--blue-light)">{k_dispatch:,}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">T2 frag %</div>
      <div class="gm-val" id="gm-frag" style="color:{frag_col}">{frag_pct}%</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Cold evictions</div>
      <div class="gm-val" id="gm-evict" style="color:var(--amber)">{evict_cnt:,}</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Peak T1</div>
      <div class="gm-val" id="gm-t1peak">{t1_peak:,} <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Peak T2</div>
      <div class="gm-val" id="gm-t2peak">{t2_peak:,} <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Peak T3</div>
      <div class="gm-val" id="gm-t3peak">{t3_peak:,} <span class="gm-sub">MB</span></div>
    </div>
  </div>
</div>

<!-- ── Feeder Vitals ── -->
<div class="card" style="margin-bottom:16px">
  <h2>Feeder Vitals
    <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; SSH direct &middot; 4s</span>
    <span id="gm-feeder-status" style="margin-left:10px;font-size:11px;color:{feeder_status_col};text-transform:none;font-weight:400">{feeder_status_txt}</span>
  </h2>

  <!-- Per-feeder GPU panel, populated by pollFeederGPU() -->
  <div id="fg-feeder-panels">
    <div style="color:var(--dim);font-size:12px;padding:8px 0">Loading feeder data…</div>
  </div>

  <!-- GreenBoost memory allocation view (from local shim_stats) -->
  <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">
    <div style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px">GreenBoost Allocation (host → feeder)</div>
    <div class="gb-metric-grid">
      <div class="gb-metric">
        <div class="gm-label">Feeder T1 cur</div>
        <div class="gm-val" id="gm-feeder-t1-cur" style="color:var(--cyan)">{feeder_t1_cur:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Feeder T1 peak</div>
        <div class="gm-val" id="gm-feeder-t1-peak">{feeder_t1_peak:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Feeder T2 cur</div>
        <div class="gm-val" id="gm-feeder-t2-cur" style="color:var(--cyan)">{feeder_t2_cur:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Feeder T2 peak</div>
        <div class="gm-val" id="gm-feeder-t2-peak">{feeder_t2_peak:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Feeder T3 cur</div>
        <div class="gm-val" id="gm-feeder-t3-cur" style="color:var(--amber)">{feeder_t3_cur:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Feeder T3 peak</div>
        <div class="gm-val" id="gm-feeder-t3-peak">{feeder_t3_peak:,} <span class="gm-sub">MB</span></div>
      </div>
      <div class="gb-metric">
        <div class="gm-label">Remote allocs</div>
        <div class="gm-val" id="gm-remote-allocs" style="color:var(--purple)">{remote_allocs:,}</div>
      </div>
    </div>
  </div>
</div>

<!-- ── NVTX Events ── -->
<div class="card" style="margin-bottom:16px">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <h2 style="margin:0">NVTX Event Log <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; /run/greenboost/nvtx_events.log &middot; 5s refresh</span></h2>
    <span style="font-size:11px;color:var(--dim)" id="nvtx-count">{len(nvtx_rows)} events</span>
  </div>
  <div style="max-height:320px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent">
    <table class="nvtx-table" id="nvtx-table">
      <thead><tr><th>Time</th><th>Event</th><th>Tier</th><th>Size</th><th>Detail</th></tr></thead>
      <tbody id="nvtx-tbody">{nvtx_html}</tbody>
    </table>
  </div>
</div>

<!-- ── Shim Injection Status ── -->
<div class="card" style="margin-bottom:16px" id="gb-shim-inj-card">
  <h2>Shim Injection Status
    <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; 5s</span>
  </h2>
  <div id="gb-shim-inj-body" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;min-height:40px">
    <div style="color:var(--dim);font-size:12px">Loading&hellip;</div>
  </div>
</div>

<!-- ── Diffuser Vitals (FLUX / SD) ── -->
<div class="card" style="margin-bottom:16px" id="gb-diffuser-card">
  <h2>Diffuser Vitals
    <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; FLUX · SD · HuggingFace pipelines &middot; 3s</span>
    <span id="gb-diff-status-badge" class="badge" style="margin-left:8px;font-size:10px">idle</span>
  </h2>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px">
    <div class="gb-metric">
      <div class="gm-label">Pipeline</div>
      <div class="gm-val" id="gm-diff-pipeline" style="color:var(--cyan)">—</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">State</div>
      <div class="gm-val" id="gm-diff-state" style="font-size:14px;color:var(--dim)">idle</div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">VRAM Alloc</div>
      <div class="gm-val" id="gm-diff-vram-alloc">0 <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">VRAM Peak</div>
      <div class="gm-val" id="gm-diff-vram-peak">0 <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">T2 Spill</div>
      <div class="gm-val" id="gm-diff-t2" style="color:var(--amber)">0 <span class="gm-sub">MB</span></div>
    </div>
    <div class="gb-metric">
      <div class="gm-label">Last Gen</div>
      <div class="gm-val" id="gm-diff-gen-s" style="color:var(--lime)">—</div>
    </div>
    <div class="gb-metric" style="grid-column:1/-1">
      <div class="gm-label">Progress</div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:4px">
        <div style="flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.08);overflow:hidden">
          <div id="gm-diff-prog-bar" style="height:100%;border-radius:3px;background:var(--violet);width:0%;transition:width .3s"></div>
        </div>
        <span id="gm-diff-prog-txt" style="font-size:11px;color:var(--dim);min-width:48px">step 0/0</span>
      </div>
    </div>
    <div class="gb-metric" style="grid-column:1/-1">
      <div class="gm-label">Last Prompt</div>
      <div id="gm-diff-prompt" style="font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">—</div>
    </div>
  </div>
</div>

<!-- ── Kernel Events + AppArmor + Service ── -->
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px">
  <div class="card">
    <h2>Kernel Events <span style="font-size:10px;color:var(--dim);text-transform:none;letter-spacing:0;font-weight:400">&middot; 10s</span></h2>
    <div id="gb-kernel-events" style="max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:2px;scrollbar-width:thin">
      <div class="empty">Loading&hellip;</div>
    </div>
  </div>
  <div class="card">
    <h2>AppArmor <span id="gb-aa-count" class="badge badge-warn" style="font-size:10px">?</span></h2>
    <div id="gb-aa-events" style="max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:2px;scrollbar-width:thin">
      <div class="empty">Loading&hellip;</div>
    </div>
    <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border)">
      <code style="font-size:10px;color:var(--amber)">sudo greenboost install-sys-configs</code>
    </div>
  </div>
  <div class="card">
    <h2>Recovery Service</h2>
    <div id="gb-service-events" style="max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:2px;scrollbar-width:thin">
      <div class="empty">Loading&hellip;</div>
    </div>
    <form method="post" action="/greenboost/clear-pool" style="margin-top:10px"
          onsubmit="return confirm('Clear T2/T3 pool? This will SIGKILL inference.')">
      <button type="submit" class="btn" style="border-color:var(--red);color:var(--red);font-size:12px">
        &curren; Clear Memory Pool
      </button>
    </form>
  </div>
</div>

<script src="/static/chart.umd.min.js"></script>
<script>
(function() {{
  // ── Chart.js sparklines ───────────────────────────────────────────────
  var charts = {{}};
  var MAX_PTS = 90;

  function makeChart(id, color, label) {{
    var el = document.getElementById(id);
    if (!el || typeof Chart === 'undefined') return null;
    return new Chart(el, {{
      type: 'line',
      data: {{
        labels: [],
        datasets: [{{
          label: label,
          data: [],
          borderColor: color,
          backgroundColor: color + '18',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.4,
          fill: true
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ display: false, min: 0, max: 100, grid: {{ display: false }} }}
        }}
      }}
    }});
  }}

  function initCharts() {{
    if (typeof Chart === 'undefined') return;
    charts.t1 = makeChart('chart-t1', '#00C4B4', 'T1');  /* electric teal */
    charts.t2 = makeChart('chart-t2', '#B0A4E3', 'T2');  /* soft lavender */
    charts.t3 = makeChart('chart-t3', '#FF5C3A', 'T3');  /* electric coral */
  }}

  function pushChart(chart, val) {{
    if (!chart) return;
    var ts = new Date().toLocaleTimeString('en', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    chart.data.labels.push(ts);
    chart.data.datasets[0].data.push(val);
    if (chart.data.labels.length > MAX_PTS) {{
      chart.data.labels.shift();
      chart.data.datasets[0].data.shift();
    }}
    chart.update('none');
  }}

  // ── Helpers ───────────────────────────────────────────────────────────
  /* Palette: ok=#00C4B4 teal, warn=#FFD000 gold, crit=#FF1478 pink */
  function _pcol(p) {{
    return p === 0 ? '#00C4B4' : (p === 1 ? '#FFD000' : '#FF1478');
  }}
  function _plab(p) {{
    return p === 0 ? 'ok' : (p === 1 ? 'warn' : 'critical');
  }}
  function _pbcls(p) {{
    return p === 0 ? 'badge badge-ok' : (p === 1 ? 'badge badge-warn' : 'badge badge-err');
  }}
  function _pct(used, total) {{
    return total ? Math.min(100, Math.round(used / total * 100)) : 0;
  }}
  function _el(id) {{ return document.getElementById(id); }}
  function _set(id, val) {{ var e = _el(id); if (e) e.textContent = val; }}
  function _style(id, prop, val) {{ var e = _el(id); if (e) e.style[prop] = val; }}
  function _cls(id, cls) {{ var e = _el(id); if (e) e.className = cls; }}

  // ── Status poll ───────────────────────────────────────────────────────
  function pollStatus() {{
    fetch('/api/greenboost/status')
      .then(function(r) {{ return r.json(); }})
      .then(function(st) {{
        var t1gb    = ((st.vram_physical_mb || 0) / 1024).toFixed(1);
        var t2used  = ((st.ram_allocated_mb || 0) / 1024).toFixed(1);
        var t2total = ((st.ram_pool_mb || 0) / 1024).toFixed(1);
        var t3used  = ((st.nvme_swap_used_mb || 0) / 1024).toFixed(1);
        var t3total = ((st.nvme_swap_total_mb || 0) / 1024).toFixed(1);
        var t2pct = _pct(st.ram_allocated_mb, st.ram_pool_mb);
        var t3pct = _pct(st.nvme_swap_used_mb, st.nvme_swap_total_mb);
        var t2c = _pcol(st.t2_pressure);
        var t3c = _pcol(st.swap_pressure);

        _set('gb-t1-lbl', t1gb + ' / ' + t1gb + ' GB (100%)');
        _set('gb-t2-lbl', t2used + ' / ' + t2total + ' GB (' + t2pct + '%)');
        _set('gb-t3-lbl', t3used + ' / ' + t3total + ' GB (' + t3pct + '%)');
        _style('gb-bar-t2', 'width', t2pct + '%'); _style('gb-bar-t2', 'background', t2c);
        _style('gb-bar-t3', 'width', t3pct + '%'); _style('gb-bar-t3', 'background', t3c);

        pushChart(charts.t1, 100);
        pushChart(charts.t2, t2pct);
        pushChart(charts.t3, t3pct);

        _set('gb-combined-val', (((st.total_combined_mb||0)/1024).toFixed(1)));
        _set('gb-bufs-val', st.active_buffers || 0);
        _set('gb-kv-val', st.kv_used_mb || 0);
        _set('gb-kv-rsv', (st.kv_reserve_mb || 0) + ' MB');
        _set('gb-kv-comp', (st.kv_compressed_mb || 0) + ' MB');
        _set('kv-used-val', (st.kv_used_mb || 0) + ' MB');
        _set('kv-t2-val', (st.kv_t2_mb || 0) + ' MB');
        _set('kv-comp-val', (st.kv_compressed_mb || 0) + ' MB');
        var tq = st.kv_compression_bits ? st.kv_compression_bits + '-bit' : 'off';
        _set('gb-tq-sub', tq); _set('kv-tq-val', tq);
        _set('gb-gpu-sub', st.gpu_name || '—');

        window._gbOOM = !!st.oom_active;
        var oomEl = _el('gb-oom-badge');
        if (oomEl) {{
          oomEl.textContent = st.oom_active ? 'ACTIVE' : 'clear';
          oomEl.className = 'badge ' + (st.oom_active ? 'badge-err' : 'badge-ok');
        }}
        _cls('gb-t2-press-badge', _pbcls(st.t2_pressure));
        _set('gb-t2-press-badge', _plab(st.t2_pressure));

        _set('flow-t1', t1gb + ' GB');
        _set('flow-t2', t2used + '/' + t2total + ' GB');
        _style('flow-t2', 'color', t2c);
        _style('flow-t2-bar', 'width', t2pct + '%'); _style('flow-t2-bar', 'background', t2c);
        _style('flow-t2-box', 'borderColor', t2c);
        _set('flow-t3', t3used + '/' + t3total + ' GB');
        _style('flow-t3', 'color', t3c);
        _style('flow-t3-bar', 'width', t3pct + '%'); _style('flow-t3-bar', 'background', t3c);
        _style('flow-t3-box', 'borderColor', t3c);

        // Alert bar
        var alertSevs = [];
        var alertMsgs = [];
        if (st.oom_active) {{ alertSevs.push('crit'); alertMsgs.push('OOM guard ACTIVE'); }}
        if (st.t2_pressure === 2) {{ alertSevs.push('crit'); alertMsgs.push('T2 critical'); }}
        else if (st.t2_pressure === 1) {{ alertSevs.push('warn'); alertMsgs.push('T2 warn'); }}
        if (st.swap_pressure === 2) {{ alertSevs.push('crit'); alertMsgs.push('T3 critical'); }}
        else if (st.swap_pressure === 1) {{ alertSevs.push('warn'); alertMsgs.push('T3 warn'); }}

        var bar = _el('gb-alert-bar');
        var items = _el('gb-alert-items');
        if (items) {{
          if (alertSevs.length === 0) {{
            items.innerHTML = '<span style="color:var(--lime);font-weight:600">✓ All systems nominal</span>';
            if (bar) bar.className = 'gb-alert-bar gb-alert-ok';
          }} else {{
            var hasCrit = alertSevs.indexOf('crit') >= 0;
            if (bar) bar.className = 'gb-alert-bar ' + (hasCrit ? 'gb-alert-crit' : 'gb-alert-warn');
            var parts = [];
            for (var i = 0; i < alertSevs.length; i++) {{
              var col = alertSevs[i] === 'crit' ? 'var(--red)' : 'var(--amber)';
              var icon = alertSevs[i] === 'crit' ? '✗' : '⚠';
              parts.push('<span style="color:' + col + '">' + icon + ' ' + alertMsgs[i] + '</span>');
            }}
            items.innerHTML = parts.join(' &nbsp;&middot;&nbsp; ');
          }}
        }}
      }})
      .catch(function() {{}});
  }}

  // ── Shim stats poll ───────────────────────────────────────────────────
  function _fmtMB(v) {{ return (v || 0).toLocaleString() + ' MB'; }}
  function pollShim() {{
    fetch('/api/greenboost/shim')
      .then(function(r) {{ return r.json(); }})
      .then(function(ss) {{
        // Phase
        var ph = ss.phase || 'UNKNOWN';
        _set('gb-phase-val', ph);
        var phaseEl = _el('gb-phase-val');
        if (phaseEl) {{
          var pmap = {{'INFERENCE':'var(--lime)','STEADY':'var(--cyan)','OOM':'var(--red)',
                       'INIT':'var(--dim)','RESET':'var(--amber)'}};
          phaseEl.style.color = pmap[ph] || 'var(--dim)';
        }}
        // VRAM headroom
        var hr = parseInt(ss.vram_headroom_mb) || 0;
        _set('gb-vram-headroom', hr);
        _style('gb-vram-headroom', 'color', hr > 500 ? 'var(--lime)' : 'var(--amber)');

        // ── Shim stat grid ────────────────────────────────────────────
        _set('gm-active-path', ss.active_path || '—');
        _set('gm-path-a',   (ss.path_a_count  || 0).toLocaleString());
        _set('gm-path-b',   (ss.path_b_count  || 0).toLocaleString());
        _set('gm-path-c',   (ss.path_c_count  || 0).toLocaleString());
        var h2d = ss.h2d_mb || 0; _set('gm-h2d', h2d.toLocaleString() + ' MB');
        var d2h = ss.d2h_mb || 0; _set('gm-d2h', d2h.toLocaleString() + ' MB');
        var kd  = ss.kernel_dispatch_count || 0;
        _set('gm-kdispatch', kd.toLocaleString());
        _style('gm-kdispatch', 'color', kd > 0 ? 'var(--lime)' : 'var(--blue-light)');
        var frag = ss.t2_pool_frag_pct || 0;
        _set('gm-frag', frag + '%');
        _style('gm-frag', 'color', frag > 60 ? 'var(--red)' : frag > 30 ? 'var(--amber)' : 'var(--lime)');
        _set('gm-evict', (ss.cold_epoch_evict_count || 0).toLocaleString());
        var t1p = (ss.tier_t1_local_peak_mb || 0) + (ss.tier_t1_feeder_peak_mb || 0);
        var t2p = (ss.tier_t2_local_peak_mb || 0) + (ss.tier_t2_feeder_peak_mb || 0);
        var t3p = (ss.tier_t3_local_peak_mb || 0) + (ss.tier_t3_feeder_peak_mb || 0);
        _set('gm-t1peak', t1p.toLocaleString() + ' MB');
        _set('gm-t2peak', t2p.toLocaleString() + ' MB');
        _set('gm-t3peak', t3p.toLocaleString() + ' MB');

        // ── Feeder vitals ─────────────────────────────────────────────
        var ft1c = ss.tier_t1_feeder_cur_mb  || 0;
        var ft1p = ss.tier_t1_feeder_peak_mb || 0;
        var ft2c = ss.tier_t2_feeder_cur_mb  || 0;
        var ft2p = ss.tier_t2_feeder_peak_mb || 0;
        var ft3c = ss.tier_t3_feeder_cur_mb  || 0;
        var ft3p = ss.tier_t3_feeder_peak_mb || 0;
        var ra   = ss.remote_alloc_count     || 0;
        _set('gm-feeder-t1-cur',   ft1c.toLocaleString() + ' MB');
        _set('gm-feeder-t1-peak',  ft1p.toLocaleString() + ' MB');
        _set('gm-feeder-t2-cur',   ft2c.toLocaleString() + ' MB');
        _set('gm-feeder-t2-peak',  ft2p.toLocaleString() + ' MB');
        _set('gm-feeder-t3-cur',   ft3c.toLocaleString() + ' MB');
        _set('gm-feeder-t3-peak',  ft3p.toLocaleString() + ' MB');
        _set('gm-remote-allocs',   ra.toLocaleString());
        var feederActive = (ft1c + ft2c) > 0;
        var fStatus = feederActive ? 'active' : (ra > 0 ? 'allocated' : 'idle');
        var fCol    = feederActive ? 'var(--lime)' : (ra > 0 ? 'var(--amber)' : 'var(--dim)');
        _set('gm-feeder-status', fStatus);
        _style('gm-feeder-status', 'color', fCol);
        _style('gm-feeder-t1-cur', 'color', ft1c > 0 ? 'var(--lime)' : 'var(--cyan)');
        _style('gm-feeder-t2-cur', 'color', ft2c > 0 ? 'var(--lime)' : 'var(--cyan)');

        // ── KV cache live ──────────────────────────────────────────────
        _set('kv-t2-val', _fmtMB(ss.kv_t2_tracked_mb));
        _set('kv-comp-val', _fmtMB(ss.kv_comp_mb || ss.kv_internal_frag_mb));
      }})
      .catch(function() {{}});
  }}

  // ── NVTX poll ─────────────────────────────────────────────────────────
  var _evCls = {{
    'ALLOC_T1':'ev-alloc','ALLOC_T2':'ev-alloc','ALLOC_T3':'ev-alloc',
    'EVICT':'ev-evict','PHASE':'ev-phase','SHIM':'ev-shim','KV':'ev-kv',
    'RESET':'ev-reset','OOM':'ev-oom'
  }};
  function _evCss(ev) {{
    var keys = Object.keys(_evCls);
    for (var i = 0; i < keys.length; i++) {{
      if (ev.indexOf(keys[i]) === 0) return _evCls[keys[i]];
    }}
    return '';
  }}
  function pollNVTX() {{
    fetch('/api/greenboost/nvtx')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        var tbody = _el('nvtx-tbody');
        var cnt = _el('nvtx-count');
        var evs = data.events || [];
        if (cnt) cnt.textContent = evs.length + ' events';
        if (!tbody || !evs.length) return;
        var html = '';
        for (var i = 0; i < evs.length; i++) {{
          var ev = evs[i];
          var ts_sec = parseInt(ev.ts_ms) / 1000;
          var d = new Date(ts_sec * 1000);
          var ts = d.toLocaleTimeString('en', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
          var detail = (ev.detail || '').substring(0, 80).replace(/&/g,'&amp;').replace(/</g,'&lt;');
          html += '<tr>'
            + '<td style="color:var(--dim)">' + ts + '</td>'
            + '<td class="' + _evCss(ev.event) + '">' + ev.event + '</td>'
            + '<td style="color:var(--text-muted)">' + ev.tier + '</td>'
            + '<td style="color:var(--cyan)">' + ev.size + '</td>'
            + '<td style="color:var(--text-muted);max-width:260px;overflow:hidden;text-overflow:ellipsis">' + detail + '</td>'
            + '</tr>';
        }}
        tbody.innerHTML = html;
      }})
      .catch(function() {{}});
  }}

  // ── Logs poll ─────────────────────────────────────────────────────────
  /* Strip ISO date prefix, keep only HH:MM:SS + host + message */
  function _trimLine(line) {{
    /* "2026-05-07T16:27:20+02:00 ncore kernel: greenboost: …" → "16:27:20  greenboost: …" */
    var m = line.match(/T(\\d\\d:\\d\\d:\\d\\d)[^ ]* \\S+ (?:kernel: |systemd\\[\\d+\\]: )?(.*)$/);
    if (m) return m[1] + '  ' + m[2];
    return line;
  }}
  function _aaShort(line) {{
    /* Extract operation + name from apparmor=DENIED operation="x" … name="y" */
    var op   = (line.match(/operation="([^"]+)"/) || [])[1] || '';
    var name = (line.match(/name="([^"]+)"/) || [])[1] || '';
    var ts   = (line.match(/T(\\d\\d:\\d\\d:\\d\\d)/) || [])[1] || '';
    if (op && name) return (ts ? ts + '  ' : '') + 'DENIED ' + op + '  ' + name.split('/').pop();
    return _trimLine(line);
  }}
  function _logLine(text, col, glass) {{
    var bg = glass ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.03)';
    return '<div style="font-size:11px;font-family:\'JetBrains Mono\',monospace;'
      + 'padding:4px 10px;border-radius:5px;box-sizing:border-box;width:100%;'
      + 'background:' + bg + ';border:1px solid rgba(255,255,255,0.07);'
      + 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
      + 'color:' + col + ';display:block;line-height:1.5">' + text + '</div>';
  }}
  function pollLogs() {{
    fetch('/api/greenboost/logs')
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        /* Update summary action items */
        var aaCount = (d.apparmor || []).length;
        var svcFails = (d.service || []).filter(function(l) {{ return l.indexOf('Failed')>=0||l.indexOf('FAILURE')>=0; }}).length;
        _updateSummary(aaCount, svcFails, d.kernel || []);
        var ts = _el('gb-summary-ts');
        if (ts) ts.textContent = 'updated ' + new Date().toLocaleTimeString();

        var ke = _el('gb-kernel-events');
        if (ke) {{
          var klines = (d.kernel || []).slice(-18).reverse();
          if (klines.length) {{
            ke.innerHTML = klines.map(function(line) {{
              var col = line.indexOf('OOM') >= 0 ? '#FF1478'
                : line.indexOf('RESET') >= 0 ? '#FFD000'
                : (line.indexOf('alloc') >= 0 || line.indexOf('pinned') >= 0) ? '#00D8E8'
                : 'rgba(255,255,255,0.42)';
              return _logLine(_trimLine(line), col);
            }}).join('');
          }} else ke.innerHTML = '<div class="empty">No kernel events</div>';
        }}

        var aa = _el('gb-aa-events');
        var ac = _el('gb-aa-count');
        if (ac) ac.textContent = (d.apparmor || []).length;
        if (aa) {{
          var aalines = (d.apparmor || []).slice(-12).reverse();
          if (aalines.length) {{
            aa.innerHTML = aalines.map(function(line) {{
              return _logLine(_aaShort(line), '#FF5C3A');
            }}).join('');
          }} else {{
            aa.innerHTML = '<div class="empty" style="color:#00C4B4">✓ No denials</div>';
          }}
        }}

        var sv = _el('gb-service-events');
        if (sv) {{
          var slines = (d.service || []).slice(-18).reverse();
          if (slines.length) {{
            sv.innerHTML = slines.map(function(line) {{
              var col = (line.indexOf('Failed') >= 0 || line.indexOf('FAILURE') >= 0) ? '#FF1478'
                : (line.indexOf('Started') >= 0 || line.indexOf('success') >= 0) ? '#00C4B4'
                : 'rgba(255,255,255,0.42)';
              return _logLine(_trimLine(line), col);
            }}).join('');
          }} else sv.innerHTML = '<div class="empty">No service events</div>';
        }}
      }})
      .catch(function() {{}});
  }}

  // ── Summary panel updater ─────────────────────────────────────────────
  function _updateSummary(aaCount, svcFails, kernelLines) {{
    var el = _el('gb-summary-body');
    if (!el) return;
    var items = [];
    /* AppArmor denials */
    if (aaCount > 0) {{
      items.push({{
        sev: aaCount > 5 ? 'crit' : 'warn',
        icon: '⚠',
        title: aaCount + ' AppArmor denial' + (aaCount>1?'s':'') + ' detected',
        action: 'Run: sudo greenboost install-sys-configs'
      }});
    }}
    /* Service failures */
    if (svcFails > 0) {{
      items.push({{
        sev: 'crit', icon: '✗',
        title: 'greenboost-recovery.service failed ' + svcFails + ' time' + (svcFails>1?'s':''),
        action: 'Check: journalctl -u greenboost-recovery -e'
      }});
    }}
    /* Recent RESET events */
    var resets = kernelLines.filter(function(l) {{ return l.indexOf('RESET')>=0; }});
    if (resets.length > 0) {{
      items.push({{
        sev: 'warn', icon: '↺',
        title: resets.length + ' kernel RESET event' + (resets.length>1?'s':'') + ' in log',
        action: 'OOM guard was cleared — buffers released'
      }});
    }}
    /* OOM guard from status */
    if (window._gbOOM) {{
      items.unshift({{
        sev: 'crit', icon: '🔴',
        title: 'OOM guard is ACTIVE — inference blocked',
        action: 'Clear pool to resume, or wait for buffers to release'
      }});
    }}
    if (items.length === 0) {{
      el.innerHTML = '<div style="display:flex;align-items:center;gap:10px;color:#00C4B4">'
        + '<span style="font-size:18px">✓</span>'
        + '<span style="font-size:13px;font-weight:500">All systems nominal — no action required</span>'
        + '</div>';
    }} else {{
      el.innerHTML = items.map(function(it) {{
        var bdr = it.sev === 'crit' ? 'rgba(255,20,120,0.4)' : 'rgba(255,208,0,0.35)';
        var col = it.sev === 'crit' ? '#FF1478' : '#FFD000';
        return '<div style="display:flex;gap:12px;align-items:flex-start;padding:10px 14px;'
          + 'border-radius:8px;border-top:2px solid ' + bdr + ';'
          + 'background:rgba(255,255,255,0.03);margin-bottom:6px">'
          + '<span style="color:' + col + ';font-size:16px;flex-shrink:0;line-height:1.3">' + it.icon + '</span>'
          + '<div><div style="font-size:13px;font-weight:600;color:' + col + '">' + it.title + '</div>'
          + '<div style="font-size:11px;color:rgba(255,255,255,0.45);margin-top:3px;font-family:\'JetBrains Mono\',monospace">'
          + it.action + '</div></div></div>';
      }}).join('');
    }}
  }}

  // ── Feeder GPU vitals (SSH-sourced) ───────────────────────────────────
  function _vramBar(used, total) {{
    if (!total) return '';
    var pct = Math.round(used / total * 100);
    var col = pct > 85 ? 'var(--red,#FF1478)' : pct > 60 ? 'var(--amber)' : 'var(--lime)';
    return '<div style="margin-top:4px;height:4px;border-radius:2px;background:rgba(255,255,255,0.08);overflow:hidden">'
      + '<div style="height:100%;border-radius:2px;background:' + col + ';width:' + pct + '%"></div></div>';
  }}
  function pollFeederGPU() {{
    fetch('/api/greenboost/feeder-gpu')
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        var panel = _el('fg-feeder-panels');
        if (!panel) return;
        var feeders = data.feeders || [];
        if (!feeders.length) {{
          panel.innerHTML = '<div style="color:var(--dim);font-size:12px;padding:4px 0">No feeders in cluster.conf</div>';
          return;
        }}
        var html = '';
        for (var i = 0; i < feeders.length; i++) {{
          var f = feeders[i];
          if (!f.reachable) {{
            html += '<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;'
              + 'background:rgba(255,20,120,0.06);border:1px solid rgba(255,20,120,0.2);margin-bottom:8px">'
              + '<span style="color:#FF1478;font-size:15px">⚠</span>'
              + '<div>'
              + '<div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.7)">'
              + (f.host || f.ip) + ' (' + f.ip + ')</div>'
              + '<div style="font-size:11px;color:var(--dim);font-family:monospace">'
              + 'SSH unreachable' + (f.error ? ' — ' + f.error : '') + '</div>'
              + '</div></div>';
            continue;
          }}
          var netdCol  = f.netd_running ? 'var(--lime)' : 'var(--red,#FF1478)';
          var netdTxt  = f.netd_running ? 'netd ●' : 'netd ✗';
          var utilCol  = f.gpu_util_pct > 80 ? 'var(--lime)' : f.gpu_util_pct > 20 ? 'var(--amber)' : 'var(--dim)';
          var tempCol  = f.temp_c > 85 ? 'var(--red,#FF1478)' : f.temp_c > 70 ? 'var(--amber)' : 'var(--lime)';
          var vramFree = f.vram_total_mb - f.vram_used_mb;
          var phaseTxt = (f.phase && f.phase !== 'UNKNOWN') ? f.phase : '—';
          var phaseCol = f.phase === 'INFERENCE' ? 'var(--lime)' : f.phase === 'STEADY' ? 'var(--cyan)' : 'var(--dim)';
          html += '<div style="padding:12px 14px;border-radius:8px;'
            + 'background:rgba(255,255,255,0.03);border:1px solid var(--border);margin-bottom:8px">'
            // header row
            + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">'
            + '<span style="font-size:12px;font-weight:600;color:var(--violet)">'
            + (f.host || f.ip) + '</span>'
            + '<span style="font-size:10px;color:var(--dim);font-family:monospace">' + f.ip + ':' + f.port + '</span>'
            + '<span style="font-size:11px;font-weight:600;color:' + netdCol + ';font-family:monospace">' + netdTxt + '</span>'
            + '</div>'
            // GPU name
            + '<div style="font-size:11px;color:var(--dim);margin-bottom:8px;font-family:monospace">'
            + (f.gpu_name || '—') + '</div>'
            // stats grid
            + '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
            // VRAM
            + '<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">'
            + '<div style="font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px">VRAM</div>'
            + '<div style="font-size:13px;font-weight:600;color:var(--cyan)">' + f.vram_used_mb.toLocaleString() + '<span style="font-size:9px;color:var(--dim)"> MB</span></div>'
            + '<div style="font-size:9px;color:var(--dim)">of ' + f.vram_total_mb.toLocaleString() + ' MB (' + vramFree.toLocaleString() + ' free)</div>'
            + _vramBar(f.vram_used_mb, f.vram_total_mb)
            + '</div>'
            // GPU util
            + '<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">'
            + '<div style="font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px">GPU UTIL</div>'
            + '<div style="font-size:13px;font-weight:600;color:' + utilCol + '">' + f.gpu_util_pct + '<span style="font-size:9px;color:var(--dim)"> %</span></div>'
            + '<div style="font-size:9px;color:' + phaseCol + '">' + phaseTxt + '</div>'
            + '</div>'
            // Temperature
            + '<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">'
            + '<div style="font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px">TEMP</div>'
            + '<div style="font-size:13px;font-weight:600;color:' + tempCol + '">' + f.temp_c + '<span style="font-size:9px;color:var(--dim)"> °C</span></div>'
            + '<div style="font-size:9px;color:var(--dim)">' + (f.power_w > 0 ? f.power_w.toFixed(1) + ' W' : '—') + '</div>'
            + '</div>'
            // Kernel dispatches
            + '<div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px">'
            + '<div style="font-size:9px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px">DISPATCHES</div>'
            + '<div style="font-size:13px;font-weight:600;color:' + (f.kernel_dispatch_count > 0 ? 'var(--lime)' : 'var(--dim)') + '">'
            + (f.kernel_dispatch_count || 0).toLocaleString() + '</div>'
            + '<div style="font-size:9px;color:var(--dim)">kernel dispatch</div>'
            + '</div>'
            + '</div>'  // end grid
            + '</div>';
        }}
        panel.innerHTML = html;
      }})
      .catch(function() {{}});
  }}

  // ── Diffuser Vitals poll ──────────────────────────────────────────────
  function pollDiffuser() {{
    fetch('/api/greenboost/diffuser')
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        var active = !!d.active;
        var badge = _el('gb-diff-status-badge');
        if (badge) {{
          badge.textContent = active ? (d.state || 'active') : 'idle';
          badge.className = 'badge ' + (active ? 'badge-ok' : '');
          badge.style.color = active ? 'var(--lime)' : 'var(--dim)';
        }}
        _set('gm-diff-pipeline', d.pipeline || (active ? 'diffuser' : '—'));
        var stateCol = d.state === 'generating' ? 'var(--violet)' : d.state === 'loading' ? 'var(--amber)' : d.state === 'ready' ? 'var(--lime)' : 'var(--dim)';
        _set('gm-diff-state', d.state || 'idle');
        _style('gm-diff-state', 'color', stateCol);
        var va = d.vram_alloc_mb || 0; _set('gm-diff-vram-alloc', va.toLocaleString() + ' MB');
        var vp = d.vram_peak_mb || 0; _set('gm-diff-vram-peak', vp.toLocaleString() + ' MB');
        var t2 = d.t2_alloc_mb || 0; _set('gm-diff-t2', t2.toLocaleString() + ' MB');
        _style('gm-diff-t2', 'color', t2 > 0 ? 'var(--amber)' : 'var(--dim)');
        var gs = d.last_gen_s || 0; _set('gm-diff-gen-s', gs > 0 ? gs.toFixed(1) + 's' : '—');
        var step = d.gen_step || 0; var total = d.gen_total_steps || 0;
        var pct = total > 0 ? Math.round(step / total * 100) : 0;
        _style('gm-diff-prog-bar', 'width', pct + '%');
        _set('gm-diff-prog-txt', 'step ' + step + '/' + total);
        _set('gm-diff-prompt', d.last_prompt || '—');
      }})
      .catch(function() {{}});
  }}

  // ── Shim Injection Status poll ──────────────────────────────────────────
  function pollShimInj() {{
    fetch('/api/greenboost/shim')
      .then(function(r) {{ return r.json(); }})
      .then(function(ss) {{
        var body = _el('gb-shim-inj-body');
        if (!body) return;
        var shimPid  = ss.pid || 0;
        var shimInit = ss.initialized == 1 || ss.initialized === true;
        var virtVram = (ss.virtual_vram_mb || 0);
        var pathB    = (ss.path_b_count || 0);
        var t2Cur    = (ss.tier_t2_local_cur_mb || 0);

        function _chip(label, val, col) {{
          return '<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);border-radius:8px;padding:10px 14px">'
            + '<div style="font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">' + label + '</div>'
            + '<div style="font-size:14px;font-weight:700;font-family:\'JetBrains Mono\',monospace;color:' + col + '">' + val + '</div></div>';
        }}
        var shimCol  = shimInit ? 'var(--lime)' : 'var(--red,#FF1478)';
        var shimTxt  = shimInit ? 'active (pid ' + shimPid + ')' : 'NOT loaded';
        var virtCol  = virtVram > 0 ? 'var(--lime)' : 'var(--dim)';
        var virtTxt  = virtVram > 0 ? Math.round(virtVram / 1024 * 10) / 10 + ' GB virtual' : '0 (shim inactive)';
        var t2Col    = t2Cur > 0 ? 'var(--amber)' : (pathB > 0 ? 'var(--lime)' : 'var(--dim)');
        var t2Txt    = t2Cur > 0 ? t2Cur.toLocaleString() + ' MB active' : (pathB > 0 ? pathB + ' B-allocs' : 'idle');
        body.innerHTML = _chip('Shim State', shimTxt, shimCol)
          + _chip('Virtual VRAM', virtTxt, virtCol)
          + _chip('T2 (Path B)', t2Txt, t2Col);
      }})
      .catch(function() {{}});
  }}

  // ── Boot ──────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', function() {{
    initCharts();
    pollStatus();
    pollShim();
    pollLogs();
    pollNVTX();
    pollFeederGPU();
    pollDiffuser();
    pollShimInj();
  }});
  setInterval(pollStatus, 2000);
  setInterval(pollShim, 4000);
  setInterval(pollNVTX, 4000);
  setInterval(pollLogs, 3000);
  setInterval(pollFeederGPU, 5000);
  setInterval(pollDiffuser, 3000);
  setInterval(pollShimInj, 5000);
}})();
</script>
"""
    return _page("GreenBoost Monitor", body, "greenboost")


# ── HTTP Handler ──────────────────────────────────────────────────────────────

def _parse_form(data: bytes) -> dict[str, str]:
    return {k: v for k, v in urlparse.parse_qsl(data.decode(errors="replace"))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _html(self, html: str, status: int = 200) -> None:
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _json_resp(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, loc: str) -> None:
        self.send_response(303)
        self.send_header("Location", loc)
        self.end_headers()

    def do_GET(self):
        p = urlparse.urlparse(self.path)
        path, qs = p.path, urlparse.parse_qs(p.query)

        # Serve local static JS/CSS files
        if path.startswith("/static/"):
            fname = path[len("/static/"):]
            static_dir = (Path(__file__).parent / "static").resolve()
            fpath = (static_dir / fname).resolve()
            if fpath.parent == static_dir and fpath.exists():
                ctype = "application/javascript" if fname.endswith(".js") else "text/css"
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)
            return

        if path == "/api/status":
            gpu = _gpu_status()
            self._json_resp({"gpu": gpu.get("available", False), "gb": _gb_available()})
            return

        if path == "/api/browse":
            browse_path = qs.get("path", [str(Path.home())])[0]
            browse_type = qs.get("type", ["dir"])[0]
            try:
                p = Path(browse_path).resolve()
                if not p.is_dir():
                    p = p.parent
                items = []
                if p.parent != p:
                    items.append({"name": "..", "path": str(p.parent), "type": "dir"})
                for child in sorted(p.iterdir(),
                                    key=lambda x: (not x.is_dir(), x.name.lower())):
                    if child.name.startswith("."):
                        continue
                    items.append({
                        "name": child.name,
                        "path": str(child),
                        "type": "dir" if child.is_dir() else "file",
                    })
                self._json_resp({"path": str(p), "items": items})
            except Exception as exc:
                self._json_resp({"path": str(Path.home()), "items": [], "error": str(exc)})
            return

        if path == "/api/factory/status":
            try:
                from greenboost_cli.workflow.factory import get_factory
                self._json_resp(get_factory().snapshot())
            except Exception as exc:
                self._json_resp({"active": False, "error": str(exc)})
            return

        if path == "/api/factory/submit":
            try:
                from greenboost_cli.workflow.factory import get_factory
                q_args = qs.get("prompt", [""])[0].strip()
                if not q_args:
                    self._json_resp({"error": "prompt required"})
                    return
                priority = int(qs.get("priority", ["10"])[0])
                priority = max(1, min(priority, 20))
                task_id = get_factory().submit(prompt=q_args[:2000], priority=priority)
                self._json_resp({"task_id": task_id, "status": "submitted"})
            except Exception as exc:
                self._json_resp({"error": str(exc)})
            return

        if path == "/api/factory/pause":
            agent = qs.get("agent", [""])[0]
            try:
                from greenboost_cli.workflow.factory import get_factory
                get_factory().pause_agent(agent)
                self._json_resp({"status": "paused", "agent": agent})
            except Exception as exc:
                self._json_resp({"error": str(exc)})
            return

        if path == "/api/factory/resume":
            agent = qs.get("agent", [""])[0]
            try:
                from greenboost_cli.workflow.factory import get_factory
                get_factory().resume_agent(agent)
                self._json_resp({"status": "resumed", "agent": agent})
            except Exception as exc:
                self._json_resp({"error": str(exc)})
            return

        if path == "/api/factory/start":
            try:
                from greenboost_cli.workflow.factory import get_factory
                workers = int(qs.get("workers", ["2"])[0])
                get_factory().start(workers=workers)
                self._json_resp({"status": "started"})
            except Exception as exc:
                self._json_resp({"error": str(exc)})
            return

        if path == "/api/factory/stop":
            try:
                from greenboost_cli.workflow.factory import get_factory
                get_factory().stop()
                self._json_resp({"status": "stopped"})
            except Exception as exc:
                self._json_resp({"error": str(exc)})
            return

        if path == "/api/greenboost/status":
            try:
                from greenboost_cli.greenboost.monitor import get_monitor
                self._json_resp(get_monitor().refresh().as_dict())
            except Exception as exc:
                self._json_resp({"loaded": False, "error": str(exc)})
            return

        if path == "/api/greenboost/cmd":
            cmd_key = qs.get("cmd", [""])[0]
            cmd_map = {
                "status": ["greenboost", "status"],
                "logs":   ["greenboost", "logs"],
                "nvtx":   ["greenboost", "nvtx-logs", "--tail", "200", "--llm"],
            }
            cmd = cmd_map.get(cmd_key)
            if cmd:
                output, elapsed = _run_gb_cmd(cmd)
                self._json_resp({"output": output, "elapsed_s": elapsed})
            else:
                self._json_resp({"error": "unknown cmd"})
            return

        if path == "/api/greenboost/shim":
            ss = _read_shim_stats()
            ph = _read_phase()
            ss.update(ph)
            self._json_resp(ss)
            return

        if path == "/api/greenboost/metrics":
            try:
                import json as _json
                data = _json.loads(_METRICS_JSON.read_text()) if _METRICS_JSON.exists() else {}
            except Exception:
                data = {}
            self._json_resp(data)
            return

        if path == "/api/greenboost/feeder-gpu":
            conf = _parse_cluster_conf()
            results = []
            for f in conf:
                d = _fetch_feeder_gpu(f["ip"], f["ssh_user"])
                d = dict(d)
                d["host"] = f["host"]
                d["port"] = f["port"]
                results.append(d)
            self._json_resp({"feeders": results})
            return

        if path == "/api/greenboost/nvtx":
            self._json_resp({"events": _read_nvtx(100)})
            return

        if path == "/api/greenboost/diffuser":
            self._json_resp(_get_diffuser_vitals())
            return

        if path == "/api/greenboost/logs":
            self._json_resp(_gb_logs_fast())
            return

        routes = {
            "/": lambda: page_dashboard(),
            "/goals": lambda: page_goals(qs),
            "/history": lambda: page_history(qs),
            "/rag": lambda: page_rag(qs),
            "/rag/search": lambda: page_rag_search(qs),
            "/pdf": lambda: page_pdf(qs),
            "/design": lambda: page_design(qs),
            "/design/intelligence": lambda: page_design_intelligence(qs),
            "/tokens":     lambda: page_tokens(qs),
            "/guidelines": lambda: page_guidelines(qs),
            "/guidelines/view": lambda: page_guidelines({**qs, "view_name": qs.get("name", [])}),
            "/factory":     lambda: page_factory(qs),
            "/system":      lambda: page_system(),
            "/greenboost":  lambda: page_greenboost(qs),
        }
        fn = routes.get(path)
        if fn:
            try:
                self._html(fn())
            except Exception:
                import traceback
                self._html(f"<pre style='color:red'>{traceback.format_exc()}</pre>", 500)
        else:
            self._html("<h1 style='color:red'>404 Not Found</h1>", 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        form = _parse_form(self.rfile.read(length))
        path = urlparse.urlparse(self.path).path

        try:
            if path == "/goals/add":
                project = form.get("project", "")
                text = form.get("text", "").strip()
                priority = int(form.get("priority", 5))
                if project and text:
                    from greenboost_cli.memory.brain import add_goal
                    pdir = GLOBAL_DIR / "projects" / project
                    pdir.mkdir(parents=True, exist_ok=True)
                    add_goal(pdir, text, priority)
                self._redirect(f"/goals?project={project}")

            elif path == "/goals/remove":
                project = form.get("project", "")
                gid = int(form.get("id", 0))
                if project and gid:
                    from greenboost_cli.memory.brain import remove_goal
                    remove_goal(GLOBAL_DIR / "projects" / project, gid)
                self._redirect(f"/goals?project={project}")

            elif path == "/history/add":
                project = form.get("project", "")
                text = form.get("text", "").strip()
                cat = form.get("category", "note")
                if project and text:
                    from greenboost_cli.memory.brain import append_history
                    pdir = GLOBAL_DIR / "projects" / project
                    pdir.mkdir(parents=True, exist_ok=True)
                    append_history(pdir, text, cat)
                self._redirect(f"/history?project={project}")

            elif path == "/rag/add":
                folder = form.get("folder", "").strip()
                project = form.get("project", "").strip() or None
                if folder:
                    from greenboost_cli.rag.engine import index_folder
                    index_folder(Path(folder), project)
                self._redirect("/rag")

            elif path == "/rag/clear":
                import shutil
                rag_dir = GLOBAL_DIR / "rag"
                if rag_dir.exists():
                    shutil.rmtree(rag_dir)
                self._redirect("/rag")

            elif path == "/rag/update":
                folder = form.get("folder", "").strip()
                do_all = form.get("all") == "1"
                force  = "force" in form
                def _update(_folder=folder, _all=do_all, _force=force):
                    try:
                        from greenboost_cli.rag.engine import update_folder, update_all
                        if _all:
                            update_all(force=_force, verbose=False)
                        elif _folder:
                            update_folder(Path(_folder), force=_force, verbose=False)
                    except Exception as e:
                        print(f"RAG update error: {e}", file=sys.stderr)
                threading.Thread(target=_update, daemon=True).start()
                msg = "RAG update started (all sources)" if do_all else f"RAG update started: {folder}"
                self._redirect(f"/rag?{urlparse.urlencode({'flash': msg})}")

            elif path == "/design/generate":
                prompt = form.get("prompt", "").strip()
                model_key = form.get("model_key", "klein-fp8")
                lora = form.get("lora", "") or None
                style = form.get("style", "glassmorphism")
                colors = form.get("colors", "deep blue violet")
                asset_types = [t for t in ["hero", "mood", "background", "illustration"] if form.get(t)]
                if not asset_types:
                    asset_types = ["hero", "mood"]
                if prompt:
                    _out_dir = GB_HOME / "design_assets"
                    def _generate(_out_dir=_out_dir, _types=asset_types,
                                  _model=model_key, _lora=lora,
                                  _style=style, _colors=colors, _prompt=prompt):
                        try:
                            from greenboost_cli.diffusion.pipeline import generate_ui_asset
                            _out_dir.mkdir(parents=True, exist_ok=True)
                            for asset_type in _types:
                                _out_path = _out_dir / f"{asset_type}_{_model}.png"
                                generate_ui_asset(
                                    asset_type=asset_type,
                                    output_path=_out_path,
                                    model_key=_model,
                                    use_lora=_lora,
                                    style=_style,
                                    colors=_colors,
                                    custom_prompt=_prompt or None,
                                )
                        except Exception as e:
                            print(f"Design generation error: {e}", file=sys.stderr)
                    threading.Thread(target=_generate, daemon=True).start()
                self._redirect("/design")

            elif path == "/pdf/convert":
                pdf_path = form.get("path", "").strip()
                output = form.get("output", "").strip() or None
                pages = form.get("pages", "").strip() or None
                page_breaks = "page_breaks" in form
                preview_only = "preview_only" in form

                if not pdf_path:
                    self._redirect("/pdf")
                    return

                try:
                    from greenboost_cli.pdf.pdf2md import convert_pdf, convert_and_save
                    if preview_only:
                        md = convert_pdf(pdf_path, pages=pages, page_breaks=page_breaks)
                        qs_out = urlparse.urlencode({"result": md[:12000], "path": ""})
                        self._redirect(f"/pdf?{qs_out}")
                    else:
                        out = convert_and_save(pdf_path, output, pages=pages, page_breaks=page_breaks)
                        md = out.read_text(encoding="utf-8")
                        qs_out = urlparse.urlencode({
                            "result": md[:12000],
                            "path": str(out),
                            "flash": f"Saved to {out}",
                        })
                        self._redirect(f"/pdf?{qs_out}")
                except Exception as e:
                    qs_out = urlparse.urlencode({"flash": f"Error: {e}"})
                    self._redirect(f"/pdf?{qs_out}")

            elif path == "/guidelines/add-file":
                project  = form.get("project", "").strip()
                filepath = form.get("path", "").strip()
                name     = form.get("name", "").strip() or None
                if project and filepath:
                    try:
                        from greenboost_cli.memory.ui_guidelines import add_guideline
                        final_name = add_guideline(filepath, name, project)
                        qs_out = urlparse.urlencode({"flash": f"Added guideline: {final_name}"})
                        self._redirect(f"/guidelines?project={project}&{qs_out}")
                        return
                    except Exception as e:
                        qs_out = urlparse.urlencode({"flash": f"Error: {e}", "project": project})
                        self._redirect(f"/guidelines?{qs_out}")
                        return
                self._redirect(f"/guidelines?project={project}")

            elif path == "/guidelines/add-content":
                project = form.get("project", "").strip()
                name    = form.get("name", "").strip()
                content = form.get("content", "").strip()
                if project and name and content:
                    try:
                        from greenboost_cli.memory.ui_guidelines import add_guideline_from_content
                        final_name = add_guideline_from_content(name, content, project)
                        qs_out = urlparse.urlencode({"flash": f"Created guideline: {final_name}"})
                        self._redirect(f"/guidelines?project={project}&{qs_out}")
                        return
                    except Exception as e:
                        qs_out = urlparse.urlencode({"flash": f"Error: {e}", "project": project})
                        self._redirect(f"/guidelines?{qs_out}")
                        return
                self._redirect(f"/guidelines?project={project}")

            elif path == "/guidelines/update":
                project = form.get("project", "").strip()
                name    = form.get("name", "").strip()
                content = form.get("content", "")
                if project and name:
                    try:
                        from greenboost_cli.memory.ui_guidelines import update_guideline
                        update_guideline(name, content, project)
                        qs_out = urlparse.urlencode({"flash": f"Updated: {name}"})
                        self._redirect(f"/guidelines?project={project}&{qs_out}")
                        return
                    except Exception as e:
                        qs_out = urlparse.urlencode({"flash": f"Error: {e}", "project": project})
                        self._redirect(f"/guidelines?{qs_out}")
                        return
                self._redirect(f"/guidelines?project={project}")

            elif path == "/guidelines/remove":
                project = form.get("project", "").strip()
                name    = form.get("name", "").strip()
                if project and name:
                    try:
                        from greenboost_cli.memory.ui_guidelines import remove_guideline
                        remove_guideline(name, project)
                    except Exception:
                        pass
                self._redirect(f"/guidelines?project={project}")

            elif path == "/guidelines/enable":
                project = form.get("project", "").strip()
                name    = form.get("name", "").strip()
                if project and name:
                    try:
                        from greenboost_cli.memory.ui_guidelines import set_active
                        set_active(name, True, project)
                    except Exception:
                        pass
                self._redirect(f"/guidelines?project={project}")

            elif path == "/guidelines/disable":
                project = form.get("project", "").strip()
                name    = form.get("name", "").strip()
                if project and name:
                    try:
                        from greenboost_cli.memory.ui_guidelines import set_active
                        set_active(name, False, project)
                    except Exception:
                        pass
                self._redirect(f"/guidelines?project={project}")

            elif path == "/pdf/batch":
                folder = form.get("folder", "").strip()
                if folder:
                    def _batch():
                        try:
                            from greenboost_cli.pdf.pdf2md import convert_and_save
                            for pdf in sorted(Path(folder).glob("**/*.pdf")):
                                try:
                                    convert_and_save(pdf)
                                except Exception as e:
                                    print(f"pdf2md batch skip {pdf}: {e}", file=sys.stderr)
                        except Exception as e:
                            print(f"pdf2md batch error: {e}", file=sys.stderr)
                    threading.Thread(target=_batch, daemon=True).start()
                self._redirect("/pdf?flash=Batch+conversion+started+in+background")

            elif path == "/greenboost/clear-pool":
                subprocess.run(
                    ["sudo", "greenboost", "clear", "memory-pool"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                self._redirect("/greenboost?flash=Memory+pool+cleared")

            elif path == "/greenboost/log-rag/index":
                label = form.get("label", "").strip()
                try:
                    from greenboost_cli.greenboost.log_rag import snapshot_and_index
                    result = snapshot_and_index(label)
                    msg = f"Snapshot captured: {result.get('snapshot','')} ({result.get('chunks',0)} chunks indexed)"
                    self._redirect("/greenboost?" + urlparse.urlencode({"flash": msg}))
                except Exception as exc:
                    self._redirect("/greenboost?" + urlparse.urlencode({"flash": f"Error: {exc}"}))

            elif path == "/greenboost/log-rag/delete":
                name = form.get("name", "").strip()
                try:
                    from greenboost_cli.greenboost.log_rag import delete_snapshot
                    delete_snapshot(name)
                    self._redirect("/greenboost?flash=Snapshot+deleted")
                except Exception as exc:
                    self._redirect("/greenboost?" + urlparse.urlencode({"flash": f"Error: {exc}"}))

            elif path == "/greenboost/log-rag/search":
                query = form.get("query", "").strip()
                self._redirect("/greenboost?" + urlparse.urlencode({"rag_q": query}))

            else:
                self._redirect("/")

        except Exception:
            import traceback
            self._html(f"<pre style='color:red'>{traceback.format_exc()}</pre>", 500)


# ── Entry point ───────────────────────────────────────────────────────────────

def _gb_clear_pool_silent() -> None:
    subprocess.run(
        ["sudo", "greenboost", "clear", "memory-pool"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run(port: int = PORT) -> None:
    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"  GreenBoost CLI dashboard  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    finally:
        _gb_clear_pool_silent()


def _port_open(port: int) -> bool:
    """Return True if something is already listening on the port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_port(port: int) -> None:
    """Kill whatever process is holding the port so we can bind it."""
    try:
        out = subprocess.check_output(
            ["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL
        ).decode().strip()
        pids = out.split()
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", pid], check=False)
            except Exception:
                pass
        time.sleep(0.3)
    except Exception:
        pass


def start_server(port: int = PORT) -> None:
    """Start dashboard in foreground (blocking). Opens browser automatically.
    If the port is already in use, kill the old process and bind it fresh."""
    import webbrowser
    import socket as _socket
    url = f"http://localhost:{port}"
    if _port_open(port):
        print(f"  Port {port} in use — stopping old process and restarting.")
        _kill_port(port)
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    print(f"  GreenBoost CLI dashboard  {url}")
    print(f"  Press Ctrl+C to stop.")
    threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    finally:
        _gb_clear_pool_silent()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()
    run(args.port)
