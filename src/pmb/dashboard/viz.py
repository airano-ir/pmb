"""Self-contained interactive memory-graph visualization.

`build_memory_html(engine)` returns a single HTML document (data + renderer
inlined, no CDN, no network) that draws the workspace's entity/association
graph as an interactive force-directed canvas: zoom/pan, drag, hover, click a
node to focus its neighbours, search, filter by kind, and re-size nodes by
mention count or recency. It is meant to be a shareable artifact — one file you
can open offline or send to someone.

Read-only: pulls from the graph store, never touches recall ranking.
"""
from __future__ import annotations

import json
import time


def build_memory_html(
    engine,
    limit: int = 300,
    max_edges: int = 4000,
    title: str | None = None,
) -> str:
    """Build the standalone HTML string for the workspace's memory graph."""
    ws = engine.workspace
    graph = engine.graph.viz_graph(ws.id, limit=limit, max_edges=max_edges)
    try:
        stats = engine.graph.stats(ws.id)
    except Exception:
        stats = {"n_entities": len(graph["nodes"]), "n_edges": len(graph["edges"]), "by_kind": {}}

    payload = {
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "stats": stats,
        "workspace": getattr(ws, "name", "workspace"),
        "shown": len(graph["nodes"]),
        "total": stats.get("n_entities", len(graph["nodes"])),
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    # Embed safely: escape `<` so a name can never break out of <script>.
    data_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")

    return (
        _TEMPLATE
        .replace("/*__DATA__*/null", data_json)
        .replace("__TITLE__", (title or f"PMB memory — {payload['workspace']}"))
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#0b0f17; --panel:#121826; --panel2:#1b2433; --line:#243043;
    --fg:#e6edf6; --muted:#8aa0b8; --accent:#4f9dff;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
    font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
  #app{display:flex;height:100%}
  #stage{position:relative;flex:1;min-width:0}
  canvas{display:block;width:100%;height:100%;cursor:grab}
  canvas.grabbing{cursor:grabbing}
  #side{width:300px;flex:none;background:var(--panel);border-left:1px solid var(--line);
    display:flex;flex-direction:column;overflow:hidden}
  .pad{padding:14px 16px}
  h1{font-size:15px;margin:0 0 2px}
  .sub{color:var(--muted);font-size:12px}
  .stat{display:flex;justify-content:space-between;padding:3px 0;color:var(--muted)}
  .stat b{color:var(--fg);font-variant-numeric:tabular-nums}
  .legend{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
  .chip{display:flex;align-items:center;gap:6px;padding:4px 9px;border:1px solid var(--line);
    border-radius:999px;cursor:pointer;font-size:12px;user-select:none;background:#0e1420}
  .chip.off{opacity:.35;text-decoration:line-through}
  .dot{width:9px;height:9px;border-radius:50%}
  .sec{border-top:1px solid var(--line)}
  .sec h2{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
  #detail{overflow:auto}
  #detail .nm{font-size:15px;font-weight:600;word-break:break-word}
  #nbrs{list-style:none;margin:8px 0 0;padding:0}
  #nbrs li{display:flex;justify-content:space-between;gap:8px;padding:4px 0;
    border-bottom:1px solid #1a2433;cursor:pointer}
  #nbrs li:hover{color:var(--accent)}
  #nbrs .w{color:var(--muted);font-variant-numeric:tabular-nums}
  #bar{position:absolute;top:12px;left:12px;right:12px;display:flex;gap:8px;align-items:center;
    pointer-events:none}
  #bar>*{pointer-events:auto}
  input,select,button{background:#0e1420;color:var(--fg);border:1px solid var(--line);
    border-radius:8px;padding:7px 10px;font:inherit;outline:none}
  input:focus,select:focus{border-color:var(--accent)}
  #q{flex:1;max-width:340px}
  button{cursor:pointer}
  button:hover{border-color:var(--accent)}
  #empty{position:absolute;inset:0;display:none;place-items:center;color:var(--muted);text-align:center}
  #tip{position:absolute;pointer-events:none;background:#0e1420;border:1px solid var(--line);
    border-radius:8px;padding:6px 9px;font-size:12px;display:none;max-width:260px;z-index:5}
  .muted{color:var(--muted)}
  .footer{color:var(--muted);font-size:11px}
</style>
</head>
<body>
<div id="app">
  <div id="stage">
    <canvas id="cv"></canvas>
    <div id="bar">
      <input id="q" placeholder="Search entities…" autocomplete="off">
      <select id="sizeby" title="Size nodes by">
        <option value="mentions">size: mentions</option>
        <option value="recency">size: recency</option>
        <option value="flat">size: flat</option>
      </select>
      <button id="reset">Reset view</button>
    </div>
    <div id="empty">No entities yet.<br><span class="muted">Use PMB for a while, then run <code>pmb regraph</code> and re-export.</span></div>
    <div id="tip"></div>
  </div>
  <aside id="side">
    <div class="pad">
      <h1 id="title">memory</h1>
      <div class="sub" id="sub"></div>
      <div style="margin-top:10px">
        <div class="stat"><span>entities (shown / total)</span><b id="s-nodes">0</b></div>
        <div class="stat"><span>connections</span><b id="s-edges">0</b></div>
      </div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="pad sec" id="detail">
      <h2>Selection</h2>
      <div id="detail-body" class="muted">Click a node to inspect it and its connections.</div>
    </div>
    <div class="pad sec footer" id="foot"></div>
  </aside>
</div>
<script>
const DB = /*__DATA__*/null;
const KIND_COLORS = {person:'#4f9dff',project:'#34d399',topic:'#fbbf24',place:'#f472b6',
  org:'#a78bfa',tool:'#22d3ee',event:'#fb923c',concept:'#f87171',other:'#94a3b8'};
function colorFor(k){ if(KIND_COLORS[k])return KIND_COLORS[k];
  let h=0; for(const c of (k||'')) h=(h*31+c.charCodeAt(0))%360; return 'hsl('+h+',60%,62%)'; }

const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const tip=document.getElementById('tip');
let W=0,H=0,DPR=Math.max(1,window.devicePixelRatio||1);
let view={x:0,y:0,k:1};                       // pan/zoom
const nodes=DB.nodes||[], edges=DB.edges||[];
const byId=new Map(nodes.map(n=>[n.id,n]));
const adj=new Map();                          // id -> [{o, w}]
edges.forEach(e=>{ if(!byId.has(e.a)||!byId.has(e.b))return;
  (adj.get(e.a)||adj.set(e.a,[]).get(e.a)).push({o:e.b,w:e.w});
  (adj.get(e.b)||adj.set(e.b,[]).get(e.b)).push({o:e.a,w:e.w}); });

// ---- node sizing ----
let now=Date.now()/1000, maxMent=1, minSeen=now, maxSeen=0;
nodes.forEach(n=>{ maxMent=Math.max(maxMent,n.mentions||0);
  if(n.last_seen){minSeen=Math.min(minSeen,n.last_seen);maxSeen=Math.max(maxSeen,n.last_seen);} });
let sizeMode='mentions';
function radius(n){
  if(sizeMode==='flat') return 6;
  if(sizeMode==='recency'){ const span=Math.max(1,maxSeen-minSeen);
    return 4+10*((n.last_seen||minSeen)-minSeen)/span; }
  return 4+9*Math.sqrt((n.mentions||0)/maxMent);
}

// ---- layout: simple force-directed ----
nodes.forEach((n,i)=>{ const a=i*2.399963; const r=20+Math.sqrt(i)*22;
  n.x=Math.cos(a)*r; n.y=Math.sin(a)*r; n.vx=0; n.vy=0; });
let alpha=1.0;
function tick(){
  if(alpha<0.02) return;
  const k=Math.sqrt((W*H)/(nodes.length+1))/3;     // ideal spacing
  for(let i=0;i<nodes.length;i++){ const a=nodes[i];
    for(let j=i+1;j<nodes.length;j++){ const b=nodes[j];
      let dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy+0.01, d=Math.sqrt(d2);
      const rep=(k*k)/d2*alpha*4;
      const fx=dx/d*rep, fy=dy/d*rep; a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy; } }
  edges.forEach(e=>{ const a=byId.get(e.a),b=byId.get(e.b); if(!a||!b)return;
    let dx=b.x-a.x, dy=b.y-a.y, d=Math.sqrt(dx*dx+dy*dy)+0.01;
    const f=(d-k)*0.02*Math.min(3,e.w)*alpha; const fx=dx/d*f, fy=dy/d*f;
    a.vx+=fx;a.vy+=fy;b.vx-=fx;b.vy-=fy; });
  nodes.forEach(n=>{ n.vx-=n.x*0.002*alpha; n.vy-=n.y*0.002*alpha;   // gravity to center
    if(n===dragNode)return; n.x+=n.vx*0.5; n.y+=n.vy*0.5; n.vx*=0.82; n.vy*=0.82; });
  alpha*=0.985;
}

// ---- filters / selection ----
const kindsOff=new Set(); let selected=null, query='';
function visible(n){ if(kindsOff.has(n.kind))return false;
  if(query && !(n.name||'').toLowerCase().includes(query))return false; return true; }
function isNeighbor(id){ if(!selected)return false; if(id===selected)return true;
  return (adj.get(selected)||[]).some(e=>e.o===id); }

// ---- render ----
function toScreen(x,y){ return [ (x*view.k)+view.x+W/2, (y*view.k)+view.y+H/2 ]; }
function fromScreen(px,py){ return [ (px-view.x-W/2)/view.k, (py-view.y-H/2)/view.k ]; }
function draw(){
  tick();
  ctx.setTransform(DPR,0,0,DPR,0,0); ctx.clearRect(0,0,W,H);
  // edges
  ctx.lineWidth=1;
  edges.forEach(e=>{ const a=byId.get(e.a),b=byId.get(e.b); if(!a||!b)return;
    if(!visible(a)||!visible(b))return;
    const foc=selected? (isNeighbor(a.id)&&isNeighbor(b.id)):true;
    const [ax,ay]=toScreen(a.x,a.y),[bx,by]=toScreen(b.x,b.y);
    ctx.strokeStyle= foc? 'rgba(120,160,210,'+Math.min(.5,.12+e.w*0.06)+')':'rgba(90,110,140,0.05)';
    ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke(); });
  // nodes
  ctx.font='12px -apple-system,Segoe UI,Roboto,sans-serif'; ctx.textAlign='center';
  nodes.forEach(n=>{ if(!visible(n))return;
    const [x,y]=toScreen(n.x,n.y), r=radius(n)*Math.sqrt(view.k);
    const foc= !selected || isNeighbor(n.id);
    ctx.globalAlpha= foc?1:0.18;
    ctx.beginPath();ctx.arc(x,y,r,0,6.2832);
    ctx.fillStyle=colorFor(n.kind);ctx.fill();
    if(n.id===selected){ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();}
    if((r>9||n.id===selected||isNeighbor(n.id)&&selected) && foc){
      ctx.globalAlpha=foc?0.95:0.2;ctx.fillStyle='#dce6f2';
      ctx.fillText((n.name||'').slice(0,26), x, y-r-4); }
  });
  ctx.globalAlpha=1;
  requestAnimationFrame(draw);
}

// ---- interaction ----
let dragNode=null,dragging=false,last=null,moved=false;
function nodeAt(px,py){ let best=null,bd=1e9;
  for(const n of nodes){ if(!visible(n))continue; const [x,y]=toScreen(n.x,n.y);
    const r=Math.max(8,radius(n)*Math.sqrt(view.k)+4); const d=(x-px)**2+(y-py)**2;
    if(d<r*r && d<bd){bd=d;best=n;} } return best; }
cv.addEventListener('mousedown',e=>{ const n=nodeAt(e.offsetX,e.offsetY); moved=false;
  if(n){dragNode=n;} dragging=true;last=[e.offsetX,e.offsetY];cv.classList.add('grabbing'); });
window.addEventListener('mousemove',e=>{
  const px=e.offsetX,py=e.offsetY;
  if(dragging){ moved=true; const dx=px-last[0],dy=py-last[1];last=[px,py];
    if(dragNode){ const [wx,wy]=fromScreen(px,py); dragNode.x=wx;dragNode.y=wy;dragNode.vx=0;dragNode.vy=0; alpha=Math.max(alpha,0.2);}
    else{ view.x+=dx;view.y+=dy; } return; }
  const n=nodeAt(px,py);
  if(n){ tip.style.display='block'; tip.style.left=(px+14)+'px'; tip.style.top=(py+12)+'px';
    tip.innerHTML='<b>'+esc(n.name)+'</b><br><span class="muted">'+esc(n.kind)+' · '+(n.mentions||0)+' mentions</span>'; cv.style.cursor='pointer'; }
  else{ tip.style.display='none'; cv.style.cursor=''; }
});
window.addEventListener('mouseup',e=>{ if(dragging&&!moved){ const n=nodeAt(e.offsetX,e.offsetY); select(n?n.id:null);}
  dragging=false;dragNode=null;cv.classList.remove('grabbing'); });
cv.addEventListener('wheel',e=>{ e.preventDefault(); const f=e.deltaY<0?1.12:0.89;
  const [wx,wy]=fromScreen(e.offsetX,e.offsetY); view.k=Math.min(5,Math.max(0.15,view.k*f));
  const [nx,ny]=toScreen(wx,wy); view.x+=e.offsetX-nx; view.y+=e.offsetY-ny; },{passive:false});

function select(id){ selected=id; renderDetail(); }
function esc(s){ const d=document.createElement('div'); d.textContent=s==null?'':s; return d.innerHTML; }

// ---- side panel ----
function renderDetail(){
  const el=document.getElementById('detail-body');
  if(selected==null){ el.className='muted';
    el.textContent='Click a node to inspect it and its connections.'; return; }
  const n=byId.get(selected); if(!n)return; el.className='';
  const nb=(adj.get(selected)||[]).slice().sort((a,b)=>b.w-a.w).slice(0,40);
  el.innerHTML='<div class="nm" style="color:'+colorFor(n.kind)+'">'+esc(n.name)+'</div>'+
    '<div class="muted" style="margin:2px 0 6px">'+esc(n.kind)+' · '+(n.mentions||0)+' mentions · '+nb.length+' connections</div>'+
    '<ul id="nbrs">'+nb.map(e=>{const o=byId.get(e.o);return o?'<li data-id="'+o.id+'"><span>'+esc(o.name)+'</span><span class="w">'+e.w+'</span></li>':'';}).join('')+'</ul>';
  el.querySelectorAll('#nbrs li').forEach(li=>li.onclick=()=>select(+li.dataset.id));
}
function renderLegend(){
  const counts=DB.stats.by_kind||{}; const kinds=[...new Set(nodes.map(n=>n.kind))].sort();
  const L=document.getElementById('legend');
  L.innerHTML=kinds.map(k=>'<div class="chip" data-k="'+esc(k)+'"><span class="dot" style="background:'+colorFor(k)+'"></span>'+esc(k)+' <span class="muted">'+(counts[k]||nodes.filter(n=>n.kind===k).length)+'</span></div>').join('');
  L.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{ const k=c.dataset.k;
    if(kindsOff.has(k)){kindsOff.delete(k);c.classList.remove('off');}else{kindsOff.add(k);c.classList.add('off');} });
}

// ---- boot ----
function resize(){ W=cv.clientWidth;H=cv.clientHeight; cv.width=W*DPR;cv.height=H*DPR; }
window.addEventListener('resize',resize);
document.getElementById('q').addEventListener('input',e=>{query=e.target.value.toLowerCase();});
document.getElementById('sizeby').addEventListener('change',e=>{sizeMode=e.target.value;alpha=Math.max(alpha,0.1);});
document.getElementById('reset').addEventListener('click',()=>{view={x:0,y:0,k:1};select(null);alpha=1;});
document.getElementById('title').textContent='__TITLE__';
document.getElementById('sub').textContent='memory graph · '+DB.generated;
document.getElementById('s-nodes').textContent=DB.shown+' / '+DB.total;
document.getElementById('s-edges').textContent=edges.length;
document.getElementById('foot').innerHTML='Generated by <b>PMB</b> · self-contained · offline. Drag to pan · scroll to zoom · click a node to focus.';
if(!nodes.length){document.getElementById('empty').style.display='grid';}
renderLegend(); renderDetail(); resize(); requestAnimationFrame(draw);
</script>
</body>
</html>
"""
