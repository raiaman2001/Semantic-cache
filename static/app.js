const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const LBL={L1_exact:"L1 exact hit",L2_semantic:"L2 semantic hit",L3_llm:"L3 LLM call"};
async function api(path,method="GET",body){
  const h={"Content-Type":"application/json"};if($("#apikey").value)h["x-api-key"]=$("#apikey").value;
  const r=await fetch("/api"+path,{method,headers:h,body:body?JSON.stringify(body):undefined});
  if(!r.ok){const e=await r.json().catch(()=>({}));toast(e.detail||"Request failed");throw new Error(e.detail)}return r.json()}
function toast(m){const t=$("#toast");t.textContent=m;t.classList.add("on");setTimeout(()=>t.classList.remove("on"),2500)}
document.querySelectorAll("#tabs button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("#tabs button,main section").forEach(e=>e.classList.remove("on"));b.classList.add("on");$("#"+b.dataset.t).classList.add("on");
  ({stats:loadStats,entries:loadEntries})[b.dataset.t]?.()});
function setMeta(box,r){
  box.querySelector(".meta").innerHTML=`<span class="badge ${r.level}">${LBL[r.level]}</span> ${r.ms} ms`+(r.level=="L2_semantic"?` · similarity ${r.score} (threshold ${r.threshold}, ${r.category})`:"")+(r.coalesced?" · shared with an identical in-flight request":"");
  if(r.level=="L2_semantic"&&r.log_id){const f=document.createElement("div");f.className="row";f.style.marginTop="8px";
    f.innerHTML=`<button class="btn ghost" data-g="1">Correct match</button><button class="btn ghost" data-g="0">Wrong match</button>`;
    f.querySelectorAll("button").forEach(x=>x.onclick=async()=>{await api("/feedback","POST",{log_id:r.log_id,good:x.dataset.g=="1"});toast(x.dataset.g=="1"?"Thanks. Threshold relaxed slightly.":"Noted. Threshold tightened.");f.remove()});box.append(f)}}
async function ask(){
  const q=$("#q").value.trim();if(!q)return;$("#go").disabled=true;
  const body={query:q,ns:$("#ns").value,doc_ver:$("#ver").value,tag:$("#tag").value,use_cache:$("#usec").checked};
  const box=document.createElement("div");box.className="msg";box.innerHTML=`<b>${esc(q)}</b><div class="meta mute">Thinking…</div><div class="ans"></div>`;
  if($("#chat .empty"))$("#chat").innerHTML="";$("#chat").prepend(box);$("#q").value="";const ans=box.querySelector(".ans");
  try{if($("#stream").checked){
      const h={"Content-Type":"application/json"};if($("#apikey").value)h["x-api-key"]=$("#apikey").value;
      const resp=await fetch("/api/ask/stream",{method:"POST",headers:h,body:JSON.stringify(body)});if(!resp.ok)throw new Error((await resp.json()).detail);
      const rd=resp.body.getReader(),dec=new TextDecoder();let buf="";
      for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const parts=buf.split("\n\n");buf=parts.pop();
        for(const p of parts){const o=JSON.parse(p.slice(6));if(o.meta)setMeta(box,o.meta);if(o.t)ans.textContent+=o.t}}
    }else{const r=await api("/ask","POST",body);setMeta(box,r);ans.textContent=r.answer}
  }catch(e){box.querySelector(".meta").textContent="Failed: "+e.message}finally{$("#go").disabled=false}}
$("#go").onclick=ask;$("#q").onkeydown=e=>e.key=="Enter"&&ask();
const tbl=(el,head,rows)=>$(el).innerHTML=rows.length?`<tr>${head.map(h=>`<th>${h}</th>`).join("")}</tr>`+rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`).join(""):`<tr><td class="empty">Nothing here yet.</td></tr>`;
async function loadStats(){const s=await api("/stats"),L=s.levels,ms=k=>L[k]?Math.round(L[k].avg_ms)+" ms":"-";
  $("#kpis").innerHTML=[["Hit rate",(s.hit_rate*100).toFixed(0)+"%"],["Requests",s.total],["Cached entries",s.entries],
    ["Tokens saved",s.tokens_saved],["Cost saved","$"+s.cost_saved.toFixed(4)],["Avg L1 / L2 / LLM",`${ms("L1_exact")} / ${ms("L2_semantic")} / ${ms("L3_llm")}`]]
    .map(([l,v])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join("");
  $("#thr").innerHTML=Object.entries(s.thresholds).map(([k,v])=>`<label>${k} <input type="number" step="0.01" min="0.5" max="0.99" id="t_${k}" value="${v}" style="width:80px"></label>`).join("")+`<button class="btn" id="savethr">Save</button>`;
  $("#savethr").onclick=async()=>{await api("/threshold","POST",{code:+$("#t_code").value,general:+$("#t_general").value});toast("Thresholds saved")};
  tbl("#recent",["Query","Result","Score","Time"],s.recent.map(r=>[esc(r.query),`<span class="badge ${r.level}">${LBL[r.level]}</span>`,r.score.toFixed(3),Math.round(r.ms)+" ms"]));
  tbl("#top",["Query","Reuses"],s.top.filter(t=>t.hits).map(t=>[esc(t.query),t.hits]))}
async function loadEntries(){const e=await api("/entries");
  tbl("#etable",["Query","Answer","Namespace","Tag","Reuses",""],e.map(x=>[esc(x.query),esc(x.answer.slice(0,90))+(x.answer.length>90?"…":""),esc(x.ns),esc(x.tag||"-"),x.hits,`<button class="btn ghost" data-d="${x.id}">Delete</button>`]));
  document.querySelectorAll("[data-d]").forEach(b=>b.onclick=async()=>{await api("/entries/"+b.dataset.d,"DELETE");loadEntries()})}
$("#clr").onclick=async()=>{if(confirm("Delete every cached entry?")){await api("/clear","POST");loadEntries();toast("Cache cleared")}};
$("#inv").onclick=async()=>{const r=await api("/invalidate","POST",{tag:$("#invtag").value});toast(r.deleted+" entries removed");loadEntries()};
$("#dowarm").onclick=async()=>{const q=$("#warm").value.split("\n");toast("Warming…");const r=await api("/warm","POST",{queries:q,ns:$("#ns").value});toast(r.warmed+" queries cached")};
$("#doeval").onclick=async()=>{const d=await api("/evaluate"),c=d.curve,W=600,H=220,x=t=>40+(t-0.5)/0.48*(W-60),y=v=>H-25-v*(H-45);
  const line=(k,col)=>`<polyline fill="none" stroke="${col}" stroke-width="2.5" points="${c.map(p=>x(p.t)+","+y(p[k])).join(" ")}"/>`;
  $("#evalout").innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Precision and recall by threshold">${[0,.5,1].map(v=>`<line x1="40" x2="${W-20}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="6" y="${y(v)+4}" fill="var(--mute)" font-size="11">${v}</text>`).join("")}
    ${c.filter((_,i)=>i%3==0).map(p=>`<text x="${x(p.t)}" y="${H-6}" fill="var(--mute)" font-size="11" text-anchor="middle">${p.t}</text>`).join("")}${line("precision","var(--acc)")}${line("recall","var(--hit)")}${line("precision_rr","var(--miss)")}
    <text x="${W-120}" y="14" fill="var(--acc)" font-size="12">Precision</text><text x="${W-60}" y="14" fill="var(--hit)" font-size="12">Recall</text><text x="${W-200}" y="14" fill="var(--miss)" font-size="12">Precision + reranker</text></svg>
    <div class="scroll"><table><tr><th>Pair</th><th>Same meaning</th><th>Similarity</th><th>Reranker</th><th>Accepted</th></tr>${d.pairs.map(p=>`<tr><td>${esc(p.a)}<br><span class="mute">${esc(p.b)}</span></td><td>${p.same?"Yes":"No"}</td><td>${p.sim}</td><td>${p.rerank}</td><td>${p.pass?"Yes":"No"}</td></tr>`).join("")}</table></div>`};
$("#dosim").onclick=async()=>{const r=await api(`/simulate?queries=${$("#sq").value}&hit_rate=${$("#sh").value}&llm_ms=${$("#sl").value}`);
  $("#simout").innerHTML=[["LLM calls saved",r.llm_calls_saved],["Time saved",r.time_saved_min+" min"],["Cost saved","$"+r.cost_saved],["Avg latency",r.avg_ms+" ms"]].map(([l,v])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join("")};

api("/config").then(c=>$("#cfg").textContent=`L1: ${c.l1} · Vectors: ${c.vectors} · Reranker: ${c.reranker?"on":"off"}`).catch(()=>{});
