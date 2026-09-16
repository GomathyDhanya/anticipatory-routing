/* Offline replay of recorded Python simulation events. No synthetic animation
   paths, routing decisions, live requests, or fabricated metric counters. */
"use strict";
(() => {
  const data = window.TRAFFIC_DEMO;
  if (!data) {
    document.getElementById("status").classList.remove("sr-only");
    document.getElementById("status").textContent = "Replay data is missing. Run python3 -m traffic_routing.map_demo from the project directory.";
    return;
  }
  const $ = id => document.getElementById(id);
  const TAU = Math.PI * 2;
  const descriptions = {
    static: "Chooses the shortest free-flow route.",
    reactive: "Responds to queues it can observe now.",
    predictive: "Adds forecasts of background arrivals.",
    anticipatory: "Also considers drivers already on their way.",
    cooperative: "Also considers the delay imposed on others.",
    rl_swarm: "Learned agents coordinate through road commitments."
  };
  const palettes = {left: "#ffc575", right: "#6aebcf"};
  const mercY = lat => Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360));
  const [west, south, east, north] = data.bounds;
  const westX = west * Math.PI / 180, northY = mercY(north);
  const worldWidth = (east-west) * Math.PI / 180, worldHeight = northY-mercY(south);
  const worldPoint = p => [p[0] * Math.PI / 180 - westX, northY - mercY(p[1])];
  const roadGeometry = {};
  for (const [id, road] of Object.entries(data.roads)) {
    const points = road.geometry.map(worldPoint), cumulative = [0];
    for (let i=1; i<points.length; i++) cumulative.push(cumulative[i-1] + Math.hypot(points[i][0]-points[i-1][0], points[i][1]-points[i-1][1]));
    roadGeometry[id] = {points, cumulative, length:cumulative.at(-1)};
  }
  const baseRoads = data.basemap.map(r => ({...r, points:r.points.map(worldPoint)}));
  let scene, time=0, end=1, running=false, lastFrame=0, lastReadout=-1;
  const view = {zoom:1, x:0, y:0};
  const panels = ["left", "right"].map(side => ({side, element:$(side+"-map"),
    base:$(side+"-map").querySelector(".base"), motion:$(side+"-map").querySelector(".motion"),
    color:palettes[side], width:0, height:0, ratio:1, stats:null, run:null}));
  let selectedRoad = Object.keys(data.roads)[0];
  const formatTime = tick => {
    const seconds=Math.floor(tick*data.tickSeconds);
    return `${String(Math.floor(seconds/60)).padStart(2,"0")}:${String(seconds%60).padStart(2,"0")}`;
  };
  const minutes = ticks => (ticks*data.tickSeconds/60).toFixed(1);
  const s = (p, panel) => {
    const scale = Math.min(panel.width/worldWidth, panel.height/worldHeight)*.89*view.zoom;
    return [(p[0]-worldWidth/2)*scale + panel.width*(.5+view.x),
            (p[1]-worldHeight/2)*scale + panel.height*(.5+view.y)];
  };
  function path(ctx, points, panel) {
    ctx.beginPath();
    points.forEach((point,index) => { const [x,y]=s(point,panel); if(index)ctx.lineTo(x,y);else ctx.moveTo(x,y); });
  }
  function text(ctx, label, x, y, size=11, color="#aec1d7", align="center") {
    ctx.font=`${size}px system-ui, sans-serif`;ctx.textAlign=align;ctx.lineJoin="round";
    ctx.lineWidth=4;ctx.strokeStyle="#111c2a";ctx.strokeText(label,x,y);ctx.fillStyle=color;ctx.fillText(label,x,y);
  }
  function paintBase(panel) {
    const ctx=panel.base.getContext("2d");ctx.setTransform(panel.ratio,0,0,panel.ratio,0,0);
    ctx.fillStyle="#111c2a";ctx.fillRect(0,0,panel.width,panel.height);
    ctx.lineCap="round";ctx.lineJoin="round";
    for (const road of baseRoads) {
      const major=/^(motorway|trunk)/.test(road.kind);
      path(ctx,road.points,panel);ctx.strokeStyle=major?"#304155":"#243447";
      ctx.lineWidth=major?2:1;ctx.stroke();
    }
    for (const label of panel.width >= 440 ? data.labels : []) {
      const [x,y]=s(worldPoint(label.point),panel);
      if (x>75 && x<panel.width-75 && y>50 && y<panel.height-25) {
        text(ctx,label.name.replace("North ","N ").replace("Expressway","Expwy").replace("Parkway","Pkwy"),x,y,9,"#778da8");
      }
    }
    for (const [point,name,symbol,dx,dy] of [[data.origin,"MILPITAS","A",-10,-16],[data.destination,"SUNNYVALE","B",12,22]]) {
      const [x,y]=s(worldPoint(point),panel);ctx.beginPath();ctx.arc(x,y,7,0,TAU);ctx.fillStyle="#e3edfa";ctx.fill();
      ctx.font="bold 9px system-ui";ctx.textAlign="center";ctx.fillStyle="#142233";ctx.fillText(symbol,x,y+3);
      text(ctx,name,x+dx,y+dy,11,"#d4e4f5",name==="MILPITAS"?"right":"left");
    }
    const scale=Math.min(panel.width/worldWidth,panel.height/worldHeight)*.89*view.zoom;
    const pxPerKm=scale/(6371*Math.cos((south+north)/2*Math.PI/180));
    const kilometers=pxPerKm>90?.5:1;
    const bar=panel.element.querySelector(".map-scale");
    bar.style.width=(pxPerKm*kilometers)+"px";bar.textContent=kilometers===1?"1 km":"500 m";
  }
  function resize() {
    for (const panel of panels) {
      const bounds=panel.element.getBoundingClientRect();
      panel.width=bounds.width;panel.height=bounds.height;panel.ratio=Math.min(window.devicePixelRatio||1,2);
      for(const canvas of [panel.base,panel.motion]) { canvas.width=Math.round(panel.width*panel.ratio);canvas.height=Math.round(panel.height*panel.ratio); }
      paintBase(panel);
    }
    render(true);
  }
  function interpolate(id, fraction, panel) {
    const geometry=roadGeometry[id];const target=Math.max(0,Math.min(1,fraction))*geometry.length;
    let i=1;while(i<geometry.cumulative.length-1 && geometry.cumulative[i]<target)i++;
    const a=geometry.points[i-1],b=geometry.points[i];
    const local=(target-geometry.cumulative[i-1])/(geometry.cumulative[i]-geometry.cumulative[i-1]||1);
    const xy=s([a[0]+(b[0]-a[0])*local,a[1]+(b[1]-a[1])*local],panel);
    return [...xy,Math.atan2(b[1]-a[1],b[0]-a[0])];
  }
  function snapshot(run,t) {
    const queues={}, incoming={}, moving=[], waiting=[];
    let completed=0,active=0;
    for(const vehicle of run.vehicles) {
      if(vehicle.arrival<=t){completed++;continue;}
      if(vehicle.departure>t)continue;
      active++;
      const index=vehicle.visits.findIndex(v=>v[1]<=t && t<v[3]);
      if(index<0)continue;
      const visit=vehicle.visits[index],edge=visit[0];
      if(t<visit[2]) {queues[edge]=(queues[edge]||0)+1;waiting.push({vehicle,visit});}
      else moving.push({vehicle,visit});
      let announced=vehicle.initialPath || vehicle.visits.map(v=>v[0]);
      if(!vehicle.initialPath) {
        const first=(run.replans || []).find(e=>e.id===vehicle.id && e.changed);
        if(first)announced=vehicle.visits.filter(v=>v[3]<=first.tick).map(v=>v[0]).concat(first.old_path);
      }
      for(const event of run.replans || []) {
        if(event.id===vehicle.id && event.tick<=t)announced=event.path;
      }
      for(const next of announced.slice(announced.indexOf(edge)+1)) {
        incoming[next]=(incoming[next]||0)+1;
      }
    }
    return {queues,incoming,moving,waiting,completed,active};
  }
  function drawCar(ctx,x,y,angle,color,waiting=false) {
    ctx.save();ctx.translate(x,y);ctx.rotate(angle);
    ctx.fillStyle=color;ctx.strokeStyle="#0a1420";ctx.lineWidth=.7;
    ctx.beginPath();ctx.rect(-2.8,-1.5,5.6,3);ctx.fill();ctx.stroke();
    if(waiting){ctx.fillStyle="#ff8280";ctx.fillRect(-2.8,-1.5,.8,3);}
    ctx.restore();
  }
  function paintTraffic(panel) {
    const ctx=panel.motion.getContext("2d");ctx.setTransform(panel.ratio,0,0,panel.ratio,0,0);ctx.clearRect(0,0,panel.width,panel.height);
    const state=snapshot(panel.run,time);panel.stats=state;
    ctx.lineCap="round";ctx.lineJoin="round";
    for(const [id,geometry] of Object.entries(roadGeometry)) {
      const queue=state.queues[id]||0, waitSeconds=Math.floor(queue/scene.capacities[id])*data.tickSeconds;
      path(ctx,geometry.points,panel);
      if(id===selectedRoad){ctx.strokeStyle="#b7d5f333";ctx.lineWidth=11;ctx.stroke();}
      ctx.strokeStyle=queue===0?"#497d8c":waitSeconds<30?"#efb15d":"#f57570";
      ctx.lineWidth=queue===0?2.1:3.9;ctx.stroke();
      if(id===scene.bottleneck){ctx.setLineDash([4,5]);ctx.strokeStyle="#f7c482";ctx.lineWidth=1.3;ctx.stroke();ctx.setLineDash([]);}
    }
    for(const {vehicle,visit} of state.moving) {
      const [x,y,angle]=interpolate(visit[0],(time-visit[2])/(visit[3]-visit[2]),panel);
      drawCar(ctx,x,y,angle,vehicle.controlled?panel.color:"#a3b4c9");
    }
    const queueOrder={};
    for(const {vehicle,visit} of state.waiting) {
      const edge=visit[0],i=queueOrder[edge]||0;queueOrder[edge]=i+1;
      const [x,y,angle]=interpolate(edge,0,panel),columns=Math.ceil(Math.sqrt(state.queues[edge]));
      // Point queues have no physical length. A compact marker cluster keeps
      // every waiting vehicle visible without inventing road spillback.
      const rows=Math.ceil(state.queues[edge]/columns),padding=columns*2.5+5;
      const cx=Math.max(padding,Math.min(panel.width-padding,x)),cy=Math.max(padding,Math.min(panel.height-padding,y));
      drawCar(ctx,cx+(i%columns-(columns-1)/2)*5,cy+(Math.floor(i/columns)-(rows-1)/2)*4,angle,vehicle.controlled?panel.color:"#a3b4c9",true);
    }
    for(const [id,count] of Object.entries(state.queues)) {
      if(count<5)continue;
      const [x,y]=interpolate(id,0,panel);text(ctx,String(count)+" waiting",x+8,y-12,10,"#ffc3a7","left");
    }
    if(scene.bottleneck){const [x,y]=interpolate(scene.bottleneck,.55,panel);text(ctx,"Simulated capacity limit",x,y-13,10,"#f2ce94");}
    for(const event of panel.run.replans || []) {
      if(!event.changed || time<event.tick || time>event.tick+3)continue;
      const [x,y]=interpolate(event.path[0],0,panel);
      ctx.beginPath();ctx.arc(x,y,8+(time-event.tick)*5,0,Math.PI*2);
      ctx.strokeStyle="#dfadff";ctx.lineWidth=2;ctx.stroke();
      text(ctx,"Rerouted",x,y-18,10,"#dfadff");
    }
  }
  function readouts() {
    $("clock").textContent=formatTime(time);$("time").value=time;
    $("time").setAttribute("aria-valuetext",`${formatTime(time)} elapsed`);
    for(const panel of panels) {
      const state=panel.stats,side=panel.side;
      $(side+"-queued").textContent=state.waiting.length;
      $(side+"-moving").textContent=state.moving.length;
      $(side+"-arrived").textContent=`${state.completed} / ${scene.vehicles}`;
      $(side+"-detail").textContent=`${state.queues[selectedRoad]||0} queued · ${state.incoming[selectedRoad]||0} committed`;
    }
  }
  function render(force=false) {
    if(!scene || !panels[0].width)return;
    panels.forEach(paintTraffic);
    if(force || Math.floor(time*4)!==lastReadout){readouts();lastReadout=Math.floor(time*4);}
  }
  function updatePolicies() {
    for(const panel of panels) {
      const policy=$(panel.side+"-policy").value;
      const replan=$(panel.side+"-planning").value==="replan";
      panel.run=(replan?scene.replanning:scene.policies)[policy];
      $(panel.side+"-description").textContent=descriptions[policy]+(replan?` · Junction planning · ${panel.run.metrics.route_changes} route changes over this run`:" · Departure planning");
      $(panel.side+"-mean").textContent=minutes(panel.run.metrics.average_travel_time)+" min";
    }
    end=Math.max(...panels.map(p=>p.run.metrics.last_arrival));time=Math.min(time,end);
    $("time").max=end;$("duration").textContent="/ "+formatTime(end);
    const baseline=panels[0].run.metrics,improved=panels[1].run.metrics;
    const change=100*(baseline.average_travel_time-improved.average_travel_time)/baseline.average_travel_time;
    $("outcome").textContent=Math.abs(change)<.05?"Same average trip time in this scenario":`${Math.abs(change).toFixed(1)}% ${change>=0?"shorter":"longer"} average trips on the right`;
    const delay=(baseline.total_delay-improved.total_delay)*data.tickSeconds/60;
    $("delay-outcome").textContent=`${Math.abs(delay).toFixed(0)} ${delay>=0?"fewer":"more"} vehicle-minutes spent queueing across all trips`;
    $("rl-note").hidden=$("right-policy").value!=="rl_swarm";
    if(data.rl)$("rl-note").textContent=`RL weights frozen · ${data.rl.metadata.training_episodes} training episodes · The evaluation below covers departure planning. Junction planning reuses those weights without retraining.`;
    render(true);
  }
  function loadScene() {
    scene=data.scenes.find(s=>s.id===$("scenario").value);
    $("report-link").href=`reports/${scene.id}/report.html`;
    $("metrics-link").href=`reports/${scene.id}/metrics.csv`;
    time=0;setRunning(false);
    const extra=scene.bottleneck?" · Hypothetical CA 237 capacity reduction":"";
    $("scene-note").textContent=`${scene.vehicles} vehicles · ${minutes(scene.departureWindow)}-minute departure window${extra}`;
    selectedRoad=scene.bottleneck || Object.keys(data.roads).find(e=>data.roads[e].is_highway) || selectedRoad;
    $("road").value=selectedRoad;updatePolicies();
    $("status").textContent=`${scene.label} loaded. Playback paused.`;
  }
  function setRunning(value) {
    if(running!==value)$("status").textContent=value?"Synchronized playback started.":"Playback paused.";
    running=value;lastFrame=0;$("play").textContent=running?"Ⅱ Pause":"▶ Play";
    $("play").setAttribute("aria-pressed",String(running));
  }
  function frame(timestamp) {
    if(running){
      if(lastFrame){time=Math.min(end,time+Math.min((timestamp-lastFrame)/1000,.15)*Number($("speed").value));render();}
      if(time>=end){setRunning(false);$("status").textContent="Comparison complete. All vehicles arrived.";}
    }
    lastFrame=timestamp;requestAnimationFrame(frame);
  }
  function refreshView(){panels.forEach(paintBase);render(true);}
  function zoom(delta){view.zoom=Math.max(.8,Math.min(5,view.zoom*delta));refreshView();}
  $("play").addEventListener("click",()=>{if(time>=end)time=0;setRunning(!running);});
  $("restart").addEventListener("click",()=>{time=0;setRunning(false);render(true);});
  $("time").addEventListener("input",()=>{time=Number($("time").value);setRunning(false);render(true);});
  for(const side of ["left","right"]) {
    const policySelect=$(side+"-policy");
    const names={static:"Static shortest path",reactive:"Reactive routing",predictive:"Predictive routing",anticipatory:"Anticipatory routing",cooperative:"Cooperative routing",rl_swarm:"RL swarm coordination"};
    for(const [value,name] of Object.entries(names)) {
      if(!Array.from(policySelect.options).some(o=>o.value===value)) {
        const option=document.createElement("option");option.value=value;option.textContent=name;policySelect.append(option);
      }
    }
    const label=document.createElement("label");
    label.textContent="Planning ";
    const select=document.createElement("select");select.id=side+"-planning";
    select.innerHTML='<option value="departure">At departure</option><option value="replan">At every junction</option>';
    select.value="replan";label.append(select);
    $(side+"-policy").parentElement.append(label);
    select.addEventListener("change",()=>{time=0;setRunning(false);updatePolicies();});
  }
  $("scenario").addEventListener("change",loadScene);
  for(const side of ["left","right"])$(side+"-policy").addEventListener("change",updatePolicies);
  $("road").addEventListener("change",()=>{selectedRoad=$("road").value;render(true);});
  document.querySelectorAll("[data-zoom]").forEach(button=>button.addEventListener("click",()=>{
    if(button.dataset.zoom==="reset"){Object.assign(view,{zoom:1,x:0,y:0});refreshView();}
    else zoom(button.dataset.zoom==="in"?1.3:1/1.3);
  }));
  function distanceToSegment(p,a,b) {
    const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;
    const f=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0;
    return Math.hypot(p[0]-a[0]-f*dx,p[1]-a[1]-f*dy);
  }
  for(const panel of panels) {
    let drag=null;
    panel.motion.addEventListener("wheel",event=>{event.preventDefault();zoom(event.deltaY<0?1.12:1/1.12);},{passive:false});
    panel.motion.addEventListener("pointerdown",event=>{drag={startX:event.clientX,startY:event.clientY,lastX:event.clientX,lastY:event.clientY};panel.motion.setPointerCapture(event.pointerId);});
    panel.motion.addEventListener("pointermove",event=>{if(!drag)return;view.x+=(event.clientX-drag.lastX)/panel.width;view.y+=(event.clientY-drag.lastY)/panel.height;drag.lastX=event.clientX;drag.lastY=event.clientY;refreshView();});
    panel.motion.addEventListener("pointerup",event=>{
      if(!drag)return;
      if(Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)<5) {
        const rect=panel.motion.getBoundingClientRect(),point=[event.clientX-rect.left,event.clientY-rect.top];
        let best=null,bestDistance=14;
        for(const [id,geometry] of Object.entries(roadGeometry)) for(let i=1;i<geometry.points.length;i++) {
          const d=distanceToSegment(point,s(geometry.points[i-1],panel),s(geometry.points[i],panel));
          if(d<bestDistance){bestDistance=d;best=id;}
        }
        if(best){selectedRoad=best;$("road").value=best;render(true);}
      }
      drag=null;
    });
    panel.motion.addEventListener("pointercancel",()=>{drag=null;});
  }
  for(const [id,road] of Object.entries(data.roads)) {
    const option=document.createElement("option");option.value=id;
    option.textContent=`${road.ref?road.ref+" · ":""}${road.name} (${road.meters} m)`;$("road").append(option);
  }
  $("source-info").textContent=`Map source timestamp: ${data.sources.osmTimestamp || "See cached source files"}. Route data and geometry are cached; playback makes no network requests.`;
  if(data.rl) {
    $("training-summary").textContent=`Parameter-sharing REINFORCE, trained from zero weights for ${data.rl.metadata.training_episodes} simulated commutes. Checkpoint selected at episode ${data.rl.metadata.selected_episode} using six validation seeds. The ${data.rl.metadata.test_seeds.length} test seeds and this demo's seed were excluded from training and selection.`;
    const names={reactive:"Reactive",anticipatory:"Anticipatory",cooperative:"Cooperative",rl_swarm:"RL swarm",rl_no_commitments:"RL with future commitments hidden",untrained_uniform:"Untrained uniform policy"};
    for(const [policy,metrics] of Object.entries(data.rl.summary)) {
      const row=document.createElement("tr"),name=document.createElement("th"),value=document.createElement("td");
      name.textContent=names[policy]||policy;
      value.textContent=`${metrics.mean_minutes_across_seeds.toFixed(2)} min (SD ${metrics.stdev_minutes_across_seeds.toFixed(2)})`;
      row.append(name,value);$("evaluation-rows").append(row);
    }
  } else $("learning").hidden=true;
  if(!data.rl) {
    $("right-policy").value="anticipatory";
    $("right-policy").querySelector('option[value="rl_swarm"]').disabled=true;
  }
  loadScene();new ResizeObserver(resize).observe($("left-map"));new ResizeObserver(resize).observe($("right-map"));
  requestAnimationFrame(frame);
  // Deliberate user-started animation also respects reduced-motion preferences.
  if(!window.matchMedia("(prefers-reduced-motion: reduce)").matches){time=90;render(true);}
})();
