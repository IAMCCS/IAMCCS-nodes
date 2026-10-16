const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const read = name => fs.readFileSync(path.join(__dirname,'../web',name),'utf8');
const source = read('iamccs_h3_swap_duration.js').replaceAll('export function','function');
const settings = read('iamccs_h3_settings_pro_ui.js');
const durationSetter = settings.slice(settings.indexOf('function setShotboardDuration('), settings.indexOf('function mirrorShotboardAuthority('));
const node = (id,type,values) => ({id,type,widgets:Object.entries(values).map(([name,value])=>({name,value})),properties:{}});
const settingsNode = node(1,'IAMCCS_ShotboardH3SettingsPro',{task_mode:'v2va_face_swap_lora',duration_seconds:14.833,frame_rate:24,h3_edit_video_source:'auto',v2v_source_offset_seconds:0});
const board = node(2,'IAMCCS_MiniMaxH3ShotPlanner',{duration_seconds:14.833,timeline_data:JSON.stringify({duration_seconds:14.833,h3_saved_settings:{duration_seconds:14.833},segments:[{id:'old',start:0,length:356}]})});
const loader = node(3,'VHS_LoadVideoPath',{video:'A.mp4',force_rate:0,skip_first_frames:0,select_every_nth:1,frame_load_cap:0});
const input = node(4,'IAMCCS_H3FaceSwapInput',{source_fps:24});
input.inputs=[{name:'source_video',link:1}];
const graph={_nodes:[settingsNode,board,loader,input],links:{1:{origin_id:3}},getNodeById:id=>graph._nodes.find(n=>n.id===id)};
graph._nodes.forEach(n=>n.graph=graph);
const widget = (n,name)=>n.widgets.find(w=>w.name===name);
let requests = 0, responseFor = async()=>({ok:true,json:async()=>({ok:true,video:{fps:24,frame_count:200}})});
const api={fetchApi:async(...args)=>{requests++;return responseFor(...args);}};
const context=vm.createContext({node:settingsNode,board,pool:[input,loader],api,widget,
  linkedShotboard:()=>board,shotboardMode:n=>widget(n,'task_mode').value,
  setValue:(n,name,value)=>{widget(n,name).value=value;},document:{dispatchEvent(){}},CustomEvent:class{},app:{},
});
vm.runInContext(source+'\n'+durationSetter+`
  const control = createSwapDuration(node,{api,mode:()=>shotboardMode(node),connected:()=>pool,
    setDuration:(seconds,reason)=>setShotboardDuration(node,seconds,reason),refresh(){}});
  globalThis.control=control;
`,context);
const set = (n,name,value)=>widget(n,name).value=value;
const state=()=>settingsNode.properties.iamccs_swap_duration;
(async()=>{
  await context.control.tick();
  assert.equal(widget(settingsNode,'duration_seconds').value,200/24);
  assert.equal(widget(board,'duration_seconds').value,200/24);
  assert.equal(JSON.parse(widget(board,'timeline_data').value).h3_saved_settings.duration_seconds,200/24);
  assert.equal(state().mode,'source');
  await context.control.tick(); assert.equal(requests,1);
  vm.runInContext('setShotboardDuration(node,5)',context);
  assert.equal(state().mode,'custom');
  await context.control.tick(); assert.equal(widget(board,'duration_seconds').value,5);
  settingsNode._iamccsSwapDurationFromShotboard(6);
  assert.equal(widget(settingsNode,'duration_seconds').value,6);
  await context.control.tick(); assert.equal(widget(board,'duration_seconds').value,6);
  set(loader,'video','B.mp4');
  await context.control.tick(); assert.equal(state().mode,'source');
  assert.equal(widget(board,'duration_seconds').value,200/24);
  // Saved automatic metadata restores the default after a workflow reload.
  set(settingsNode,'duration_seconds',14.833); await context.control.tick();
  assert.equal(widget(board,'duration_seconds').value,200/24);
  // Source trim and loader cap are part of the source identity.
  set(loader,'skip_first_frames',24); set(loader,'select_every_nth',2); set(loader,'frame_load_cap',60);
  await context.control.tick(); assert.equal(widget(board,'duration_seconds').value,60/24);
  set(settingsNode,'v2v_source_offset_seconds',0.5);
  await context.control.tick(); assert.equal(widget(board,'duration_seconds').value,2);
  const prompter=node(5,'IAMCCS_Prompter',{project_data:JSON.stringify({edit_video:{path:'P.mp4',start_seconds:1,duration_seconds:3}})});
  context.pool.push(prompter); await context.control.tick();
  assert.equal(widget(board,'duration_seconds').value,2.5);
  // Backend policy must ignore the Prompter's video.
  set(settingsNode,'h3_edit_video_source','backend'); await context.control.tick();
  assert.equal(widget(board,'duration_seconds').value,2);
  // Follow the actual native-subgraph input even with other video loaders present.
  const subgraph={links:{9:{origin_id:-10,origin_slot:1}},getNodeById(){return null;}};
  const parent=node(10,'native-subgraph',{});parent.graph=graph;parent.subgraph=subgraph;
  parent.inputs=[{name:'other'},{name:'swap_video',link:1}];
  input.graph=subgraph;input.inputs[0].link=9;
  const unrelated=node(11,'VHS_LoadVideoPath',{video:'wrong.mp4'});
  context.pool.push(parent,unrelated);
  vm.runInContext('globalThis.selectedSource = swapVideoSource(node,pool);',context);
  assert.equal(context.selectedSource.path,'B.mp4');
  input.graph=graph;input.inputs[0].link=1;
  // A late response for a replaced source must not restore its duration.
  let release;
  responseFor=()=>new Promise(resolve=>{release=resolve;});
  set(loader,'video','slow.mp4'); const slow=context.control.tick();
  responseFor=async()=>({ok:true,json:async()=>({ok:true,video:{fps:24,frame_count:48}})});
  set(loader,'video','latest.mp4'); set(loader,'skip_first_frames',0);set(loader,'select_every_nth',1);set(loader,'frame_load_cap',0);set(settingsNode,'v2v_source_offset_seconds',0);
  await context.control.tick(); assert.equal(widget(board,'duration_seconds').value,2);
  release({ok:true,json:async()=>({ok:true,video:{fps:24,frame_count:900}})}); await slow;
  assert.equal(widget(board,'duration_seconds').value,2);
  // Queue waits for source metadata before serializing either node.
  const order=[];
  const app={graph,graphToPrompt(){order.push('serialize');return widget(board,'duration_seconds').value;}};
  context.app=app; vm.runInContext('installSwapDurationQueueGuard(app)',context);
  responseFor=async()=>{order.push('metadata');return {ok:true,json:async()=>({ok:true,video:{fps:24,frame_count:72}})};};
  set(loader,'video','queue.mp4'); assert.equal(await app.graphToPrompt(),3);
  assert.deepEqual(order,['metadata','serialize']);
  // Errors do not trigger repeated metadata requests or serialize stale defaults.
  responseFor=async()=>({ok:false,json:async()=>({ok:false,error:'file missing'})});
  set(loader,'video','missing.mp4'); await context.control.tick(); const count=requests;
  await context.control.tick(); assert.equal(requests,count);
  await assert.rejects(()=>app.graphToPrompt(),/SwapChar duration: file missing/);
  const afterQueue=requests;
  set(settingsNode,'task_mode','i2va');
  await app.graphToPrompt(); assert.equal(requests,afterQueue);
  context.control.dispose();
  assert.equal(settingsNode._iamccsSwapDurationReady,undefined);
  console.log('PASS: SwapChar video default, bidirectional custom sync, loader trim, source selection, reload, stale response, queue ordering, isolation');
})().catch(error=>{console.error(error);process.exitCode=1;});

const shotboard = read('iamccs_minimax_h3_shotboard_ui.js');
const floorCode=shotboard.slice(shotboard.indexOf('    const durationFloorSeconds ='),shotboard.indexOf('    const showTimelineNotice ='));
const floor=vm.createContext({node:{},getWidget:()=>({value:'v2va_face_swap_lora'}),durationFloorFrames:()=>356,getFps:()=>24});
vm.runInContext(floorCode+'globalThis.getFloor=durationFloorSeconds;',floor);
assert.equal(floor.getFloor(),0);
floor.getWidget=()=>({value:'longvid_multishot_hardcut'});
assert.equal(floor.getFloor(),14.833);
