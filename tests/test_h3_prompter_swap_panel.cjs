const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/iamccs_prompter_ui.js'),'utf8');
const extract=(from,to)=>source.slice(source.indexOf(from),source.indexOf(to,source.indexOf(from)));
class Element {
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';this.textContent='';}
  appendChild(child){this.children.push(child);return child;}
  append(...children){children.forEach(child=>this.appendChild(child));}
  replaceChildren(...children){this.children=children;}
  addEventListener(){}
  setAttribute(key,value){this[key]=value;}
}
const makeNode=(type,values={},title='')=>({type,title,widgets:Object.entries(values).map(([name,value])=>({name,value}))});
function panel(nodes,mode='v2va_face_swap_lora',video=null){
  let commits=0;
  const center=new Element('main');
  const modeButtons=new Map([['v2va_face_swap_lora',{}],['ref2va_vfx_edit',{}]]);
  const context=vm.createContext({
    app:{graph:{_nodes:nodes}},project:{task_mode:'t2va',sections:{},edit_video:video,edit_video_source:'auto'},
    document:{createElement:tag=>new Element(tag)},
    Option:function(label,value){return {textContent:label,value};},
    api:{fetchApi(){throw Error('Opening the panel must not run AI or inspect video');}},
    aiStatus:{},SWAPCHAR_REQUEST:'Replace the character',SWAPCHAR_PROMPT:'Current swap prompt',VFX_EDIT_PROMPT:'Current VFX prompt',
    commit(){commits++;},snapshotProject(){},renderControls(){},renderPreview(){},modeButtons,center,
  });
  vm.runInContext(extract('function widget(','function setWidget(')+
    extract('function el(','function downloadProject(')+
    extract('function nodeType(','function shotboardsForPrompter(')+
    extract('    const editGraphNodes =','    const renderSections =')+
    '\nconst renderSections=()=>{center.replaceChildren();center.appendChild(renderEditMedia());};\n'+
    extract('    modeButtons.forEach((item, key) => {','    targetButtons.forEach(')+
    '\nglobalThis.mediaConfig=editMediaConfig;',context);
  modeButtons.get(mode).onclick();
  assert.equal(commits,1);
  assert.equal(center.children.length,1);
  assert.ok(center.children[0].children.length>5,'Swap media controls must finish rendering');
  return {context,card:center.children[0],config:context.mediaConfig()};
}
// Missing optional settings, video and image loaders used to throw after the
// central panel was cleared. Standalone authoring must remain usable.
panel([]);
panel([],'ref2va_vfx_edit');
for(const type of ['VHS_LoadVideo','VHS_LoadVideoPath']){
  const {config,card}=panel([makeNode(type,{video:'source.mp4'})]);
  assert.equal(config.backend_video.path,'source.mp4');
  assert.equal(card.children.find(child=>child.tag==='video').src,'/iamccs/prompter/edit-video/view?path=source.mp4');
}
const settings=makeNode('IAMCCS_ShotboardH3SettingsPro',{duration_seconds:8.333,h3_edit_video_source:'backend'});
const {config}=panel([settings,makeNode('VHS_LoadVideo',{video:'actual.mp4'})]);
assert.equal(config.video_source,'backend');
assert.equal(config.backend_video.duration_seconds,8.333);
// Prompter-owned video also renders its trim inputs without backend loaders.
const own=panel([],'v2va_face_swap_lora',{path:'own.mp4',start_seconds:1,duration_seconds:3});
assert.equal(own.config.video.path,'own.mp4');
const vfx=panel([makeNode('IAMCCS_H3VFXInput',{video_path:'vfx.mp4'})],'ref2va_vfx_edit');
assert.equal(vfx.config.backend_video.path,'vfx.mp4');
// Demonstrate that the old optional-node lookup reproduced the reported failure.
assert.throws(()=>vm.runInNewContext('function widget(node,name){return node.widgets?.find(w=>w.name===name)}; widget(undefined,"video")'),/widgets/);
console.log('PASS: SwapChar/VFX mode buttons render central media panel with missing optional nodes, both VHS video loaders and Prompter clips');
