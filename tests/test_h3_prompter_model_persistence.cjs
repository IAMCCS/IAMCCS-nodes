const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const source = fs.readFileSync(require('path').join(__dirname, '..', 'web', 'iamccs_prompter_ui.js'), 'utf8');
const extract = (start, end) => source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
function select() {
    return { options: [], selected: '', get value() { return this.selected; },
        set value(value) { this.selected = this.options.some(o => o.value === value) ? value : ''; },
        replaceChildren(...options) { this.options = options; this.selected = options[0]?.value || ''; },
        appendChild(option) { this.options.push(option); } };
}
const controls = {node:{properties:{iamccs_prompter_ai:{provider:'ollama',model:'chosen-writer',vision_model:'chosen-vision'}}},
    aiProvider:{value:'ollama'},aiBaseUrl:{value:''},aiModel:{value:''},aiTemperature:{value:''},
    aiModelPicker:select(),aiVisionModel:select(),aiDefaults:{ollama:{baseUrl:'http://local',model:''}},
    refreshModelsBtn:{},connectOllamaBtn:{style:{}},aiProviderChip:{classList:{toggle(){}}},aiStatus:{},
    Option:function(label,value){this.text=label;this.value=value;},
    api:{fetchApi:async()=>({ok:true,json:async()=>({ok:true,models:[{name:'first-model'},{name:'chosen-vision'}]})})}};
const browserPrefs = new Map();
controls.localStorage = {getItem:key=>browserPrefs.get(key) || null,setItem:(key,value)=>browserPrefs.set(key,value)};
controls.aiVisionList = controls.aiVisionModel;
const context = vm.createContext(controls);
vm.runInContext(extract('    const AI_PREFS_KEY =','    const renderAIProviderChrome = () => {') +
    extract('    let aiModelsRevision = 0;','    aiModelPicker.onchange = () => {') +
    '\nglobalThis.restore = restoreAISettings; globalThis.persist = persistAI; globalThis.refresh = loadOllamaModels;', context);
(async () => {
    assert.equal(controls.aiModel.value,'chosen-writer');
    assert.equal(controls.aiVisionModel.value,'chosen-vision');
    await context.refresh();
    assert.equal(controls.aiModel.value,'chosen-writer','missing saved writer must not switch to first model');
    assert.equal(controls.aiVisionModel.value,'chosen-vision','rebuilding dropdown must preserve vision');
    assert.equal(controls.node.properties.iamccs_prompter_ai.model,'chosen-writer');
    controls.node.properties.iamccs_prompter_ai.model='reloaded-writer';
    context.restore();
    await context.refresh();
    assert.equal(controls.aiModel.value,'reloaded-writer','workflow configuration must restore writer');
    controls.aiModel.value='session-writer';
    context.persist();
    controls.node.properties.iamccs_prompter_ai={};
    controls.aiModel.value='';
    context.restore();
    assert.equal(controls.aiModel.value,'session-writer','browser session preferences must restore writer without a saved workflow');
    assert.match(source,/const aiVisionModel = el\("select"\)/);
    assert.doesNotMatch(source,/visionLabel\.appendChild\(aiVisionList\)/);
    console.log('PASS: one vision dropdown; saved writer/vision survive refresh and configuration');
})().catch(error=>{console.error(error);process.exitCode=1;});
