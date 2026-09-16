import {applyAdvisorValues, advisorSnapshot, h3TaskFamily} from './iamccs_h3_advisor.js';

export function slaValues(specs, task, selected = '') {
    const options = specs.turbo_lora_name?.[1] || {};
    if (!options.iamccs_sla_available) throw new Error('Installa H3 SLA Attention sul server e aggiorna i modelli.');
    const assets = (options.iamccs_h3_assets || []).filter(a => a.recipe === 'lightx_768_sla_4' && a.family === h3TaskFamily(task) && a.native && !a.error && !a.conflict);
    const asset = assets.find(a => a.name === selected) || assets[0];
    const recipe = options.iamccs_h3_recipes?.lightx_768_sla_4;
    if (!asset || !recipe) throw new Error('Serve una LoRA Turbo SLA ComfyUI compatibile con questo mode. Vedi LightX2V / Minimax-h3-Turbo-SLA; aggiorna dopo installazione. Ref2VA e Face Swap richiedono adapter della propria famiglia.');
    return {...recipe.values, turbo_lora_name: asset.name, performance_profile: 'custom'};
}

export function createH3SLAPanel({getSource, getBoard, getSpecs, notify, refreshModels}) {
    const root = document.createElement('section'); root.className = 'iamccs-h3-sla';
    root.style.cssText = 'grid-column:1/-1;padding:18px;border:1px solid #82704c;border-radius:14px;background:linear-gradient(120deg,#242a2d,#161b20);color:#eee6d7;display:grid;gap:12px;min-width:0';
    const title = document.createElement('strong'); title.textContent = 'TURBO SLA  /  4 STEP';
    const caption = document.createElement('div'); caption.textContent = 'Adapter installato · ricetta completa · attenzione sparsa';
    const model = document.createElement('select'); model.style.cssText = 'width:100%;min-width:0'; model.setAttribute('aria-label','Turbo SLA modello installato');
    const state = advisorSnapshot(getSource(),getBoard()).settings;
    const specs = getSpecs();
    model.append(new Option('Seleziona una LoRA Turbo SLA compatibile',''));
    for (const a of specs.turbo_lora_name?.[1]?.iamccs_h3_assets || []) if (a.recipe === 'lightx_768_sla_4' && a.family === h3TaskFamily(state.task_mode) && !a.error) model.append(new Option(a.name,a.name));
    model.value = state.turbo_lora_name || '';
    const status = document.createElement('div'); status.setAttribute('role','status');
    const apply = values => applyAdvisorValues(getSource(),getBoard(),values,getSpecs(),notify);
    const enable = document.createElement('input'); enable.type='checkbox'; enable.checked=state.acceleration === 'h3_sla';
    const label = document.createElement('label'); label.append(enable,document.createTextNode(' Attiva Turbo SLA e applica la ricetta a 4 step'));
    const activate = () => {try {apply(slaValues(getSpecs(),state.task_mode,model.value));} catch(e) {status.textContent=e.message; enable.checked=state.acceleration==='h3_sla';}};
    enable.onchange=()=>enable.checked ? activate() : apply({acceleration:'h3_sage',turbo_mode:'off',steps:20});
    model.onchange=activate;
    const button=document.createElement('button'); button.type='button';button.textContent='Applica modello + 4 step';button.onclick=activate;
    const refresh=document.createElement('button');refresh.type='button';refresh.textContent='Aggiorna modelli dal server';refresh.onclick=async()=>{try {await refreshModels();notify();}catch(e){status.textContent=e.message;}};
    root.append(title,caption,label,model,button,refresh);
    for (const [name,text] of [['h3_sla_sparsity','Sparsity · quota di blocchi saltati (0,85 = 85%). Più alta riduce il calcolo; confronta la qualità.'],['h3_sla_dense_last_steps','Step finali dense · ultimi step senza sparsità. 0 mantiene la ricetta SLA; aumentare costa tempo.']]) {
        const spec=specs[name]; if(!spec) continue;
        const field=document.createElement('label');field.textContent=text;
        const input=document.createElement('input');input.type='number';input.min=spec[1].min;input.max=spec[1].max;input.step=spec[1].step||1;input.value=state[name]??spec[1].default;input.disabled=state.acceleration!=='h3_sla';input.style.cssText='display:block;width:100%;margin-top:6px';
        input.onchange=()=>{try{apply({[name]:Number(input.value)});}catch(e){status.textContent=e.message;}};field.append(input);root.append(field);
    }
    status.textContent = state.acceleration==='h3_sla' ? `Attivo · ${state.steps} step · video shift ${state.shift_video} / audio ${state.shift_audio}` : 'Seleziona e applica: nessun render automatico.';
    root.append(status);return root;
}
