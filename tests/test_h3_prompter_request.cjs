const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'web', 'iamccs_prompter_ui.js'), 'utf8');
const start = source.indexOf('    const performAIRewrite = async');
const end = source.indexOf('    const renderLocalPrompts = () => {', start);
assert(start > 0 && end > start);

function fixture() {
    const state = {request: 'Una donna posa una tazza.', task_mode: 't2va', sections: {scene: 'Old scene', camera: 'Old camera', acting: 'Tired hands'}};
    const context = {
        project: state, node: {}, MODE_META: {t2va: {sections: [['scene'], ['camera'], ['acting']]}},
        activePromptKey: null, activePromptArea: null, rewriteBtn: {}, aiScope: {value: 'active_field'},
        aiDirection: {value: ''}, aiStatus: {}, aiProvider: {value: 'qwen_native'}, aiBaseUrl: {value: ''},
        aiModel: {value: '__workflow__'}, aiApiKey: {value: ''}, aiVisionModel: {value: ''},
        aiTemperature: {value: '0.35'}, aiVisualFiles: [], isEditMode: () => false,
        swapcharSignature: () => '', persistAI() {}, commit() {}, beginAiBusy: () => () => {},
        buildAIImagePayload: async () => [], nativeClipConfig: () => ({profile: '4b_v3.1'}),
        snapshotProject() {}, renderControls() {}, renderSections() {}, renderPreview() {},
        workspace: {showReview() {context.reviewCount = (context.reviewCount || 0) + 1;}},
        fetchAI: async (_url, options) => {
            context.sent = JSON.parse(options.body);
            return {ok: true, json: async () => ({ok: true, sections: {scene: 'A woman lowers a cup onto the table.', camera: 'Fixed eye-level framing.'}})};
        },
    };
    vm.createContext(context);
    vm.runInContext(source.slice(start, end) + '\nglobalThis.rewrite = performAIRewrite;', context);
    return context;
}

(async () => {
    const request = fixture();
    assert.equal(await request.rewrite({narrativeRequest: request.project.request, directTargetKeys: ['scene', 'camera']}), true);
    assert.equal(request.sent.request_text, 'Una donna posa una tazza.');
    assert.deepEqual(request.sent.sections, {scene: '', camera: '', acting: 'Tired hands'});
    assert.equal(request.project.sections.scene, 'A woman lowers a cup onto the table.');
    assert.equal(request.project.sections.acting, 'Tired hands');
    assert.equal(request.project.request, 'Una donna posa una tazza.');
    assert.equal(request.reviewCount, 1, 'REQUEST must reveal the compiled H3 prompt');

    const field = fixture();
    assert.equal(await field.rewrite({directTargetKeys: ['scene', 'camera']}), true);
    assert.equal(field.sent.request_text, null);
    assert.equal(field.sent.sections.scene, 'Old scene');
    assert.equal(field.reviewCount, undefined, 'a field rewrite keeps the current view');

    const edited = fixture();
    edited.fetchAI = async () => {
        edited.project.sections.scene = 'User edit while waiting';
        return {ok: true, json: async () => ({ok: true, sections: {scene: 'Late AI result'}})};
    };
    await edited.rewrite({narrativeRequest: edited.project.request, directTargetKeys: ['scene', 'camera']});
    assert.equal(edited.project.sections.scene, 'User edit while waiting');
    assert.match(edited.aiStatus.textContent, /Your edits were kept/);
    console.log('PASS: REQUEST compiles from its brief, field rewrite retains context, late AI cannot overwrite edits');
})().catch(error => {console.error(error); process.exitCode = 1;});
