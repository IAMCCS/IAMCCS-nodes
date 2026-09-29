const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.join(__dirname, '..', 'web', 'iamccs_h3_settings_pro_ui.js'), 'utf8');
function need(fragment, message) {
  if (!src.includes(fragment)) throw new Error(message + `\nMissing: ${fragment}`);
}
need('LONG TAKE · EXTEND', 'Extension must remain a distinct mode-control child.');
need('SAVED TAKE · CONTINUATION', 'Saved-take continuation must remain a distinct mode-control child.');
need('LONG TAKE CONTINUOUS · IMAGE', 'Extended AV must be a selectable assistant mode.');
need('["fl2va_stable", "fl2va_continuous"].includes(String(key)) ? "fl2va" : key', 'Only historical FL2VA aliases may collapse to fl2va.');
if (src.includes('String(key).startsWith("fl2va_") ? "fl2va" : key')) throw new Error('Broad fl2va_* alias is forbidden: it would collapse fl2va_extended_av.');
need('["fl2va_extended_av", "t2va_continuous"].includes(String(key))', 'Both long-take presets need the B1 production contract.');
need('function normalizeExtendedAvProductionContract(node)', 'Extended AV needs one canonical production-contract normalizer.');
need('h3_auto_extend_mode: "off"', 'Extended AV must disable legacy auto-extend.');
need('flf_overlap_frames: 0', 'Extended AV must disable legacy FLF overlap.');
need('extended_av_pin_mode: "masked"', 'Extended AV production path must lock MASKED extend.');
need('extended_av_mask_profile: "exact"', 'Extended AV production path must lock EXACT hold.');
need('normalizeExtendedAvProductionContract(node);', 'Selecting Extended AV must apply the canonical locked contract.');
need('["extended_av_custom_window_frames", "extended_av_custom_overlap_frames"].includes(name)', 'Custom window controls must be conditionally visible.');
need('!== "custom") return false', 'Custom window controls must hide for preset profiles.');
need('"extended_av_level_lock_mode"', 'Deferred level-lock field must remain explicitly classified.');
need('"extended_av_joint_refine_mode"', 'Deferred joint-refine field must remain explicitly classified.');
need('ENGINES', 'Mode-specific panels must live under the ENGINES rail folder.');
need('h3p-mode-folder', 'Mode controls need a distinct visual folder style.');
need('h3p-subtab', 'Mode controls need nested child styling.');
need('iamccs_h3_settings_pro_mode_controls_open', 'Mode controls folder state must persist.');
console.log('Extended AV UI semantics OK: independent mode, nested ENGINES, legacy isolation, custom-only fields, deferred controls hidden.');
