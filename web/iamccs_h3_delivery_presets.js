// All native axes are multiples of 32. Delivery axes are exact output sizes.
// Legal grids are not a VRAM guarantee. Applying a preset is an explicit edit.
export const H3_DELIVERY_PAIRS = [
    {id:"hd_safe", label:"LIGHT · 640×384 → 1280×768", native:[640,384], delivery:[1280,768], rtx:false},
    {id:"hd_plus", label:"BALANCED · 768×448 → 1536×896", native:[768,448], delivery:[1536,896], rtx:false},
    {id:"wide_plus", label:"DETAIL · 864×480 → 1728×960", native:[864,480], delivery:[1728,960], rtx:false},
    {id:"fhd_light", label:"FHD LIGHT · 736×416 → 1920×1080", native:[736,416], delivery:[1920,1080], rtx:false},
    {id:"fhd_detail", label:"FHD DETAIL · 960×544 → 1920×1080", native:[960,544], delivery:[1920,1080], rtx:false},
    {id:"uhd_rtx", label:"UHD RTX · 736×416 → H3 FHD → RTX 3840×2160", native:[736,416], delivery:[3840,2160], rtx:true},
];
export function deliveryPairValues(pair, route) {
    return {width:pair.native[0], height:pair.native[1], image_width:pair.native[0], image_height:pair.native[1],
        upscale_width:pair.delivery[0], upscale_height:pair.delivery[1], upscale_enabled:true,
        ...(route === "rtx_final" ? {h3_upres_rtx_enabled:true} : {}),
        ...(route === "h3_pixel_refine" || route === "h3_latent_upres" ? {h3_upres_rtx_enabled:pair.rtx} : {}),
        ...(route === "ltx23" ? {ltx_4k_enabled:pair.rtx} : {})};
}
export function h3ModeBaselines(task) {
    const lock = ["ref2vid_lipsync", "longvid_guided_lipsync", "longvid_ref2vid_lipsync"].includes(task);
    return [
        {id:"lipsync",label:"AUDIO DRIVE · aligned AudioBoard",values:{audio_mode:"h3_custom_audio_drive"}},
        ...(!lock ? [{id:"native_audio",label:"NATIVE AUDIO · H3 generates sound",values:{audio_mode:"h3_native_generated"}}] : []),
        ...(task === "fl2va" ? [{id:"continuity",label:"FLF · native AV continuity",values:{flf_continuity_mode:"native_av_context",flf_continuity_tail_frames:"22",flf_continuity_audio:true}}] : []),
    ];
}
