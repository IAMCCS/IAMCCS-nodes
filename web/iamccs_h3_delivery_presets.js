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
    const common = {sampler_name:"res_multistep",scheduler:"simple",shift_video:12,shift_audio:3};
    const choices = [
        {id:"exact_xx60",label:"EXACT RTX xx60 · 8–12 GB · 4B / 124f / 5.17s",values:{...common,steps:8,turbo_mode:"early_8_10",turbo_strength:0.7,turbo_sampler_mode:"res_multistep_stock",acceleration:"h3_exact",motion_context_window_frames:124,h3_exact_profile:"rtx_xx60_8_12gb_124",h3_exact_chunk_rows:2048,h3_exact_precision_mode:"Preserve native",h3_exact_qkv_streaming:"Auto",h3_exact_attention_memory:"Standard",h3_clipproj_profile:"4b_v3.1",h3_clipproj_load_mode:"dynamic"}},
        {id:"exact_xx70",label:"EXACT RTX xx70 · 12–16 GB · 4B / 209f / 8.71s",values:{...common,steps:8,turbo_mode:"early_8_10",turbo_strength:0.7,turbo_sampler_mode:"res_multistep_stock",acceleration:"h3_exact",motion_context_window_frames:209,h3_exact_profile:"rtx_xx70_12_16gb_209",h3_exact_chunk_rows:2048,h3_exact_precision_mode:"Preserve native",h3_exact_qkv_streaming:"Auto",h3_exact_attention_memory:"Standard",h3_clipproj_profile:"4b_v3.1",h3_clipproj_load_mode:"dynamic"}},
        {id:"exact_xx80",label:"EXACT RTX xx80 · 16–24 GB · 4B / 294f / 12.25s",values:{...common,steps:8,turbo_mode:"early_8_10",turbo_strength:0.7,turbo_sampler_mode:"res_multistep_stock",acceleration:"h3_exact",motion_context_window_frames:294,h3_exact_profile:"rtx_xx80_16_24gb_294",h3_exact_chunk_rows:4096,h3_exact_precision_mode:"Preserve native",h3_exact_qkv_streaming:"Auto",h3_exact_attention_memory:"Standard",h3_clipproj_profile:"4b_v3.1",h3_clipproj_load_mode:"dynamic"}},
        {id:"exact_xx90",label:"EXACT RTX xx90 · 24 GB+ · 8B / 362f / 15.08s",values:{...common,steps:8,turbo_mode:"early_8_10",turbo_strength:0.7,turbo_sampler_mode:"res_multistep_stock",acceleration:"h3_exact",motion_context_window_frames:362,h3_exact_profile:"rtx_xx90_24gb_362",h3_exact_chunk_rows:8192,h3_exact_precision_mode:"Preserve native",h3_exact_qkv_streaming:"Auto",h3_exact_attention_memory:"Standard",h3_clipproj_profile:"8b_v3.1",h3_clipproj_load_mode:"dynamic"}},
        {id:"balanced",label:"DEFAULT · balanced / 12 steps",values:{...common,steps:12}},
        {id:"low_ram",label:"LOW RAM · 640×384 / CPU text encode",values:{...common,width:640,height:384,image_width:640,image_height:384,text_encoder_device:"cpu_direct",performance_profile:"low_vram_balanced"}},
        {id:"lipsync",label:"LIPSYNC · aligned AudioBoard / visible speaker",values:{audio_mode:"h3_custom_audio_drive"}},
        ...(!lock ? [{id:"native_audio",label:"NATIVE AUDIO · H3 generates sound / no locked recording",values:{audio_mode:"h3_native_generated"}}] : []),
        {id:"turbo",label:"PDD TURBO · 8 steps / task-matched Acc LoRA",values:{...common,steps:8,sampler_name:"euler",scheduler:"simple",denoise:1.0,acceleration:"pdd_native_8step",turbo_mode:"off",pdd_strength:1.0}},
    ];
    if (task === "v2va_object_swap") {
        choices.unshift(
            {
                id:"v2v_12gb",
                label:"V2V 12 GB · 640×384 / ≤124f recommended",
                values:{
                    ...common, width:640, height:384, image_width:640, image_height:384,
                    steps:12, text_encoder_device:"cpu_direct", performance_profile:"low_vram_balanced",
                    acceleration:"low_vram_auto", ref_image_size:"match", v2v_guide_mode:"raw_only",
                    v2v_source_fit:"native_adapt", v2v_source_end_policy:"hold_last_for_grid",
                },
            },
            {
                id:"v2v_high_vram",
                label:"V2V HIGH VRAM · 960×544 / 16 steps",
                values:{
                    ...common, width:960, height:544, image_width:960, image_height:544,
                    steps:16, text_encoder_device:"gpu_auto", performance_profile:"custom",
                    acceleration:"low_vram_auto", ref_image_size:"match", v2v_guide_mode:"raw_only",
                    v2v_source_fit:"native_adapt", v2v_source_end_policy:"hold_last_for_grid",
                },
            },
        );
    }
    if (task === "longvid_motion_context") {
        choices.unshift(
            {id:"mc_xx60", label:"MOTION RTX xx60 · 8–12 GB · 124f / 5.17s", values:{...common,motion_context_window_frames:124,h3_exact_profile:"rtx_xx60_8_12gb_124",performance_profile:"low_vram_balanced",acceleration:"comfy_kitchen"}},
            {id:"mc_xx70", label:"MOTION RTX xx70 · 12–16 GB · 209f / 8.71s", values:{...common,motion_context_window_frames:209,h3_exact_profile:"rtx_xx70_12_16gb_209",performance_profile:"custom",acceleration:"comfy_kitchen"}},
            {id:"mc_xx80", label:"MOTION RTX xx80 · 16–24 GB · 294f / 12.25s", values:{...common,motion_context_window_frames:294,h3_exact_profile:"rtx_xx80_16_24gb_294",performance_profile:"custom",acceleration:"comfy_kitchen"}},
            {id:"mc_xx90", label:"MOTION RTX xx90 · 24 GB+ · 362f / 15.08s", values:{...common,motion_context_window_frames:362,h3_exact_profile:"rtx_xx90_24gb_362",performance_profile:"custom",acceleration:"comfy_kitchen"}},
        );
    }
    if (task === "fl2va") choices.push({id:"continuity",label:"FLF FILM · 22-frame native AV continuity",values:{flf_continuity_mode:"native_av_context",flf_continuity_tail_frames:"22",flf_continuity_audio:true}});
    return choices;
}
