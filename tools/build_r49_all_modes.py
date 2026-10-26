"""Build a prepared R49 variant from the last saved full workflow, without changing its source."""

import argparse
import copy
import json
import uuid
from pathlib import Path


def inline_group(workflow, node_id):
    node = next(n for n in workflow['nodes'] if n['id'] == node_id)
    definition = next(d for d in workflow['definitions']['subgraphs'] if d['id'] == node['type'])
    links = workflow['links']
    incoming = {link[4]: (link[1], link[2]) for link in links if link[3] == node_id}
    outputs = {link['target_slot']: (link['origin_id'], link['origin_slot'])
               for link in definition['links'] if link['target_id'] == -20}

    def origin(slot):
        source, index = outputs[slot]
        return incoming[index] if source == -10 else (source, index)

    expanded = []
    for link in links:
        if link[3] == node_id:
            continue
        source, slot = origin(link[2]) if link[1] == node_id else (link[1], link[2])
        expanded.append([0, source, slot, link[3], link[4], link[5]])
    for link in definition['links']:
        if link['target_id'] == -20:
            continue
        if link['origin_id'] == -10:
            if link['origin_slot'] not in incoming:
                continue
            source, slot = incoming[link['origin_slot']]
        else:
            source, slot = link['origin_id'], link['origin_slot']
        expanded.append([0, source, slot, link['target_id'], link['target_slot'], link['type']])
    workflow['nodes'] = [n for n in workflow['nodes'] if n['id'] != node_id] + copy.deepcopy(definition['nodes'])
    workflow['links'] = expanded


def validate(workflow, object_info):
    flat = copy.deepcopy(workflow)
    definitions = {d['id'] for d in flat['definitions']['subgraphs']}
    while any(n['type'] in definitions for n in flat['nodes']):
        inline_group(flat, next(n['id'] for n in flat['nodes'] if n['type'] in definitions))
    nodes = {n['id']: n for n in flat['nodes']}
    if len(nodes) != len(flat['nodes']):
        raise ValueError('Duplicate expanded node IDs')
    dependencies = {node_id: set() for node_id in nodes}
    seen = set()
    for _, origin_id, origin_slot, target_id, target_slot, _ in flat['links']:
        output = nodes[origin_id]['outputs'][origin_slot]
        target = nodes[target_id]['inputs'][target_slot]
        if (target_id, target_slot) in seen:
            raise ValueError(f'Duplicate expanded input: {target_id}/{target_slot}')
        seen.add((target_id, target_slot))
        if not set(output['type'].split(',')) & set(target['type'].split(',')) and '*' not in (output['type'], target['type']):
            raise ValueError(f'Incompatible link: {origin_id}/{output["name"]} -> {target_id}/{target["name"]}')
        dependencies[target_id].add(origin_id)
    remaining = set(nodes)
    while remaining:
        ready = {node_id for node_id in remaining if not dependencies[node_id] & remaining}
        if not ready:
            raise ValueError(f'Workflow cycle: {sorted(remaining)}')
        remaining -= ready
    for node in nodes.values():
        if node['type'] == 'MarkdownNote':
            continue
        if node['type'] not in object_info:
            raise ValueError(f'Node is not installed: {node["type"]}')
        schema = object_info[node['type']]['input']
        specs = {**schema.get('required', {}), **schema.get('optional', {})}
        for name, value in node.get('widgets_values_named', {}).items():
            spec = specs.get(name)
            if spec and isinstance(spec[0], list) and value not in spec[0]:
                raise ValueError(f'Unavailable widget choice: {node["id"]}/{name}={value}')
    return len(nodes), len(flat['links'])


def build(source, object_info):
    workflow = copy.deepcopy(source)
    workflow['id'] = str(uuid.uuid4())
    for node_id in (1003, 1004, 1031):
        inline_group(workflow, node_id)
    nodes = {node['id']: node for node in workflow['nodes']}
    edges = workflow['links']
    next_id = max(nodes) + 1

    def connect(source_id, output, target_id, name):
        target = nodes[target_id]
        slot = next(i for i, inp in enumerate(target['inputs']) if inp['name'] == name)
        edges[:] = [e for e in edges if (e[3], e[4]) != (target_id, slot)]
        edges.append([0, source_id, output, target_id, slot, target['inputs'][slot]['type']])

    def disconnect(target_id, name):
        slot = next(i for i, inp in enumerate(nodes[target_id]['inputs']) if inp['name'] == name)
        edges[:] = [e for e in edges if (e[3], e[4]) != (target_id, slot)]

    def widget_values(node_id):
        node = nodes[node_id]
        named = node.get('widgets_values_named')
        if isinstance(named, dict):
            return named
        schema = object_info[node['type']]['input']
        names = [name for category in ('required', 'optional') for name, spec in schema.get(category, {}).items()
                 if isinstance(spec[0], list) or spec[0] in ('INT', 'FLOAT', 'BOOLEAN', 'STRING') and not spec[1].get('forceInput')]
        values = node.get('widgets_values') or []
        return {name: value for name, value in zip(names, values)}

    def set_values(node_id, changes):
        node = nodes[node_id]
        values = widget_values(node_id)
        values.update(changes)
        node['widgets_values_named'] = values
        schema = object_info[node['type']]['input']
        names = [name for category in ('required', 'optional') for name, spec in schema.get(category, {}).items()
                 if isinstance(spec[0], list) or spec[0] in ('INT', 'FLOAT', 'BOOLEAN', 'STRING') and not spec[1].get('forceInput')]
        for name in changes:
            if name not in names:
                names.append(name)
        node['widgets_values'] = [values.get(name) for name in names]

    def loader(kind, title, position, filename, node_id=None):
        nonlocal next_id
        if node_id is None:
            node_id, next_id = next_id, next_id + 1
        image = kind == 'LoadImage'
        node = {'id': node_id, 'type': kind, 'title': title, 'pos': position,
                'size': [360, 340 if image else 540], 'flags': {}, 'order': 0, 'mode': 0,
                'inputs': [] if image else [{'name': 'meta_batch', 'type': 'VHS_BatchManager', 'link': None},
                                            {'name': 'vae', 'type': 'VAE', 'link': None}],
                'outputs': [{'name': name, 'type': typ, 'links': []}
                            for name, typ in zip(object_info[kind]['output_name'], object_info[kind]['output'])],
                'properties': {'Node name for S&R': kind, 'iamccs_r49_prepared_media': True},
                'widgets_values': [filename, 'image'] if image else {
                    'video': filename, 'force_rate': 24, 'custom_width': 0, 'custom_height': 0,
                    'frame_load_cap': 0, 'skip_first_frames': 0, 'select_every_nth': 1,
                    'format': 'None', 'choose video to upload': 'image'}}
        if node_id in nodes:
            workflow['nodes'].remove(nodes[node_id])
        workflow['nodes'].append(node)
        nodes[node_id] = node
        return node_id

    # All media gates now follow the compiled Shotboard plan. Control preprocessing
    # no longer feeds back into Shotboard through the modular authoring input.
    disconnect(809, 'h3_fun_control_input')
    connect(7, 0, 626, 'cine_linx')
    connect(626, 0, 819, 'cine_linx')
    connect(819, 0, 586, 'cine_linx')
    connect(586, 0, 1021, 'cine_linx')
    connect(1021, 0, 1030, 'cine_linx')
    for name in ('reference_audio', 'custom_audio'):
        disconnect(586, name)

    image_names = object_info['LoadImage']['input']['required']['image'][0]
    defaults = ['ref_001 (12).png', 'ref_002 (13).png', 'ref_001b (1).png', 'ref_002b.png']
    pictures = []
    for index in range(4):
        filename = defaults[index] if defaults[index] in image_names else image_names[0]
        pic = loader('LoadImage', f'PICTURE {index+1} / upload reference image / mode-selected',
                     [-6100 + index*440, -1300], filename, (952, 1022)[index] if index < 2 else None)
        pictures.append(pic)
        connect(pic, 0, 586, f'reference_image_{index+1}')
        connect(pic, 0, 1021, f'reference_image_{index+1}')
    connect(pictures[0], 0, 819, 'reference_face')
    disconnect(819, 'reference_face_2')
    # Subject Swap uses SAM3 unless an actual per-frame source mask is supplied.
    # A reference-image alpha mask has neither the source geometry nor duration.
    disconnect(819, 'source_mask')

    video_names = object_info['VHS_LoadVideo']['input']['required']['video'][0]
    video_file = '1_00001-audio.mp4' if '1_00001-audio.mp4' in video_names else video_names[0]
    reference = loader('VHS_LoadVideo', 'REF VIDEO / upload / Ref2VA and Ref2Vid video role', [-6100, -650], video_file)
    connect(reference, 0, 586, 'reference_video')
    connect(reference, 2, 586, 'reference_video_audio')
    control = loader('VHS_LoadVideo', 'CONTROLNET VIDEO / upload / active only in ControlNet mode', [-5660, -650], video_file)
    connect(control, 0, 626, 'source_video')
    swap = loader('VHS_LoadVideo', 'SWAP VIDEO / upload / Subject and Object Swap modes', [-5220, -650], video_file, 951)
    for target in (819, 1021):
        connect(swap, 0, target, 'source_video')
        connect(swap, 2, target, 'source_audio')
    vfx = loader('VHS_LoadVideo', 'VFX VIDEO / upload / active only in Ref2VA VFX Edit', [-4780, -650], video_file)
    for name, typ in (('source_video', 'IMAGE'), ('source_audio', 'AUDIO')):
        nodes[1030]['inputs'].append({'name': name, 'type': typ, 'link': None})
    connect(vfx, 0, 1030, 'source_video')
    connect(vfx, 2, 1030, 'source_audio')
    for name in ('depth_guide', 'pose_guide'):
        guide = loader('VHS_LoadVideo', f'OBJECT SWAP / {name.upper()} / upload preprocessed video',
                       [-6100 if name == 'depth_guide' else -5660, 100], video_file)
        connect(guide, 0, 1021, name)
    set_values(1030, {'video_path': '', 'source_fps': 24.0})
    set_values(626, {'active_mode_only': True, 'preprocessor': 'from_iamccs_settings'})
    set_values(586, {'active_mode_only': True, 'task_override': 'from_shotboard'})
    set_values(1021, {'active_mode_only': True})
    set_values(811, {'reference_role_1': 'subject_identity', 'reference_role_2': 'subject_identity',
                     'reference_role_3': 'disabled', 'reference_role_4': 'disabled',
                     'h3_edit_video_source': 'backend'})
    nodes[811].setdefault('properties', {})['iamccs_r49_all_modes'] = True
    nodes[820].setdefault('properties', {})['iamccs_r49_preview_modes'] = ['v2va_controlnet']
    nodes[820]['mode'] = 2

    # Keep the media adapters visible next to their prepared inputs.
    for node_id, position in {626: [-5200, 100], 819: [-4500, 100], 586: [-6100, 900],
                              1021: [-5300, 900], 1030: [-4500, 900], 595: [1900, 1750],
                              591: [2350, 1750], 588: [3200, 1750]}.items():
        nodes[node_id]['pos'] = position
    nodes[621]['widgets_values'] = [
        '# R49 ALL MODES / PREPARED MEDIA\n\n'
        'Select MODE in Settings PRO. Each media branch is requested only by its active mode.\n\n'
        '- PICTURE 1–4: image upload loaders. In REF2VA / REF2VID ENGINE select the roles or 3 SUBJECTS. Disabled references are skipped.\n'
        '- REF VIDEO: enable its role in the Ref2VA/Ref2Vid engine.\n'
        '- CONTROLNET VIDEO: choose ControlNet mode, kind and preprocessing in its engine.\n'
        '- SWAP VIDEO: Subject Swap / Object Swap. PICTURE 1 is the target identity.\n'
        '- VFX VIDEO: choose Ref2VA VFX Edit and supply a vfx_edit: prompt.\n'
        '- Custom performance audio belongs to Shotboard. Tag Editor Render remains the explicit TTS action.\n\n'
        'The loaders contain example selections; replace them with your own media. '
        'Model/LoRA requirements still depend on the selected mode. Save your edited workflow normally.'
    ]
    editor = nodes.get(1032)
    if editor:
        editor.setdefault('properties', {})['iamccs_dialogue_audio_render'] = {'stage': 'shotboard'}

    # Rebuild both directions of the link index; native subgraph internals retain their IDs.
    for node in workflow['nodes']:
        for inp in node.get('inputs', []):
            inp['link'] = None
        for out in node.get('outputs', []):
            out['links'] = []
    for link_id, edge in enumerate(edges, 1):
        edge[0] = link_id
        nodes[edge[1]]['outputs'][edge[2]]['links'].append(link_id)
        target = nodes[edge[3]]['inputs'][edge[4]]
        if target['link'] is not None:
            raise ValueError(f'Duplicate input link: {edge[3]} / {target["name"]}')
        target['link'] = link_id
    workflow['last_node_id'] = max(nodes)
    workflow['last_link_id'] = len(edges)
    workflow.setdefault('extra', {})['iamccs_r49_variant'] = 'all_modes_prepared_media_2026-10-15'
    workflow['extra']['ds'] = {'scale': 0.7, 'offset': [140, 40]}
    workflow['groups'].extend([
        {'title': 'MEDIA / PICTURE 1–4 / ROLES IN REF2VA ENGINE', 'bounding': [-6140, -1380, 1740, 420], 'color': '#3d685b', 'font_size': 24},
        {'title': 'MEDIA / VIDEO UPLOADS / MODE-SELECTED', 'bounding': [-6140, -730, 1740, 690], 'color': '#355b78', 'font_size': 24},
        {'title': 'INPUT GATES / ONLY THE SELECTED MODE LOADS ITS MEDIA', 'bounding': [-6140, 20, 2240, 1550], 'color': '#66573c', 'font_size': 22},
    ])
    return workflow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('object_info', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(json.loads(args.source.read_text(encoding='utf-8')), json.loads(args.object_info.read_text(encoding='utf-8')))
    checked = validate(result, json.loads(args.object_info.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {args.output}: {len(result["nodes"])} root nodes, {len(result["links"])} links; expanded {checked[0]} nodes/{checked[1]} links checked')
