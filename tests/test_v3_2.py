import importlib.util
p='iamccs_prompt_q21_enh.py'
s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m._auto_mode('change the pose of image 1 to match image 2',[1,2])=='pose_text_only'
assert m._auto_mode('change pose and keep the background of image 2',[1,2])=='subject1_on_canvas2'
assert m._normalize_user_refs('image 1 on image 2',{1:2,2:1})=='<image2> on <image1>'
print('V3.2 tests passed')
