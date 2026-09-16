"""IAMCCS manual post-generation seam review; no sampling or automatic export."""
import re
from .iamccs_ahead_seam_editor import run_path, inspect_run

class IAMCCS_AheadControlRoom:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'generation_report':('STRING',{'forceInput':True})}}
    RETURN_TYPES=('STRING',)
    RETURN_NAMES=('generation_report',)
    FUNCTION='open_ready'
    OUTPUT_NODE=True
    CATEGORY='IAMCCS/Cine/Post Production'

    def open_ready(self, generation_report):
        match=re.search(r'checkpoints=.*?[\\/]([0-9a-f]{32})(?:\s*\||\s*$)',generation_report)
        if not match:
            raise ValueError('Connect the LatentGoAhead report output to IAMCCS Ahead Control Room.')
        run=match.group(1)
        inspect_run(run_path(run))
        return {'ui':{'iamccs_ahead_run':[run]},'result':(generation_report,)}

NODE_CLASS_MAPPINGS={'IAMCCS_AheadControlRoom':IAMCCS_AheadControlRoom}
NODE_DISPLAY_NAME_MAPPINGS={'IAMCCS_AheadControlRoom':'IAMCCS Ahead Control Room · Seam Review'}
