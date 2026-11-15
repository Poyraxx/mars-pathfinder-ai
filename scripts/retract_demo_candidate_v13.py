"""Record a real-image regression; never weaken tests or deploy the candidate."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
    assert hashlib.sha256((R/'data/prepared/navigation.onnx').read_bytes()).hexdigest()=='6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0'
    p=R/'data/research/live-demo-v13/release-result.json';r=json.loads(p.read_text('utf8'));r['installed']=False;r['all_gates_passed']=False;r['gates']['real_image_regression_tests_passed']=False;r['gates']['source_overlay_semantics_acceptable']=False;r['retracted']=True;r['reason']='Actual real-image tests and paired source inspection exposed missed visible foreground rocks in FLA_402284021; the proposed partial corridor crosses that rock area. Weak flat-ground labels had included flat-topped rock surfaces, so aggregate partial-label improvement was misleading. Original v11 restored; original tests remain unchanged.';p.write_text(json.dumps(r,indent=2),encoding='utf8')
    (p.parent/'release-retraction.json').write_text(json.dumps({'rejected_candidate_sha256':r['candidate_sha256'],'restored_original_sha256':'6dcf488232cc3d46cfe7250c1bbfe1311437c9f3f65555cd6858708602ec82b0','original_weights_and_card_restored':True,'original_fixture_tests_not_weakened':True,'reason':r['reason'],'next_work':'Manually review original-model detection/weak-ground conflicts across all181 demo frames; correct visible rock positives and uncertainty before retraining from original weights.'},indent=2),encoding='utf8');print('Recorded rejection and verified original restored',flush=True)
if __name__=='__main__':main()
