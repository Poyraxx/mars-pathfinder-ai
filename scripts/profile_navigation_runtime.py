"""Production-only CPU/peak-memory check; never imports the training stack."""
import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def peak_memory_bytes():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return int(counters.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True);ap.add_argument('--navigation-only',action='store_true')
    ap.add_argument('--warm-scenario',action='store_true');args = ap.parse_args()
    os.environ['MARS_NAVIGATION_MODEL_PATH' if args.navigation_only else 'MARS_MODEL_PATH'] = str(args.model.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import data, local_navigation, perception
    data.load_regions()
    rows = []
    for prefix in ['FLA_401492677', 'FLA_404772660', 'FLA_402284021']:
        frame = next(f for f in data.frame_catalog() if f['id'].startswith(prefix))
        tick = time.perf_counter(); result = local_navigation.compute(frame['id'])
        rows.append({'frame_id':frame['id'], 'status':result['status'], 'elapsed_s':time.perf_counter()-tick,
                     'path_points':len(result.get('path_xy',[])), 'ribbons':len(result['ribbons'])})
    if args.navigation_only:
        # The unrelated global classifier can be requested in the same server.
        # Include its workspace too when checking the free-host memory budget.
        perception.infer(rows[-1]['frame_id'])
    warmed=None
    if args.warm_scenario:
        from backend import main, replay
        scenario=replay.scenarios()[0]['id']
        tick=time.perf_counter();result=main._scenario_local(scenario,None)
        warmed={'id':scenario,'frames':len(result['frames']),'elapsed_s':time.perf_counter()-tick}
    assert not any(name in sys.modules for name in ['torch', 'transformers']), 'Training stack leaked into runtime'
    report = {'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'input_shape':(perception.get_navigation_session() if args.navigation_only else perception.get_session()).get_inputs()[0].shape, 'provider':'CPUExecutionProvider',
              'peak_working_set_mib':peak_memory_bytes()/1024**2, 'platform':sys.platform,
              'training_stack_imported':False, 'routes':rows,'warmed_scenario':warmed,
              'note':'One process, regions and three real local routes. Host allocator and full warmed server caches can differ; this is not a Render memory guarantee.'}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2), 'utf8'); print(json.dumps(report), flush=True)


if __name__ == '__main__':main()
