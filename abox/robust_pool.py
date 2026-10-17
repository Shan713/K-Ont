"""Crash-tolerant, resumable process-pool map for long CPU jobs (CHGNet relaxations, Steps 41/44).

A worker that dies (out of memory, a native crash on one structure) breaks a whole ProcessPoolExecutor
and loses every pending result. Here jobs run in chunks with a fresh pool each; results are handed to
`save` after every chunk; when a chunk breaks, its jobs are retried one per fresh single-worker pool, and a
job that kills its worker again gets `failed` as its result instead of stopping the run.
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool


def robust_map(fn, jobs, workers, save, initializer=None, initargs=(), chunk=48, failed=None, log=print, key_of=None):
    """fn(job) -> (key, value); returns {key: value}; save(results) is called after every chunk"""
    out = {}
    for i in range(0, len(jobs), chunk):
        part = jobs[i:i + chunk]
        try:
            with ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as ex:
                out.update(ex.map(fn, part, chunksize=1))
        except BrokenProcessPool:
            log(f"  chunk {i // chunk}: a worker died; retrying its {len(part)} jobs one by one")
            for job in part:
                try:
                    with ProcessPoolExecutor(1, initializer=initializer, initargs=initargs) as ex:
                        k, v = ex.submit(fn, job).result()
                    out[k] = v
                except BrokenProcessPool:
                    log("  job killed its worker; marked failed")
                    out[key_of(job) if key_of else job[0]] = failed
        save(out)
        log(f"  {min(i + chunk, len(jobs))}/{len(jobs)} done")
    return out
