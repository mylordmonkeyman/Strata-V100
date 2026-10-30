#!/usr/bin/env python3
"""Private paired prefill/verification routing screen; original MMQ only."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from qpn_full_model_ab import benchmark_env, sha, logits, read_summary
from qpn_r1_prefill_convert import decode_private_prefill
from qpn_r1_verify_convert import decode_private_verify
from qpn_resident_r1 import build_groups, score_holdout
from qpn_resident_cache_feasibility import PACKED_BLOB, MIB
from qpn_r1_verify_probe import MODEL


def normalize_request(prefill, verify, expected):
    if sum(c['tokens'] for c in prefill['prefill']) != expected:
        raise ValueError('incomplete prefill capture')
    residents = None
    chunks = []
    for c in prefill['prefill']:
        if c['stream_all'] or len(c['layers']) != 48:
            raise ValueError('unsupported prefill capture')
        chunks.append({l['layer']: dict(l['counts']) for l in c['layers']})
        current = {(l['layer'], e) for l in c['layers'] for e in l['resident_experts']}
        if residents is not None and residents != current:
            raise ValueError('resident identities changed across chunks')
        residents = current
    windows = verify['verify']
    # Reaching the capture cap cannot establish complete request coverage.
    if not windows or len(windows) >= 32:
        raise ValueError('missing or potentially truncated verification capture')
    if any(w['position'] < expected for w in windows):
        raise ValueError('verification includes unclassified prompt-tail work')
    native_windows = [{l['layer']: dict(l['counts']) for l in w['layers']} for w in windows]
    # Internal scorer key only: all executed post-prefill verification consumes
    # canonical weights, including speculative inputs eventually rejected.
    return {'prefill': chunks, 'decode': native_windows}, residents


def score_requests(train, heldout, resident):
    groups = build_groups(train, resident)
    results = []
    for mib in (128, 256, 512, 737, 1024, 2048):
        n = min(len(groups), (mib*MIB//PACKED_BLOB)//16)
        selected = {p for _, _, group in groups[:n] for p in group}
        values = score_holdout(heldout, selected, resident)
        # Never export verification as accepted-token decode evidence.
        values['verification_routed_entries'] = values.pop('decode_routed_entries', 0)
        values['verification_distinct_packed_experts_to_restore'] = values.pop('decode_distinct_packed_experts_to_restore')
        values['optimistic_verification_restore_bytes'] = values.pop('optimistic_decode_restore_bytes')
        results.append(dict(replacement_payload_budget_mib=mib,
                            selected_packed_experts=len(selected), **values))
    return results


def prepare_prompts(a, private):
    sys.path.insert(0, str(a.source))
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    tokenizer = Tokenizer.from_gguf(a.prepared/'dense.gguf')
    template = ChatTemplate(a.prepared/'pack/tokenizer/chat_template.jinja')
    # Separate authored subject matter. No slicing/shuffling the pinned prompt
    # into a nominally held-out copy; no real customer text is published.
    texts = [
        '\n'.join(f'Experiment {i}: A compiler processes a program with {i+3} functions. '
                  f'The engineer measures parsing, optimization and code generation separately. '
                  f'There are {i*7+11} input records and {i%9+2} repeated queries. '
                  'Explain how caching and dependency ordering affect total runtime.' for i in range(90)),
        '\n'.join(f'Field note {i}: Garden bed {i+1} receives {i%8+2} hours of sunlight. '
                  f'The soil sample contains {i%17+6} percent clay. '
                  'A botanist tracks root growth, rainfall, flowering and pollinator visits. '
                  'Explain how to distinguish seasonal variation from treatment effects.' for i in range(90))]
    result=[]
    for i, (text, count) in enumerate(zip(texts, (1024, 2048))):
        full=tokenizer.encode(template.render([{'role':'user','content':text}], enable_thinking=False), parse_special=True)
        if len(full) <= count:
            raise ValueError('generated prompt too short')
        ids=full[:count-32]+full[-32:]
        path=private/f'prompt-{i}.txt'
        path.write_text(','.join(map(str,ids))); path.chmod(0o600)
        result.append((path, count))
    if sha(result[0][0]) == sha(result[1][0]):
        raise ValueError('identical prompt inputs')
    return result


def run(a):
    if sha(a.prepared/'pack/experts.bin') != MODEL:
        raise ValueError('model SHA mismatch')
    base=benchmark_env()
    base.update(STRATA_PREFILL_NVFP4='w4a8', STRATA_PREFILL_CPU_SHARE='0',
                STRATA_QPN_PREFILL_EXPERIMENTAL='0', STRATA_PREFILL_STREAM_MIN='4096',
                STRATA_PREFILL_GROUP_GATHER='1')
    captured=[]; metrics=[]
    with tempfile.TemporaryDirectory(prefix='qpn-joint-private-',dir=a.state) as directory:
        private=Path(directory); private.chmod(0o700)
        prompts=prepare_prompts(a,private)
        for i,(tokens,count) in enumerate(prompts):
            command=[str(a.binary),'--pack',str(a.prepared/'pack'),'--native',str(a.prepared/'dense.gguf'),
                     '--native-dense-gguf',str(a.prepared/'dense.gguf'),'--ple-gguf',str(a.prepared/'ple-nvfp4.gguf'),
                     '--tokens-file',str(tokens),'--max-context','4096','--max-new','16','--greedy',
                     '--expert-cache','auto','--expert-profile',str(a.source/'data/expert-profile.bin'),
                     '--resident-budget-gib','24','--pool-workers','16','--prefill','1024','--spec','2',
                     '--suffix-draft','0','--adapt-every','0','--vram-reserve-mib','1536','--stats']
            hashes=[]; records=[]
            pf=private/f'{i}.qpr1'; vf=private/f'{i}.qdv1'
            for trace in (False,True):
                log=private/f'{i}-{trace}.log'; first=private/f'{i}-{trace}.f32'
                env=base.copy(); env['STRATA_DUMP_FIRST_LOGITS']=str(first)
                if trace:
                    env['STRATA_QPN_R1_PRIVATE_PREFILL_CAPTURE']=str(pf)
                    env['STRATA_QPN_R1_PRIVATE_VERIFY_CAPTURE']=str(vf)
                with log.open('w') as output:
                    result=subprocess.run(['numactl','--cpunodebind=0','--membind=0',*command],
                                          env=env,stdout=output,stderr=subprocess.STDOUT,timeout=900)
                if result.returncode:
                    raise RuntimeError('model arm failed; private details suppressed')
                raw=log.read_text(errors='replace')
                if re.search(r'non[- ]finite|illegal memory access|CUDA error|device-side assert',raw,re.I):
                    raise RuntimeError('model safety failure')
                summary=read_summary(log)
                if (summary['prefill_tokens'] != count-1 or summary['qpn_groups'] != 0 or
                    summary['decode_toks_per_s'] is None):
                    raise RuntimeError('missing work or unexpected QPN execution')
                if len(logits(first)) != 248320:
                    raise RuntimeError('invalid logit shape')
                hashes.append(sha(first))
                records.append({k:summary[k] for k in ('prefill_tokens','prefill_ms','ttft_ms','decode_toks_per_s')})
            if hashes[0] != hashes[1]:
                raise RuntimeError('joint capture changed first logits')
            prefill=decode_private_prefill(pf); verify=decode_private_verify(vf)
            captured.append(normalize_request(prefill,verify,count-1))
            metrics.append(dict(prompt_sha256=sha(tokens),input_tokens=count,
                                prefill_capture_sha256=sha(pf),verification_capture_sha256=sha(vf),
                                verification_windows=len(verify['verify']),
                                first_logit_equal=True, diagnostic_control_trace_timings=records))
        train, train_res=captured[0]; heldout, heldout_res=captured[1]
        # Train-only admission uses observed residency; reject changing identities
        # rather than selecting experts using held-out information.
        if train_res != heldout_res:
            raise ValueError('resident identities changed between requests')
        residents=train_res
        scenarios=score_requests(train,heldout,residents)
        report=dict(schema=1,scope='independently authored synthetic-prompt routing screen; not representative qualification',
                    source_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
                    model_sha256=MODEL,binary_sha256=sha(a.binary),
                    profile_sha256=sha(a.source/'data/expert-profile.bin'),
                    template_sha256=sha(a.prepared/'pack/tokenizer/chat_template.jinja'),control_flags={k:base[k] for k in base if k.startswith('STRATA_')},
                    requests=metrics,common_observed_resident_experts=len(residents),scenarios=scenarios,
                    performance_qualified=False,model_quality_qualified=False,actual_qpn_decode_executed=False,
                    actual_accepted_token_decode_qualified=False,
                    limitations=['different synthetic topics and lengths; one train/heldout pair only',
                                 'arbitrary regrouping upper bound; actual group membership not captured',
                                 'static observed resident identities required; dynamic cache not qualified',
                                 'native post-prefill verification includes rejected speculative inputs',
                                 'first-logit parity only; generated-token parity unqualified',
                                 'fresh processes, uncontrolled host cache; instrumented times not speedups',
                                 'exposed compute and canonical restoration latency remain unmeasured'])
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','prepared','state','binary','report'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:
        run(a)
    except Exception:
        # Tokenizer/parser errors can include private content. Never expose it.
        raise SystemExit('Joint capture failed; private diagnostics suppressed') from None
    print('Joint routing screen complete; aggregate report only')

if __name__=='__main__':main()
