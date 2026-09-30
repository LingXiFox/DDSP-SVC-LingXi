"""Reassess the existing Stage-3 grid against an unapproved diagnostic standard."""
import json
from pathlib import Path

RELATIVE_INCREASE_CAP = 0.10
OUT = Path('reports/timbre_blend_stage3_standard_draft.json')


def main():
    assert not OUT.exists(), f'refusing existing report: {OUT}'
    diagnostic = json.loads(Path('reports/timbre_blend_stage3_diagnostics.json').read_text())
    old = json.loads(Path('reports/timbre_blend_stage3_recipe_screen.json').read_text())
    native = diagnostic['native_virtual']['by_public_spk']
    distributions = diagnostic['pure_public_conversion']
    absolute = distributions['different_singer_pairwise']['p95']
    same_unit = distributions['different_singer_grouped']['p95']
    assert len(old['results']) == 12 and len(native) == 12 and absolute > same_unit
    rows = []
    for row in old['results']:
        assert len(row['public_scores']) == 12
        failures = []
        stricter_failures = []
        per_singer = {}
        for i in range(1, 13):
            key = str(i)
            similarity = row['public_scores'][key]['mean']
            baseline = native[key]['mean']
            assert baseline > 0
            increase = similarity / baseline - 1
            if increase > RELATIVE_INCREASE_CAP:
                failures.append({'spk_id': i, 'reason': 'relative_increase', 'increase': increase,
                                 'actual_similarity': similarity, 'native_similarity': baseline})
            if similarity >= absolute:
                failures.append({'spk_id': i, 'reason': 'absolute_pairwise_p95',
                                 'actual_similarity': similarity, 'ceiling': absolute})
            if similarity >= same_unit:
                stricter_failures.append(i)
            per_singer[key] = {'recipe_similarity': similarity, 'native_similarity': baseline,
                               'relative_increase': increase}
        rows.append({'recipe_id': row['recipe']['id'], 'virtual_weight': row['recipe']['virtual_weight'],
                     't_start': row['recipe']['t_start'], 'passes_draft': not failures,
                     'failures': failures, 'public_singer_detail': per_singer,
                     'fails_grouped_p95_alternative_spk_ids': stricter_failures})
    result = {'status': 'PROPOSAL_ONLY_PENDING_USER_CONFIRMATION',
              'relative_increase_cap': RELATIVE_INCREASE_CAP,
              'relative_rule': 'For each public Y: recipe_mean(Y) / native_virtual_mean(Y) - 1 <= 0.10. Negative increases pass. This 10% cap is a proposed policy choice, not a confidence bound derived from 12 samples.',
              'absolute_limit_strictly_below': absolute,
              'absolute_rule': 'For every public Y: recipe_mean(Y) < the pairwise different-speaker p95 from 36 pure-public conversions x 11 other public singers x four references (=1584 cosines). This distribution is of individual cosines, while the gate compares 12-cosine means; the alternative matched-unit p95 is reported explicitly.',
              'matched_unit_p95_strictly_below_alternative': same_unit,
              'matched_unit_rule': 'More conservative, apples-to-apples option: 132 conversion-target vs other-reference-singer means, each averaged across three output clips x four reference clips.',
              'passing_recipe_ids': [r['recipe_id'] for r in rows if r['passes_draft']],
              'failed_recipe_ids': [r['recipe_id'] for r in rows if not r['passes_draft']],
              'source_reports': ['reports/timbre_blend_stage3_diagnostics.json',
                                 'reports/timbre_blend_stage3_recipe_screen.json'],
              'results': rows,
              'caveat': 'A proposal is not an approved compliance rule. No pass can be inferred from a failed rule, and no new inference was performed.'}
    assert len(result['passing_recipe_ids']) + len(result['failed_recipe_ids']) == 12
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print('DRAFT', row['recipe_id'], 'PASS' if row['passes_draft'] else 'FAIL',
              'absolute', sorted({r['spk_id'] for r in row['failures'] if r['reason'] == 'absolute_pairwise_p95'}),
              'relative', sorted({r['spk_id'] for r in row['failures'] if r['reason'] == 'relative_increase'}), flush=True)
    print('DRAFT_RESULT', 'pass', result['passing_recipe_ids'], 'fail', result['failed_recipe_ids'], flush=True)


if __name__ == '__main__':
    main()
