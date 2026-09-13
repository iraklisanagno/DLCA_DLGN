"""Prepare a bounded, additive factorial and full-S replication matrix."""
from .common import ROOT, OLD, REPO, read, sha, verify_incumbent, write_json


def prepare():
    verify_incumbent()
    old = read(OLD / 'summary/fourth_round_results.json')
    entries = []
    for seed in range(3):
        reference = next(r for r in old['rows'] if r['phase'] == 'mechanism' and
            r['coordinate'] == 'dense_m' and r['family'] == 'u2' and r['seed'] == seed)
        cfg = read(REPO / reference['output'] / 'training_config.json')
        for arm in ['SS', 'SR', 'RS', 'RR']:
            name = f'ur1_factorial_dense_m_{arm.lower()}_seed{seed}'
            settings = {k: v for k, v in cfg.items() if k not in {'config', 'output'}}
            if arm == 'SS':
                output = reference['output']
            else:
                output = str((ROOT / 'results' / name).relative_to(REPO))
                settings['ur1_arm'] = arm
            settings['output'] = output
            path = ROOT / 'configs' / f'{name}.json'
            write_json(path, settings)
            entries.append(dict(name=name, phase='factorial', arm=arm, seed=seed, reuse=arm == 'SS',
                output=output, config=str(path.relative_to(REPO)),
                reference_output=reference['output'], reference_artifacts=reference['artifact_hashes']))
    sources = dict(random='second_full_conv_cifar10_s_random_seed0', u2='second_final_u2_conv_cifar10_s_seed0')
    for seed in [1, 2]:
        for family, source in sources.items():
            directory = OLD / 'results' / source
            cfg = read(directory / 'training_config.json')
            assert cfg['num_iterations'] == 350000 and cfg['architecture'] == 'ClgnCifar10PaperSmall'
            name = f'ur1_full_conv_s_{family}_seed{seed}'
            settings = {k: v for k, v in cfg.items() if k not in {'config', 'output'}}
            settings.update(seed=seed, topology_seed=seed, ur1_arm='identity',
                output=str((ROOT / 'results' / name).relative_to(REPO)))
            path = ROOT / 'configs' / f'{name}.json'
            write_json(path, settings)
            entries.append(dict(name=name, phase='full_s', family=family, seed=seed, reuse=False,
                output=settings['output'], config=str(path.relative_to(REPO)),
                reference_output=str(directory.relative_to(REPO)),
                reference_config_sha256=sha(directory / 'training_config.json')))
    payload = dict(name='UR1', scope='validation_training_only_no_test_queries', entries=entries,
        protocol_sha256=sha(ROOT / 'PROTOCOL.md'),
        stages_3b_4='deferred_pending_complete_matrix_and_explicit_decision')
    write_json(ROOT / 'matrix.json', payload)
    return payload


if __name__ == '__main__':
    print('Prepared',len(prepare()['entries']),'logical training cells (3 reused, 13 new)')
