"""UR1 tests live outside the frozen incumbent test/hash namespace."""
import itertools
import numpy as np
import pytest
import torch

from torchlogix.parametrization import RawLUTParametrization
from torchlogix.topology import generate_dense_topology, image_input_semantics, packed_identity
from .construction import nominal_indices, degree_null, factorial_indices, validate_indices
from .functional_support_v2 import relevance, layer_analysis


def test_all_truth_tables_and_argmax_ties():
    tables = np.array(list(itertools.product([0, 1], repeat=4)))
    expected = []
    for table in tables:
        expected.append([any(table[x] != table[x ^ 2] for x in range(4)),
                         any(table[x] != table[x ^ 1] for x in range(4))])
    np.testing.assert_array_equal(relevance(tables), np.array(expected).T)
    assert np.bincount(relevance(tables).sum(0)).tolist() == [2, 4, 10]
    param = RawLUTParametrization(2)
    np.testing.assert_array_equal(param.get_luts(torch.eye(16)).numpy(), tables)
    np.testing.assert_array_equal(param.get_luts(torch.zeros(1, 16)).numpy(), [[0, 0, 0, 0]])
    weights = torch.zeros(1, 16); weights[0, 3:5] = 1
    np.testing.assert_array_equal(param.get_luts(weights).numpy(), [[0, 0, 1, 1]])


def test_exhaustive_global_dependence_is_upper_bounded():
    rng = np.random.default_rng(17)
    inputs = np.array(list(itertools.product([0, 1], repeat=4)))
    for _ in range(30):
        parents = [np.array([rng.choice(4, 2, replace=False) for _ in range(4)]).T for _ in range(3)]
        tables = [rng.integers(0, 2, (4, 4)) for _ in range(3)]
        _, support = layer_analysis(parents, tables, np.arange(4))
        values = inputs.copy()
        for edges, luts in zip(parents, tables):
            values = luts[np.arange(4)[None, :], 2 * values[:, edges[0]] + values[:, edges[1]]]
        for source in range(4):
            flipped = np.arange(16) ^ (1 << (3 - source))
            dependent = np.any(values != values[flipped], axis=0)
            assert np.all(~dependent | ((support[:, 0] >> np.uint64(source)) & 1).astype(bool))


def test_reconvergent_cancellation_and_threshold_source_collapse():
    # Two identical projections, then XOR: constant globally but two paths.
    parents = [np.array([[0, 0], [1, 1]]), np.array([[0], [1]])]
    tables = [np.array([[0, 0, 1, 1]] * 2), np.array([[0, 1, 1, 0]])]
    report, support = layer_analysis(parents, tables, np.array([0, 1]))
    assert support.tolist() == [[1]]
    assert report['input_bit_path_reachable_fraction'] == .5
    assert report['layers'][0]['unary_fraction'] == 1
    report, support = layer_analysis([np.array([[0], [1]])], [np.array([[0, 1, 1, 1]])], np.array([0, 0]))
    assert support.tolist() == [[1]] and report['n_sources'] == 1
    report, support = layer_analysis([np.array([[0], [1]])], [np.zeros((1, 4), int)], np.arange(2))
    assert support.tolist() == [[0]] and report['input_bit_path_reachable_fraction'] == 0
    assert report['layers'][0]['output_path_reachable_gate_fraction'] == 1


def test_native_random_repeated_parent_is_retained():
    report, support = layer_analysis([np.array([[0], [0]])], [np.array([[0, 1, 1, 0]])], np.arange(2))
    assert support.tolist() == [[1]]  # XOR(x,x) is constant; path support is only an upper bound.
    assert report['layers'][0]['locally_relevant_edges'] == 2


@pytest.mark.parametrize('n', range(2, 18))
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_nominal_identity_bounds_and_rng(n, seed):
    state = np.random.get_state()
    for width in sorted({1, n // 2, n, n + 1, 3 * n + 1}):
        for depth in [0, 3]:
            actual = nominal_indices(n, width, layer_index=depth, topology_seed=seed)
            old = generate_dense_topology(n, width, strategy='semantic_multiscale_nominal',
                topology_seed=seed, layer_index=depth, input_ancestry=packed_identity(n),
                allow_partial_input_coverage=True)
            np.testing.assert_array_equal(actual, old.indices)
            validate_indices(actual, n)
    current = np.random.get_state()
    assert state[0] == current[0] and np.array_equal(state[1], current[1]) and state[2:] == current[2:]


@pytest.mark.parametrize('shape', [(3, 1, 1, 3), (3, 4, 4, 3), (8, 2, 2, 1)])
def test_semantic_nominal_identity(shape):
    semantics = image_input_semantics(*shape, layout='channel_interleaved')
    for seed in range(3):
        for width in [1, 11, 128]:
            actual = nominal_indices(semantics.n_inputs, width, topology_seed=seed, input_semantics=semantics)
            old = generate_dense_topology(semantics.n_inputs, width, strategy='semantic_multiscale_nominal',
                topology_seed=seed, input_semantics=semantics, input_ancestry=semantics.source_ancestry(),
                allow_partial_input_coverage=True)
            np.testing.assert_array_equal(actual, old.indices)


def test_degree_null_and_factorial_components():
    reference = [nominal_indices(32, 128, topology_seed=3)] * 3
    state = np.random.get_state()
    arms = {arm: factorial_indices(reference, [32] * 3, arm, 2)[0] for arm in ['SS', 'SR', 'RS', 'RR']}
    for depth in range(3):
        a, record = degree_null(reference[depth], 32, seed=7)
        b, record2 = degree_null(reference[depth], 32, seed=7)
        np.testing.assert_array_equal(a, b); assert record == record2
        assert record['changed_pair_multiset']
        for slot in range(2):
            np.testing.assert_array_equal(np.bincount(a[slot], minlength=32), np.bincount(reference[depth][slot], minlength=32))
        np.testing.assert_array_equal(arms['SS'][depth], reference[depth])
        for left, right in ([('SS', 'SR'), ('RS', 'RR')] if depth == 0 else [('SS', 'RS'), ('SR', 'RR')]):
            np.testing.assert_array_equal(arms[left][depth], arms[right][depth])
    current = np.random.get_state()
    assert np.array_equal(state[1], current[1]) and state[2:] == current[2:]


def test_reject_invalid_semantics():
    for value in [np.array([[0], [0]]), np.array([[0], [2]]), np.array([[0.], [1.]])]:
        with pytest.raises(ValueError): validate_indices(value, 2)
    with pytest.raises(ValueError): relevance([[0, 1, 2, 0]])
    with pytest.raises(ValueError): layer_analysis([], [], np.arange(2))
    with pytest.raises(ValueError): layer_analysis([np.array([[0], [1]])], [np.zeros((2, 4))], np.arange(2))
    with pytest.raises(ValueError): nominal_indices(1, 4)
    with pytest.raises(ValueError): degree_null(np.array([[0], [1]]), 2, seed=0, rounds_per_slot=0)


def test_model_intervention_parameters_rng_metadata_and_metrics():
    from torchlogix.layers import LogicDense
    from .train_adapter import intervene, tensor_digest, analyze_intervened
    from torchlogix.topology import model_topology_metadata
    model=torch.nn.Sequential(*[LogicDense(32,32,connections_kwargs={'init_method':'semantic_multiscale_balanced','topology_seed':1,'layer_index':d}) for d in range(4)])
    initial=tensor_digest(model.parameters());rng=torch.get_rng_state()
    receipt=intervene(model,'RR',1)
    assert tensor_digest(model.parameters())==initial and torch.equal(rng,torch.get_rng_state())
    assert all(r['changed_pair_multiset'] for r in receipt['layers'])
    rows=analyze_intervened(model)
    assert len(rows)==4 and all(r['strategy']=='ur1_degree_null' for r in rows)
    assert all(r['strategy']=='ur1_degree_null' for r in model_topology_metadata(model))
    assert all(m.connections.strategy=='ur1_degree_null' for m in model)
    for r,m in zip(rows,model):
        assert r['in_dim']==32 and r['out_dim']==32
        validate_indices(m.connections.indices.numpy(),32)


def test_statistics_include_all_five_contrasts_and_ties():
    from .report import paired
    result=paired([-1,0,1])
    assert result['mean_pp']==0 and result['sample_sd_pp']==1
    assert result['wins']==result['ties']==result['losses']==1
    assert result['ci95_pp'][0]==pytest.approx(-2.4841377117)
    with pytest.raises(ValueError):paired([1,2])


def test_dataset_path_is_explicit_and_no_download(monkeypatch,tmp_path):
    from .train_adapter import dataset_integrity
    monkeypatch.delenv('DATASET_PATH',raising=False)
    with pytest.raises(RuntimeError,match='Set DATASET_PATH'):dataset_integrity()
    monkeypatch.setenv('DATASET_PATH',str(tmp_path))
    with pytest.raises(RuntimeError,match='refusing automatic download'):dataset_integrity()
    assert list(tmp_path.iterdir())==[]
