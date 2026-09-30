"""Read-only summaries of the active selection tree; no motion configuration."""
from collections import Counter

NAMES = {0: 'KEEP', 1: 'SHARED', 2: 'SPLIT'}


def summarize(packet):
    if packet.get('mode') == 'learned_selective':
        return summarize_learned(packet)
    state = packet['state']
    states = state['states'].cpu().tolist()
    evidence = state['evidence'].cpu().tolist()
    owner = state['owner'].cpu().tolist()
    children, depths = packet['children'], packet['depths']
    terminals = set(owner)
    # Only ancestors of current owners are active. Collapsed descendants are stale.
    parent = {child: node for node, row in enumerate(children) for child in row}
    active = set(terminals)
    for node in terminals:
        while node in parent:
            node = parent[node]
            active.add(node)

    def counts(nodes):
        result = dict(Counter(NAMES[states[n]] for n in nodes))
        return {name: result.get(name, 0) for name in NAMES.values()}

    def group(nodes):
        nodes = list(nodes)
        count = counts(nodes)
        return dict(denominator=len(nodes), counts=count,
                    fractions={k: v / len(nodes) if nodes else 0 for k, v in count.items()},
                    unknown=sum(evidence[n] == 0 for n in nodes),
                    depth_counts=dict(sorted(Counter(depths[n] for n in nodes).items())),
                    depth_by_state={name: dict(sorted(Counter(depths[n] for n in nodes if states[n] == value).items()))
                                    for value, name in NAMES.items()},
                    unknown_depth_counts=dict(sorted(Counter(depths[n] for n in nodes if evidence[n] == 0).items())))

    coverage = {}
    records = packet['comparisons']
    for kind, internal in [('leaf', False), ('internal', True)]:
        eligible = [n for n, row in enumerate(children) if bool(row) == internal and evidence[n] == 2]
        compared = [r for r in records if bool(children[r['node']]) == internal]
        budget = packet['config']['max_' + kind + '_comparisons']
        coverage[kind] = dict(eligible_changed=len(eligible), compared=len(compared), budget=budget,
                              coverage=len(compared) / len(eligible) if eligible else None,
                              budget_utilization=len(compared) / budget,
                              shared_blocked=sum(bool(r.get('shared_blocked')) for r in compared),
                              shared_blocked_fraction=sum(bool(r.get('shared_blocked')) for r in compared) / len(compared) if compared else None,
                              blocked_reasons=dict(Counter(reason for r in compared if r.get('shared_blocked')
                                                           for reason in r.get('shared_blocked_reasons', ['unknown_in_legacy_record']))))
    events = packet.get('topology_events', [])
    clone_added = sum(e.get('added', 0) for e in events if e['event'] == 'clone')
    net = sum(e.get('after', 0) - e.get('before', 0) for e in events if 'after' in e)
    return dict(active_tree=group(sorted(active)), terminal_regions=group(sorted(terminals)),
                gs_weighted=group(owner), comparison_coverage=coverage,
                final_review=packet.get('final_review'),
                topology=dict(base_gs=len(owner), clone_added=clone_added,
                              incremental_net_change=net,
                              incoming_merge_added=sum(e.get('merged', 0) for e in events),
                              incoming_merge_pruned=sum(e.get('pruned', 0) for e in events),
                              incoming_merge_pruned_keep=sum(e.get('pruned_keep', 0) for e in events),
                              incoming_merge_pruned_unknown=sum(e.get('pruned_unknown', 0) for e in events),
                              events=events,
                              note='Net events cannot recover simultaneous gross additions/deletions.'),
                semantics='SPLIT is an internal decision, not a GS owner category. UNKNOWN is evidence, overlaps KEEP.',
                active_nodes=sorted(active), terminal_nodes=sorted(terminals))


def concise(summary):
    if summary.get('mode') == 'learned_selective':
        active = summary['active_tree']
        gs = summary['gs_weighted']
        return ('active ' + '/'.join(f'{name}={active["fractions"][name]:.1%}' for name in NAMES.values()) +
                '; GS ' + '/'.join(f'{name}={gs["fractions"][name]:.1%}' for name in ('KEEP', 'SHARED')) +
                f'; supported candidates={summary["observed_candidate_nodes"]}/{summary["candidate_nodes"]}')
    def render(key):
        row = summary[key]
        values = '/'.join(f'{k}={row["counts"][k]}({row["fractions"][k]:.1%})'
                          for k in NAMES.values() if key == 'active_tree' or k != 'SPLIT')
        return values + f' n={row["denominator"]} UNKNOWN={row["unknown"]}'

    coverage = summary['comparison_coverage']
    compared = []
    for kind in ('leaf', 'internal'):
        row = coverage[kind]
        fraction = f'{row["coverage"]:.1%}' if row['coverage'] is not None else 'n/a'
        compared.append(f'{kind}={row["compared"]}/{row["eligible_changed"]}({fraction}),budget={row["budget"]}')
    internal = coverage['internal']
    review = summary['final_review'] or {}
    topology = summary['topology']
    return ('active ' + render('active_tree') + '; terminals ' + render('terminal_regions') +
            '; GS ' + render('gs_weighted') +
            '\nSHARED terminal depths=' + str(summary['terminal_regions']['depth_by_state']['SHARED']) +
            '; comparisons ' + '; '.join(compared) +
            f'; internal blocked={internal["shared_blocked"]}/{internal["compared"]}' +
            f'\nreview rollback={review.get("rolled_back", "pending")} all_KEEP={review.get("all_keep_fallback", "pending")}; '
            f'GS clone+={topology["clone_added"]} incremental_net={topology["incremental_net_change"]} '
            f'incoming_merge+={topology["incoming_merge_added"]} prune-={topology["incoming_merge_pruned"]} '
            f'pruned_KEEP={topology["incoming_merge_pruned_keep"]} UNKNOWN={topology["incoming_merge_pruned_unknown"]}')


def summarize_learned(packet):
    state = packet['state']
    states = state['states'].cpu().tolist()
    owner = state['owner'].cpu().tolist()
    terminals = set(owner)
    parent = {child: node for node, row in enumerate(packet['children']) for child in row}
    active = set(terminals)
    for n in terminals:
        while n in parent:
            n = parent[n]
            active.add(n)
    def group(nodes):
        nodes = list(nodes)
        counts = {name: sum(states[n] == value for n in nodes) for value, name in NAMES.items()}
        return dict(denominator=len(nodes), counts=counts,
                    fractions={name: count / len(nodes) if nodes else 0 for name, count in counts.items()},
                    depth_by_state={name: dict(sorted(Counter(packet['depths'][n] for n in nodes if states[n] == value).items()))
                                               for value, name in NAMES.items()})
    fixed_leaf = packet['config'].get('fixed_leaf_ablation', False)
    return dict(mode='learned_selective', fixed_leaf_ablation=fixed_leaf, active_tree=group(sorted(active)),
                terminal_regions=group(sorted(terminals)), gs_weighted=group(owner),
                active_nodes=sorted(active), terminal_nodes=sorted(terminals),
                observed_candidate_nodes=int((state['observation_count'] > 0).sum()),
                candidate_nodes=len(states), last_metrics=packet.get('last_metrics', {}),
                stage_seconds=packet.get('stage_seconds', {}),
                final_review=packet.get('final_review'),
                semantics=('Fixed real tree leaves all use their own SHARED rigid motion; internal nodes SPLIT. No gates or structure costs. '
                           if fixed_leaf else 'Per-frame optimized gates. ') + 'SPLIT counts internal active regions, never Gaussian owners. '
                          'Observation is renderer geometric support, not physical visibility or change evidence. '
                          + ('' if fixed_leaf else 'Expected costs are effective reachable counts per initial root; unsupported gates are detached SHARED and still charged.'))
