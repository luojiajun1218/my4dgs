"""Budgeted attribute selection under the unchanged motion frontier."""
import numpy as np

def strict_full_point_budget(population, requested):
    """Maximum selected points strictly below 10% of the whole population."""
    if (isinstance(population, bool) or not isinstance(population, int) or population < 1 or
            isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise ValueError('Invalid population or full-attribute point budget')
    limit = min(requested, (population - 1) // 10)
    if limit < 1:
        raise ValueError('Population too small for nonempty strict-below-10-percent selection')
    return limit

def select_recursive_error_regions(paths, owners, numerator, support, budget):
    paths=np.asarray(paths); owners=np.asarray(owners); numerator=np.asarray(numerator); support=np.asarray(support)
    if (paths.ndim != 2 or owners.shape != (len(paths),) or paths.dtype.kind not in 'iu' or
        owners.dtype.kind not in 'iu' or numerator.shape != owners.shape or support.shape != owners.shape or
        not np.isfinite(numerator).all() or not np.isfinite(support).all() or
        np.any(numerator < 0) or np.any(support < 0) or isinstance(budget,bool) or not isinstance(budget,int) or budget < 1 or
        np.any((paths == owners[:,None]).sum(axis=1) != 1)):
        raise ValueError('Invalid regional selection inputs')
    labels=np.full(len(owners),-1,dtype=np.int64); counts={}; oversized=[]
    def walk(rows,node,depth):
        if len(rows)<=budget or depth==paths.shape[1]-1:
            if node in counts: raise ValueError('Candidate node belongs to multiple owners')
            labels[rows]=node;counts[node]=len(rows)
            if len(rows)>budget: oversized.append(node)
            return
        children=paths[rows,depth+1]
        for child in np.unique(children): walk(rows[children==child],int(child),depth+1)
    for owner in np.unique(owners):
        rows=np.flatnonzero(owners==owner);depth=int(np.flatnonzero(paths[rows[0]]==owner)[0]);walk(rows,int(owner),depth)
    if np.any(labels<0): raise ValueError('Selection partition left points uncovered')
    nodes=np.array(sorted(counts),dtype=np.int64)
    inv=np.searchsorted(nodes,labels)
    sums=np.bincount(inv,weights=numerator,minlength=len(nodes));vis=np.bincount(inv,weights=support,minlength=len(nodes))
    sizes=np.array([counts[int(n)] for n in nodes],dtype=np.int64)
    scores=sums/sizes
    order=sorted((i for i in range(len(nodes)) if vis[i]>0 and sizes[i]<=budget),key=lambda i:(-scores[i],int(nodes[i])))
    chosen=[];total=0
    for i in order:
        if total+int(sizes[i])<=budget: chosen.append(int(nodes[i]));total+=int(sizes[i])
    selected=np.isin(labels,np.asarray(chosen,dtype=np.int64))
    if int(selected.sum())!=total or total>budget:raise RuntimeError('Selection budget exceeded')
    return dict(mask=selected,nodes=chosen,candidate_count=len(nodes),visible_count=len(order),oversized_count=len(oversized),
                selected_count=total,scores={str(int(nodes[i])):float(scores[i]) for i in order})
