"""CPU metrics for matched causal populations; exact empirical OT, not Sinkhorn."""

import json

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import kron, eye, vstack
import torch


def squared_cost(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Features must be nonempty [samples, dimension] matrices")
    if not len(x) or not len(y) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Empty or nonfinite features")
    return np.maximum((x*x).sum(1)[:, None] + (y*y).sum(1)[None, :] - 2*x@y.T, 0)


def exact_ot(x, y):
    cost = squared_cost(x, y)
    b, n = cost.shape
    if b == n:
        rows, columns = linear_sum_assignment(cost)
        plan = np.zeros_like(cost)
        plan[rows, columns] = 1 / b
        solver = "scipy.linear_sum_assignment; uniform equal-size empirical measures"
    else:
        constraints = vstack([kron(eye(b), np.ones((1, n))),
                              kron(np.ones((1, b)), eye(n))], format="csr")
        target = np.r_[np.full(b, 1/b), np.full(n, 1/n)]
        solution = linprog(cost.ravel(), A_eq=constraints, b_eq=target,
                           bounds=(0, None), method="highs")
        if not solution.success:
            raise RuntimeError(solution.message)
        plan = solution.x.reshape(b, n)
        solver = "scipy.linprog HiGHS; uniform empirical measures"
    residual = max(abs(plan.sum(1)-1/b).max(), abs(plan.sum(0)-1/n).max())
    if residual > 1e-8 or plan.min() < -1e-10:
        raise ArithmeticError("OT marginal constraints failed")
    return float((cost*plan).sum()), plan, solver


def gaussian_w2(x, y):
    """PSD moment surrogate with population (1/B) covariance; low-rank exact algebra."""
    xc, yc = x-x.mean(0), y-y.mean(0)
    cross = xc @ yc.T / np.sqrt(len(x)*len(y))
    trace_root = np.linalg.svd(cross, compute_uv=False).sum()
    return float(max(0, np.square(x.mean(0)-y.mean(0)).sum()
                     + np.square(xc).sum()/len(x) + np.square(yc).sum()/len(y)
                     - 2*trace_root))


def moments(x):
    centered = x-x.mean(0)
    values = np.square(np.linalg.svd(centered, compute_uv=False))/len(x)
    trace = values.sum()
    probabilities = values / max(trace, 1e-30)
    return {"mean_norm": float(np.linalg.norm(x.mean(0))),
            "mean_sample_norm": float(np.linalg.norm(x, axis=1).mean()),
            "variance_trace": float(trace),
            "effective_rank": float(np.exp(-(probabilities*np.log(probabilities+1e-30)).sum()))
                if trace > 1e-20 else 0.0}


def tcost(x, y):
    return ((x*x).sum(1)[:, None]+(y*y).sum(1)[None, :]-2*x@y.T).clamp_min(0)


def cosine(x, y):
    return float((x*y).sum() / (x.norm()*y.norm()).clamp_min(1e-30))


def kernel_probe(x, y, sigma2, plan, reference=None, text=None):
    """Autograd includes both generated sides of k_gg; report raw and self-normalized."""
    z = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    gt = torch.tensor(y, dtype=torch.float64)
    if reference is None:
        joint = z
        kgr = torch.exp(-tcost(z, gt)/(2*sigma2)).mean()
        krr = torch.exp(-tcost(gt, gt)/(2*sigma2)).mean()
        dimension = z.shape[1]
    else:
        beta = float(reference["beta"])
        joint = torch.cat([z, beta*torch.tensor(text, dtype=torch.float64)], 1)
        landmarks = reference["Z"].double()
        sigma2 = float(reference["sigma"])**2
        kgr = (torch.exp(-tcost(joint, landmarks)/(2*sigma2)) @ reference["alpha"].double()).mean()
        krr = z.new_tensor(float(reference["k_rr"]))
        dimension = joint.shape[1]
    kgg = torch.exp(-tcost(joint, joint)/(2*sigma2)).mean()
    raw = kgg - 2*kgr + krr
    attr, = torch.autograd.grad(-2*kgr, z, retain_graph=True)
    rep, = torch.autograd.grad(kgg, z, retain_graph=True)
    grad, = torch.autograd.grad(raw, z)
    mass = torch.tensor(plan, dtype=torch.float64)
    ot_grad = 2*(mass.sum(1)[:, None]*z.detach()-mass@gt)
    paired_grad = 2*(z.detach()-gt)/len(z) if z.shape == gt.shape else None
    result = {"raw_biased_mmd2": float(raw.detach()), "k_gg": float(kgg.detach()),
              "k_gr": float(kgr.detach()), "k_rr": float(krr), "sigma2": sigma2,
              "kernel_feature_dimension": dimension,
              "visual_gradient_frobenius": float(grad.norm()),
              "attraction_gradient_frobenius": float(attr.norm()),
              "repulsion_gradient_frobenius": float(rep.norm()),
              "attraction_repulsion_norm_ratio": float(attr.norm()/rep.norm().clamp_min(1e-30)),
              "gradient_cosine_with_visual_ot": cosine(grad, ot_grad),
              "attraction_cosine_with_visual_ot": cosine(attr, ot_grad),
              "self_normalized_visual_gradient_frobenius": float(grad.norm()/(raw.detach().abs()+1e-7))}
    if paired_grad is not None:
        result["gradient_cosine_with_paired"] = cosine(grad, paired_grad)
    return result


def compare(x, y, sigma2, paired=True, reference=None, text=None, gt_text=None):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    value, plan, solver = exact_ot(x, y)
    result = {"empirical_w2_squared": value, "solver": solver,
              "gaussian_moment_w2_squared": gaussian_w2(x, y), "moments": moments(x),
              "visual_mmd": kernel_probe(x, y, sigma2, plan),
              "ot_plan": plan.tolist()}
    mean_shift = float(np.square(x.mean(0)-y.mean(0)).sum())
    result["squared_mean_shift"] = mean_shift
    result["centered_empirical_w2_squared"] = max(0., value-mean_shift)
    if paired:
        result.update(paired_squared_distance=float(np.square(x-y).sum(1).mean()),
                      paired_cosine=float(((x*y).sum(1)/np.maximum(
                          np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1), 1e-30)).mean()),
                      visual_ot_same_prompt_mass=float(np.trace(plan)))
        paired_gradient = (x-y)/len(x)
        ot_gradient = plan.sum(1)[:, None]*x-plan@y
        result["paired_ot_gradient_cosine"] = float((paired_gradient*ot_gradient).sum()/max(
            np.linalg.norm(paired_gradient)*np.linalg.norm(ot_gradient), 1e-30))
    if reference is not None:
        beta = float(reference["beta"])
        joint_cost, joint_plan, _ = exact_ot(np.c_[x, beta*text], np.c_[y, beta*gt_text])
        result["joint_empirical_w2_squared"] = joint_cost
        result["joint_plan_visual_cost"] = float((joint_plan*squared_cost(x, y)).sum())
        result["joint_plan_text_cost"] = float((joint_plan*squared_cost(beta*text, beta*gt_text)).sum())
        if paired:
            result["joint_ot_same_prompt_mass"] = float(np.trace(joint_plan))
        result["official_joint_nystrom_mmd"] = kernel_probe(x, y, sigma2, plan, reference, text)
    return result


def load_rows(directory, indices, field="global"):
    rows = [torch.load(directory / f"row{i:04d}.pt", map_location="cpu", weights_only=True)
            for i in indices]
    return np.concatenate([r[field].float().numpy() for r in rows], axis=0).astype(np.float64), rows


def boundary_summary(rows):
    if not rows or 'boundary' not in rows[0]:
        return None
    result = {}
    for space in ('rgb', 'latent'):
        values = [row['boundary'][space] for row in rows]
        b = float(np.mean([v['boundary_mean'] for v in values]))
        w = float(np.mean([v['interior_mean'] for v in values]))
        result[space] = {'boundary_mean': b, 'interior_mean': w,
            'ratio_of_means': b/max(w, 1e-12),
            'median_video_ratio': float(np.median([v['ratio'] for v in values])),
            'video_ratios': [v['ratio'] for v in values],
            'mean_transition_magnitudes': np.mean([v['transition_magnitudes'] for v in values], axis=0).tolist()}
    return result


def evaluate(output, reference_path=None, initializations=None):
    initializations = initializations or ['random', 'native', 'tf600']
    torch.set_num_threads(4)
    manifest = json.loads((output/"manifest.json").read_text())
    count, indices = manifest["count"], manifest["indices"]
    y, rows = load_rows(output/"gt", indices[:count])
    y_null, null_rows = load_rows(output/"gt", indices[count:])
    text = np.concatenate([r["text"].numpy() for r in rows])
    null_text = np.concatenate([r["text"].numpy() for r in null_rows])
    distances = squared_cost(y, y)
    sigma2 = float(np.median(distances[np.triu_indices(count, 1)]))
    if sigma2 <= 0:
        raise ValueError("Degenerate GT-only median kernel bandwidth")
    reference = torch.load(reference_path, map_location="cpu", weights_only=False) if reference_path else None
    results = {"protocol": manifest,
               "metric_notes": {"primary": "exact finite-sample W2^2 on raw pooled V-JEPA features",
                                "mmd_bandwidth": "sigma^2 = median off-diagonal squared GT distance, fixed across all groups",
                                "mmd_sigma2": sigma2, "paired": "same prompt; SF samples are not ground-truth aligned trajectories",
                                "official_mmd": "separate frozen joint Nyström reference; not the OT empirical target",
                                "repeats": "noise repeats on one fixed prompt subset; not independent dataset/training repeats"},
               "gt_moments": moments(y),
               "gt_vs_disjoint_gt": compare(y_null, y, sigma2, paired=False,
                                            reference=reference, text=null_text, gt_text=text),
               "gt_boundary": boundary_summary(rows), "initializations": {}}
    populations = {}
    for initialization in initializations:
        per_seed = {}
        for seed in manifest["seeds"]:
            directory = output/initialization/f"seed{seed}"
            if not all((directory/f"row{i:04d}.pt").exists() for i in indices[:count]):
                continue
            x, generated_rows = load_rows(directory, indices[:count])
            populations[(initialization, seed)] = x
            entry = compare(x, y, sigma2, reference=reference, text=text, gt_text=text)
            entry['boundary'] = boundary_summary(generated_rows)
            # Temporal pooling stays secondary: squared distance averaged over tubelets.
            temporal = np.concatenate([r["temporal"].float().numpy() for r in generated_rows])
            temporal_gt = np.concatenate([r["temporal"].float().numpy() for r in rows])
            scale = np.sqrt(temporal.shape[1])
            entry["temporal_empirical_w2_squared"] = exact_ot(
                temporal.reshape(count, -1)/scale, temporal_gt.reshape(count, -1)/scale)[0]
            entry["latent_rms_mean"] = float(np.mean([r["latent_rms"] for r in generated_rows]))
            entry["rgb_frame_delta_mean"] = float(np.mean([r["rgb_frame_delta"] for r in generated_rows]))
            entry["variance_trace_ratio_to_gt"] = entry["moments"]["variance_trace"]/results["gt_moments"]["variance_trace"]
            per_seed[str(seed)] = entry
        if per_seed:
            values = [v["empirical_w2_squared"] for v in per_seed.values()]
            results["initializations"][initialization] = {"per_seed": per_seed,
                "w2_squared_mean": float(np.mean(values)), "w2_squared_min": min(values),
                "w2_squared_max": max(values), "complete_seeds": len(per_seed)}
    results["complete"] = all(results["initializations"].get(k, {}).get("complete_seeds") == len(manifest["seeds"])
                              for k in initializations)
    if results["complete"]:
        subset_count = min(8, count)
        subsets = [np.random.default_rng(700+i).choice(count, subset_count, replace=False) for i in range(32)]
        repeated = {k: [exact_ot(populations[(k, seed)][subset], y[subset])[0]
                       for seed in manifest["seeds"] for subset in subsets]
                    for k in initializations}
        results["subsample_sensitivity"] = {
            "note": f"32 shared subsets per noise seed, sampled from the same {count} prompts; dependent repeats, not a population confidence interval",
            "count": subset_count, "subsets": [s.tolist() for s in subsets],
            "w2_squared": {k: {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "values": v}
                           for k, v in repeated.items()},
        }
        for left, right in [('tf100', 'native'), ('tf600', 'native'), ('native', 'random'), ('sf_paired', 'sf_mmd')]:
            if left in repeated and right in repeated:
                results['subsample_sensitivity'][f'fraction_{left}_below_{right}'] = float(np.mean(
                    np.array(repeated[left]) < np.array(repeated[right])))
        if 'tf600' in repeated and 'native' in repeated:
            results['subsample_sensitivity']['fraction_cca_below_native'] = results['subsample_sensitivity']['fraction_tf600_below_native']
    path = output/"metrics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)
    print(json.dumps({"complete": results["complete"], "gt_vs_gt_w2_squared": results["gt_vs_disjoint_gt"]["empirical_w2_squared"],
                      "groups": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                                 for k, v in results["initializations"].items()}}, indent=2))
    return results


def self_test():
    x = np.array([[0., 0.], [1., 0.], [0., 2.]])
    assert abs(exact_ot(x, x[::-1])[0]) < 1e-12
    assert abs(exact_ot(x, x+[3., -4.])[0]-25) < 1e-10
    assert abs(gaussian_w2(x, x+[3., -4.])-25) < 1e-10
    assert abs(exact_ot(np.array([[0.], [2.]]), np.array([[1.]]))[0]-1) < 1e-10
    value, plan, _ = exact_ot(x, x+[.3, -.4])
    assert abs(value-.25) < 1e-10
    eps = 1e-5
    z = x+[.3, -.4]
    perturb = np.zeros_like(z)
    perturb[0, 0] = eps
    finite_difference = (exact_ot(z+perturb, x)[0]-exact_ot(z-perturb, x)[0])/(2*eps)
    expected = 2*(plan.sum(1)[:, None]*z-plan@x)[0, 0]
    assert abs(finite_difference-expected) < 1e-8
    probe = kernel_probe(x, x, 1., exact_ot(x, x)[1])
    assert abs(probe["raw_biased_mmd2"]) < 1e-12
    assert probe["visual_gradient_frobenius"] < 1e-12
    # Analytic one-dimensional Gaussian moments with unequal sample counts.
    a, b = np.array([[-1.], [1.]]), np.array([[-2.], [0.], [2.]])
    assert abs(gaussian_w2(a, b)-(1-np.sqrt(8/3))**2) < 1e-10
    print("PASS: identity/permutation, translation, rectangular marginals, OT finite-difference gradient, MMD cancellation, Gaussian moments")
