import torch


def random_score_region_torch(graph, budget, node_type="note"):
    """
    Takes a Pytorch Geometric Heterogeneous Score graph and samples a random region of a given budget.

    If the budget is larger than the number of nodes, it returns all nodes.
    If the graph is heterogeneous, it samples from the given node type.
    """
    if budget >= graph[node_type].num_nodes:
        return torch.arange(graph[node_type].num_nodes)
    else:
        num_nodes = graph[node_type].num_nodes
        onsets = graph[node_type].onset_div
        random_note = torch.randint(0, num_nodes - budget, (1,))[0]
        min_index = torch.where(onsets == onsets[random_note])[0].min()
        max_budget_index = min_index + budget
        max_index_candidates = torch.where(onsets == onsets[max_budget_index])[0]
        max_index = max_index_candidates.max()
        # if max_index
        if max_index > max_budget_index:
            max_index = max_index_candidates.min() - 1
        note_out = torch.arange(min_index, max_index, dtype=torch.long)
        out = {"note": note_out}
        if node_type == "note":
            # fetch score section for beats and measures too, if they are included in the graph.
            if "beat" in graph.node_types:
                beats = graph["note"].beat_cluster[note_out]
                out["beat"] = torch.arange(beats.min(), beats.max(), dtype=torch.long)
            if "measure" in graph.node_types:
                beats = graph["note"].measure_cluster[note_out]
                out["beat"] = torch.arange(beats.min(), beats.max(), dtype=torch.long)
        return out







