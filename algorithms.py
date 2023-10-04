from copy import deepcopy

def dfs_all_paths(graph, start, path=[], all_paths=[]):

    # If the current node is not in the graph, return an empty list
    if start not in graph.get_all_nodes():
        print('ERROR - Node not in graph')
        return []

    flight_neighbours = [n for n in graph.get_neighbours(start) if n[1] is not None]
    neighbour_map = {}
    
    for neigh in flight_neighbours:
        ground_neighbours = set()
        for gn in [
            n
            for n in graph.get_outgoing_nodes(neigh[0].get_name())
            if n.get_time() > neigh[0].get_time() 
        ]:
            for gn2 in ground_neighbours:
                if repr(gn) == repr(gn2):
                    break
            ground_neighbours.add(gn)
        neighbour_map[neigh] = ground_neighbours

    # BASE CASE
    import pdb; pdb.set_trace()
    
    if not list(neighbour_map.values()):
        import pdb; pdb.set_trace()
        return all_paths + path
    
    for neighbour, ground_neighs in neighbour_map.items():
        if neighbour[1] not in path:
            path += [neighbour[1]]
            for gn in ground_neighs:
                all_paths_copy = deepcopy(all_paths)
                all_paths.append(dfs_all_paths(graph, gn, path, all_paths_copy))
                
    return all_paths
                    