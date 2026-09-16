import { app } from '/scripts/app.js';

// app.graph._nodes exposes only the current graph level. Resolve through the
// root graph first so execution events from nodes inside subgraphs are routed
// to the node that actually produced them.
export function allGraphNodes() {
    const nodes=[];const seenGraphs=new Set();const seenNodes=new Set();
    const visit=graph=>{
        if(!graph||seenGraphs.has(graph))return;seenGraphs.add(graph);
        let values=[];
        if(Array.isArray(graph._nodes))values=graph._nodes;
        else if(Array.isArray(graph.nodes))values=graph.nodes;
        else if(graph.nodes instanceof Map)values=[...graph.nodes.values()];
        for(const node of values){
            if(!node||seenNodes.has(node))continue;seenNodes.add(node);nodes.push(node);
            for(const child of [node.subgraph,node.innerGraph,node.graphData?.graph])if(child&&child!==graph)visit(child);
        }
        const subgraphs=graph.subgraphs||graph.definitions?.subgraphs;
        if(Array.isArray(subgraphs))for(const child of subgraphs)visit(child);
        else if(subgraphs instanceof Map)for(const child of subgraphs.values())visit(child);
    };
    for(const graph of [app.rootGraph,app.graph,app.canvas?.graph,app.graph?.rootGraph])visit(graph);
    return nodes;
}

export function findRuntimeNode(runtimeId, predicate=()=>true) {
    const id=String(runtimeId??'');
    for(const graph of [app.rootGraph,app.graph,app.canvas?.graph,app.graph?.rootGraph]){
        const node=graph?.getNodeById?.(runtimeId)||graph?.getNodeById?.(id);
        if(node&&predicate(node))return node;
    }
    const nodes=allGraphNodes().filter(predicate);
    const exact=nodes.filter(n=>String(n.id)===id);
    if(exact.length===1)return exact[0];
    const tail=id.split(/[/:.]/).filter(Boolean).at(-1);
    const suffix=nodes.filter(n=>String(n.id)===tail);
    return suffix.length===1?suffix[0]:null;
}
