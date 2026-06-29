import networkx as nx
from pyvis.network import Network

def create_knowledge_graph(row):

    G = nx.Graph()

    herb = row["Herb"]
    compound = row["Compound"]
    drug = row["Drug"]
    drug_class = row["Drug_Class"]
    disease = row["Target_Disease"]
    effect = row["Interaction_Effect"]
    mechanism = row["Mechanism"]
    risk = row["Risk"]

    # Nodes

    G.add_node(drug,
               color="#ef4444",
               title="Drug")

    G.add_node(herb,
               color="#10b981",
               title="Herb")

    G.add_node(compound,
               color="#f59e0b",
               title="Compound")

    G.add_node(drug_class,
               color="#3b82f6",
               title="Drug Class")

    G.add_node(disease,
               color="#8b5cf6",
               title="Disease")

    G.add_node(effect,
               color="#f97316",
               title="Interaction Effect")

    G.add_node(mechanism,
               color="#ec4899",
               title="Mechanism")

    G.add_node(risk,
               color="#dc2626",
               title="Risk")

    # Relationships

    G.add_edge(herb, compound,
               label="contains")

    G.add_edge(drug, drug_class,
               label="belongs_to")

    G.add_edge(herb, disease,
               label="used_for")

    G.add_edge(drug, herb,
               label="interacts_with")

    G.add_edge(drug, effect,
               label="causes")

    G.add_edge(effect, mechanism,
               label="via")

    G.add_edge(effect, risk,
               label="risk_level")

    return G


def save_graph(G):

    net = Network(
        height="700px",
        width="100%",
        bgcolor="#0B1220",
        font_color="white"
    )

    net.from_nx(G)

    net.repulsion(
        node_distance=250,
        spring_length=200
    )

    net.save_graph("knowledge_graph.html")

    return "knowledge_graph.html"