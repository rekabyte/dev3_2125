import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import sys
import copy


#Implementation de l'algo de Ford-Fulkerson pris d'ici: https://www.programiz.com/dsa/ford-fulkerson-algorithm
class Graph:

    def __init__(self, graph):
        self.graph = graph
        self. ROW = len(graph)


    # Using BFS as a searching algorithm 
    def searching_algo_BFS(self, s, t, parent):

        visited = [False] * (self.ROW)
        queue = []

        queue.append(s)
        visited[s] = True

        while queue:

            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return True if visited[t] else False

    # Applying fordfulkerson algorithm
    def ford_fulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.searching_algo_BFS(source, sink, parent):

            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Adding the path flows
            max_flow += path_flow

            # Updating the residual values of edges
            v = sink
            while(v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow


#Cree la matrice d'adjacence du graph a partir du fichier
def create_matrice_adjacence():
    file = open(sys.argv[1], "r")
    line = file.readline()
    line_elements = line.split(';')

    #Contiendera une liste de nos arcs:
    arcs = []

    matrice_taille = int(line_elements[0])
    nbre_arc_a_enlever = int(line_elements[1])

    #Cree une matrice d'adjacence de taille matrix_size*matrix_size:
    matrice_adjacence = [[0]*matrice_taille for _ in range(matrice_taille)]

    #On skip la premiere ligne:
    line = file.readline()

    #Index de la source du graph (correspond au premier noeud du fichier)
    index_source = int(line.split(';')[0])

    #Pour chaque ligne du fichier:
    while line:
        line_elements = line.split(';')
        #Enlever les \n qui sont en trop lors de la lecture des lignes
        line_elements = [line_elements.strip().replace('\n', '') for line_elements in line_elements]

        #Pour chaque element entre deux ';' dans la ligne:
        for index, element in enumerate(line_elements):

            #Ici on selectionne le premier nombre (qui correspond au sommet qui envoie des flots):
            if index == 0:
                sommet = int(line_elements[0])
                continue
            
            autre_sommet = int(element.split("(")[0])
            poids = int(element.split("(")[1].replace(")", ""))

            matrice_adjacence[sommet][autre_sommet] = poids

            #Liste d'arcs sans doublons:
            arcs.append((sommet, autre_sommet))

        #Index du puits (correspond au dernier noeud du fichier)
        index_puits = int(line.split(';')[0])

        line = file.readline()
    #Debug:
    #for ligne in matrice_adjacence:
        #print(ligne)
    file.close()
    
    return matrice_adjacence, arcs, nbre_arc_a_enlever,index_source, index_puits

#Visualise le graph et le sauvegarde sous le nom: "graph.png"
def save_graph(matrice_adjacence):
    G = nx.DiGraph(np.array(matrice_adjacence))
    #layout = nx.fruchterman_reingold_layout(G)
    layout = nx.spring_layout(G, seed=0)
    nx.draw(G, layout, node_size=1000, with_labels=True,font_weight='bold',font_size=15)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos=layout,edge_labels=labels)
    #plot.show()
    plot.savefig("Graph.png")

#Affiche et retourne le flot max qui peut passer par le graph (on precise quel est le sommet source et le sommet puits)
def calculer_flot_max(graph, source, sink):
    g = copy.deepcopy(graph)
    g = Graph(g)
    flot_max = g.ford_fulkerson(source, sink)
    return flot_max

def algo_retour_arriere(matrice_adj, arcs_du_graph):
    global dernier_flot, source, puits, arcs

    if (dernier_flot < flot_max) or (dernier_flot == 0):
        #print("condition de if rencontree")
        return matrice_adj
    
    prochains_graphs = []
    for arc in arcs_du_graph :
        sommet = arc[0]
        autre_sommet = arc[1]

        graph_a_explorer = copy.deepcopy(matrice_adj)

        #On enleve l'arc du graph (c.a.d de la matrice d'adjacence)
        graph_a_explorer[sommet][autre_sommet] = 0
        #On enleve l'arc de la liste des arcs
        nouveau_arcs = copy.deepcopy(arcs_du_graph)
        nouveau_arcs.remove((sommet, autre_sommet))
        prochains_graphs.append((graph_a_explorer, nouveau_arcs))
        
    for tuple in prochains_graphs:
        graph_actuel = tuple[0]
        arcs_actuels = tuple[1]

        dernier_flot = calculer_flot_max(graph_actuel, source, puits)
        if(dernier_flot < flot_max) or (dernier_flot == 0):
            resultat = algo_retour_arriere(graph_actuel, arcs_actuels)
            if resultat is not None:
                arc_enleves = [x for x in arcs if x not in arcs_actuels]
                print("Flot max de ce graphe: %d" %dernier_flot)
                print("Arc enelevees: ", arc_enleves)
                for line in graph_actuel:
                    print(line)
                print()
                print()
                #return resultat


    


matrice_adjacence, arcs, nbre_arcs_a_enlever, source, puits = create_matrice_adjacence()

flot_max = calculer_flot_max(matrice_adjacence, source, puits)
dernier_flot = flot_max

#print("Nombre d'arcs a enlever: ", nbre_arcs_a_enlever)
print("Flot max de base: %d" %flot_max)
algo_retour_arriere(matrice_adjacence, arcs)